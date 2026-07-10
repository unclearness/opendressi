// Multi-view silhouette optimization: deform an icosphere toward the bunny
// using vertex-position gradients, with two selectable techniques:
//
//   --technique=hardsoftras (default)
//     HardSoftRas (Dressi paper Alg.2, K=1): soft rasterization of
//     GPU-enlarged triangles; the silhouette probability is
//     sigmoid(dist / (r/7)) with the paper's Eq.5 edge/non-edge split.
//
//   --technique=aa
//     Screen-space AA (Dr.Hair, arXiv:2403.17496 Eq.3-5): hard
//     rasterization plus a differentiable blend across triangle-ID
//     boundaries; gradients flow through the pixel-to-edge distance.
//
// The whole iteration lives on the GPU (the paper's transfer model: upload
// once, download at the end): the 3D positions are the optimizer target,
// per-view projections / HardSoftRas soft geometry / uniform-Laplacian and
// normal-consistency regularizers / Adam (with GPU-resident state via
// DressiAD::addUpdate) are all graph ops, and the executor refreshes the
// vertex buffers from the computed clip images. Per-iteration CPU traffic
// is zero; loss logging and the live viewers read back small debug data at
// intervals only.

#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "../common/asset_utils.h"
#include "../common/geometry_utils.h"
#include "../common/vk_viewer.h"
#include "dressi/dressi.h"

using namespace dressi;
using namespace dressi_examples;

namespace {

struct Options {
    std::string data_dir = "data/bunny";
    std::string technique = "hardsoftras";
    uint32_t n_views = 8;
    uint32_t screen = 128;
    int n_iters = 300;
    uint32_t sphere_level = 3;
    float radius_px = 3.f;
    uint32_t samples = 8;  // stochastic backward samples per face
    uint32_t peels = 1;    // hardsoftras depth-peeling layers (K)
    float lr = 0.01f;
    // Regularizer weights relative to the data-gradient RMS; negative =
    // per-technique default (hardsoftras: 0.15/0.4 -- its sigmoid-band
    // gradients are weaker, so lighter smoothing fits noticeably better;
    // aa: 0.5/0.5)
    float laplacian = -1.f;
    float normal = -1.f;
};

Options ParseArgs(int argc, char* argv[]) {
    Options opt;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        std::string key, value;
        if (const size_t eq = a.find('=');
            a.rfind("--", 0) == 0 && eq != std::string::npos) {
            key = a.substr(0, eq);
            value = a.substr(eq + 1);
        }
        if (key == "--technique") {
            opt.technique = value;
        } else if (key == "--views") {
            opt.n_views = uint32_t(std::stoul(value));
        } else if (key == "--screen") {
            opt.screen = uint32_t(std::stoul(value));
        } else if (key == "--iters") {
            opt.n_iters = std::stoi(value);
        } else if (key == "--sphere-level") {
            opt.sphere_level = uint32_t(std::stoul(value));
        } else if (key == "--samples") {
            opt.samples = uint32_t(std::stoul(value));
        } else if (key == "--peels") {
            opt.peels = std::max(1u, uint32_t(std::stoul(value)));
        } else if (key == "--lr") {
            opt.lr = std::stof(value);
        } else if (key == "--laplacian") {
            opt.laplacian = std::stof(value);
        } else if (key == "--normal") {
            opt.normal = std::stof(value);
        } else if (a.rfind("--", 0) != 0) {
            opt.data_dir = a;
        } else {
            throw DressiError("unknown option: " + a);
        }
    }
    if (opt.technique != "hardsoftras" && opt.technique != "aa") {
        throw DressiError("--technique must be hardsoftras or aa");
    }
    const bool hs = opt.technique == "hardsoftras";
    if (opt.laplacian < 0.f) {
        opt.laplacian = hs ? 0.15f : 0.5f;
    }
    if (opt.normal < 0.f) {
        opt.normal = hs ? 0.4f : 0.5f;
    }
    return opt;
}

struct View {
    Mat4 mvp;
    Variable tgt_clip{nullptr};  // static bunny clip leaf
    Variable target{nullptr};
    Variable pred{nullptr};
};

float MaskIoU(const CpuImage& a, const CpuImage& b) {
    size_t inter = 0, uni = 0;
    for (size_t i = 0; i < a.data.size(); i++) {
        const bool pa = a.data[i] > 0.5f;
        const bool pb = b.data[i] > 0.5f;
        inter += (pa && pb);
        uni += (pa || pb);
    }
    return uni ? float(inter) / float(uni) : 1.f;
}

}  // namespace

int main(int argc, char* argv[]) try {
    spdlog::cfg::load_env_levels();
    const Options opt = ParseArgs(argc, argv);
    const ImgSize screen = {opt.screen, opt.screen};
    const bool hardsoft = opt.technique == "hardsoftras";
    const std::string out_dir = "silopt_out_" + opt.technique;
    std::filesystem::create_directories(out_dir);

    // ------------------------- Assets / geometry -------------------------
    const Mesh bunny = LoadObjMesh(opt.data_dir + "/bunny.obj");
    Mesh sphere = GenerateIcosphere(opt.sphere_level);
    for (float& p : sphere.pos.data) {
        p *= 0.55f;
    }
    const uint32_t n_verts = sphere.numVertices();
    const uint32_t n_faces = sphere.numFaces();
    spdlog::info("technique {}: bunny {}v/{}f -> sphere {}v/{}f, {} views, "
                 "{}x{} screen",
                 opt.technique, bunny.numVertices(), bunny.numFaces(),
                 n_verts, n_faces, opt.n_views, screen.w, screen.h);

    // --------------------------- Static leaves ---------------------------
    // Topology / adjacency textures (uploaded once)
    const CpuImage b_vf_img = VertexFacesTex(bunny.faces,
                                             bunny.numVertices());
    const CpuImage s_vf_img = VertexFacesTex(sphere.faces, n_verts);
    const CpuImage s_vn_img = VertexNeighborsTex(sphere.faces, n_verts);
    const CpuImage s_fa_img = FaceNeighborsTex(sphere.faces);
    // Bunny (target)
    Variable b_ones(FLOAT, {bunny.numVertices(), 1});
    Variable b_faces_i(IVEC3, {bunny.numFaces(), 1});
    Variable b_faces_f(VEC3, {bunny.numFaces(), 1});
    Variable b_vtx_faces(FLOAT, b_vf_img.getImgSize());
    // Sphere topology
    Variable s_ones(FLOAT, {n_verts, 1});
    Variable s_faces_i(IVEC3, {n_faces, 1});
    Variable s_faces_f(VEC3, {n_faces, 1});
    Variable s_vtx_faces(FLOAT, s_vf_img.getImgSize());
    Variable s_vtx_nbrs(FLOAT, s_vn_img.getImgSize());
    Variable s_face_adj(VEC3, {n_faces, 1});
    Variable s_face_id(FLOAT, {n_faces * 3, 1});   // hardsoftras
    Variable s_faces_soft(IVEC3, {n_faces, 1});    // hardsoftras

    // Optimization state (GPU-resident; uploaded once, updated by the
    // end-of-frame copy-backs)
    Variable pos(VEC3, {n_verts, 1});     // the optimizer target
    Variable adam_m(VEC3, {n_verts, 1});
    Variable adam_v(VEC3, {n_verts, 1});
    Variable adam_t(FLOAT, {1, 1});
    // Static seed for the frozen target branch (adam_t would make it
    // dynamic and defeat the reactive-cache pruning)
    Variable zero_seed(FLOAT, {1, 1});

    // ------------------------------ Views --------------------------------
    const Mat4 proj =
            PerspectiveVk(45.f * 3.14159265f / 180.f, 1.f, 0.1f, 10.f);
    std::vector<View> views(opt.n_views);
    Variables view_losses;
    const float sigma_scale = 7.f / opt.radius_px;
    for (uint32_t v = 0; v < opt.n_views; v++) {
        View& view = views[v];
        const float ang = 2.f * 3.14159265f * float(v) / float(opt.n_views);
        const float elev = (v % 2 == 0) ? 0.55f : -0.35f;
        const std::array<float, 3> eye = {1.6f * std::sin(ang), elev,
                                          1.6f * std::cos(ang)};
        view.mvp = Mul(proj, LookAt(eye, {0, 0, 0}, {0, 1, 0}));

        // Target: bunny silhouette rendered in-graph, frozen (the static
        // branch is pruned by the reactive cache after warm-up)
        view.tgt_clip = Variable(VEC4, {bunny.numVertices(), 1});
        Variable t_mask =
                F::Rasterize(view.tgt_clip, b_ones, b_faces_i, screen);
        if (hardsoft) {
            view.target = F::StopGradient(t_mask);  // hard 0/1 mask
        } else {
            Variable t_tri = F::RasterizeFaceId(view.tgt_clip, b_ones,
                                                b_faces_i, screen);
            // AA'd target so pred can match it exactly at convergence
            view.target = F::StopGradient(
                    F::AntiAlias(t_mask, t_tri, view.tgt_clip, b_faces_f,
                                 b_vtx_faces, zero_seed, 0));
        }

        // Prediction: the per-view projection is IN-GRAPH, so the clip
        // positions are computed variables and the executor refreshes the
        // vertex buffers from their images each frame
        Variable clip_v = TransformToClipVar(pos, view.mvp);
        if (hardsoft) {
            // Paper-faithful HardSoftRas: the rasterizer only supplies the
            // (non-differentiable) face-ID buffer; the pixel-to-edge
            // distance is built from ordinary image-space ops over
            // FaceFetch'd corner positions, so the whole backward is
            // generated by the AD ("no special backward declarations").
            // FaceFetch's backward is the paper's inverse-table-style
            // stochastic gather (Alg.1 philosophy).
            Variable soft_clip =
                    F::SoftClip(clip_v, s_faces_f, screen, opt.radius_px);
            // Per-face corner clip positions (differentiable, exact bwd)
            Variable tri0 =
                    F::LookupFaces(clip_v, s_faces_f, s_vtx_faces, 0);
            Variable tri1 =
                    F::LookupFaces(clip_v, s_faces_f, s_vtx_faces, 1);
            Variable tri2 =
                    F::LookupFaces(clip_v, s_faces_f, s_vtx_faces, 2);
            Variable pc = F::ScreenCoord(screen);
            Variable zero = F::Float(0.f);
            Variable one = F::Float(1.f);

            // D_k = mask * sigmoid(dist / sigma) built from a face-ID
            // buffer with image-space ops (backward: FaceFetch sampling +
            // AD; adam_t doubles as the per-iteration jitter seed)
            const auto soft_prob = [&](const Variable& idx) {
                Variable mask = F::Greater(idx, F::Float(0.5f));
                const auto fetch = [&](const Variable& tri) {
                    return F::FaceFetch(tri, idx, tri0, tri1, tri2, adam_t,
                                        opt.radius_px, opt.samples);
                };
                const auto screen_of = [&](const Variable& c) {
                    Variable w = F::GetW(c) + (1.f - mask);
                    Variable sx = (F::GetX(c) / w * 0.5f + 0.5f) *
                                  float(screen.w);
                    Variable sy = (F::GetY(c) / w * 0.5f + 0.5f) *
                                  float(screen.h);
                    return F::Vec2(sx, sy);
                };
                Variable s0 = screen_of(fetch(tri0));
                Variable s1 = screen_of(fetch(tri1));
                Variable s2 = screen_of(fetch(tri2));
                const auto edge_dist = [&](const Variable& a,
                                           const Variable& b) {
                    Variable e = b - a;
                    Variable t = F::Clamp(F::Dot(pc - a, e) /
                                                  (F::Dot(e, e) + 1e-12f),
                                          zero, one);
                    Variable dd = pc - (a + t * e);
                    return F::Sqrt(F::Dot(dd, dd) + 1e-12f);
                };
                const auto edge_cross = [&](const Variable& a,
                                            const Variable& b) {
                    Variable e = b - a;
                    return F::GetX(e) * (F::GetY(pc) - F::GetY(a)) -
                           F::GetY(e) * (F::GetX(pc) - F::GetX(a));
                };
                Variable dmin = F::Min(F::Min(edge_dist(s0, s1),
                                              edge_dist(s1, s2)),
                                       edge_dist(s2, s0));
                Variable cr0 = edge_cross(s0, s1);
                Variable cr1 = edge_cross(s1, s2);
                Variable cr2 = edge_cross(s2, s0);
                Variable inside =
                        F::Min(F::Step(zero, cr0) * F::Step(zero, cr1) *
                                               F::Step(zero, cr2) +
                                       F::Step(zero, -cr0) *
                                               F::Step(zero, -cr1) *
                                               F::Step(zero, -cr2),
                               one);
                Variable dist = (inside * 2.f - 1.f) * dmin;
                return mask * F::Sigmoid(dist * sigma_scale);
            };

            // Alg.2 depth peeling: peel k discards fragments at or in
            // front of peel k-1's shifted depth; Eq.6 silhouette blending
            // I_s = 1 - prod_k (1 - D_k)
            Variable one_minus = one;
            Variable prev_shift(nullptr);
            for (uint32_t peel = 0; peel < opt.peels; peel++) {
                Variable rs = F::StopGradient(
                        peel == 0
                                ? F::RasterizeSoft(soft_clip, s_face_id,
                                                   s_faces_soft, clip_v,
                                                   s_faces_f, s_vtx_faces,
                                                   screen, opt.radius_px)
                                : F::RasterizeSoft(soft_clip, s_face_id,
                                                   s_faces_soft, clip_v,
                                                   s_faces_f, s_vtx_faces,
                                                   prev_shift, screen,
                                                   opt.radius_px));
                // Next peel's threshold: this layer's Eq.3 shifted depth
                // rebuilt from the output channels (0 at background)
                Variable dist_c = F::GetX(rs);
                Variable cov = F::GetW(rs);
                Variable ge = F::Step(zero, dist_c);
                prev_shift =
                        cov * (ge * (F::Clamp(F::GetZ(rs), zero, one) *
                                     0.5f) +
                               (1.f - ge) *
                                       (F::Clamp(-dist_c *
                                                         (1.f /
                                                          opt.radius_px),
                                                 zero, one) *
                                                0.5f +
                                        0.5f));
                Variable idx = (F::GetY(rs) + 1.f) * cov;
                one_minus = one_minus * (1.f - soft_prob(idx));
            }
            Variable soft_sil = 1.f - one_minus;
            // Paper Eq.5 spirit: non-edge pixels keep the hard value (kills
            // the interior seams of the per-face distance); the sigmoid
            // band supplies the gradients around the silhouette
            Variable hard =
                    F::Rasterize(clip_v, s_ones, s_faces_i, screen);
            view.pred = F::Max(F::StopGradient(hard), soft_sil);
        } else {
            Variable mask = F::Rasterize(clip_v, s_ones, s_faces_i, screen);
            Variable tri = F::RasterizeFaceId(clip_v, s_ones, s_faces_i,
                                              screen);
            // Stochastic vertex backward (paper pattern), jittered by the
            // in-graph iteration counter
            view.pred = F::AntiAlias(mask, tri, clip_v, s_faces_f,
                                     s_vtx_faces, adam_t, opt.samples);
        }
        // Per-pixel loss IMAGE (no reduction to {1,1}, per the paper);
        // the 1/N scale keeps gradients equal to a per-view Mean
        Variable diff = view.target - view.pred;
        view_losses.push_back(diff * diff *
                              (1.f / float(screen.w * screen.h)));
    }
    Variable loss = F::SumPixelWise(view_losses);  // {S,S} loss image
    Variable pos_mut = pos;
    pos_mut.setRequiresGradRecursively();

    // ----------------- In-graph Adam + regularizers ----------------------
    DressiAD ad;
    ad.setLossVar(loss);
    bool state_registered = false;
    ad.setOptimizer([&](Variables xs, Variables gxs) {
        DRESSI_CHECK(xs.size() == 1 && xs[0] == pos,
                     "expected pos as the only optimizer target");
        // Regularizers, weighted relative to the data-gradient RMS so
        // smoothing fades as the fit converges
        const auto rms = [](const Variable& x) {
            return F::Sqrt(F::Mean(x * x) + 1e-24f);  // FLOAT {1,1}
        };
        Variable g = gxs[0];
        Variable data_rms = rms(g);
        Variable g_lap = pos - F::VertexNeighborMean(pos, s_vtx_nbrs);
        Variable g_nrm = F::NormalConsistencyVertexGrad(
                F::NormalConsistencyFaceTerm(pos, s_faces_f, s_face_adj),
                pos, s_faces_f, s_vtx_faces);
        g = g + g_lap * (data_rms / rms(g_lap) * opt.laplacian) +
            g_nrm * (data_rms / rms(g_nrm) * opt.normal);
        // Adam with GPU-resident state
        const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        Variable t_new = adam_t + 1.f;
        Variable m_new = adam_m * b1 + g * (1.f - b1);
        Variable v_new = adam_v * b2 + g * g * (1.f - b2);
        Variable bc1 = 1.f - F::Pow(F::Float(b1), t_new);
        Variable bc2 = 1.f - F::Pow(F::Float(b2), t_new);
        Variable step = (m_new / bc1) /
                        (F::Sqrt(v_new / bc2) + eps) * opt.lr;
        if (!state_registered) {
            ad.addUpdate(adam_m, m_new);
            ad.addUpdate(adam_v, v_new);
            ad.addUpdate(adam_t, t_new);
            state_registered = true;
        }
        return Variables{pos - step};
    });
    for (auto& view : views) {
        ad.markOutput(view.target);
        ad.markOutput(view.pred);
    }

    // ------------------- One-time uploads (then GPU-only) -----------------
    const CpuImage b_ones_img(bunny.numVertices(), 1, 1, 1.f);
    const CpuImage s_ones_img(n_verts, 1, 1, 1.f);
    ad.sendImg(b_ones, b_ones_img);
    ad.sendImg(b_faces_i, bunny.faces);
    ad.sendImg(b_faces_f, bunny.faces);
    ad.sendImg(b_vtx_faces, b_vf_img);
    ad.sendImg(s_ones, s_ones_img);
    ad.sendImg(s_faces_i, sphere.faces);
    ad.sendImg(s_faces_f, sphere.faces);
    ad.sendImg(s_vtx_faces, s_vf_img);
    ad.sendImg(s_vtx_nbrs, s_vn_img);
    ad.sendImg(s_face_adj, s_fa_img);
    for (auto& view : views) {
        ad.sendImg(view.tgt_clip, TransformToClip(bunny.pos, view.mvp));
    }
    if (hardsoft) {
        // Constant per-face vertex attribute / sequential soft faces
        CpuImage face_id_img(n_faces * 3, 1, 1);
        CpuImage faces_soft_img(n_faces, 1, 3);
        for (uint32_t f = 0; f < n_faces; f++) {
            for (uint32_t k = 0; k < 3; k++) {
                face_id_img.at(f * 3 + k, 0, 0) = float(f);
                faces_soft_img.at(f, 0, k) = float(f * 3 + k);
            }
        }
        ad.sendImg(s_face_id, face_id_img);
        ad.sendImg(s_faces_soft, faces_soft_img);
    }
    ad.sendImg(pos, sphere.pos);
    ad.sendImg(adam_m, CpuImage(n_verts, 1, 3, 0.f));
    ad.sendImg(adam_v, CpuImage(n_verts, 1, 3, 0.f));
    ad.sendImg(adam_t, CpuImage(1, 1, 1, 0.f));
    if (!hardsoft) {
        ad.sendImg(zero_seed, CpuImage(1, 1, 1, 0.f));
    }

    // ------------------------------ Viewer --------------------------------
    const uint32_t tile_cols = 4;
    const uint32_t tile_rows = (opt.n_views + tile_cols - 1) / tile_cols;
    const uint32_t win_w = screen.w * tile_cols;
    VkViewer viewer_target(win_w, screen.h * tile_rows,
                           "bunny silhouette targets");
    VkViewer viewer_pred(win_w, screen.h * tile_rows,
                         "optimized silhouette");
    // Fixed side-by-side layout (deterministic placement for recording)
    viewer_target.setPosition(80, 80);
    viewer_pred.setPosition(80 + int(win_w) + 16, 80);
    bool viewer_open = viewer_target.valid() && viewer_pred.valid();
    if (!viewer_open) {
        spdlog::warn("live viewer unavailable; continuing headless");
    }
    const int view_interval = 5;
    CpuImage target_tile;

    // ---------------------------- Optimization ----------------------------
    // The loop body is execStep() alone; loss logging and the viewers are
    // small debug readbacks at intervals
    const auto loss_value = [&]() {
        const CpuImage li = ad.recvImg(loss);
        double s = 0.0;
        for (float v : li.data) {
            s += v;
        }
        return float(s);
    };

    using Clock = std::chrono::steady_clock;
    double opt_ms = 0.0;
    double view_ms = 0.0;
    float first_loss = 0.f;
    float last_loss = 0.f;
    for (int iter = 0; iter < opt.n_iters; iter++) {
        const auto t0 = Clock::now();
        ad.execStep();
        opt_ms += std::chrono::duration<double, std::milli>(Clock::now() -
                                                            t0)
                          .count();

        if (iter == 0) {
            first_loss = loss_value();
            std::vector<CpuImage> targets, preds;
            for (auto& view : views) {
                targets.push_back(ad.recvImg(view.target));
                preds.push_back(ad.recvImg(view.pred));
            }
            target_tile = TileImages(targets, tile_cols);
            SaveImagePng(out_dir + "/targets.png", target_tile);
            SaveImagePng(out_dir + "/pred_first.png",
                         TileImages(preds, tile_cols));
        }
        if (viewer_open && iter % view_interval == 0) {
            const auto tv0 = Clock::now();
            std::vector<CpuImage> preds;
            for (auto& view : views) {
                preds.push_back(ad.recvImg(view.pred));
            }
            viewer_pred.setTitle(fmt::format("optimized silhouette  iter {}",
                                             iter));
            viewer_open = viewer_pred.update(TileImages(preds, tile_cols)) &&
                          viewer_target.update(target_tile);
            view_ms += std::chrono::duration<double, std::milli>(
                               Clock::now() - tv0)
                               .count();
        }
        if (iter % 20 == 0 || iter == opt.n_iters - 1) {
            spdlog::info("iter {:4d}  loss {:.8f}  opt {:.1f} ms/iter"
                         "  (view {:.1f} ms/iter excluded)",
                         iter, loss_value(), opt_ms / double(iter + 1),
                         view_ms / double(iter + 1));
        }
    }

    // --------------------- Final download / outputs ----------------------
    ad.execStep();
    last_loss = loss_value();
    const CpuImage final_pos = ad.recvImg(pos);
    sphere.pos.data = final_pos.data;

    std::vector<CpuImage> preds;
    float mean_iou = 0.f;
    for (auto& view : views) {
        preds.push_back(ad.recvImg(view.pred));
        mean_iou += MaskIoU(preds.back(), ad.recvImg(view.target));
    }
    mean_iou /= float(opt.n_views);
    SaveImagePng(out_dir + "/pred_last.png", TileImages(preds, tile_cols));
    SaveObjMesh(out_dir + "/optimized.obj", sphere);

    const auto face_adj = BuildFaceAdjacency(sphere.faces);
    const uint32_t flipped =
            CountFlippedFacePairs(sphere.pos, sphere.faces, face_adj);
    spdlog::info("loss {:.8f} -> {:.8f} ({:.1f}x), mean silhouette IoU"
                 " {:.4f}, flipped face pairs {} / {}",
                 first_loss, last_loss,
                 first_loss / std::max(last_loss, 1e-12f), mean_iou,
                 flipped, face_adj.size());
    spdlog::info("outputs in {}/", out_dir);
    // Note: the hardsoftras loss has an inherent floor (the sigmoid ramp of
    // pred vs the hard 0/1 target along the silhouette band)
    return (last_loss < first_loss / 5.f && mean_iou > 0.85f) ? 0 : 1;
} catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 2;
}
