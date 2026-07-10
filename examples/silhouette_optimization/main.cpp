// Multi-view silhouette optimization: deform an icosphere toward the bunny
// using vertex-position gradients, with two selectable techniques:
//
//   --technique=hardsoftras (default)
//     HardSoftRas (Dressi paper Alg.2, K=1): CPU-enlarged soft triangles
//     are hardware-rasterized; the fragment shader outputs the signed
//     screen-space distance to the hard triangle and the silhouette
//     probability is sigmoid(dist / (r/7)). Gradients flow through the
//     dist channel back to the hard clip positions.
//
//   --technique=aa
//     Screen-space AA (Dr.Hair, arXiv:2403.17496 Eq.3-5): hard
//     rasterization plus a differentiable blend across triangle-ID
//     boundaries; gradients flow through the pixel-to-edge distance.
//
// Both paths read the per-view clip-position gradients back each
// iteration (identity optimizer capturing the backward graph), chain-rule
// them through the fixed projections, and run CPU Adam with a relative
// uniform-Laplacian regularizer on the shared 3D vertex positions.

#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
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
    float lr = 0.01f;
    float laplacian = 0.5f;  // relative to the data-gradient RMS
    float normal = 0.5f;     // normal consistency, same relative scale
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
    return opt;
}

struct View {
    Mat4 mvp;
    // Bunny target branch (static leaves, sent once)
    Variable tgt_clip{nullptr};
    Variable target{nullptr};
    // Sphere branch (per-iteration leaves)
    Variable clip{nullptr};       // {V,1} VEC4, requires grad
    Variable soft_clip{nullptr};  // {3F,1} VEC4 (hardsoftras only)
    Variable pred{nullptr};
    Variable grad{nullptr};  // captured d loss / d clip
    CpuImage clip_img;
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

float Rms(const CpuImage& img) {
    double s = 0.0;
    for (float v : img.data) {
        s += double(v) * v;
    }
    return float(std::sqrt(s / double(img.data.size())));
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

    // -------------------------- Shared leaves ----------------------------
    // Bunny (target) static leaves
    Variable b_ones(FLOAT, {bunny.numVertices(), 1});
    Variable b_faces_i(IVEC3, {bunny.numFaces(), 1});
    Variable b_faces_f(VEC3, {bunny.numFaces(), 1});
    // Sphere static leaves
    Variable s_ones(FLOAT, {n_verts, 1});
    Variable s_faces_i(IVEC3, {n_faces, 1});
    Variable s_faces_f(VEC3, {n_faces, 1});
    Variable s_face_id(FLOAT, {n_faces * 3, 1});   // hardsoftras
    Variable s_faces_soft(IVEC3, {n_faces, 1});    // hardsoftras

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

        // Target: bunny silhouette rendered in-graph, frozen
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
                    F::AntiAlias(t_mask, t_tri, view.tgt_clip, b_faces_f));
        }

        // Prediction: sphere silhouette, differentiable in the clip leaf
        view.clip = Variable(VEC4, {n_verts, 1});
        if (hardsoft) {
            view.soft_clip = Variable(VEC4, {n_faces * 3, 1});
            Variable out = F::RasterizeSoft(view.soft_clip, s_face_id,
                                            s_faces_soft, view.clip,
                                            s_faces_f, screen,
                                            opt.radius_px);
            Variable soft_sil = F::StopGradient(F::GetW(out)) *
                                F::Sigmoid(F::GetX(out) * sigma_scale);
            // Paper Eq.5 spirit: non-edge pixels keep the hard value (kills
            // the interior seams of the per-face distance at K=1); the
            // sigmoid band supplies the gradients around the silhouette.
            // view.clip doubles as the vertex-buffer leaf here.
            Variable hard =
                    F::Rasterize(view.clip, s_ones, s_faces_i, screen);
            view.pred = F::Max(F::StopGradient(hard), soft_sil);
        } else {
            Variable mask = F::Rasterize(view.clip, s_ones, s_faces_i,
                                         screen);
            Variable tri = F::RasterizeFaceId(view.clip, s_ones, s_faces_i,
                                              screen);
            view.pred = F::AntiAlias(mask, tri, view.clip, s_faces_f);
        }
        Variable diff = view.target - view.pred;
        view_losses.push_back(F::Mean(diff * diff));

        Variable clip_mut = view.clip;
        clip_mut.setRequiresGradRecursively();
    }
    Variable loss = F::SumPixelWise(view_losses);

    // -------------------- Dressi-AD setup (grad capture) ------------------
    DressiAD ad;
    ad.setLossVar(loss);
    bool grads_marked = false;
    ad.setOptimizer([&](Variables xs, Variables gxs) {
        Variables updated;
        for (size_t i = 0; i < xs.size(); i++) {
            updated.push_back(xs[i] + F::Float(0.f) * gxs[i]);
        }
        if (!grads_marked) {
            for (size_t i = 0; i < xs.size(); i++) {
                for (auto& view : views) {
                    if (xs[i] == view.clip) {
                        view.grad = gxs[i];
                        ad.markOutput(gxs[i]);
                    }
                }
            }
            grads_marked = true;
        }
        return updated;
    });
    for (auto& view : views) {
        ad.markOutput(view.target);
        ad.markOutput(view.pred);
    }

    // Static uploads (clean after the first iteration -> reactive cache
    // prunes the target branch)
    const CpuImage b_ones_img(bunny.numVertices(), 1, 1, 1.f);
    const CpuImage s_ones_img(n_verts, 1, 1, 1.f);
    ad.sendImg(b_ones, b_ones_img);
    ad.sendImg(b_faces_i, bunny.faces);
    ad.sendImg(b_faces_f, bunny.faces);
    ad.sendImg(s_ones, s_ones_img);
    ad.sendImg(s_faces_i, sphere.faces);
    ad.sendImg(s_faces_f, sphere.faces);
    for (auto& view : views) {
        ad.sendImg(view.tgt_clip, TransformToClip(bunny.pos, view.mvp));
    }
    if (hardsoft) {
        // face_id / sequential soft faces are constant; soft clip changes
        const SoftGeometry sg0 = BuildSoftGeometry(
                TransformToClip(sphere.pos, views[0].mvp), sphere.faces,
                screen, opt.radius_px);
        ad.sendImg(s_face_id, sg0.face_id);
        ad.sendImg(s_faces_soft, sg0.faces);
    }

    // Per-iteration sphere leaf upload from the current 3D positions
    const auto send_sphere = [&]() {
        for (auto& view : views) {
            view.clip_img = TransformToClip(sphere.pos, view.mvp);
            ad.sendImg(view.clip, view.clip_img);
            if (hardsoft) {
                const SoftGeometry sg = BuildSoftGeometry(
                        view.clip_img, sphere.faces, screen, opt.radius_px);
                ad.sendImg(view.soft_clip, sg.clip);
            }
        }
    };
    send_sphere();

    // ------------------------------ Viewer --------------------------------
    const uint32_t tile_cols = 4;
    const uint32_t tile_rows = (opt.n_views + tile_cols - 1) / tile_cols;
    VkViewer viewer_target(screen.w * tile_cols, screen.h * tile_rows,
                           "bunny silhouette targets");
    VkViewer viewer_pred(screen.w * tile_cols, screen.h * tile_rows,
                         "optimized silhouette");
    bool viewer_open = viewer_target.valid() && viewer_pred.valid();
    if (!viewer_open) {
        spdlog::warn("live viewer unavailable; continuing headless");
    }
    const int view_interval = 5;
    CpuImage target_tile;

    // ---------------------------- Optimization ----------------------------
    const auto adj = BuildVertexAdjacency(sphere.faces, n_verts);
    const auto face_adj = BuildFaceAdjacency(sphere.faces);
    AdamState adam;
    std::vector<float> grad_acc(size_t(n_verts) * 3, 0.f);

    using Clock = std::chrono::steady_clock;
    double opt_ms = 0.0;
    double view_ms = 0.0;
    float first_loss = 0.f;
    float last_loss = 0.f;
    for (int iter = 0; iter < opt.n_iters; iter++) {
        const auto t0 = Clock::now();
        ad.execStep();
        last_loss = ad.recvImg(loss).data[0];
        if (iter == 0) {
            first_loss = last_loss;
        }

        // d loss / d 3D position, summed over views
        std::fill(grad_acc.begin(), grad_acc.end(), 0.f);
        for (auto& view : views) {
            const CpuImage g_clip = ad.recvImg(view.grad);
            const CpuImage g_pos = ChainRuleClipToPos(g_clip, view.mvp);
            for (size_t i = 0; i < grad_acc.size(); i++) {
                grad_acc[i] += g_pos.data[i];
            }
        }
        // Regularizers (uniform Laplacian + normal consistency), weighted
        // relative to the data gradient so smoothing fades as the fit
        // converges. The Laplacian evens vertex spacing; normal consistency
        // penalizes creases and flipped faces the Laplacian cannot see.
        CpuImage g_data(n_verts, 1, 3);
        g_data.data = grad_acc;
        const float data_rms = Rms(g_data);
        const CpuImage g_lap = UniformLaplacianGrad(sphere.pos, adj, 1.f);
        const float lap_scale =
                opt.laplacian * data_rms / std::max(Rms(g_lap), 1e-12f);
        const CpuImage g_nrm =
                NormalConsistencyGrad(sphere.pos, sphere.faces, face_adj,
                                      1.f);
        const float nrm_scale =
                opt.normal * data_rms / std::max(Rms(g_nrm), 1e-12f);
        for (size_t i = 0; i < grad_acc.size(); i++) {
            grad_acc[i] += lap_scale * g_lap.data[i] +
                           nrm_scale * g_nrm.data[i];
        }
        AdamStep(adam, sphere.pos.data, grad_acc, opt.lr);
        send_sphere();
        const auto t1 = Clock::now();
        opt_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (iter == 0) {
            std::vector<CpuImage> targets;
            for (auto& view : views) {
                targets.push_back(ad.recvImg(view.target));
            }
            target_tile = TileImages(targets, tile_cols);
            SaveImagePng(out_dir + "/targets.png", target_tile);
            std::vector<CpuImage> preds;
            for (auto& view : views) {
                preds.push_back(ad.recvImg(view.pred));
            }
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
                         iter, last_loss, opt_ms / double(iter + 1),
                         view_ms / double(iter + 1));
        }
    }

    // ------------------------------ Outputs -------------------------------
    ad.execStep();
    last_loss = ad.recvImg(loss).data[0];
    std::vector<CpuImage> preds;
    float mean_iou = 0.f;
    for (auto& view : views) {
        preds.push_back(ad.recvImg(view.pred));
        mean_iou += MaskIoU(preds.back(), ad.recvImg(view.target));
    }
    mean_iou /= float(opt.n_views);
    SaveImagePng(out_dir + "/pred_last.png", TileImages(preds, tile_cols));
    SaveObjMesh(out_dir + "/optimized.obj", sphere);

    const uint32_t flipped =
            CountFlippedFacePairs(sphere.pos, sphere.faces, face_adj);
    spdlog::info("loss {:.8f} -> {:.8f} ({:.1f}x), mean silhouette IoU"
                 " {:.4f}, flipped face pairs {} / {}",
                 first_loss, last_loss,
                 first_loss / std::max(last_loss, 1e-12f), mean_iou,
                 flipped, face_adj.size());
    spdlog::info("outputs in {}/", out_dir);
    // Note: the hardsoftras loss has an inherent floor (the sigmoid ramp of
    // pred vs the hard 0/1 target along the silhouette band), so judge by
    // the reduction factor plus the achieved mask IoU
    return (last_loss < first_loss / 5.f && mean_iou > 0.85f) ? 0 : 1;
} catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 2;
}
