// Multi-view environment-map optimization on DamagedHelmet: recover the
// (small) equirectangular HDR env map from rendered images while geometry,
// materials and the light stay at ground truth. The forward shader is
// exactly the viewer's; per view the G-buffer is rasterized once and shaded
// twice (GT env maps -> target, optimized env -> pred).
//
// Gradient paths into the env leaf (the paper has no recipe here; this is
// the documented Dressi-style solution):
//   - background pixels: EquirectSample(env, view_ray) -> exact bilinear
//     transpose (each env texel scans the screen; O(env * pixels), which is
//     why the OPTIMIZED env is a small leaf, 64x32 by default)
//   - diffuse IBL: EquirectSample(irr, n) -> transpose -> IrradianceConv
//     transpose (exact: the conv is a deterministic linear map) -> AvgPool
//     backward -> env
//   - specular IBL: the deterministic F::PrefilterConv chain (GGX-NDF
//     convolution, exact transpose) — the helmet is mostly metal, so the
//     glossy reflections constrain large parts of the sphere that the
//     background rays never see
// Because the env leaf is dirty every iteration, the IBL precompute of the
// optimized branch is NOT pruned (it re-runs per iter); the GT branch stays
// static and is pruned as usual.

#include "run.h"

#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "../common/asset_utils.h"
#include "../common/bench.h"
#include "../common/pbr_graph.h"
#include "dressi/dressi.h"

using namespace dressi;
using namespace dressi_examples;

namespace {

CpuImage Vec3Img(float x, float y, float z) {
    CpuImage img(1, 1, 3);
    img.data = {x, y, z};
    return img;
}

CpuImage DownsampleTo(CpuImage img, uint32_t max_w) {
    while (img.width > max_w) {
        img = DownsampleHalf(img);
    }
    return img;
}

// Reinhard + gamma for saving HDR maps as viewable PNGs
CpuImage TonemapForView(CpuImage img) {
    for (float& v : img.data) {
        v = std::pow(v / (v + 1.f), 1.f / 2.2f);
    }
    return img;
}

// Nearest upscale so the tiny env maps are inspectable
CpuImage Upscale(const CpuImage& img, uint32_t k) {
    CpuImage out(img.width * k, img.height * k, img.channels);
    for (uint32_t y = 0; y < out.height; y++) {
        for (uint32_t x = 0; x < out.width; x++) {
            for (uint32_t c = 0; c < img.channels; c++) {
                out.at(x, y, c) = img.at(x / k, y / k, c);
            }
        }
    }
    return out;
}

struct View {
    Variable clip{nullptr};
    PbrFrameLeaves frame;
    Variable target{nullptr};
    Variable pred{nullptr};
    CpuImage clip_img;
};

}  // namespace

int dressi_examples::RunPbrEnvmapOptimization(
        const std::vector<std::string>& args, ExampleHost& host) try {
    spdlog::cfg::load_env_levels();

    std::string mesh_path = "data/DamagedHelmet/glTF/DamagedHelmet.gltf";
    std::string env_path = "data/suburban_garden_4k.exr";
    std::string out_dir = "pbrenv_out";
    uint32_t size = 192, env_w = 64, n_views = 4;
    int n_iters = 1500;
    int view_interval = 1;  // live-viewer refresh cadence (iters); 0 = off
    float lr = 0.05f;
    // Spatial smoothness prior on the recovered env (0 = off). The 4-view
    // inverse is underdetermined: weakly observed texels (equirect poles +
    // the sphere's nadir) fill with per-texel noise that the >=0 clamp
    // rectifies into black speckle. A 2x2-block-variance penalty pulls each
    // texel toward its local mean, filling those holes.
    float env_reg = 0.05f;
    // Non-negativity clamp on the optimized radiance (physically correct).
    // --clamp=0 is a diagnostic: without it the underconstrained texels
    // random-walk around their init instead of piling up at black.
    bool clamp_nonneg = true;
    for (const std::string& arg : args) {
        if (arg.rfind("--mesh=", 0) == 0) {
            mesh_path = arg.substr(7);
        } else if (arg.rfind("--env=", 0) == 0) {
            env_path = arg.substr(6);
        } else if (arg.rfind("--out-dir=", 0) == 0) {
            out_dir = arg.substr(10);
        } else if (arg.rfind("--size=", 0) == 0) {
            size = uint32_t(std::stoi(arg.substr(7)));
        } else if (arg.rfind("--envres=", 0) == 0) {
            env_w = uint32_t(std::stoi(arg.substr(9)));
        } else if (arg.rfind("--views=", 0) == 0) {
            n_views = uint32_t(std::stoi(arg.substr(8)));
        } else if (arg.rfind("--iters=", 0) == 0) {
            n_iters = std::stoi(arg.substr(8));
        } else if (arg.rfind("--view-interval=", 0) == 0) {
            view_interval = std::stoi(arg.substr(16));
        } else if (arg.rfind("--lr=", 0) == 0) {
            lr = std::stof(arg.substr(5));
        } else if (arg.rfind("--env-reg=", 0) == 0) {
            env_reg = std::stof(arg.substr(10));
        } else if (arg.rfind("--clamp=", 0) == 0) {
            clamp_nonneg = std::stoi(arg.substr(8)) != 0;
        }
    }
    const ImgSize screen = {size, size};
    std::filesystem::create_directories(out_dir);

    // ------------------------- Load assets (CPU) -------------------------
    const GltfScene gltf = LoadGltfScene(mesh_path);
    const CpuImage env_gt_img = DownsampleTo(LoadImageExr(env_path), env_w);
    const ImgSize env_size = env_gt_img.getImgSize();
    const uint32_t tex_max = 512;
    const CpuImage m_albedo = DownsampleTo(SrgbToLinear(gltf.base_color),
                                           tex_max);
    const CpuImage m_mr = DownsampleTo(gltf.metallic_roughness, tex_max);
    const CpuImage m_nrm = DownsampleTo(gltf.normal_map, tex_max);
    const CpuImage m_ao = DownsampleTo(gltf.occlusion, tex_max);
    const CpuImage m_emis = DownsampleTo(SrgbToLinear(gltf.emissive),
                                         tex_max);
    spdlog::info("mesh: {} verts, {} faces; optimizing env at {}x{}",
                 gltf.mesh.numVertices(), gltf.mesh.numFaces(), env_size.w,
                 env_size.h);

    const uint32_t n_verts = gltf.mesh.numVertices();
    CpuImage uvh_img(n_verts, 1, 4);
    CpuImage tangent_img(n_verts, 1, 3);
    for (uint32_t v = 0; v < n_verts; v++) {
        uvh_img.at(v, 0, 0) = gltf.mesh.uv.at(v, 0, 0);
        uvh_img.at(v, 0, 1) = gltf.mesh.uv.at(v, 0, 1);
        uvh_img.at(v, 0, 2) = 1.f;
        uvh_img.at(v, 0, 3) = gltf.tangent.at(v, 0, 3);
        for (uint32_t c = 0; c < 3; c++) {
            tangent_img.at(v, 0, c) = gltf.tangent.at(v, 0, c);
        }
    }

    // ------------------------ Graph leaf variables ------------------------
    PbrGeometryLeaves geom;
    geom.uvh = Variable(VEC4, {n_verts, 1});
    geom.normal = Variable(VEC3, {n_verts, 1});
    geom.tangent = Variable(VEC3, {n_verts, 1});
    geom.wpos = Variable(VEC3, {n_verts, 1});
    geom.faces = Variable(IVEC3, {gltf.mesh.numFaces(), 1});

    PbrMaterialLeaves mat;
    mat.albedo = Variable(VEC3, m_albedo.getImgSize());
    mat.mr = Variable(VEC3, m_mr.getImgSize());
    mat.normal_map = Variable(VEC3, m_nrm.getImgSize());
    mat.ao = Variable(VEC3, m_ao.getImgSize());
    mat.emissive = Variable(VEC3, m_emis.getImgSize());
    // No texture gradients here -> never-uploaded dummy inverse-UV table
    Variable dummy_inv_uv(VEC4, m_albedo.getImgSize());

    // Both branches use the deterministic (differentiable) prefilter so
    // no formulation mismatch is pushed into the recovered env; the GT
    // branch is static and pruned after warmup regardless
    Variable env_gt(VEC3, env_size);
    Variable env_opt(VEC3, env_size);
    const PbrIblMaps ibl_gt = BuildPbrIblMaps(env_gt, true);
    const PbrIblMaps ibl_opt = BuildPbrIblMaps(env_opt, true);

    const float fov = 45.f * 3.14159265f / 180.f;
    const float tan_half = std::tan(fov * 0.5f);
    const Mat4 proj = PerspectiveVk(fov, 1.f, 0.1f, 10.f);
    Variable light_pos(VEC3, {1, 1});
    Variable light_col(VEC3, {1, 1});

    // Alternate the elevation sign so the background rays cover both the
    // upper and lower halves of the sphere (all-above cameras never see
    // the sky and the recovery stays dark there)
    const auto view_eye = [&](uint32_t vi) {
        const float ang = 2.f * 3.14159265f * float(vi) / float(n_views);
        const float ey = (vi % 2 == 0) ? 0.85f : -0.85f;
        return std::array<float, 3>{2.2f * std::sin(ang) * 0.92f, ey,
                                    2.2f * std::cos(ang) * 0.92f};
    };

    std::vector<View> views(n_views);
    Variables view_losses;
    for (uint32_t vi = 0; vi < n_views; vi++) {
        View& view = views[vi];
        const std::array<float, 3> eye = view_eye(vi);
        const Mat4 vm = LookAt(eye, {0, 0, 0}, {0, 1, 0});
        view.clip = Variable(VEC4, {n_verts, 1});
        view.clip_img = TransformToClip(gltf.mesh.pos, Mul(proj, vm));

        view.frame.cam_pos = Variable(VEC3, {1, 1});
        view.frame.cam_fwd = Variable(VEC3, {1, 1});
        view.frame.cam_right = Variable(VEC3, {1, 1});
        view.frame.cam_up = Variable(VEC3, {1, 1});
        view.frame.light_pos = light_pos;
        view.frame.light_col = light_col;

        PbrGeometryLeaves vgeom = geom;
        vgeom.clip = view.clip;
        const PbrGBuffer gbuf = BuildPbrGBuffer(vgeom, screen);
        const PbrOutputs target = BuildPbrShade(gbuf, mat, view.frame,
                                                ibl_gt, dummy_inv_uv, screen);
        const PbrOutputs pred = BuildPbrShade(gbuf, mat, view.frame, ibl_opt,
                                              dummy_inv_uv, screen);
        view.target = target.ldr;
        view.pred = pred.ldr;
        Variable diff = F::StopGradient(target.mapped) - pred.mapped;
        view_losses.push_back(F::Mean(diff * diff));
    }
    Variable loss = F::SumPixelWise(view_losses);

    // Env smoothness prior: mean per-2x2-block variance of the env,
    // Var[x] = E[x^2] - E[x]^2 evaluated per non-overlapping block with
    // AvgPool2x2. Minimizing it pulls each texel toward its local mean, so
    // the underconstrained (noisy, black-speckled) poles/nadir are filled
    // from their neighbors. Added to the loss => its gradient rides the
    // same Adam step as the render gradient. Built BEFORE the requires-grad
    // pass below so the pass propagates through it.
    if (env_reg > 0.f) {
        Variable pool_sq = F::AvgPool2x2(env_opt * env_opt);
        Variable pool = F::AvgPool2x2(env_opt);
        Variable block_var = pool_sq - pool * pool;
        loss = loss + F::Float(env_reg) * F::Mean(block_var);
    }

    Variable env_mut = env_opt;
    env_mut.setRequiresGradRecursively();

    // ------------------- In-graph Adam (GPU-resident state) ---------------
    Variable adam_m(VEC3, env_size);
    Variable adam_v(VEC3, env_size);
    Variable adam_t(FLOAT, {1, 1});
    DressiAD ad;
    ad.setLossVar(loss);
    bool state_registered = false;
    ad.setOptimizer([&](Variables xs, Variables gxs) {
        DRESSI_CHECK(xs.size() == 1 && xs[0] == env_opt,
                     "expected the env map as the only optimizer target");
        const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        Variable g = gxs[0];
        Variable t_new = adam_t + 1.f;
        Variable m_new = adam_m * b1 + g * (1.f - b1);
        Variable v_new = adam_v * b2 + g * g * (1.f - b2);
        Variable bc1 = 1.f - F::Pow(F::Float(b1), t_new);
        Variable bc2 = 1.f - F::Pow(F::Float(b2), t_new);
        Variable step = (m_new / bc1) / (F::Sqrt(v_new / bc2) + eps) * lr;
        if (!state_registered) {
            ad.addUpdate(adam_m, m_new);
            ad.addUpdate(adam_v, v_new);
            ad.addUpdate(adam_t, t_new);
            state_registered = true;
        }
        // Radiance stays non-negative (no upper bound: HDR). --clamp=0
        // drops the floor for the black-vs-gray diagnostic.
        Variable updated = env_opt - step;
        if (clamp_nonneg) {
            updated = F::Max(updated, F::Float(0.f));
        }
        return Variables{updated};
    });
    for (auto& view : views) {
        ad.markOutput(view.target);
        ad.markOutput(view.pred);
    }

    // ------------------------ One-time uploads ---------------------------
    ad.sendImg(geom.uvh, uvh_img);
    ad.sendImg(geom.normal, gltf.normal);
    ad.sendImg(geom.tangent, tangent_img);
    ad.sendImg(geom.wpos, gltf.mesh.pos);
    ad.sendImg(geom.faces, gltf.mesh.faces);
    ad.sendImg(mat.albedo, m_albedo);
    ad.sendImg(mat.mr, m_mr);
    ad.sendImg(mat.normal_map, m_nrm);
    ad.sendImg(mat.ao, m_ao);
    ad.sendImg(mat.emissive, m_emis);
    ad.sendImg(env_gt, env_gt_img);
    ad.sendImg(light_pos, Vec3Img(2.f, 2.f, 2.f));
    ad.sendImg(light_col, Vec3Img(20.f, 20.f, 20.f));
    // Gray HDR init around the scene's average brightness
    ad.sendImg(env_opt, CpuImage(env_size.w, env_size.h, 3, 0.3f));
    ad.sendImg(adam_m, CpuImage(env_size.w, env_size.h, 3, 0.f));
    ad.sendImg(adam_v, CpuImage(env_size.w, env_size.h, 3, 0.f));
    ad.sendImg(adam_t, CpuImage(1, 1, 1, 0.f));
    for (uint32_t vi = 0; vi < n_views; vi++) {
        View& view = views[vi];
        const std::array<float, 3> eye = view_eye(vi);
        const Mat4 vm = LookAt(eye, {0, 0, 0}, {0, 1, 0});
        ad.sendImg(view.clip, view.clip_img);
        ad.sendImg(view.frame.cam_pos, Vec3Img(eye[0], eye[1], eye[2]));
        ad.sendImg(view.frame.cam_fwd, Vec3Img(-vm[2], -vm[6], -vm[10]));
        ad.sendImg(view.frame.cam_right,
                   Vec3Img(vm[0] * tan_half, vm[4] * tan_half,
                           vm[8] * tan_half));
        ad.sendImg(view.frame.cam_up,
                   Vec3Img(vm[1] * tan_half, vm[5] * tan_half,
                           vm[9] * tan_half));
    }

    // --------------------------- Optimization ----------------------------
    // The env leaf is the only dirty leaf: re-send it? No — the optimizer
    // writes it in place, which marks it dynamic. Nothing else changes per
    // iteration, so no per-iteration uploads at all.
    // PSNR on Reinhard-tonemapped radiance (the raw HDR MSE is dominated
    // by the sun texels), over components the gradients actually touched
    const auto psnr_vs_gt = [&]() {
        const CpuImage rec = ad.recvImg(env_opt);
        double se = 0.0;
        size_t cnt = 0;
        for (size_t i = 0; i < rec.data.size(); i++) {
            if (std::abs(rec.data[i] - 0.3f) > 1e-3f) {  // touched by grads
                const double a = rec.data[i] / (1.f + rec.data[i]);
                const double b = env_gt_img.data[i] /
                                 (1.f + env_gt_img.data[i]);
                se += (a - b) * (a - b);
                cnt++;
            }
        }
        const double mse = cnt ? se / double(cnt) : 1.0;
        return std::pair<double, size_t>(
                10.0 * std::log10(1.0 / (mse + 1e-12)), cnt);
    };

    // Live view: every input view tiled (GT targets fixed, optimized
    // renders refreshing) plus the env map under optimization stacked
    // under its ground truth (both Reinhard-tonemapped and upscaled)
    const uint32_t tile_cols = 3;
    const uint32_t tile_rows = (n_views + tile_cols - 1) / tile_cols;
    const uint32_t win_w = screen.w * tile_cols;
    const uint32_t win_h = screen.h * tile_rows;
    const uint32_t env_up = std::max(1u, 512u / env_size.w);
    auto viewer_target =
            host.makeViewer(win_w, win_h, "targets (GT env, fixed views)");
    auto viewer_pred = host.makeViewer(win_w, win_h, "optimized-env render");
    auto viewer_env =
            host.makeViewer(env_size.w * env_up, env_size.h * env_up * 2,
                            "env map (GT / optimized)");
    viewer_target->setPosition(60, 60);
    viewer_pred->setPosition(60 + int(win_w) + 16, 60);
    viewer_env->setPosition(60, 60 + int(win_h) + 46);
    bool viewer_open = viewer_target->valid() && viewer_pred->valid() &&
                       viewer_env->valid();
    CpuImage target_tile;
    const CpuImage env_gt_disp = Upscale(TonemapForView(env_gt_img), env_up);

    using Clock = std::chrono::steady_clock;
    std::vector<double> iter_ms;  // steady-state samples (>= warmup)
    std::vector<double> warmup_samples;  // the excluded build/rebuild iters
    const int warmup = 20;
    const auto median_ms = [&]() {
        if (iter_ms.empty()) {
            return 0.0;
        }
        std::vector<double> s = iter_ms;
        std::nth_element(s.begin(), s.begin() + s.size() / 2, s.end());
        return s[s.size() / 2];
    };
    for (int iter = 0; iter < n_iters; iter++) {
        if (host.cancelled()) {
            break;
        }
        const auto t0 = Clock::now();
        ad.execStep();
        const double ms = std::chrono::duration<double, std::milli>(
                                  Clock::now() - t0)
                                  .count();
        if (iter >= warmup) {
            iter_ms.push_back(ms);
        } else {
            warmup_samples.push_back(ms);
        }

        if (iter == 0) {
            SaveImagePng(out_dir + "/target_view0.png",
                         ad.recvImg(views[0].target));
            SaveImagePng(out_dir + "/render_first_view0.png",
                         ad.recvImg(views[0].pred));
            std::vector<CpuImage> targets;
            for (auto& view : views) {
                targets.push_back(ad.recvImg(view.target));
            }
            target_tile = TileImages(targets, tile_cols);
        }
        if (viewer_open && view_interval > 0 && iter % view_interval == 0) {
            std::vector<CpuImage> preds;
            for (auto& view : views) {
                preds.push_back(ad.recvImg(view.pred));
            }
            viewer_pred->setTitle(
                    fmt::format("optimized-env render  iter {}", iter));
            viewer_open =
                    viewer_target->update(target_tile) &&
                    viewer_pred->update(TileImages(preds, tile_cols)) &&
                    viewer_env->update(TileImages(
                            {env_gt_disp,
                             Upscale(TonemapForView(ad.recvImg(env_opt)),
                                     env_up)},
                            1));
        }
        if (iter % 200 == 0 || iter == n_iters - 1) {
            const auto [psnr, cnt] = psnr_vs_gt();
            spdlog::info(
                    "iter {:4d}  loss {:.6f}  env PSNR {:.2f} dB over {} "
                    "components  median {:.2f} ms/iter (excl. {}-iter "
                    "warmup)",
                    iter, ad.recvImg(loss).data[0], psnr, cnt, median_ms(),
                    warmup);
        }
    }

    // ------------------------------ Outputs -------------------------------
    const auto [psnr, cnt] = psnr_vs_gt();
    SaveImagePng(out_dir + "/env_gt.png",
                 Upscale(TonemapForView(env_gt_img), 8));
    SaveImagePng(out_dir + "/env_recovered.png",
                 Upscale(TonemapForView(ad.recvImg(env_opt)), 8));
    SaveImagePng(out_dir + "/render_last_view0.png",
                 ad.recvImg(views[0].pred));

    // Diagnostic (see --clamp): the >=0 clamp makes underconstrained texels
    // pile up at exactly 0 (black speckle); --clamp=0 lets them spread
    // (gray noise, some negative). Report the distribution and save a
    // linear-grayscale view that renders 0 as black (unlike the Reinhard
    // tonemap, which NaNs on the negatives that appear with --clamp=0).
    {
        const CpuImage rec = ad.recvImg(env_opt);
        float mn = 1e30f, mx = -1e30f;
        double sum = 0.0;
        size_t n_black = 0, n_neg = 0;
        for (float v : rec.data) {
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            sum += v;
            if (v < 0.02f) n_black++;
            if (v < 0.f) n_neg++;
        }
        const double N = double(rec.data.size());
        spdlog::info("env stats (clamp={}): min {:.3f} max {:.2f} mean {:.3f}"
                     " | near-black(<0.02) {:.1f}% | negative {:.1f}%",
                     clamp_nonneg, mn, mx, sum / N,
                     100.0 * double(n_black) / N, 100.0 * double(n_neg) / N);
        // Diverging view: non-negative radiance -> grayscale, NEGATIVE
        // radiance -> red. With --clamp=1 the underconstrained texels are
        // black (0); with --clamp=0 the very same texels light up red,
        // i.e. the black speckle is where the env wants to go negative.
        const auto g22 = [](float x) {
            x = x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
            return std::pow(x, 1.f / 2.2f);
        };
        CpuImage lin(rec.width, rec.height, 3);
        for (uint32_t i = 0; i < rec.width * rec.height; i++) {
            const float v = rec.data[i * rec.channels];  // R channel proxy
            if (v < 0.f) {
                lin.data[i * 3 + 0] = g22(-v / 0.6f);
                lin.data[i * 3 + 1] = 0.f;
                lin.data[i * 3 + 2] = 0.f;
            } else {
                const float g = g22(v / 0.6f);
                lin.data[i * 3 + 0] = g;
                lin.data[i * 3 + 1] = g;
                lin.data[i * 3 + 2] = g;
            }
        }
        SaveImagePng(out_dir + "/env_recovered_signed.png", Upscale(lin, 8));
    }
    spdlog::info("final loss {:.6f}, env PSNR {:.2f} dB over {} components",
                 ad.recvImg(loss).data[0], psnr, cnt);

    // Benchmark record for scripts/bench_summary.py
    const double steady_median = MedianMs(iter_ms);
    const double warmup_ms = WarmupMs(warmup_samples, steady_median);
    const double first_build_ms =
            warmup_samples.empty() ? 0.0 : warmup_samples.front();
    spdlog::info("one-time build/warmup {:.1f} ms (first build {:.1f} ms)",
                 warmup_ms, first_build_ms);
    {
        BenchRecord rec("pbr_envmap_optimization", ad.getDeviceName());
        rec.addPacking(ad.getFuncCount(), ad.getSubStageCount(),
                       ad.getStageCount());
        rec.add("screen", int64_t(size));
        rec.add("envres", int64_t(env_size.w));
        rec.add("views", int64_t(n_views));
        rec.add("iters", int64_t(n_iters));
        rec.add("median_ms_per_iter", steady_median);
        rec.add("warmup_excluded", int64_t(warmup));
        rec.add("warmup_ms", warmup_ms, 1);
        rec.add("first_build_ms", first_build_ms, 1);
        rec.add("psnr_db", psnr, 2);
        rec.save(out_dir + "/bench.json");
    }
    spdlog::info("outputs in {}/", out_dir);
    return (psnr > 10.0 && cnt > 0) ? 0 : 1;
} catch (const std::exception& e) {
    spdlog::error("error: {}", e.what());
    return 2;
}
