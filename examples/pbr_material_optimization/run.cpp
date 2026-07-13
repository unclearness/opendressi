// Multi-view PBR material optimization on DamagedHelmet: recover ONE
// material texture (default: albedo) from rendered images while every
// other input (other materials, env map, light, geometry) stays at ground
// truth. The forward shader is exactly the viewer's
// (common/pbr_graph.{h,cpp}); per view the G-buffer is rasterized once and
// shaded twice (GT materials -> target, optimized materials -> pred).
// Gradients reach the texture through F::TextureBilinear's tent-weighted
// inverse-UV gather; Sobol sub-pixel jitter sweeps texels across
// iterations (the paper's coverage countermeasure). Optimizer = in-graph
// Adam with GPU-resident state (addUpdate), so nothing crosses PCIe per
// iteration except the jittered clip positions.

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
#include "../common/sobol.h"
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

struct View {
    Variable clip{nullptr};
    Variable inv_uv_attr{nullptr};
    Variable inv_uv{nullptr};
    PbrFrameLeaves frame;
    Variable target{nullptr};
    Variable pred{nullptr};
    CpuImage clip_img;
    CpuImage inv_uv_attr_img;
};

}  // namespace

int dressi_examples::RunPbrMaterialOptimization(
        const std::vector<std::string>& args, ExampleHost& host) try {
    spdlog::cfg::load_env_levels();

    std::string mesh_path = "data/DamagedHelmet/glTF/DamagedHelmet.gltf";
    std::string env_path = "data/suburban_garden_4k.exr";
    std::string out_dir = "pbrmat_out";
    std::string optimize = "albedo";  // or "mr"
    uint32_t size = 256, tex_max = 512, n_views = 6;
    int n_iters = 1200;  // albedo PSNR ~20.7 dB (21.2 at 2000; slow log climb)
    int view_interval = 1;  // live-viewer refresh cadence (iters); 0 = off
    float lr = 0.02f;
    for (const std::string& arg : args) {
        if (arg.rfind("--mesh=", 0) == 0) {
            mesh_path = arg.substr(7);
        } else if (arg.rfind("--env=", 0) == 0) {
            env_path = arg.substr(6);
        } else if (arg.rfind("--out-dir=", 0) == 0) {
            out_dir = arg.substr(10);
        } else if (arg.rfind("--optimize=", 0) == 0) {
            optimize = arg.substr(11);
        } else if (arg.rfind("--size=", 0) == 0) {
            size = uint32_t(std::stoi(arg.substr(7)));
        } else if (arg.rfind("--tex=", 0) == 0) {
            tex_max = uint32_t(std::stoi(arg.substr(6)));
        } else if (arg.rfind("--views=", 0) == 0) {
            n_views = uint32_t(std::stoul(arg.substr(8)));
        } else if (arg.rfind("--iters=", 0) == 0) {
            n_iters = std::stoi(arg.substr(8));
        } else if (arg.rfind("--view-interval=", 0) == 0) {
            view_interval = std::stoi(arg.substr(16));
        } else if (arg.rfind("--lr=", 0) == 0) {
            lr = std::stof(arg.substr(5));
        }
    }
    DRESSI_CHECK(optimize == "albedo" || optimize == "mr",
                 "--optimize must be albedo or mr");
    const ImgSize screen = {size, size};
    std::filesystem::create_directories(out_dir);

    // ------------------------- Load assets (CPU) -------------------------
    const GltfScene gltf = LoadGltfScene(mesh_path);
    CpuImage env_img = DownsampleTo(LoadImageExr(env_path), 512);
    const CpuImage gt_albedo = DownsampleTo(SrgbToLinear(gltf.base_color),
                                            tex_max);
    const CpuImage gt_mr_img = DownsampleTo(gltf.metallic_roughness, tex_max);
    const CpuImage gt_nrm = DownsampleTo(gltf.normal_map, tex_max);
    const CpuImage gt_ao = DownsampleTo(gltf.occlusion, tex_max);
    const CpuImage gt_emis = DownsampleTo(SrgbToLinear(gltf.emissive),
                                          tex_max);
    const ImgSize tex_size = gt_albedo.getImgSize();
    const CpuImage& gt_opt = optimize == "albedo" ? gt_albedo : gt_mr_img;
    spdlog::info("mesh: {} verts, {} faces; optimizing {} at {}x{}",
                 gltf.mesh.numVertices(), gltf.mesh.numFaces(), optimize,
                 tex_size.w, tex_size.h);

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
    PbrGeometryLeaves geom;  // clip is per-view (set below)
    geom.uvh = Variable(VEC4, {n_verts, 1});
    geom.normal = Variable(VEC3, {n_verts, 1});
    geom.tangent = Variable(VEC3, {n_verts, 1});
    geom.wpos = Variable(VEC3, {n_verts, 1});
    geom.faces = Variable(IVEC3, {gltf.mesh.numFaces(), 1});

    PbrMaterialLeaves gt_mat;
    gt_mat.albedo = Variable(VEC3, tex_size);
    gt_mat.mr = Variable(VEC3, tex_size);
    gt_mat.normal_map = Variable(VEC3, tex_size);
    gt_mat.ao = Variable(VEC3, tex_size);
    gt_mat.emissive = Variable(VEC3, tex_size);

    Variable opt_tex(VEC3, tex_size);  // the one optimized texture
    PbrMaterialLeaves opt_mat = gt_mat;
    (optimize == "albedo" ? opt_mat.albedo : opt_mat.mr) = opt_tex;

    Variable env(VEC3, env_img.getImgSize());
    const PbrIblMaps ibl = BuildPbrIblMaps(env);
    Variable uv_clip(VEC4, {n_verts, 1});  // UV-space positions (inverse UV)
    const CpuImage uv_clip_img = UvAsClip(gltf.mesh.uv);

    // Cameras on a circle, slightly elevated; one shared point light
    const float fov = 45.f * 3.14159265f / 180.f;
    const float tan_half = std::tan(fov * 0.5f);
    const Mat4 proj = PerspectiveVk(fov, 1.f, 0.1f, 10.f);
    Variable light_pos(VEC3, {1, 1});
    Variable light_col(VEC3, {1, 1});

    std::vector<View> views(n_views);
    Variables view_losses;
    for (uint32_t vi = 0; vi < n_views; vi++) {
        View& view = views[vi];
        const float ang = 2.f * 3.14159265f * float(vi) / float(n_views);
        const std::array<float, 3> eye = {2.2f * std::sin(ang) * 0.92f,
                                          0.85f,
                                          2.2f * std::cos(ang) * 0.92f};
        const Mat4 vm = LookAt(eye, {0, 0, 0}, {0, 1, 0});
        view.clip = Variable(VEC4, {n_verts, 1});
        view.inv_uv_attr = Variable(VEC4, {n_verts, 1});
        view.clip_img = TransformToClip(gltf.mesh.pos, Mul(proj, vm));
        view.inv_uv_attr_img = ScreenAttr(view.clip_img, screen);

        view.frame.cam_pos = Variable(VEC3, {1, 1});
        view.frame.cam_fwd = Variable(VEC3, {1, 1});
        view.frame.cam_right = Variable(VEC3, {1, 1});
        view.frame.cam_up = Variable(VEC3, {1, 1});
        view.frame.light_pos = light_pos;
        view.frame.light_col = light_col;

        PbrGeometryLeaves vgeom = geom;
        vgeom.clip = view.clip;
        const PbrGBuffer gbuf = BuildPbrGBuffer(vgeom, screen);
        view.inv_uv = F::Rasterize(uv_clip, view.inv_uv_attr, geom.faces,
                                   tex_size);
        const PbrOutputs target = BuildPbrShade(gbuf, gt_mat, view.frame,
                                                ibl, view.inv_uv, screen);
        const PbrOutputs pred = BuildPbrShade(gbuf, opt_mat, view.frame, ibl,
                                              view.inv_uv, screen);
        view.target = target.ldr;
        view.pred = pred.ldr;
        // Loss on the pre-gamma tonemapped image (the gamma Pow has an
        // infinite gradient at 0)
        Variable diff = F::StopGradient(target.mapped) - pred.mapped;
        view_losses.push_back(F::Mean(diff * diff));
    }
    Variable loss = F::SumPixelWise(view_losses);

    Variable opt_mut = opt_tex;
    opt_mut.setRequiresGradRecursively();

    // ------------------- In-graph Adam (GPU-resident state) ---------------
    Variable adam_m(VEC3, tex_size);
    Variable adam_v(VEC3, tex_size);
    Variable adam_t(FLOAT, {1, 1});
    DressiAD ad;
    ad.setLossVar(loss);
    bool state_registered = false;
    ad.setOptimizer([&](Variables xs, Variables gxs) {
        DRESSI_CHECK(xs.size() == 1 && xs[0] == opt_tex,
                     "expected the optimized texture as the only target");
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
        // Keep the recovered texture in a physical range
        return Variables{F::Clamp(opt_tex - step, F::Float(0.f),
                                  F::Float(1.f))};
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
    ad.sendImg(gt_mat.albedo, gt_albedo);
    ad.sendImg(gt_mat.mr, gt_mr_img);
    ad.sendImg(gt_mat.normal_map, gt_nrm);
    ad.sendImg(gt_mat.ao, gt_ao);
    ad.sendImg(gt_mat.emissive, gt_emis);
    ad.sendImg(env, env_img);
    ad.sendImg(uv_clip, uv_clip_img);
    ad.sendImg(light_pos, Vec3Img(2.f, 2.f, 2.f));
    ad.sendImg(light_col, Vec3Img(20.f, 20.f, 20.f));
    ad.sendImg(opt_tex, CpuImage(tex_size.w, tex_size.h, 3, 0.5f));
    ad.sendImg(adam_m, CpuImage(tex_size.w, tex_size.h, 3, 0.f));
    ad.sendImg(adam_v, CpuImage(tex_size.w, tex_size.h, 3, 0.f));
    ad.sendImg(adam_t, CpuImage(1, 1, 1, 0.f));
    for (auto& view : views) {
        ad.sendImg(view.inv_uv_attr, view.inv_uv_attr_img);
    }

    // Per-view camera uniforms (static)
    for (uint32_t vi = 0; vi < n_views; vi++) {
        View& view = views[vi];
        const float ang = 2.f * 3.14159265f * float(vi) / float(n_views);
        const std::array<float, 3> eye = {2.2f * std::sin(ang) * 0.92f,
                                          0.85f,
                                          2.2f * std::cos(ang) * 0.92f};
        const Mat4 vm = LookAt(eye, {0, 0, 0}, {0, 1, 0});
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
    const auto send_jitter = [&](int iter) {
        const auto s = Sobol2D(uint32_t(iter));
        const float jx = (s[0] - 0.5f) * 2.f / float(screen.w);
        const float jy = (s[1] - 0.5f) * 2.f / float(screen.h);
        std::vector<std::pair<Variable, CpuImageView>> sends;
        static std::vector<CpuImage> jittered(views.size());
        for (size_t i = 0; i < views.size(); i++) {
            jittered[i] = views[i].clip_img;
            for (uint32_t v = 0; v < jittered[i].width; v++) {
                const float w = jittered[i].at(v, 0, 3);
                jittered[i].at(v, 0, 0) += jx * w;
                jittered[i].at(v, 0, 1) += jy * w;
            }
            sends.emplace_back(views[i].clip, CpuImageView(jittered[i]));
        }
        ad.sendImgs(sends);
    };

    const auto psnr_vs_gt = [&]() {
        const CpuImage rec = ad.recvImg(opt_tex);
        double se = 0.0;
        size_t cnt = 0;
        for (size_t i = 0; i < rec.data.size(); i++) {
            // Only texels the optimizer moved (covered by some view)
            if (std::abs(rec.data[i] - 0.5f) > 1e-3f) {
                const double d = double(rec.data[i]) - double(gt_opt.data[i]);
                se += d * d;
                cnt++;
            }
        }
        const double mse = cnt ? se / double(cnt) : 1.0;
        return std::pair<double, size_t>(
                10.0 * std::log10(1.0 / (mse + 1e-12)), cnt);
    };

    // Live view: every input view tiled (GT targets fixed, optimized
    // renders refreshing) plus the texture atlas under optimization
    // (gamma-encoded for display — the graph works in linear space)
    const auto display_gamma = [](CpuImage img) {
        for (float& v : img.data) {
            v = std::pow(std::max(v, 0.f), 1.f / 2.2f);
        }
        return img;
    };
    const uint32_t tile_cols = 3;
    const uint32_t tile_rows = (n_views + tile_cols - 1) / tile_cols;
    const uint32_t win_w = screen.w * tile_cols;
    const uint32_t win_h = screen.h * tile_rows;
    auto viewer_target =
            host.makeViewer(win_w, win_h, "targets (GT, fixed views)");
    auto viewer_pred = host.makeViewer(win_w, win_h, "optimized render");
    auto viewer_tex = host.makeViewer(
            512, 256, "texture " + optimize + " (GT | optimized)");
    viewer_target->setPosition(60, 60);
    viewer_pred->setPosition(60 + int(win_w) + 16, 60);
    viewer_tex->setPosition(60, 60 + int(win_h) + 46);
    bool viewer_open = viewer_target->valid() && viewer_pred->valid() &&
                       viewer_tex->valid();
    CpuImage target_tile;
    const CpuImage gt_tex_disp = display_gamma(gt_opt);

    using Clock = std::chrono::steady_clock;
    double opt_ms = 0.0;
    std::vector<double> opt_samples;  // steady-state jitter+execStep ms
    std::vector<double> warmup_samples;  // the excluded build/rebuild iters
    const int opt_warmup = std::min(20, n_iters / 2);
    for (int iter = 0; iter < n_iters; iter++) {
        if (host.cancelled()) {
            break;
        }
        const auto t0 = Clock::now();
        send_jitter(iter);
        ad.execStep();
        const double iter_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - t0)
                        .count();
        opt_ms += iter_ms;
        if (iter >= opt_warmup) {
            opt_samples.push_back(iter_ms);
        } else {
            warmup_samples.push_back(iter_ms);
        }

        if (iter == 0) {
            SaveImagePng(out_dir + "/target_view0.png",
                         ad.recvImg(views[0].target));
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
                    fmt::format("optimized render  iter {}", iter));
            viewer_open =
                    viewer_target->update(target_tile) &&
                    viewer_pred->update(TileImages(preds, tile_cols)) &&
                    viewer_tex->update(TileImages(
                            {gt_tex_disp,
                             display_gamma(ad.recvImg(opt_tex))},
                            2));
        }
        if (iter % 200 == 0 || iter == n_iters - 1) {
            const auto [psnr, cnt] = psnr_vs_gt();
            spdlog::info(
                    "iter {:4d}  loss {:.6f}  {} PSNR {:.2f} dB over {} "
                    "texels  opt {:.2f} ms/iter",
                    iter, ad.recvImg(loss).data[0], optimize, psnr, cnt,
                    opt_ms / double(iter + 1));
        }
    }

    // ------------------------------ Outputs -------------------------------
    for (auto& view : views) {
        ad.sendImg(view.clip, view.clip_img);  // zero jitter
    }
    ad.execStep();
    const auto [psnr, cnt] = psnr_vs_gt();
    SaveImagePng(out_dir + "/recovered_" + optimize + ".png",
                 ad.recvImg(opt_tex));
    SaveImagePng(out_dir + "/render_last_view0.png",
                 ad.recvImg(views[0].pred));
    spdlog::info("final loss {:.6f}, {} PSNR {:.2f} dB over {} texels",
                 ad.recvImg(loss).data[0], optimize, psnr, cnt);

    // Benchmark record for scripts/bench_summary.py
    const double median_ms = MedianMs(opt_samples);
    const double warmup_ms = WarmupMs(warmup_samples, median_ms);
    const double first_build_ms =
            warmup_samples.empty() ? 0.0 : warmup_samples.front();
    spdlog::info("median steady-state {:.3f} ms/iter ({} warmup excluded); "
                 "one-time build/warmup {:.1f} ms (first build {:.1f} ms)",
                 median_ms, opt_warmup, warmup_ms, first_build_ms);
    {
        BenchRecord rec("pbr_material_optimization", ad.getDeviceName());
        rec.addPacking(ad.getFuncCount(), ad.getSubStageCount(),
                       ad.getStageCount());
        rec.add("optimize", optimize);
        rec.add("screen", int64_t(size));
        rec.add("tex", int64_t(tex_max));
        rec.add("views", int64_t(n_views));
        rec.add("iters", int64_t(n_iters));
        rec.add("median_ms_per_iter", median_ms);
        rec.add("warmup_excluded", int64_t(opt_warmup));
        rec.add("warmup_ms", warmup_ms, 1);
        rec.add("first_build_ms", first_build_ms, 1);
        rec.add("psnr_db", psnr, 2);
        rec.save(out_dir + "/bench.json");
    }
    spdlog::info("outputs in {}/", out_dir);
    return (psnr > 20.0 && cnt > 0) ? 0 : 1;
} catch (const std::exception& e) {
    spdlog::error("error: {}", e.what());
    return 2;
}
