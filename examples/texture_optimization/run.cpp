// Multi-view texture optimization on the bunny mesh with deferred unlit
// shading and Sobol-jittered sampling.
//
// G-buffer channels (screen-space UV + coverage mask) come from
// non-differentiable rasterization; shading samples the texture, which is
// differentiable through the inverse-UV backward. Every iteration the
// sampling UVs are offset by a quasi-random Sobol point (the paper's
// countermeasure against texels that never receive updates), and the target
// is rendered in-graph with the GT texture under the same jitter, so each
// iteration constrains a fresh subset of texels. Starting from a black
// texture, the summed per-view MSE densely recovers the GT atlas.

#include "run.h"

#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "../common/asset_utils.h"
#include "../common/bench.h"
#include "../common/sobol.h"
#include "dressi/dressi.h"

using namespace dressi;
using namespace dressi_examples;

namespace {

struct View {
    Variable clip{nullptr};         // {V,1} VEC4 clip positions
    Variable inv_uv_attr{nullptr};  // {V,1} VEC4 inverse-UV screen attr
    Variable target{nullptr};       // {screen} VEC3 in-graph GT render
    Variable pred{nullptr};         // {screen} VEC3 optimized render
    CpuImage clip_img;
    CpuImage inv_uv_attr_img;
};

}  // namespace

int dressi_examples::RunTextureOptimization(
        const std::vector<std::string>& args, ExampleHost& host) try {
    spdlog::cfg::load_env_levels();  // e.g. SPDLOG_LEVEL=debug for timings

    std::string data_dir = "data/bunny";
    std::string out_dir = "texopt_out";
    uint32_t n_views = 6;
    int n_iters = 1500;  // 98.7% texel coverage (99.4% at 4000; jitter-limited)
    int view_interval = 1;  // live-viewer refresh cadence (iters); 0 = off
    for (const std::string& arg : args) {
        if (arg.rfind("--data-dir=", 0) == 0) {
            data_dir = arg.substr(11);
        } else if (arg.rfind("--out-dir=", 0) == 0) {
            out_dir = arg.substr(10);
        } else if (arg.rfind("--views=", 0) == 0) {
            n_views = uint32_t(std::stoul(arg.substr(8)));
        } else if (arg.rfind("--iters=", 0) == 0) {
            n_iters = std::stoi(arg.substr(8));
        } else if (arg.rfind("--view-interval=", 0) == 0) {
            view_interval = std::stoi(arg.substr(16));
        } else if (arg.rfind("--", 0) != 0) {
            data_dir = arg;  // positional data dir (back-compat)
        } else {
            throw DressiError("unknown option: " + arg);
        }
    }
    const ImgSize screen = {256, 256};

    std::filesystem::create_directories(out_dir);

    // ------------------------- Load assets (CPU) -------------------------
    const Mesh mesh = LoadObjMesh(data_dir + "/bunny.obj");
    const CpuImage gt_tex_img = LoadImageRgb(data_dir + "/bunny-atlas.jpg");
    const ImgSize tex_size = {gt_tex_img.width, gt_tex_img.height};
    spdlog::info("mesh: {} verts, {} faces; atlas: {}x{}",
                 mesh.numVertices(), mesh.numFaces(), tex_size.w, tex_size.h);

    // ------------------------ Graph leaf variables ------------------------
    Variable vtx_uv(VEC2, {mesh.numVertices(), 1});
    Variable faces(IVEC3, {mesh.numFaces(), 1});
    Variable uv_clip(VEC4, {mesh.numVertices(), 1});  // UV-space positions
    Variable ones(FLOAT, {mesh.numVertices(), 1});
    Variable gt_tex(VEC3, tex_size);  // GT atlas (no gradient)
    Variable tex(VEC3, tex_size);     // optimized texture

    const CpuImage uv_clip_img = UvAsClip(mesh.uv);
    const CpuImage ones_img(mesh.numVertices(), 1, 1, 1.f);

    // Cameras on a circle around the object, slightly elevated
    std::vector<View> views(n_views);
    const Mat4 proj = PerspectiveVk(45.f * 3.14159265f / 180.f, 1.f, 0.1f,
                                    10.f);
    for (uint32_t v = 0; v < n_views; v++) {
        const float ang = 2.f * 3.14159265f * float(v) / float(n_views);
        const std::array<float, 3> eye = {1.6f * std::sin(ang), 0.5f,
                                          1.6f * std::cos(ang)};
        const Mat4 mvp = Mul(proj, LookAt(eye, {0, 0, 0}, {0, 1, 0}));
        views[v].clip = Variable(VEC4, {mesh.numVertices(), 1});
        views[v].inv_uv_attr = Variable(VEC4, {mesh.numVertices(), 1});
        views[v].clip_img = TransformToClip(mesh.pos, mvp);
        views[v].inv_uv_attr_img = ScreenAttr(views[v].clip_img, screen);
    }

    // ------------------- Deferred unlit rendering graph -------------------
    // G-buffer per view: screen-space UV + coverage mask. Shading samples
    // both textures at the jittered UV; the GT-textured render is the
    // per-iteration target.
    Variables view_losses;
    for (auto& view : views) {
        Variable g_uv = F::Rasterize(view.clip, vtx_uv, faces, screen);
        Variable g_mask = F::Rasterize(view.clip, ones, faces, screen);
        Variable inv_uv =
                F::Rasterize(uv_clip, view.inv_uv_attr, faces, tex_size);
        view.target = g_mask * F::Texture(gt_tex, g_uv, inv_uv);
        view.pred = g_mask * F::Texture(tex, g_uv, inv_uv);
        Variable diff = F::StopGradient(view.target) - view.pred;
        view_losses.push_back(F::Mean(diff * diff));
    }
    Variable loss = F::SumPixelWise(view_losses);

    Variable tex_mut = tex;
    tex_mut.setRequiresGradRecursively();

    // MSE mean divides by N. Under jitter a texel typically receives its
    // gradient from one view at a time, so scale per single contribution
    // while keeping the worst case (all views hitting at once) stable (<2).
    const float lr = 0.6f * float(screen.w * screen.h * 3) / (2.f * n_views);

    DressiAD ad;
    ad.setLossVar(loss);
    ad.setOptimizer([lr](Variables xs, Variables gxs) {
        Variables updated;
        for (size_t i = 0; i < xs.size(); i++) {
            updated.push_back(xs[i] - gxs[i] * lr);
        }
        return updated;
    });
    for (auto& view : views) {
        ad.markOutput(view.target);
        ad.markOutput(view.pred);
    }

    ad.sendImg(vtx_uv, mesh.uv);
    ad.sendImg(faces, mesh.faces);
    ad.sendImg(uv_clip, uv_clip_img);
    ad.sendImg(ones, ones_img);
    for (auto& view : views) {
        ad.sendImg(view.clip, view.clip_img);
        ad.sendImg(view.inv_uv_attr, view.inv_uv_attr_img);
    }
    ad.sendImg(gt_tex, gt_tex_img);
    ad.sendImg(tex, CpuImage(tex_size.w, tex_size.h, 3, 0.f));  // black

    // --------------------------- Optimization ----------------------------
    // Sub-pixel camera jitter (TAA style): offsets the projection by less
    // than half a pixel so each pixel center sweeps its surface footprint
    // across iterations. Colors stay attached to the geometry (a surface
    // point always samples the texture at its own UV), so the pattern does
    // not wobble; texels are covered densely over time.
    const auto send_jitter = [&](int iter) {
        const auto s = Sobol2D(uint32_t(iter));
        const float jx = (s[0] - 0.5f) * 2.f / float(screen.w);
        const float jy = (s[1] - 0.5f) * 2.f / float(screen.h);
        for (auto& view : views) {
            CpuImage jittered = view.clip_img;
            for (uint32_t i = 0; i < jittered.width; i++) {
                const float w = jittered.at(i, 0, 3);
                jittered.at(i, 0, 0) += jx * w;
                jittered.at(i, 0, 1) += jy * w;
            }
            ad.sendImg(view.clip, jittered);
        }
    };

    // Live view (glad + GLFW): all views tiled at fixed viewpoints, GT
    // targets in one window and the current renders in another. Display
    // cost is excluded from optimization timing.
    const uint32_t tile_cols = 3;
    const uint32_t tile_rows = (n_views + tile_cols - 1) / tile_cols;
    const uint32_t win_w = 256 * tile_cols;
    const uint32_t win_h = 256 * tile_rows;
    auto viewer_target =
            host.makeViewer(win_w, win_h, "targets (GT, fixed views)");
    auto viewer_pred = host.makeViewer(win_w, win_h, "optimized render");
    auto viewer_tex =
            host.makeViewer(512, 256, "texture atlas (GT | optimized)");
    // Fixed side-by-side layout (deterministic placement for recording)
    viewer_target->setPosition(80, 80);
    viewer_pred->setPosition(80 + int(win_w) + 16, 80);
    viewer_tex->setPosition(80, 80 + int(win_h) + 46);
    bool viewer_open = viewer_target->valid() && viewer_pred->valid() &&
                       viewer_tex->valid();
    if (!viewer_open) {
        spdlog::warn("live viewer unavailable; continuing headless");
    }
    CpuImage target_tile;

    using Clock = std::chrono::steady_clock;
    double opt_ms = 0.0;   // execStep only
    double view_ms = 0.0;  // recvImg + tiling + window updates
    // Cross-device benchmark metric: MEDIAN steady-state jitter+execStep
    // ms/iter, excluding the build/rebuild warmup (see CLAUDE.md
    // "Benchmarking"); the running mean above stays for progress logs.
    std::vector<double> opt_samples;
    std::vector<double> warmup_samples;  // the excluded build/rebuild iters
    const int opt_warmup = std::min(20, n_iters / 2);
    for (int iter = 0; iter < n_iters; iter++) {
        if (host.cancelled()) {
            break;
        }
        const auto t0 = Clock::now();
        send_jitter(iter);
        ad.execStep();
        const auto t1 = Clock::now();
        const double iter_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
        opt_ms += iter_ms;
        if (iter >= opt_warmup) {
            opt_samples.push_back(iter_ms);
        } else {
            warmup_samples.push_back(iter_ms);
        }

        if (iter == 0) {
            std::vector<CpuImage> targets;
            for (uint32_t v = 0; v < n_views; v++) {
                targets.push_back(ad.recvImg(views[v].target));
                SaveImagePng(fmt::format("{}/target_view{}.png", out_dir, v),
                             targets.back());
            }
            target_tile = TileImages(targets, tile_cols);
            SaveImagePng(out_dir + "/render_first_view0.png",
                         ad.recvImg(views[0].pred));
        }

        if (viewer_open && view_interval > 0 && iter % view_interval == 0) {
            const auto tv0 = Clock::now();
            std::vector<CpuImage> preds;
            for (uint32_t v = 0; v < n_views; v++) {
                preds.push_back(ad.recvImg(views[v].pred));
            }
            viewer_pred->setTitle(
                    fmt::format("optimized render  iter {}", iter));
            viewer_open = viewer_pred->update(TileImages(preds, tile_cols)) &&
                          viewer_target->update(target_tile) &&
                          viewer_tex->update(TileImages(
                                  {gt_tex_img, ad.recvImg(tex)}, 2));
            view_ms += std::chrono::duration<double, std::milli>(
                               Clock::now() - tv0)
                               .count();
        }

        if (iter % 100 == 0 || iter == n_iters - 1) {
            spdlog::info(
                    "iter {:4d}  loss {:.8f}  stages {}  opt {:.2f} ms/iter"
                    "  (view {:.2f} ms/iter excluded)",
                    iter, ad.recvImg(loss).data[0], ad.getStageCount(),
                    opt_ms / double(iter + 1), view_ms / double(iter + 1));
        }
    }

    // ------------------------------ Outputs -------------------------------
    // Final evaluation and renders at zero jitter (clean pixel centers)
    {
        for (auto& view : views) {
            ad.sendImg(view.clip, view.clip_img);
        }
        ad.execStep();
    }
    const float final_loss = ad.recvImg(loss).data[0];
    const CpuImage recovered = ad.recvImg(tex);
    SaveImagePng(out_dir + "/recovered_texture.png", recovered);
    for (uint32_t v = 0; v < n_views; v++) {
        SaveImagePng(fmt::format("{}/render_last_view{}.png", out_dir, v),
                     ad.recvImg(views[v].pred));
    }

    // Texels recovered to the GT atlas (texels observed under some jitter
    // in some view; the rest stay black)
    size_t updated = 0;
    size_t recovered_ok = 0;
    for (size_t p = 0; p < size_t(tex_size.w) * tex_size.h; p++) {
        float rec_mag = 0.f;
        float err = 0.f;
        for (uint32_t c = 0; c < 3; c++) {
            rec_mag += recovered.data[p * 3 + c];
            err = std::max(err, std::abs(recovered.data[p * 3 + c] -
                                         gt_tex_img.data[p * 3 + c]));
        }
        if (rec_mag > 0.01f) {
            updated++;
            if (err < 0.1f) {
                recovered_ok++;
            }
        }
    }
    spdlog::info("final loss {:.8f}", final_loss);
    spdlog::info("recovered texels: {} / {} updated ({:.1f}% accurate)",
                 recovered_ok, updated,
                 updated ? 100.f * float(recovered_ok) / float(updated)
                         : 0.f);

    // Benchmark record for scripts/bench_summary.py
    const double median_ms = MedianMs(opt_samples);
    const double warmup_ms = WarmupMs(warmup_samples, median_ms);
    const double first_build_ms =
            warmup_samples.empty() ? 0.0 : warmup_samples.front();
    spdlog::info("median steady-state {:.3f} ms/iter ({} warmup excluded); "
                 "one-time build/warmup {:.1f} ms (first build {:.1f} ms)",
                 median_ms, opt_warmup, warmup_ms, first_build_ms);
    {
        BenchRecord rec("texture_optimization", ad.getDeviceName());
        rec.addPacking(ad.getFuncCount(), ad.getSubStageCount(),
                       ad.getStageCount());
        rec.add("screen", int64_t(screen.w));
        rec.add("views", int64_t(n_views));
        rec.add("iters", int64_t(n_iters));
        rec.add("median_ms_per_iter", median_ms);
        rec.add("warmup_excluded", int64_t(opt_warmup));
        rec.add("warmup_ms", warmup_ms, 1);
        rec.add("first_build_ms", first_build_ms, 1);
        rec.add("final_loss", double(final_loss), 8);
        rec.add("accurate_pct",
                updated ? 100.0 * double(recovered_ok) / double(updated)
                        : 0.0,
                1);
        rec.save(out_dir + "/bench.json");
    }
    spdlog::info("outputs in {}/", out_dir);

    const bool ok = final_loss < 1e-4f && updated > 0 &&
                    float(recovered_ok) / float(updated) > 0.95f;
    return ok ? 0 : 1;
} catch (const std::exception& e) {
    spdlog::error("error: {}", e.what());
    return 2;
}
