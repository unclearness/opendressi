// Image fitting example: the optimization loop shape of the Dressi paper's
// Appendix A.1, without rasterization. A 64x64 VEC3 parameter image is
// optimized by SGD so that a nonlinearly tone-mapped version of it matches a
// procedural target image.

#include "run.h"

#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>

#include "../common/bench.h"
#include "dressi/dressi.h"

using namespace dressi;

int dressi_examples::RunImageFitting(const std::vector<std::string>& args,
                                     ExampleHost& host) try {
    spdlog::cfg::load_env_levels();  // e.g. SPDLOG_LEVEL=debug for timings
    std::string out_dir = "imgfit_out";  // holds only bench.json
    for (const std::string& arg : args) {
        if (arg.rfind("--out-dir=", 0) == 0) {
            out_dir = arg.substr(10);
        }
    }
    const uint32_t W = 64, H = 64;

    // Create top variables of the computational graph
    Variable param(VEC3, {W, H});
    Variable target(VEC3, {W, H});

    // Build a computational graph: render = Reinhard tonemap of the parameter
    Variable rendered = param / (param + 1.f);
    Variable diff = target - rendered;
    Variable loss = F::Mean(diff * diff);

    // Mark the parameter image as the optimization target
    Variable param_mut = param;
    param_mut.setRequiresGradRecursively();

    // MSE mean divides by N; rescale so effective per-pixel steps stay ~1
    const float lr = 0.6f * float(W * H * 3);

    DressiAD dressi_ad;
    dressi_ad.setLossVar(loss);
    dressi_ad.setOptimizer([lr](Variables xs, Variables gxs) {
        // SGD for all inputs
        Variables updated_xs;
        for (size_t i = 0; i < xs.size(); i++) {
            updated_xs.push_back(xs[i] - gxs[i] * lr);
        }
        return updated_xs;
    });

    // Procedural target in [0.1, 0.6] (reachable tonemapped range)
    CpuImage target_img(W, H, 3);
    for (uint32_t y = 0; y < H; y++) {
        for (uint32_t x = 0; x < W; x++) {
            target_img.at(x, y, 0) = 0.35f + 0.25f * std::sin(0.2f * float(x));
            target_img.at(x, y, 1) = 0.35f + 0.25f * std::cos(0.15f * float(y));
            target_img.at(x, y, 2) =
                    0.35f + 0.25f * std::sin(0.1f * float(x + y));
        }
    }
    CpuImage init_img(W, H, 3, 0.5f);

    dressi_ad.sendImg(param, init_img);
    dressi_ad.sendImg(target, target_img);

    // Optimization iterations (median steady-state timing for bench.json)
    using Clock = std::chrono::steady_clock;
    const int n_iters = 150;  // loss hits the 1e-6 noise floor by ~100 iters
    const int opt_warmup = std::min(20, n_iters / 2);
    std::vector<double> opt_samples;
    std::vector<double> warmup_samples;  // the excluded build/rebuild iters
    for (int iter = 0; iter < n_iters; iter++) {
        if (host.cancelled()) {
            break;
        }
        const auto t0 = Clock::now();
        dressi_ad.execStep();
        const double iter_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - t0)
                        .count();
        if (iter >= opt_warmup) {
            opt_samples.push_back(iter_ms);
        } else {
            warmup_samples.push_back(iter_ms);
        }
        if (iter % 50 == 0 || iter == n_iters - 1) {
            spdlog::info("iter {:4d}  loss {:.6f}", iter,
                         dressi_ad.recvImg(loss).data[0]);
        }
    }

    // Verify: the tone-mapped optimized parameter should match the target.
    // (`rendered` itself is fused inside a shader by RSP and has no image,
    // so receive the parameter and apply the tonemap on the CPU.)
    const CpuImage result = dressi_ad.recvImg(param);
    float max_err = 0.f;
    for (size_t i = 0; i < result.data.size(); i++) {
        const float r = result.data[i] / (result.data[i] + 1.f);
        max_err = std::max(max_err, std::abs(r - target_img.data[i]));
    }
    spdlog::info("max |rendered - target| = {:.4f}", max_err);

    const double median_ms = MedianMs(opt_samples);
    const double warmup_ms = WarmupMs(warmup_samples, median_ms);
    const double first_build_ms =
            warmup_samples.empty() ? 0.0 : warmup_samples.front();
    spdlog::info("median steady-state {:.3f} ms/iter ({} warmup excluded); "
                 "one-time build/warmup {:.1f} ms (first build {:.1f} ms)",
                 median_ms, opt_warmup, warmup_ms, first_build_ms);
    std::filesystem::create_directories(out_dir);
    BenchRecord rec("image_fitting", dressi_ad.getDeviceName());
    rec.addPacking(dressi_ad.getFuncCount(), dressi_ad.getSubStageCount(),
                   dressi_ad.getStageCount());
    rec.add("screen", int64_t(W));
    rec.add("iters", int64_t(n_iters));
    rec.add("median_ms_per_iter", median_ms);
    rec.add("warmup_excluded", int64_t(opt_warmup));
    rec.add("warmup_ms", warmup_ms, 1);
    rec.add("first_build_ms", first_build_ms, 1);
    rec.add("max_err", double(max_err), 4);
    rec.save(out_dir + "/bench.json");

    return max_err < 0.05f ? 0 : 1;
} catch (const std::exception& e) {
    spdlog::error("error: {}", e.what());
    return 2;
}
