// GPU parity for the IBL split-sum ops (equirect sampling, precompute
// convolutions, BRDF LUT, bilinear texture sampling) against the CPU
// evaluator, in both packing modes, plus a TextureBilinear recovery loop
// exercising the bilinear inverse-UV backward on the GPU.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>

#include "core/cpu_eval.h"
#include "dressi/dressi.h"

using namespace dressi;

namespace {

CpuImage MakeImage(uint32_t w, uint32_t h, uint32_t c,
                   const std::function<float(size_t)>& gen) {
    CpuImage img(w, h, c);
    for (size_t i = 0; i < img.data.size(); i++) {
        img.data[i] = gen(i);
    }
    return img;
}

void ExpectImagesNear(const CpuImage& a, const CpuImage& b, float tol) {
    ASSERT_EQ(a.width, b.width);
    ASSERT_EQ(a.height, b.height);
    ASSERT_EQ(a.channels, b.channels);
    for (size_t i = 0; i < a.data.size(); i++) {
        ASSERT_NEAR(a.data[i], b.data[i], tol) << "index " << i;
    }
}

// Smooth uv map inside the clamp-free region of a 4x4 texture (matches the
// CPU gradient test in test_backward.cpp) + its synthetic inverse-UV table
CpuImage InteriorUvMap(ImgSize screen) {
    CpuImage uv(screen.w, screen.h, 2);
    for (uint32_t py = 0; py < screen.h; py++) {
        for (uint32_t px = 0; px < screen.w; px++) {
            uv.at(px, py, 0) =
                    0.125f + 0.75f * (float(px) + 0.5f) / float(screen.w);
            uv.at(px, py, 1) =
                    0.125f + 0.75f * (float(py) + 0.5f) / float(screen.h);
        }
    }
    return uv;
}

CpuImage SyntheticInvUv(ImgSize tex_size, ImgSize screen) {
    CpuImage inv(tex_size.w, tex_size.h, 4);
    const float step = 0.75f * float(tex_size.w) / float(screen.w);
    for (uint32_t ty = 0; ty < tex_size.h; ty++) {
        for (uint32_t tx = 0; tx < tex_size.w; tx++) {
            const auto anchor = [&](uint32_t t) {
                float p = float(t) / step;
                if (p > float(screen.w) - 1.f) {
                    p = float(screen.w) - 1.f;
                }
                return std::floor(p) + 0.5f;
            };
            inv.at(tx, ty, 0) = anchor(tx);
            inv.at(tx, ty, 1) = anchor(ty);
            inv.at(tx, ty, 2) = 1.f;
        }
    }
    return inv;
}

}  // namespace

class IblGpu : public ::testing::TestWithParam<DressiAD::PackingMode> {};

TEST_P(IblGpu, ForwardParity) {
    const ImgSize env_size = {16, 8};
    const ImgSize dir_size = {8, 8};
    Variable env(VEC3, env_size);
    Variable dir(VEC3, dir_size);
    Variable uv(VEC2, dir_size);
    Variable inv_uv(VEC4, env_size);

    Variable eq = F::EquirectSample(env, dir);
    Variable bil = F::TextureBilinear(env, uv, inv_uv);
    Variable pooled = F::AvgPool2x2(env);  // 8x4
    Variable irr = F::IrradianceConv(pooled, {8, 4});
    Variable pre = F::PrefilterEnv(env, {8, 4}, 0.6f, 32);
    Variable lut = F::BrdfIntegrationLut({16, 16}, 64);
    Variable loss = F::Mean(eq) + F::Mean(bil) + F::Mean(irr) +
                    F::Mean(pre) + F::Mean(lut);

    const CpuImage env_img = MakeImage(env_size.w, env_size.h, 3, [](size_t i) {
        return 0.05f + 0.4f * float((i * 13) % 17) / 17.f;
    });
    // Directions spanning all octants (plus a zero background direction)
    CpuImage dir_img(dir_size.w, dir_size.h, 3);
    for (uint32_t py = 0; py < dir_size.h; py++) {
        for (uint32_t px = 0; px < dir_size.w; px++) {
            dir_img.at(px, py, 0) = std::sin(float(px) * 0.83f + 0.2f);
            dir_img.at(px, py, 1) = std::cos(float(py) * 0.57f);
            dir_img.at(px, py, 2) = std::sin(float(px + py) * 0.31f - 1.f);
        }
    }
    dir_img.at(0, 0, 0) = dir_img.at(0, 0, 1) = dir_img.at(0, 0, 2) = 0.f;
    const CpuImage uv_img = InteriorUvMap(dir_size);
    const CpuImage inv_img = SyntheticInvUv(env_size, dir_size);

    DressiAD ad;
    ad.setPackingMode(GetParam());
    ad.setLossVar(loss);
    ad.markOutput(eq);
    ad.markOutput(bil);
    ad.markOutput(irr);
    ad.markOutput(pre);
    ad.markOutput(lut);
    ad.sendImg(env, env_img);
    ad.sendImg(dir, dir_img);
    ad.sendImg(uv, uv_img);
    ad.sendImg(inv_uv, inv_img);
    ad.execStep();

    CpuEvaluator ev;
    ev.bindImage(env, env_img);
    ev.bindImage(dir, dir_img);
    ev.bindImage(uv, uv_img);
    ev.bindImage(inv_uv, inv_img);

    // Integer (Hammersley) paths are bit-identical; transcendentals differ
    // by ULPs -> 1e-3
    ExpectImagesNear(ad.recvImg(eq), CpuImageFromTensor(ev.eval(eq)), 1e-3f);
    ExpectImagesNear(ad.recvImg(bil), CpuImageFromTensor(ev.eval(bil)), 1e-4f);
    ExpectImagesNear(ad.recvImg(irr), CpuImageFromTensor(ev.eval(irr)), 1e-3f);
    ExpectImagesNear(ad.recvImg(pre), CpuImageFromTensor(ev.eval(pre)), 1e-3f);
    ExpectImagesNear(ad.recvImg(lut), CpuImageFromTensor(ev.eval(lut)), 1e-3f);
}

// Static-env reactive behavior: with the env leaf sent once and a dynamic
// dir leaf re-sent every iteration, the precompute branch must keep serving
// its cached images across the fast/full rebuilds (iters 2 and 8).
TEST_P(IblGpu, PrecomputeStaysCachedAcrossRebuilds) {
    const ImgSize env_size = {16, 8};
    const ImgSize dir_size = {8, 8};
    Variable env(VEC3, env_size);
    Variable dir(VEC3, dir_size);

    Variable pooled = F::AvgPool2x2(env);
    Variable irr = F::IrradianceConv(pooled, {8, 4});
    Variable pre = F::PrefilterEnv(env, {8, 4}, 0.4f, 32);
    Variable shade = F::EquirectSample(irr, dir) + F::EquirectSample(pre, dir);
    Variable loss = F::Mean(shade);

    const CpuImage env_img = MakeImage(env_size.w, env_size.h, 3, [](size_t i) {
        return 0.1f + 0.3f * float((i * 7) % 13) / 13.f;
    });
    const auto dir_at = [&](int iter) {
        return MakeImage(dir_size.w, dir_size.h, 3, [iter](size_t i) {
            return std::sin(float(i) * 0.13f + float(iter) * 0.05f) + 0.1f;
        });
    };

    DressiAD ad;
    ad.setPackingMode(GetParam());
    ad.setLossVar(loss);
    ad.markOutput(shade);
    ad.sendImg(env, env_img);

    CpuImage first;
    for (int iter = 0; iter < 12; iter++) {
        ad.sendImg(dir, dir_at(0));  // constant dir -> identical results
        ad.execStep();
        if (iter == 0) {
            first = ad.recvImg(shade);
        }
    }
    // After the fast (iter 2) and full (iter 8) rebuilds pruned the static
    // precompute, the cached irr/pre images must still feed the shading
    ExpectImagesNear(ad.recvImg(shade), first, 1e-5f);

    CpuEvaluator ev;
    ev.bindImage(env, env_img);
    ev.bindImage(dir, dir_at(0));
    ExpectImagesNear(ad.recvImg(shade), CpuImageFromTensor(ev.eval(shade)),
                     1e-3f);
}

// End-to-end bilinear texture recovery: validates the tent-weighted
// inverse-UV gather (__gather_inv_uv_bilinear__) on the GPU.
TEST_P(IblGpu, TextureBilinearRecovery) {
    const ImgSize tex_size = {4, 4};
    const ImgSize screen = {16, 16};
    Variable tex(VEC3, tex_size);
    Variable uv(VEC2, screen);
    Variable inv_uv(VEC4, tex_size);
    Variable target(VEC3, screen);
    Variable pred = F::TextureBilinear(tex, uv, inv_uv);
    Variable diff = pred - target;
    Variable loss = F::Mean(diff * diff);
    Variable tex_mut = tex;
    tex_mut.setRequiresGradRecursively();

    const CpuImage gt_tex = MakeImage(tex_size.w, tex_size.h, 3, [](size_t i) {
        return 0.15f + 0.7f * float((i * 11) % 9) / 9.f;
    });
    const CpuImage uv_img = InteriorUvMap(screen);
    const CpuImage inv_img = SyntheticInvUv(tex_size, screen);

    // GT render on the CPU oracle
    CpuEvaluator ev;
    ev.bindImage(tex, gt_tex);
    ev.bindImage(uv, uv_img);
    const CpuImage gt_render = CpuImageFromTensor(ev.eval(pred));

    // Unlike the nearest gather (one matching pixel per texel), the
    // bilinear gather accumulates EVERY contributing pixel (~sum w^2 per
    // texel), so the stable step is bounded by N / sum(w^2), not N
    const float lr = 15.f;
    DressiAD ad;
    ad.setPackingMode(GetParam());
    ad.setLossVar(loss);
    ad.setOptimizer([lr](Variables xs, Variables gxs) {
        Variables updated;
        for (size_t i = 0; i < xs.size(); i++) {
            updated.push_back(xs[i] - gxs[i] * lr);
        }
        return updated;
    });
    ad.sendImg(tex, CpuImage(tex_size.w, tex_size.h, 3, 0.f));
    ad.sendImg(uv, uv_img);
    ad.sendImg(inv_uv, inv_img);
    ad.sendImg(target, gt_render);

    for (int iter = 0; iter < 500; iter++) {
        ad.execStep();
    }
    EXPECT_LT(ad.recvImg(loss).data[0], 1e-5f);

    const CpuImage recovered = ad.recvImg(tex);
    float max_err = 0.f;
    for (size_t i = 0; i < recovered.data.size(); i++) {
        max_err = std::max(max_err,
                           std::abs(recovered.data[i] - gt_tex.data[i]));
    }
    EXPECT_LT(max_err, 0.05f);
}

INSTANTIATE_TEST_SUITE_P(
        PackingModes, IblGpu,
        ::testing::Values(DressiAD::PackingMode::Naive,
                          DressiAD::PackingMode::RSP),
        [](const ::testing::TestParamInfo<DressiAD::PackingMode>& info) {
            return info.param == DressiAD::PackingMode::Naive ? "Naive"
                                                              : "RSP";
        });
