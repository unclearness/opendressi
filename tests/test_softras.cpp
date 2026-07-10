// HardSoftRas CPU tests: signed-distance forward profile and (from B2 on)
// finite-difference gradient checks of the dist-channel backward.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "core/build_backward.h"
#include "core/cpu_eval.h"
#include "dressi/dressi.h"

using namespace dressi;

namespace {

constexpr uint32_t kScreen = 16;
constexpr float kRadius = 3.f;

CpuTensor MakeTensor(VType vtype, ImgSize size, std::vector<float> data) {
    CpuTensor t;
    t.vtype = vtype;
    t.size = size;
    t.data = std::move(data);
    return t;
}

// Clip position for a screen-space point (w = 1): clip = s / S * 2 - 1
std::array<float, 4> ClipOfScreen(float sx, float sy, float z, float w = 1.f) {
    return {(sx / kScreen * 2.f - 1.f) * w, (sy / kScreen * 2.f - 1.f) * w,
            z * w, w};
}

// A soft-rasterization scene: hard triangles + soft triangles enlarged
// about the screen-space centroid by `scale` (w is uniform per face here,
// so scaling clip xy about the centroid scales screen coords identically).
struct SoftScene {
    Variable soft_clip, face_id, faces_soft, hard_clip, faces_tex;
    CpuTensor t_soft_clip, t_face_id, t_faces_soft, t_hard_clip, t_faces_tex;

    SoftScene(const std::vector<std::array<float, 4>>& hard_verts,
              const std::vector<std::array<uint32_t, 3>>& faces, float scale)
        : soft_clip(VEC4, {uint32_t(faces.size()) * 3, 1}),
          face_id(FLOAT, {uint32_t(faces.size()) * 3, 1}),
          faces_soft(IVEC3, {uint32_t(faces.size()), 1}),
          hard_clip(VEC4, {uint32_t(hard_verts.size()), 1}),
          faces_tex(VEC3, {uint32_t(faces.size()), 1}) {
        const uint32_t n_faces = uint32_t(faces.size());
        t_soft_clip = MakeTensor(VEC4, {n_faces * 3, 1},
                                 std::vector<float>(n_faces * 12, 0.f));
        t_face_id = MakeTensor(FLOAT, {n_faces * 3, 1},
                               std::vector<float>(n_faces * 3, 0.f));
        t_faces_soft = MakeTensor(IVEC3, {n_faces, 1},
                                  std::vector<float>(n_faces * 3, 0.f));
        t_faces_tex = MakeTensor(VEC3, {n_faces, 1},
                                 std::vector<float>(n_faces * 3, 0.f));
        std::vector<float> hc;
        for (const auto& v : hard_verts) {
            hc.insert(hc.end(), v.begin(), v.end());
        }
        t_hard_clip = MakeTensor(VEC4, {uint32_t(hard_verts.size()), 1},
                                 std::move(hc));

        for (uint32_t f = 0; f < n_faces; f++) {
            float cx = 0.f, cy = 0.f;
            for (int k = 0; k < 3; k++) {
                const auto& v = hard_verts[faces[f][size_t(k)]];
                cx += v[0] / v[3] / 3.f;
                cy += v[1] / v[3] / 3.f;
            }
            for (int k = 0; k < 3; k++) {
                const auto& v = hard_verts[faces[f][size_t(k)]];
                const uint32_t sv = f * 3 + uint32_t(k);
                const float w = v[3];
                t_soft_clip.data[sv * 4 + 0] =
                        (cx + (v[0] / w - cx) * scale) * w;
                t_soft_clip.data[sv * 4 + 1] =
                        (cy + (v[1] / w - cy) * scale) * w;
                t_soft_clip.data[sv * 4 + 2] = v[2];
                t_soft_clip.data[sv * 4 + 3] = w;
                t_face_id.data[sv] = float(f);
                t_faces_soft.data[f * 3 + uint32_t(k)] = float(sv);
                t_faces_tex.data[f * 3 + uint32_t(k)] =
                        float(faces[f][size_t(k)]);
            }
        }
    }

    void bind(CpuEvaluator& ev) const {
        ev.bind(soft_clip, t_soft_clip);
        ev.bind(face_id, t_face_id);
        ev.bind(faces_soft, t_faces_soft);
        ev.bind(hard_clip, t_hard_clip);
        ev.bind(faces_tex, t_faces_tex);
    }

    Variable rasterize(float radius = kRadius) const {
        return F::RasterizeSoft(soft_clip, face_id, faces_soft, hard_clip,
                                faces_tex, {kScreen, kScreen}, radius);
    }
};

float EvalLoss(const Variable& loss, CpuEvaluator& ev) {
    ev.clearCache();
    return ev.eval(loss).data[0];
}

// Central finite differences vs the analytic gradient for `wrt`, with the
// perturbation applied to `wrt_tensor` (re-bound each probe)
void CheckGradAgainstFd(const Variable& loss, const SoftScene& scene,
                        const Variable& wrt, CpuTensor wrt_tensor,
                        const CpuTensor& target_tensor,
                        const Variable& target, float h = 1e-3f,
                        float rel_tol = 2e-2f, float abs_tol = 2e-3f) {
    Variable wrt_mut = wrt;
    wrt_mut.setRequiresGradRecursively();
    auto [input_vars, input_grad_vars] = BuildBackward(loss);
    Variable grad_var(nullptr);
    for (size_t i = 0; i < input_vars.size(); i++) {
        if (input_vars[i] == wrt) {
            grad_var = input_grad_vars[i];
        }
    }
    ASSERT_TRUE(grad_var);
    ASSERT_EQ(grad_var.getVType(), wrt.getVType());
    ASSERT_EQ(grad_var.getImgSize(), wrt.getImgSize());

    CpuEvaluator ev;
    scene.bind(ev);
    ev.bind(target, target_tensor);
    ev.bind(wrt, wrt_tensor);
    const CpuTensor analytic = ev.eval(grad_var);

    for (size_t i = 0; i < wrt_tensor.data.size(); i++) {
        const float orig = wrt_tensor.data[i];
        wrt_tensor.data[i] = orig + h;
        ev.bind(wrt, wrt_tensor);
        const float f_plus = EvalLoss(loss, ev);
        wrt_tensor.data[i] = orig - h;
        ev.bind(wrt, wrt_tensor);
        const float f_minus = EvalLoss(loss, ev);
        wrt_tensor.data[i] = orig;
        ev.bind(wrt, wrt_tensor);

        const float numeric = (f_plus - f_minus) / (2.f * h);
        const float tol = abs_tol + rel_tol * std::abs(numeric);
        EXPECT_NEAR(analytic.data[i], numeric, tol) << "component " << i;
        if (i % 4 == 2) {  // clip z never influences the dist channel
            EXPECT_FLOAT_EQ(analytic.data[i], 0.f);
        }
    }
}

// pred = coverage * sigmoid(dist / sigma), loss = Mean((target - pred)^2)
Variable SilhouetteLoss(const SoftScene& scene, const Variable& target,
                        float radius = kRadius) {
    Variable out = scene.rasterize(radius);
    Variable pred = F::StopGradient(F::GetW(out)) *
                    F::Sigmoid(F::GetX(out) * (7.f / radius));
    Variable diff = F::StopGradient(target) - pred;
    return F::Mean(diff * diff);
}

}  // namespace

TEST(SoftRas, ForwardDistProfile) {
    // Right triangle with screen corners (4,4), (12,4), (4,12), z = 0.5
    SoftScene scene({ClipOfScreen(4, 4, 0.5f), ClipOfScreen(12, 4, 0.5f),
                     ClipOfScreen(4, 12, 0.5f)},
                    {{0, 1, 2}}, 1.5f);
    CpuEvaluator ev;
    scene.bind(ev);
    const CpuTensor out = ev.eval(scene.rasterize());
    ASSERT_EQ(out.vtype, VEC4);

    const auto px = [&](uint32_t x, uint32_t y, uint32_t c) {
        return out.data[(size_t(y) * kScreen + x) * 4 + c];
    };

    // Interior pixel (6,6), center (6.5, 6.5): min distance over the edges
    // y=4 (2.5), x=4 (2.5), x+y=16 (3/sqrt(2))
    EXPECT_NEAR(px(6, 6, 0), 3.f / std::sqrt(2.f), 1e-4f);
    EXPECT_FLOAT_EQ(px(6, 6, 1), 0.f);        // face id
    EXPECT_NEAR(px(6, 6, 2), 0.5f, 1e-5f);    // hard z
    EXPECT_FLOAT_EQ(px(6, 6, 3), 1.f);        // coverage

    // Soft-rim pixel (3,6), center (3.5, 6.5): 0.5 px outside edge x=4
    EXPECT_NEAR(px(3, 6, 0), -0.5f, 1e-4f);
    EXPECT_FLOAT_EQ(px(3, 6, 3), 1.f);

    // Far background stays cleared
    for (uint32_t c = 0; c < 4; c++) {
        EXPECT_FLOAT_EQ(px(14, 14, c), 0.f);
    }
}

TEST(SoftRas, DepthShiftNearWins) {
    // Two identical screen triangles at different depths; the far one is
    // listed second (draw order must not matter, only the shifted depth)
    SoftScene scene({ClipOfScreen(4, 4, 0.2f), ClipOfScreen(12, 4, 0.2f),
                     ClipOfScreen(4, 12, 0.2f), ClipOfScreen(4, 4, 0.8f),
                     ClipOfScreen(12, 4, 0.8f), ClipOfScreen(4, 12, 0.8f)},
                    {{3, 4, 5}, {0, 1, 2}}, 1.5f);
    CpuEvaluator ev;
    scene.bind(ev);
    const CpuTensor out = ev.eval(scene.rasterize());

    const size_t o = (size_t(6) * kScreen + 6) * 4;
    EXPECT_FLOAT_EQ(out.data[o + 1], 1.f);       // near face id
    EXPECT_NEAR(out.data[o + 2], 0.2f, 1e-5f);   // near hard z
}

TEST(SoftRas, DistGradMatchesFiniteDifferences) {
    // Generic-position triangle (edges avoid pixel centers), w = 1
    SoftScene scene({ClipOfScreen(4.3f, 3.8f, 0.5f),
                     ClipOfScreen(12.4f, 5.2f, 0.5f),
                     ClipOfScreen(5.1f, 12.6f, 0.5f)},
                    {{0, 1, 2}}, 1.6f);
    Variable target(FLOAT, {kScreen, kScreen});
    CpuTensor t_target = MakeTensor(
            FLOAT, {kScreen, kScreen},
            std::vector<float>(size_t(kScreen) * kScreen, 0.f));
    for (size_t i = 0; i < t_target.data.size(); i++) {
        t_target.data[i] = 0.2f + 0.5f * float(i % 3);
    }
    Variable loss = SilhouetteLoss(scene, target);
    CheckGradAgainstFd(loss, scene, scene.hard_clip, scene.t_hard_clip,
                       t_target, target);
}

TEST(SoftRas, DistGradPerspectiveW) {
    // One vertex at w = 1.3 exercises the -x/w^2 Jacobian terms
    auto v1 = ClipOfScreen(12.4f, 5.2f, 0.5f, 1.3f);
    SoftScene scene({ClipOfScreen(4.3f, 3.8f, 0.5f), v1,
                     ClipOfScreen(5.1f, 12.6f, 0.5f)},
                    {{0, 1, 2}}, 1.6f);
    Variable target(FLOAT, {kScreen, kScreen});
    CpuTensor t_target = MakeTensor(
            FLOAT, {kScreen, kScreen},
            std::vector<float>(size_t(kScreen) * kScreen, 0.4f));
    Variable loss = SilhouetteLoss(scene, target);
    CheckGradAgainstFd(loss, scene, scene.hard_clip, scene.t_hard_clip,
                       t_target, target);
}
