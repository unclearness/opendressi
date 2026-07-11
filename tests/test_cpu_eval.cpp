// CPU evaluator tests: per-op forward values against analytic references,
// including a PBR-shaped composite (GGX + Smith + Fresnel-Schlick).

#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include "core/cpu_eval.h"
#include "dressi/dressi.h"

using namespace dressi;

namespace {

CpuTensor ScalarTensor(float v) {
    CpuTensor t;
    t.vtype = FLOAT;
    t.size = {1, 1};
    t.data = {v};
    return t;
}

CpuTensor Vec3Tensor(float x, float y, float z) {
    CpuTensor t;
    t.vtype = VEC3;
    t.size = {1, 1};
    t.data = {x, y, z};
    return t;
}

float EvalScalar1(Variable (*op)(const Variable&), float x) {
    Variable vx(FLOAT, {1, 1});
    Variable y = op(vx);
    CpuEvaluator ev;
    ev.bind(vx, ScalarTensor(x));
    return ev.eval(y).data[0];
}

float EvalScalar2(Variable (*op)(const Variable&, const Variable&), float a,
                  float b) {
    Variable va(FLOAT, {1, 1});
    Variable vb(FLOAT, {1, 1});
    Variable y = op(va, vb);
    CpuEvaluator ev;
    ev.bind(va, ScalarTensor(a));
    ev.bind(vb, ScalarTensor(b));
    return ev.eval(y).data[0];
}

}  // namespace

TEST(CpuEval, UnaryMath) {
    EXPECT_FLOAT_EQ(EvalScalar1(F::Sin, 0.5f), std::sin(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::Cos, 0.5f), std::cos(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::Tan, 0.5f), std::tan(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::Asin, 0.5f), std::asin(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::Acos, 0.5f), std::acos(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::Atan, 0.5f), std::atan(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::Exp, 0.5f), std::exp(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::Exp2, 0.5f), std::exp2(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::Log, 0.5f), std::log(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::Log2, 0.5f), std::log2(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::Sqrt, 0.5f), std::sqrt(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::InverseSqrt, 0.5f), 1.f / std::sqrt(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::Abs, -0.5f), 0.5f);
    EXPECT_FLOAT_EQ(EvalScalar1(F::Sign, -0.5f), -1.f);
    EXPECT_FLOAT_EQ(EvalScalar1(F::Floor, 1.7f), 1.f);
    EXPECT_FLOAT_EQ(EvalScalar1(F::Fract, 1.7f), 0.7f);
}

TEST(CpuEval, BinaryMath) {
    EXPECT_FLOAT_EQ(EvalScalar2(F::Add, 2.f, 3.f), 5.f);
    EXPECT_FLOAT_EQ(EvalScalar2(F::Sub, 2.f, 3.f), -1.f);
    EXPECT_FLOAT_EQ(EvalScalar2(F::Mul, 2.f, 3.f), 6.f);
    EXPECT_FLOAT_EQ(EvalScalar2(F::Div, 3.f, 2.f), 1.5f);
    EXPECT_FLOAT_EQ(EvalScalar2(F::Pow, 2.f, 3.f), 8.f);
    EXPECT_FLOAT_EQ(EvalScalar2(F::Max, 2.f, 3.f), 3.f);
    EXPECT_FLOAT_EQ(EvalScalar2(F::Min, 2.f, 3.f), 2.f);
    EXPECT_FLOAT_EQ(EvalScalar2(F::Mod, 7.f, 3.f), 1.f);
    EXPECT_FLOAT_EQ(EvalScalar2(F::Atan2, 1.f, 2.f), std::atan2(1.f, 2.f));
    EXPECT_FLOAT_EQ(EvalScalar2(F::Step, 2.f, 3.f), 1.f);
    EXPECT_FLOAT_EQ(EvalScalar2(F::Step, 3.f, 2.f), 0.f);
}

TEST(CpuEval, OperatorsAndConstants) {
    Variable x(FLOAT, {1, 1});
    Variable y = (2.f * x + 1.f) / (x - 4.f);
    CpuEvaluator ev;
    ev.bind(x, ScalarTensor(2.f));
    EXPECT_FLOAT_EQ(ev.eval(y).data[0], (2.f * 2.f + 1.f) / (2.f - 4.f));
}

TEST(CpuEval, VectorOps) {
    Variable a(VEC3, {1, 1});
    Variable b(VEC3, {1, 1});
    CpuEvaluator ev;
    ev.bind(a, Vec3Tensor(1.f, 2.f, 3.f));
    ev.bind(b, Vec3Tensor(4.f, -5.f, 6.f));

    EXPECT_FLOAT_EQ(ev.eval(F::Dot(a, b)).data[0], 1.f * 4 - 2.f * 5 + 3.f * 6);

    auto cross = ev.eval(F::Cross(a, b));
    EXPECT_FLOAT_EQ(cross.data[0], 2.f * 6 - 3.f * (-5));
    EXPECT_FLOAT_EQ(cross.data[1], 3.f * 4 - 1.f * 6);
    EXPECT_FLOAT_EQ(cross.data[2], 1.f * (-5) - 2.f * 4);

    const float len = std::sqrt(1.f + 4.f + 9.f);
    EXPECT_FLOAT_EQ(ev.eval(F::Length(a)).data[0], len);

    auto norm = ev.eval(F::Normalize(a));
    EXPECT_FLOAT_EQ(norm.data[1], 2.f / len);

    // reflect(I,N) with N normalized
    Variable n = F::Normalize(b);
    auto refl = ev.eval(F::Reflect(a, n));
    // reference: I - 2*dot(N,I)*N
    auto nn = ev.eval(n);
    const float d = nn.data[0] * 1.f + nn.data[1] * 2.f + nn.data[2] * 3.f;
    EXPECT_NEAR(refl.data[0], 1.f - 2.f * d * nn.data[0], 1e-5f);
    EXPECT_NEAR(refl.data[2], 3.f - 2.f * d * nn.data[2], 1e-5f);

    EXPECT_FLOAT_EQ(ev.eval(F::CompSum(a)).data[0], 6.f);
    EXPECT_FLOAT_EQ(ev.eval(F::GetY(a)).data[0], 2.f);

    auto v4 = ev.eval(F::Vec4(a, F::GetX(b)));
    EXPECT_FLOAT_EQ(v4.data[3], 4.f);
}

TEST(CpuEval, ScalarVectorBroadcast) {
    Variable s(FLOAT, {1, 1});
    Variable v(VEC3, {1, 1});
    CpuEvaluator ev;
    ev.bind(s, ScalarTensor(2.f));
    ev.bind(v, Vec3Tensor(1.f, 2.f, 3.f));
    auto y = ev.eval(s * v);
    EXPECT_EQ(y.vtype, VEC3);
    EXPECT_FLOAT_EQ(y.data[2], 6.f);
}

TEST(CpuEval, MatrixOps) {
    Variable m(MAT3, {1, 1});
    Variable v(VEC3, {1, 1});
    CpuEvaluator ev;
    // Column-major: columns (1,0,0), (0,2,0), (0,0,4) with m[2][1]=3
    // i.e. col2 = (0,3,4)
    CpuTensor mt;
    mt.vtype = MAT3;
    mt.size = {1, 1};
    mt.data = {1, 0, 0, 0, 2, 0, 0, 3, 4};
    ev.bind(m, mt);
    ev.bind(v, Vec3Tensor(1.f, 1.f, 1.f));

    auto mv = ev.eval(F::MatMul(m, v));
    EXPECT_FLOAT_EQ(mv.data[0], 1.f);
    EXPECT_FLOAT_EQ(mv.data[1], 2.f + 3.f);
    EXPECT_FLOAT_EQ(mv.data[2], 4.f);

    auto mtj = ev.eval(F::Transpose(m));
    EXPECT_FLOAT_EQ(mtj.data[1 * 3 + 2], 3.f);  // transposed position

    auto outer = ev.eval(F::OuterProduct(v, v));
    EXPECT_EQ(outer.vtype, MAT3);
    EXPECT_FLOAT_EQ(outer.data[0], 1.f);
}

TEST(CpuEval, SpatialImageOpsAndReductions) {
    const uint32_t W = 37, H = 23;
    Variable img(FLOAT, {W, H});
    CpuTensor t;
    t.vtype = FLOAT;
    t.size = {W, H};
    t.data.resize(size_t(W) * H);
    double ref_sum = 0.0;
    for (size_t i = 0; i < t.data.size(); i++) {
        t.data[i] = float(i % 17) * 0.25f - 1.f;
        ref_sum += t.data[i];
    }
    CpuEvaluator ev;
    ev.bind(img, t);

    auto sum = ev.eval(F::Sum(img));
    EXPECT_EQ(sum.size, (ImgSize{1, 1}));
    EXPECT_NEAR(sum.data[0], float(ref_sum), 1e-2f);

    auto mean = ev.eval(F::Mean(img));
    EXPECT_NEAR(mean.data[0], float(ref_sum / (W * H)), 1e-5f);
}

TEST(CpuEval, SumPixelWise) {
    Variable a(FLOAT, {4, 4});
    Variable b(FLOAT, {4, 4});
    Variable c(FLOAT, {1, 1});
    Variable y = F::SumPixelWise({a, b, c});
    EXPECT_EQ(y.getImgSize(), (ImgSize{4, 4}));

    CpuEvaluator ev;
    CpuTensor ta, tb;
    ta.vtype = tb.vtype = FLOAT;
    ta.size = tb.size = {4, 4};
    ta.data.assign(16, 1.f);
    tb.data.assign(16, 2.f);
    ev.bind(a, ta);
    ev.bind(b, tb);
    ev.bind(c, ScalarTensor(10.f));
    EXPECT_FLOAT_EQ(ev.eval(y).data[5], 13.f);
}

// PBR-shaped composite: Cook-Torrance GGX specular for a single direction
// setup, evaluated per-pixel over roughness values.
TEST(CpuEval, PbrCompositeExpressible) {
    const uint32_t W = 8, H = 8;
    Variable n(VEC3, {W, H});      // normal
    Variable v(VEC3, {1, 1});      // view dir
    Variable l(VEC3, {1, 1});      // light dir
    Variable rough(FLOAT, {W, H});
    Variable f0(VEC3, {1, 1});

    Variable h = F::Normalize(v + l);
    Variable n_dot_h = F::Max(F::Dot(n, h), F::Float(0.f));
    Variable n_dot_v = F::Max(F::Dot(n, v), F::Float(1e-4f));
    Variable n_dot_l = F::Max(F::Dot(n, l), F::Float(1e-4f));
    Variable v_dot_h = F::Max(F::Dot(v, h), F::Float(0.f));

    // GGX normal distribution
    Variable alpha = rough * rough;
    Variable alpha2 = alpha * alpha;
    Variable denom_d = n_dot_h * n_dot_h * (alpha2 - 1.f) + 1.f;
    Variable dist =
            alpha2 / (F::Float(3.14159265f) * denom_d * denom_d);

    // Smith GGX geometry (Schlick-GGX)
    Variable k = (rough + 1.f) * (rough + 1.f) / 8.f;
    Variable g_v = n_dot_v / (n_dot_v * (1.f - k) + k);
    Variable g_l = n_dot_l / (n_dot_l * (1.f - k) + k);
    Variable geom = g_v * g_l;

    // Fresnel-Schlick
    Variable fresnel =
            f0 + (1.f - f0) * F::Pow(1.f - v_dot_h, F::Float(5.f));

    Variable spec = fresnel * (dist * geom /
                               (4.f * n_dot_v * n_dot_l + 1e-4f));
    // Tone map (Reinhard) + gamma
    Variable mapped = spec / (spec + 1.f);
    Variable ldr = F::Pow(mapped, F::Float(1.f / 2.2f));

    EXPECT_EQ(ldr.getVType(), VEC3);
    EXPECT_EQ(ldr.getImgSize(), (ImgSize{W, H}));

    CpuEvaluator ev;
    CpuTensor tn, tr;
    tn.vtype = VEC3;
    tn.size = {W, H};
    tn.data.resize(size_t(W) * H * 3);
    for (size_t p = 0; p < size_t(W) * H; p++) {
        tn.data[p * 3 + 0] = 0.f;
        tn.data[p * 3 + 1] = 0.f;
        tn.data[p * 3 + 2] = 1.f;
    }
    tr.vtype = FLOAT;
    tr.size = {W, H};
    tr.data.resize(size_t(W) * H);
    for (size_t p = 0; p < tr.data.size(); p++) {
        tr.data[p] = 0.1f + 0.8f * float(p) / float(tr.data.size());
    }
    ev.bind(n, tn);
    ev.bind(rough, tr);
    ev.bind(v, Vec3Tensor(0.f, 0.f, 1.f));
    ev.bind(l, Vec3Tensor(0.f, 0.f, 1.f));
    ev.bind(f0, Vec3Tensor(0.04f, 0.04f, 0.04f));

    auto out = ev.eval(ldr);
    ASSERT_EQ(out.data.size(), size_t(W) * H * 3);
    for (float c : out.data) {
        EXPECT_TRUE(std::isfinite(c));
        EXPECT_GE(c, 0.f);
        EXPECT_LE(c, 1.f);
    }
    // Rougher pixels produce weaker mirror-direction specular
    EXPECT_GT(out.data[0], out.data[(size_t(W) * H - 1) * 3]);
}

// ------------------------- CPU reference rasterizer --------------------------

namespace {

CpuTensor ClipTensor(const std::vector<std::array<float, 4>>& verts) {
    CpuTensor t;
    t.vtype = VEC4;
    t.size = {uint32_t(verts.size()), 1};
    for (const auto& v : verts) {
        t.data.insert(t.data.end(), v.begin(), v.end());
    }
    return t;
}

CpuTensor FacesTensor(const std::vector<std::array<float, 3>>& faces) {
    CpuTensor t;
    t.vtype = IVEC3;
    t.size = {uint32_t(faces.size()), 1};
    for (const auto& f : faces) {
        t.data.insert(t.data.end(), f.begin(), f.end());
    }
    return t;
}

}  // namespace

TEST(CpuEval, RasterizeHardQuad) {
    // Fullscreen quad: mask = 1 everywhere, UV interpolates pixel centers
    const ImgSize screen = {8, 8};
    Variable pos(VEC4, {4, 1});
    Variable uv(VEC2, {4, 1});
    Variable faces(IVEC3, {2, 1});

    const float p[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    CpuTensor tp, tu;
    tp.vtype = VEC4;
    tp.size = {4, 1};
    tu.vtype = VEC2;
    tu.size = {4, 1};
    for (int i = 0; i < 4; i++) {
        tp.data.insert(tp.data.end(), {p[i][0], p[i][1], 0.5f, 1.f});
        tu.data.insert(tu.data.end(),
                       {(p[i][0] + 1.f) * 0.5f, (p[i][1] + 1.f) * 0.5f});
    }
    CpuTensor tf = FacesTensor({{0, 1, 2}, {0, 2, 3}});

    Variable g_uv = F::Rasterize(pos, uv, faces, screen);
    CpuEvaluator ev;
    ev.bind(pos, tp);
    ev.bind(uv, tu);
    ev.bind(faces, tf);
    const CpuTensor out = ev.eval(g_uv);

    for (uint32_t y = 0; y < screen.h; y++) {
        for (uint32_t x = 0; x < screen.w; x++) {
            const size_t o = (size_t(y) * screen.w + x) * 2;
            ASSERT_NEAR(out.data[o + 0], (float(x) + 0.5f) / screen.w, 1e-5f);
            ASSERT_NEAR(out.data[o + 1], (float(y) + 0.5f) / screen.h, 1e-5f);
        }
    }
}

TEST(CpuEval, RasterizeHardPerspective) {
    // One triangle with a non-uniform w: interpolation must be
    // perspective-correct (u/w and 1/w affine in screen space)
    const ImgSize screen = {8, 8};
    const float w1 = 3.f;
    // Screen targets: s0=(0,0), s1=(8,0), s2=(0,8); clip = (s/S*2-1)*w
    CpuTensor tp = ClipTensor({{-1.f, -1.f, 0.25f, 1.f},
                               {1.f * w1, -1.f * w1, 0.25f * w1, w1},
                               {-1.f, 1.f, 0.25f, 1.f}});
    CpuTensor tu;
    tu.vtype = FLOAT;
    tu.size = {3, 1};
    tu.data = {0.f, 1.f, 0.f};
    CpuTensor tf = FacesTensor({{0, 1, 2}});

    Variable pos(VEC4, {3, 1});
    Variable attr(FLOAT, {3, 1});
    Variable faces(IVEC3, {1, 1});
    Variable g = F::Rasterize(pos, attr, faces, screen);
    CpuEvaluator ev;
    ev.bind(pos, tp);
    ev.bind(attr, tu);
    ev.bind(faces, tf);
    const CpuTensor out = ev.eval(g);

    // Probe interior pixels; expected from the affine u/w, 1/w derivation
    const float s[3][2] = {{0, 0}, {8, 0}, {0, 8}};
    const float u[3] = {0.f, 1.f, 0.f};
    const float w[3] = {1.f, w1, 1.f};
    for (uint32_t y = 0; y < 3; y++) {
        for (uint32_t x = 0; x < 3; x++) {
            const float px = float(x) + 0.5f;
            const float py = float(y) + 0.5f;
            const float area = (s[1][0] - s[0][0]) * (s[2][1] - s[0][1]) -
                               (s[1][1] - s[0][1]) * (s[2][0] - s[0][0]);
            const float l0 = ((s[1][0] - px) * (s[2][1] - py) -
                              (s[1][1] - py) * (s[2][0] - px)) /
                             area;
            const float l1 = ((s[2][0] - px) * (s[0][1] - py) -
                              (s[2][1] - py) * (s[0][0] - px)) /
                             area;
            const float l2 = 1.f - l0 - l1;
            if (l0 < 0.f || l1 < 0.f || l2 < 0.f) {
                continue;
            }
            const float num = l0 * u[0] / w[0] + l1 * u[1] / w[1] +
                              l2 * u[2] / w[2];
            const float den = l0 / w[0] + l1 / w[1] + l2 / w[2];
            ASSERT_NEAR(out.data[size_t(y) * screen.w + x], num / den, 1e-5f)
                    << "pixel " << x << "," << y;
        }
    }
    // A pixel with equal screen-barycentric weight of v1 must NOT match the
    // non-perspective (affine) interpolation
    EXPECT_GT(std::abs(out.data[size_t(1) * screen.w + 1] - 0.1875f), 1e-3f);
}

TEST(CpuEval, RasterizeDepthAndBackground) {
    // Two overlapping fullscreen triangles at different depths: the nearer
    // one wins; a half-covering quad leaves background pixels at 0
    const ImgSize screen = {8, 8};
    CpuTensor tp = ClipTensor({// far triangle (z=0.8), attr 1
                               {-1, -1, 0.8f, 1},
                               {3, -1, 0.8f, 1},
                               {-1, 3, 0.8f, 1},
                               // near triangle (z=0.2), attr 2
                               {-1, -1, 0.2f, 1},
                               {3, -1, 0.2f, 1},
                               {-1, 3, 0.2f, 1}});
    CpuTensor ta;
    ta.vtype = FLOAT;
    ta.size = {6, 1};
    ta.data = {1, 1, 1, 2, 2, 2};
    // Near triangle drawn FIRST: depth test (not draw order) must decide
    CpuTensor tf = FacesTensor({{3, 4, 5}, {0, 1, 2}});

    Variable pos(VEC4, {6, 1});
    Variable attr(FLOAT, {6, 1});
    Variable faces(IVEC3, {2, 1});
    Variable g = F::Rasterize(pos, attr, faces, screen);
    CpuEvaluator ev;
    ev.bind(pos, tp);
    ev.bind(attr, ta);
    ev.bind(faces, tf);
    const CpuTensor out = ev.eval(g);
    EXPECT_FLOAT_EQ(out.data[0], 2.f);  // near wins

    // Half-covering quad: right half is background
    CpuTensor tp2 = ClipTensor({{-1, -1, 0.5f, 1},
                                {0, -1, 0.5f, 1},
                                {0, 1, 0.5f, 1},
                                {-1, 1, 0.5f, 1}});
    CpuTensor ta2;
    ta2.vtype = FLOAT;
    ta2.size = {4, 1};
    ta2.data = {1, 1, 1, 1};
    CpuTensor tf2 = FacesTensor({{0, 1, 2}, {0, 2, 3}});
    Variable pos2(VEC4, {4, 1});
    Variable attr2(FLOAT, {4, 1});
    Variable faces2(IVEC3, {2, 1});
    Variable g2 = F::Rasterize(pos2, attr2, faces2, screen);
    CpuEvaluator ev2;
    ev2.bind(pos2, tp2);
    ev2.bind(attr2, ta2);
    ev2.bind(faces2, tf2);
    const CpuTensor out2 = ev2.eval(g2);
    for (uint32_t y = 0; y < screen.h; y++) {
        EXPECT_FLOAT_EQ(out2.data[size_t(y) * screen.w + 1], 1.f);
        EXPECT_FLOAT_EQ(out2.data[size_t(y) * screen.w + 6], 0.f);
    }
}

// --------------------------------- IBL ops -----------------------------------

namespace {

CpuTensor FillTensor(VType vtype, ImgSize size,
                     const std::function<float(size_t)>& gen) {
    CpuTensor t;
    t.vtype = vtype;
    t.size = size;
    t.data.resize(size_t(size.w) * size.h * NumComponents(vtype));
    for (size_t i = 0; i < t.data.size(); i++) {
        t.data[i] = gen(i);
    }
    return t;
}

}  // namespace

TEST(CpuEval, EquirectSampleConstantEnvAndZeroDir) {
    const ImgSize env_size = {8, 4};
    Variable env(VEC3, env_size);
    Variable dir(VEC3, {6, 1});
    Variable out = F::EquirectSample(env, dir);
    EXPECT_EQ(out.getVType(), VEC3);
    EXPECT_EQ(out.getImgSize(), (ImgSize{6, 1}));

    CpuEvaluator ev;
    ev.bind(env, FillTensor(VEC3, env_size, [](size_t i) {
                return float(i % 3) * 0.25f + 0.1f;  // (0.1, 0.35, 0.6)
            }));
    // Poles, seam-adjacent, axis dirs, unnormalized, and zero (background)
    CpuTensor dirs;
    dirs.vtype = VEC3;
    dirs.size = {6, 1};
    dirs.data = {0.f, 1.f,  0.f,   0.f, -1.f, 0.f,  -1.f, 0.f, 1e-4f,
                 2.f, 0.5f, -1.f,  0.f, 0.f,  1.f,  0.f,  0.f, 0.f};
    ev.bind(dir, dirs);

    const CpuTensor y = ev.eval(out);
    for (uint32_t p = 0; p < 5; p++) {
        EXPECT_NEAR(y.data[p * 3 + 0], 0.1f, 1e-5f) << p;
        EXPECT_NEAR(y.data[p * 3 + 1], 0.35f, 1e-5f) << p;
        EXPECT_NEAR(y.data[p * 3 + 2], 0.6f, 1e-5f) << p;
    }
    // Zero-length direction returns 0 (rasterizer background)
    for (uint32_t c = 0; c < 3; c++) {
        EXPECT_FLOAT_EQ(y.data[5 * 3 + c], 0.f);
    }
}

TEST(CpuEval, EquirectSampleSeamContinuity) {
    const ImgSize env_size = {16, 8};
    Variable env(VEC3, env_size);
    Variable dir(VEC3, {2, 1});
    Variable out = F::EquirectSample(env, dir);

    CpuEvaluator ev;
    ev.bind(env, FillTensor(VEC3, env_size, [](size_t i) {
                return 0.05f + 0.13f * float((i * 7) % 29) / 29.f;
            }));
    // phi = pi - eps vs -pi + eps: the u = 1/0 wrap must interpolate the
    // same two texel columns
    const float eps = 1e-4f;
    CpuTensor dirs;
    dirs.vtype = VEC3;
    dirs.size = {2, 1};
    dirs.data = {-std::cos(eps), 0.2f, std::sin(eps),
                 -std::cos(eps), 0.2f, -std::sin(eps)};
    ev.bind(dir, dirs);

    const CpuTensor y = ev.eval(out);
    for (uint32_t c = 0; c < 3; c++) {
        EXPECT_NEAR(y.data[c], y.data[3 + c], 1e-3f) << "channel " << c;
    }
}

TEST(CpuEval, TextureBilinearExactInterpolation) {
    const ImgSize tex_size = {2, 2};
    Variable tex(FLOAT, tex_size);
    Variable uv(VEC2, {4, 1});
    Variable inv_uv(VEC4, tex_size);  // forward ignores it
    Variable out = F::TextureBilinear(tex, uv, inv_uv);

    CpuEvaluator ev;
    CpuTensor tex_t;
    tex_t.vtype = FLOAT;
    tex_t.size = tex_size;
    tex_t.data = {0.f, 1.f, 2.f, 3.f};
    ev.bind(tex, tex_t);
    ev.bind(inv_uv, FillTensor(VEC4, tex_size, [](size_t) { return 0.f; }));
    CpuTensor uvs;
    uvs.vtype = VEC2;
    uvs.size = {4, 1};
    uvs.data = {0.25f, 0.25f,   // texel (0,0) center -> 0
                0.75f, 0.75f,   // texel (1,1) center -> 3
                0.5f,  0.5f,    // center of all four -> 1.5
                0.5f,  0.25f};  // midpoint of top row -> 0.5
    ev.bind(uv, uvs);

    const CpuTensor y = ev.eval(out);
    EXPECT_NEAR(y.data[0], 0.f, 1e-6f);
    EXPECT_NEAR(y.data[1], 3.f, 1e-6f);
    EXPECT_NEAR(y.data[2], 1.5f, 1e-6f);
    EXPECT_NEAR(y.data[3], 0.5f, 1e-6f);
}

TEST(CpuEval, IrradianceConvConstantEnv) {
    const ImgSize env_size = {16, 8};
    const ImgSize out_size = {8, 4};
    Variable env(VEC3, env_size);
    Variable irr = F::IrradianceConv(env, out_size);
    EXPECT_EQ(irr.getVType(), VEC3);
    EXPECT_EQ(irr.getImgSize(), out_size);

    CpuEvaluator ev;
    ev.bind(env, FillTensor(VEC3, env_size,
                            [](size_t i) { return i % 3 == 0 ? 0.8f : 0.3f; }));
    const CpuTensor y = ev.eval(irr);
    // Stored E(N)/pi: a constant environment L0 maps to exactly L0
    // (up to the midpoint-rule discretization of the 16x8 source)
    for (size_t p = 0; p < y.numPixels(); p++) {
        EXPECT_NEAR(y.data[p * 3 + 0], 0.8f, 0.8f * 0.02f) << p;
        EXPECT_NEAR(y.data[p * 3 + 1], 0.3f, 0.3f * 0.02f) << p;
        EXPECT_NEAR(y.data[p * 3 + 2], 0.3f, 0.3f * 0.02f) << p;
    }
}

TEST(CpuEval, PrefilterEnvConstantEnv) {
    const ImgSize env_size = {16, 8};
    Variable env(VEC3, env_size);
    CpuTensor env_t = FillTensor(VEC3, env_size, [](size_t i) {
        return i % 3 == 1 ? 0.7f : 0.2f;
    });
    for (float rough : {0.f, 0.5f, 1.f}) {
        Variable pref = F::PrefilterEnv(env, {4, 2}, rough, 16);
        CpuEvaluator ev;
        ev.bind(env, env_t);
        const CpuTensor y = ev.eval(pref);
        // NoL-weighted average of a constant is exactly the constant
        for (size_t p = 0; p < y.numPixels(); p++) {
            EXPECT_NEAR(y.data[p * 3 + 0], 0.2f, 1e-4f) << rough << " " << p;
            EXPECT_NEAR(y.data[p * 3 + 1], 0.7f, 1e-4f) << rough << " " << p;
            EXPECT_NEAR(y.data[p * 3 + 2], 0.2f, 1e-4f) << rough << " " << p;
        }
    }
}

TEST(CpuEval, BrdfIntegrationLutShape) {
    const ImgSize size = {8, 8};
    Variable lut = F::BrdfIntegrationLut(size, 64);
    EXPECT_EQ(lut.getVType(), VEC2);
    EXPECT_EQ(lut.getImgSize(), size);

    CpuEvaluator ev;
    const CpuTensor y = ev.eval(lut);
    for (size_t p = 0; p < y.numPixels(); p++) {
        const float a = y.data[p * 2 + 0];
        const float b = y.data[p * 2 + 1];
        EXPECT_TRUE(std::isfinite(a) && std::isfinite(b)) << p;
        EXPECT_GE(a, 0.f);
        EXPECT_LE(a, 1.05f);
        EXPECT_GE(b, 0.f);
        EXPECT_LE(b, 1.05f);
    }
    // High NdotV, low roughness: full specular scale, no bias
    const size_t corner = (0 * size.w + (size.w - 1)) * 2;
    EXPECT_GT(y.data[corner + 0], 0.9f);
    EXPECT_LT(y.data[corner + 1], 0.05f);
    // The scale term decays with roughness at fixed mid NdotV
    const uint32_t mid_x = size.w / 2;
    const float a_smooth = y.data[(0 * size.w + mid_x) * 2];
    const float a_rough = y.data[(size_t(size.h - 1) * size.w + mid_x) * 2];
    EXPECT_GT(a_smooth, a_rough);
}

TEST(CpuEval, AvgPool2x2Exact) {
    Variable x(FLOAT, {4, 2});
    Variable y = F::AvgPool2x2(x);
    EXPECT_EQ(y.getImgSize(), (ImgSize{2, 1}));

    CpuEvaluator ev;
    CpuTensor t;
    t.vtype = FLOAT;
    t.size = {4, 2};
    t.data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    ev.bind(x, t);
    const CpuTensor out = ev.eval(y);
    EXPECT_FLOAT_EQ(out.data[0], (1.f + 2.f + 5.f + 6.f) * 0.25f);
    EXPECT_FLOAT_EQ(out.data[1], (3.f + 4.f + 7.f + 8.f) * 0.25f);
}

TEST(CpuEval, IblBuildersConstantEnv) {
    const ImgSize env_size = {64, 32};
    Variable env(VEC3, env_size);
    Variable dir(VEC3, {2, 1});
    Variable rough(FLOAT, {1, 1});

    Variable irr = BuildIrradianceSample(env, {8, 4});
    std::vector<Variable> pref = BuildPrefEnvironmentSample(env, 3, {8, 4}, 16);
    Variable lut = BuildBrdfIntegrationMap({8, 8}, 32);
    Variable s_irr = SampleIrradiance(irr, dir);
    Variable s_pref = SamplePrefEnvironment(pref, dir, rough);

    CpuEvaluator ev;
    ev.bind(env, FillTensor(VEC3, env_size, [](size_t i) {
                return i % 3 == 0 ? 0.5f : (i % 3 == 1 ? 0.2f : 0.3f);
            }));
    CpuTensor dirs;
    dirs.vtype = VEC3;
    dirs.size = {2, 1};
    dirs.data = {0.3f, 0.8f, -0.5f, -1.f, 0.2f, 0.4f};
    ev.bind(dir, dirs);
    ev.bind(rough, ScalarTensor(0.3f));

    // Constant env survives pooling exactly, irradiance within the
    // midpoint-rule error, prefilter exactly; the LOD tent weights sum to 1
    const float expect[3] = {0.5f, 0.2f, 0.3f};
    const CpuTensor yi = ev.eval(s_irr);
    const CpuTensor yp = ev.eval(s_pref);
    for (uint32_t p = 0; p < 2; p++) {
        for (uint32_t c = 0; c < 3; c++) {
            EXPECT_NEAR(yi.data[p * 3 + c], expect[c], expect[c] * 0.02f);
            EXPECT_NEAR(yp.data[p * 3 + c], expect[c], 1e-3f);
        }
    }
    const CpuTensor yl = ev.eval(lut);
    for (float v : yl.data) {
        EXPECT_TRUE(std::isfinite(v));
    }
}
