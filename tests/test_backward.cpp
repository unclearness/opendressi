// BuildBackward tests: analytic gradients vs central finite differences via
// the CPU evaluator, plus structural tests (diamond accumulation, pruning,
// gradient barriers, broadcast reduction).

#include <gtest/gtest.h>

#include <cmath>
#include <functional>

#include "core/build_backward.h"
#include "core/cpu_eval.h"
#include "dressi/dressi.h"

using namespace dressi;

namespace {

struct Binding {
    Variable var;
    CpuTensor value;
};

CpuTensor MakeTensor(VType vtype, ImgSize size,
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

float EvalLoss(const Variable& loss, const std::vector<Binding>& binds) {
    CpuEvaluator ev;
    for (const auto& b : binds) {
        ev.bind(b.var, b.value);
    }
    return ev.eval(loss).data[0];
}

// Central finite differences vs the analytic gradient graph for `wrt`
void CheckGrad(const Variable& loss, std::vector<Binding> binds,
               const Variable& wrt, float h = 1e-3f, float rel_tol = 2e-2f,
               float abs_tol = 2e-3f) {
    Variable wrt_mut = wrt;
    wrt_mut.setRequiresGradRecursively();

    auto [input_vars, input_grad_vars] = BuildBackward(loss);
    Variable grad_var(nullptr);
    for (size_t i = 0; i < input_vars.size(); i++) {
        if (input_vars[i] == wrt) {
            grad_var = input_grad_vars[i];
        }
    }
    ASSERT_TRUE(grad_var) << "No gradient variable for wrt";
    ASSERT_EQ(grad_var.getVType(), wrt.getVType());
    ASSERT_EQ(grad_var.getImgSize(), wrt.getImgSize());

    CpuEvaluator ev;
    for (const auto& b : binds) {
        ev.bind(b.var, b.value);
    }
    const CpuTensor analytic = ev.eval(grad_var);

    size_t wrt_idx = binds.size();
    for (size_t i = 0; i < binds.size(); i++) {
        if (binds[i].var == wrt) {
            wrt_idx = i;
        }
    }
    ASSERT_LT(wrt_idx, binds.size());

    for (size_t i = 0; i < binds[wrt_idx].value.data.size(); i++) {
        const float orig = binds[wrt_idx].value.data[i];
        binds[wrt_idx].value.data[i] = orig + h;
        const float f_plus = EvalLoss(loss, binds);
        binds[wrt_idx].value.data[i] = orig - h;
        const float f_minus = EvalLoss(loss, binds);
        binds[wrt_idx].value.data[i] = orig;

        const float numeric = (f_plus - f_minus) / (2.f * h);
        const float a = analytic.data[i];
        const float tol = abs_tol + rel_tol * std::abs(numeric);
        EXPECT_NEAR(a, numeric, tol) << "component " << i;
    }
}

}  // namespace

TEST(Backward, UnaryMathChain) {
    Variable x(FLOAT, {1, 1});
    Variable loss = F::Sin(F::Exp(x * 0.5f)) + F::Log(x + 3.f) * F::Sqrt(x + 2.f);
    CheckGrad(loss, {{x, MakeTensor(FLOAT, {1, 1},
                                    [](size_t) { return 0.7f; })}},
              x);
}

TEST(Backward, TrigInverse) {
    Variable x(FLOAT, {1, 1});
    Variable loss = F::Asin(x) + F::Acos(x) * F::Atan(x) + F::Tan(x);
    CheckGrad(loss, {{x, MakeTensor(FLOAT, {1, 1},
                                    [](size_t) { return 0.3f; })}},
              x);
}

TEST(Backward, PowExpLog) {
    Variable x(FLOAT, {1, 1});
    Variable p(FLOAT, {1, 1});
    Variable loss = F::Pow(x, p) + F::Exp2(x) + F::Log2(x) +
                    F::InverseSqrt(x);
    auto bx = MakeTensor(FLOAT, {1, 1}, [](size_t) { return 1.3f; });
    auto bp = MakeTensor(FLOAT, {1, 1}, [](size_t) { return 2.5f; });
    CheckGrad(loss, {{x, bx}, {p, bp}}, x);
    CheckGrad(loss, {{x, bx}, {p, bp}}, p);
}

TEST(Backward, Sigmoid) {
    Variable x(FLOAT, {1, 1});
    Variable loss = F::Sigmoid(x * 2.f - 0.5f) + F::Sigmoid(x) * x;
    CheckGrad(loss, {{x, MakeTensor(FLOAT, {1, 1},
                                    [](size_t) { return 0.4f; })}},
              x);
}

TEST(Backward, MinMaxClampMixAbs) {
    Variable a(FLOAT, {1, 1});
    Variable b(FLOAT, {1, 1});
    Variable loss = F::Max(a, b) + F::Min(a * 2.f, b) +
                    F::Clamp(a, F::Float(0.f), F::Float(1.f)) +
                    F::Mix(a, b, F::Float(0.3f)) + F::Abs(a - b) +
                    F::SmoothStep(F::Float(0.f), F::Float(1.f), a);
    auto ba = MakeTensor(FLOAT, {1, 1}, [](size_t) { return 0.4f; });
    auto bb = MakeTensor(FLOAT, {1, 1}, [](size_t) { return 0.9f; });
    CheckGrad(loss, {{a, ba}, {b, bb}}, a);
    CheckGrad(loss, {{a, ba}, {b, bb}}, b);
}

TEST(Backward, MixWithVariableT) {
    Variable a(VEC3, {1, 1});
    Variable t(FLOAT, {1, 1});
    Variable loss = F::CompSum(F::Mix(a, a * 2.f, t));
    auto ba = MakeTensor(VEC3, {1, 1},
                         [](size_t i) { return 0.5f + 0.2f * float(i); });
    auto bt = MakeTensor(FLOAT, {1, 1}, [](size_t) { return 0.35f; });
    CheckGrad(loss, {{a, ba}, {t, bt}}, t);
}

TEST(Backward, VectorOps) {
    Variable a(VEC3, {1, 1});
    Variable b(VEC3, {1, 1});
    Variable loss = F::Dot(a, b) + F::Length(a) +
                    F::Dot(F::Normalize(a), b) +
                    F::CompSum(F::Cross(a, b)) + F::Distance(a, b) +
                    F::CompSum(F::Reflect(a, F::Normalize(b)));
    auto ba = MakeTensor(VEC3, {1, 1},
                         [](size_t i) { return 0.4f + 0.3f * float(i); });
    auto bb = MakeTensor(VEC3, {1, 1},
                         [](size_t i) { return 1.1f - 0.4f * float(i); });
    CheckGrad(loss, {{a, ba}, {b, bb}}, a);
    CheckGrad(loss, {{a, ba}, {b, bb}}, b);
}

TEST(Backward, ComposeExtract) {
    Variable x(FLOAT, {1, 1});
    Variable y(FLOAT, {1, 1});
    Variable v = F::Vec3(x, y, x * y);
    Variable loss = F::Dot(v, v) + F::GetZ(v) * F::GetX(v);
    auto bx = MakeTensor(FLOAT, {1, 1}, [](size_t) { return 0.8f; });
    auto by = MakeTensor(FLOAT, {1, 1}, [](size_t) { return -0.6f; });
    CheckGrad(loss, {{x, bx}, {y, by}}, x);
    CheckGrad(loss, {{x, bx}, {y, by}}, y);
}

TEST(Backward, MatrixOps) {
    Variable m(MAT3, {1, 1});
    Variable v(VEC3, {1, 1});
    Variable loss = F::CompSum(F::MatMul(m, v)) +
                    F::CompSum(F::MatMul(F::Transpose(m), v)) +
                    F::Dot(F::MatMul(m, v), v);
    auto bm = MakeTensor(MAT3, {1, 1},
                         [](size_t i) { return 0.1f * float(i) - 0.3f; });
    auto bv = MakeTensor(VEC3, {1, 1},
                         [](size_t i) { return 0.5f + 0.25f * float(i); });
    CheckGrad(loss, {{m, bm}, {v, bv}}, m);
    CheckGrad(loss, {{m, bm}, {v, bv}}, v);
}

TEST(Backward, OuterProductGrad) {
    Variable u(VEC3, {1, 1});
    Variable v(VEC3, {1, 1});
    Variable w(VEC3, {1, 1});
    Variable loss = F::CompSum(F::MatMul(F::OuterProduct(u, v), w));
    auto bu = MakeTensor(VEC3, {1, 1},
                         [](size_t i) { return 0.3f + 0.1f * float(i); });
    auto bv = MakeTensor(VEC3, {1, 1},
                         [](size_t i) { return -0.2f + 0.4f * float(i); });
    auto bw = MakeTensor(VEC3, {1, 1},
                         [](size_t i) { return 1.f - 0.3f * float(i); });
    CheckGrad(loss, {{u, bu}, {v, bv}, {w, bw}}, u);
    CheckGrad(loss, {{u, bu}, {v, bv}, {w, bw}}, v);
}

TEST(Backward, DiamondAccumulation) {
    // x feeds two paths that rejoin: d/dx (x*x + sin(x)*x) must accumulate
    Variable x(FLOAT, {1, 1});
    Variable a = x * x;
    Variable b = F::Sin(x) * x;
    Variable loss = a + b;
    CheckGrad(loss, {{x, MakeTensor(FLOAT, {1, 1},
                                    [](size_t) { return 0.9f; })}},
              x);
}

TEST(Backward, MeanOverImage) {
    const uint32_t W = 13, H = 7;
    Variable img(VEC3, {W, H});
    Variable target(VEC3, {W, H});
    Variable loss = F::Mean(F::Abs(target - img));

    auto bi = MakeTensor(VEC3, {W, H}, [](size_t i) {
        return 0.3f + 0.001f * float(i % 91);
    });
    auto bt = MakeTensor(VEC3, {W, H}, [](size_t i) {
        return 0.8f - 0.002f * float(i % 53);
    });
    CheckGrad(loss, {{img, bi}, {target, bt}}, img, 1e-3f, 2e-2f, 1e-4f);
}

TEST(Backward, UniformBroadcastReducesGrad) {
    // A {1,1} scalar multiplied into an image: gradient must spatially sum
    const uint32_t W = 5, H = 4;
    Variable img(FLOAT, {W, H});
    Variable scale(FLOAT, {1, 1});
    Variable loss = F::Mean(img * scale);

    auto bi = MakeTensor(FLOAT, {W, H},
                         [](size_t i) { return 0.1f * float(i % 7); });
    auto bs = MakeTensor(FLOAT, {1, 1}, [](size_t) { return 1.7f; });
    CheckGrad(loss, {{img, bi}, {scale, bs}}, scale);
}

TEST(Backward, ScalarIntoVectorReducesGrad) {
    // A scalar broadcast into a vector op: gradient must component-sum
    Variable s(FLOAT, {1, 1});
    Variable v(VEC3, {1, 1});
    Variable loss = F::CompSum(s * v);
    auto bs = MakeTensor(FLOAT, {1, 1}, [](size_t) { return 0.6f; });
    auto bv = MakeTensor(VEC3, {1, 1},
                         [](size_t i) { return 1.f + float(i); });
    CheckGrad(loss, {{s, bs}, {v, bv}}, s);
}

TEST(Backward, RequiresGradPruning) {
    Variable x(FLOAT, {1, 1});
    Variable c(FLOAT, {1, 1});  // not marked
    Variable loss = x * c;
    Variable x_mut = x;
    x_mut.setRequiresGradRecursively();

    auto [input_vars, input_grad_vars] = BuildBackward(loss);
    ASSERT_EQ(input_vars.size(), 1u);
    EXPECT_EQ(input_vars[0], x);
}

TEST(Backward, StopGradientBarrier) {
    Variable x(FLOAT, {1, 1});
    Variable loss = F::StopGradient(x * 2.f) + x;
    Variable x_mut = x;
    x_mut.setRequiresGradRecursively();

    auto [input_vars, input_grad_vars] = BuildBackward(loss);
    ASSERT_EQ(input_vars.size(), 1u);

    CpuEvaluator ev;
    ev.bind(x, MakeTensor(FLOAT, {1, 1}, [](size_t) { return 0.5f; }));
    // Only the direct path contributes: d/dx = 1
    EXPECT_FLOAT_EQ(ev.eval(input_grad_vars[0]).data[0], 1.f);
}

TEST(Backward, ImageLossMatchesReducedScalarLoss) {
    // A non-reduced image loss seeds gradient 1 at every pixel, which is
    // the derivative of the implicit sum: gradients must match the
    // explicitly reduced CompSum(Sum(...)) scalar loss.
    const uint32_t W = 6, H = 5;
    const auto bind_img = MakeTensor(FLOAT, {W, H}, [](size_t i) {
        return 0.1f + 0.07f * float(i % 9);
    });
    const auto bind_tgt = MakeTensor(FLOAT, {W, H}, [](size_t i) {
        return 0.9f - 0.05f * float(i % 7);
    });

    const auto grad_of = [&](bool reduce) {
        Variable img(FLOAT, {W, H});
        Variable target(FLOAT, {W, H});
        Variable diff = F::StopGradient(target) - F::Sigmoid(img);
        Variable loss = diff * diff;
        if (reduce) {
            loss = F::CompSum(F::Sum(loss));
        }
        Variable img_mut = img;
        img_mut.setRequiresGradRecursively();
        auto [input_vars, input_grad_vars] = BuildBackward(loss);
        CpuEvaluator ev;
        ev.bind(img, bind_img);
        ev.bind(target, bind_tgt);
        for (size_t i = 0; i < input_vars.size(); i++) {
            if (input_vars[i] == img) {
                return ev.eval(input_grad_vars[i]);
            }
        }
        throw std::runtime_error("no grad");
    };

    const CpuTensor g_img_loss = grad_of(false);
    const CpuTensor g_scalar_loss = grad_of(true);
    ASSERT_EQ(g_img_loss.data.size(), g_scalar_loss.data.size());
    for (size_t i = 0; i < g_img_loss.data.size(); i++) {
        EXPECT_NEAR(g_img_loss.data[i], g_scalar_loss.data[i], 1e-6f)
                << "component " << i;
    }
}

TEST(Backward, IntLossRejected) {
    Variable img(IVEC3, {4, 1});
    Variable img_mut = img;
    img_mut.setRequiresGradRecursively();
    EXPECT_THROW(BuildBackward(img), DressiError);
}

TEST(Backward, PbrCompositeGradients) {
    // Gradient flows through a GGX+Fresnel style expression w.r.t. roughness
    Variable rough(FLOAT, {1, 1});
    Variable n_dot_h(FLOAT, {1, 1});

    Variable alpha2 = rough * rough * rough * rough;
    Variable denom = n_dot_h * n_dot_h * (alpha2 - 1.f) + 1.f;
    Variable dist = alpha2 / (F::Float(3.14159265f) * denom * denom);
    Variable loss = dist / (dist + 1.f);

    auto br = MakeTensor(FLOAT, {1, 1}, [](size_t) { return 0.5f; });
    auto bh = MakeTensor(FLOAT, {1, 1}, [](size_t) { return 0.95f; });
    CheckGrad(loss, {{rough, br}, {n_dot_h, bh}}, rough);
    CheckGrad(loss, {{rough, br}, {n_dot_h, bh}}, n_dot_h);
}

TEST(Backward, FaceFetchExactBackward) {
    // n_samples == 0: the exact per-face bbox scan must equal the true VJP
    // g_attr[f] = sum over pixels with idx == f+1 of gy(p). FaceFetch is
    // linear in attr, so central differences are exact.
    const uint32_t W = 8, H = 6;
    Variable attr(FLOAT, {2, 1});
    Variable idx(FLOAT, {W, H});
    Variable p0(VEC4, {2, 1});
    Variable p1(VEC4, {2, 1});
    Variable p2(VEC4, {2, 1});
    Variable seed(FLOAT, {1, 1});
    Variable wimg(FLOAT, {W, H});
    Variable out = F::FaceFetch(attr, idx, p0, p1, p2, seed, 1.f,
                                /*n_samples=*/0);
    Variable loss = F::Mean(out * wimg);

    // Screen-space corners (w = 1): face 0 bbox [1,5]x[1,4], face 1 bbox
    // [3,7]x[2,5]; the id assignments below stay inside those bboxes
    const auto clip = [&](float sx, float sy, int c) {
        const float v[4] = {sx / float(W) * 2.f - 1.f,
                            sy / float(H) * 2.f - 1.f, 0.5f, 1.f};
        return v[c];
    };
    std::vector<float> d0, d1, d2;
    const float c0[2][2] = {{1.f, 1.f}, {3.f, 2.f}};   // corner 0 of f0, f1
    const float c1[2][2] = {{5.f, 1.f}, {7.f, 2.f}};
    const float c2[2][2] = {{1.f, 4.f}, {7.f, 5.f}};
    for (int f = 0; f < 2; f++) {
        for (int c = 0; c < 4; c++) {
            d0.push_back(clip(c0[f][0], c0[f][1], c));
            d1.push_back(clip(c1[f][0], c1[f][1], c));
            d2.push_back(clip(c2[f][0], c2[f][1], c));
        }
    }
    std::vector<float> idx_data(size_t(W) * H, 0.f);
    idx_data[size_t(1) * W + 2] = 1.f;
    idx_data[size_t(2) * W + 3] = 1.f;
    idx_data[size_t(0) * W + 5] = 1.f;
    idx_data[size_t(3) * W + 4] = 2.f;
    idx_data[size_t(2) * W + 6] = 2.f;
    idx_data[size_t(5) * W + 7] = 2.f;

    std::vector<Binding> binds = {
            {attr, MakeTensor(FLOAT, {2, 1},
                              [](size_t i) { return 0.4f + 0.3f * float(i); })},
            {idx, MakeTensor(FLOAT, {W, H},
                             [&](size_t i) { return idx_data[i]; })},
            {p0, MakeTensor(VEC4, {2, 1}, [&](size_t i) { return d0[i]; })},
            {p1, MakeTensor(VEC4, {2, 1}, [&](size_t i) { return d1[i]; })},
            {p2, MakeTensor(VEC4, {2, 1}, [&](size_t i) { return d2[i]; })},
            {seed, MakeTensor(FLOAT, {1, 1}, [](size_t) { return 0.f; })},
            {wimg, MakeTensor(FLOAT, {W, H}, [](size_t i) {
                 return 0.1f + 0.05f * float(i % 13);
             })}};
    CheckGrad(loss, binds, attr, 1e-3f, 1e-3f, 1e-4f);
}

TEST(Backward, TraverseFuncsTopoOrder) {
    Variable x(FLOAT, {1, 1});
    Variable y = F::Sin(x) + F::Cos(x);
    Functions funcs = TraverseFuncs({y});
    ASSERT_GE(funcs.size(), 3u);
    for (size_t i = 1; i < funcs.size(); i++) {
        EXPECT_LT(funcs[i - 1].id(), funcs[i].id());
    }
}

TEST(Backward, TextureBilinearTextureGrad) {
    // Synthetic deferred setup: an 8x8 screen samples a 4x4 texture with a
    // smooth uv map kept strictly inside the clamp-free region, and the
    // inverse-UV table anchors each texel at its nearest screen pixel so
    // the +-3 gather window sees every contributing pixel (the tent-weight
    // gather is exact under these conditions).
    const ImgSize tex_size = {4, 4};
    const ImgSize screen = {8, 8};
    Variable tex(FLOAT, tex_size);
    Variable uv(VEC2, screen);
    Variable inv_uv(VEC4, tex_size);
    Variable out = F::TextureBilinear(tex, uv, inv_uv);
    Variable loss = F::Mean(out * out + out);

    auto tex_t = MakeTensor(FLOAT, tex_size, [](size_t i) {
        return 0.2f + 0.11f * float((i * 5) % 7);
    });
    CpuTensor uv_t;
    uv_t.vtype = VEC2;
    uv_t.size = screen;
    uv_t.data.resize(size_t(screen.w) * screen.h * 2);
    for (uint32_t py = 0; py < screen.h; py++) {
        for (uint32_t px = 0; px < screen.w; px++) {
            const size_t o = (size_t(py) * screen.w + px) * 2;
            uv_t.data[o + 0] = 0.125f + 0.75f * (float(px) + 0.5f) / 8.f;
            uv_t.data[o + 1] = 0.125f + 0.75f * (float(py) + 0.5f) / 8.f;
        }
    }
    CpuTensor inv_t;
    inv_t.vtype = VEC4;
    inv_t.size = tex_size;
    inv_t.data.assign(size_t(tex_size.w) * tex_size.h * 4, 0.f);
    for (uint32_t ty = 0; ty < tex_size.h; ty++) {
        for (uint32_t tx = 0; tx < tex_size.w; tx++) {
            // Screen pixel whose sampling coordinate is nearest this texel
            // (st = 0.375 * (p + 0.5), so p + 0.5 = t / 0.375)
            const auto anchor = [](uint32_t t) {
                float p = float(t) / 0.375f;
                if (p > 7.f) {
                    p = 7.f;
                }
                return p;
            };
            const size_t o = (size_t(ty) * tex_size.w + tx) * 4;
            inv_t.data[o + 0] = std::floor(anchor(tx)) + 0.5f;
            inv_t.data[o + 1] = std::floor(anchor(ty)) + 0.5f;
            inv_t.data[o + 2] = 1.f;
        }
    }

    CheckGrad(loss, {{tex, tex_t}, {uv, uv_t}, {inv_uv, inv_t}}, tex);
}
