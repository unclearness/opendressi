#include "dressi/f.h"

#include <cmath>
#include <cstdio>
#include <string>

#include "core/infer.h"
#include "core/node.h"
#include "core/op_builder.h"

namespace dressi {
namespace {

std::string FloatLiteral(float fv) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.9g", double(fv));
    std::string s(buf);
    // Ensure a float-typed literal in GLSL
    if (s.find_first_of(".eEnif") == std::string::npos) {
        s += ".0";
    }
    return s;
}

Variable NullBwd(const Variables&, const Variable&, const Variable&,
                 uint32_t) {
    return nullptr;
}

// Contiguous component extraction {y}={x0}.<swizzle>; (internal; also the
// backward building block of Compose)
Variable ExtractComps(const Variable& v, uint32_t ofs, uint32_t n);

// Generic float-vector composition from scalar/vector parts (internal;
// public F::Vec2/3/4 wrap it)
Variable Compose(const Variables& parts);

Variable ComposeReplicate(const Variable& scalar, uint32_t n) {
    if (n == 1) {
        return scalar;
    }
    return Compose(Variables(n, scalar));
}

Variable ExtractComps(const Variable& v, uint32_t ofs, uint32_t n) {
    const uint32_t src_n = NumComponents(v.getVType());
    DRESSI_CHECK(ofs + n <= src_n, "ExtractComps out of range");
    if (n == src_n) {
        return v;
    }
    static const char* kSwizzle = "xyzw";
    const std::string swz(kSwizzle + ofs, kSwizzle + ofs + n);
    OpDesc desc;
    desc.name = "ExtractComps";
    desc.fwd_code = "{y}={x0}." + swz + ";";
    desc.infer = FixedInfer(VecVTypeOf(n));
    desc.bwd = [ofs, n, src_n](const Variables& /*xs*/, const Variable& /*y*/,
                               const Variable& gy,
                               uint32_t /*bwd_idx*/) -> Variable {
        // Scatter gy back into a zero-padded vector of the source width
        Variables parts;
        for (uint32_t j = 0; j < src_n; j++) {
            if (ofs <= j && j < ofs + n) {
                parts.push_back(n == 1 ? gy : ExtractComps(gy, j - ofs, 1));
            } else {
                parts.push_back(F::Float(0.f));
            }
        }
        return Compose(parts);
    };
    desc.cpu = ElementwiseCpu(
            [ofs, n](const std::vector<const float*>& xs,
                     const std::vector<uint32_t>&, float* y, uint32_t) {
                for (uint32_t c = 0; c < n; c++) {
                    y[c] = xs[0][ofs + c];
                }
            },
            VecVTypeOf(n));
    return MakeOp(std::move(desc), {v});
}

Variable Compose(const Variables& parts) {
    uint32_t total = 0;
    for (const auto& p : parts) {
        const VType t = p.getVType();
        DRESSI_CHECK(!IsIntVType(t) && !IsMatrixVType(t),
                     "Compose expects float scalar/vector parts");
        total += NumComponents(t);
    }
    DRESSI_CHECK(2 <= total && total <= 4, "Compose: invalid component count");
    const VType out_vtype = VecVTypeOf(total);

    std::string code = std::string("{y}=") + GlslTypeName(out_vtype) + "(";
    for (size_t i = 0; i < parts.size(); i++) {
        code += (i ? "," : "");
        code += "{x" + std::to_string(i) + "}";
    }
    code += ");";

    std::vector<uint32_t> offsets(parts.size());
    std::vector<uint32_t> widths(parts.size());
    uint32_t ofs = 0;
    for (size_t i = 0; i < parts.size(); i++) {
        offsets[i] = ofs;
        widths[i] = NumComponents(parts[i].getVType());
        ofs += widths[i];
    }

    OpDesc desc;
    desc.name = "Compose";
    desc.fwd_code = code;
    desc.infer = FixedInfer(out_vtype);
    desc.bwd = [offsets, widths](const Variables& /*xs*/, const Variable&,
                                 const Variable& gy,
                                 uint32_t bwd_idx) -> Variable {
        return ExtractComps(gy, offsets[bwd_idx], widths[bwd_idx]);
    };
    desc.cpu = ElementwiseCpu(
            [widths](const std::vector<const float*>& xs,
                     const std::vector<uint32_t>&, float* y, uint32_t) {
                uint32_t o = 0;
                for (size_t i = 0; i < xs.size(); i++) {
                    for (uint32_t c = 0; c < widths[i]; c++) {
                        y[o++] = xs[i][c];
                    }
                }
            },
            out_vtype);
    return MakeOp(std::move(desc), parts);
}

// Scalar-only comparison producing 0/1 FLOAT
Variable CompareOp(const char* name, const char* op, const Variable& x0,
                   const Variable& x1, float (*cpu)(float, float)) {
    OpDesc desc;
    desc.name = name;
    desc.fwd_code = std::string("{y}=float({x0}") + op + "{x1});";
    desc.infer = [](const Variables& xs) -> std::pair<VType, ImgSize> {
        DRESSI_CHECK(xs[0].getVType() == FLOAT && xs[1].getVType() == FLOAT,
                     "Comparison ops require scalar FLOAT inputs");
        return {FLOAT, InferCommonImgSize(xs)};
    };
    desc.bwd = NullBwd;
    desc.cpu = MapCpu2(cpu);
    return MakeOp(std::move(desc), {x0, x1});
}

}  // namespace

namespace F {

// --------------------------------- Constants ---------------------------------
Variable Float(float fv) {
    OpDesc desc;
    desc.name = "Float";
    desc.fwd_code = "{y}=" + FloatLiteral(fv) + ";";
    desc.infer = FixedInfer(FLOAT);
    desc.bwd = NullBwd;
    desc.const_val = std::vector<float>{fv};
    return MakeOp(std::move(desc), {});
}

namespace {
Variable ConstVec(const char* name, std::vector<float> vals) {
    const VType vtype = VecVTypeOf(uint32_t(vals.size()));
    std::string code = std::string("{y}=") + GlslTypeName(vtype) + "(";
    for (size_t i = 0; i < vals.size(); i++) {
        code += (i ? "," : "") + FloatLiteral(vals[i]);
    }
    code += ");";
    OpDesc desc;
    desc.name = name;
    desc.fwd_code = code;
    desc.infer = FixedInfer(vtype);
    desc.bwd = NullBwd;
    desc.const_val = std::move(vals);
    return MakeOp(std::move(desc), {});
}
}  // namespace

Variable Vec2(float x, float y) {
    return ConstVec("Vec2", {x, y});
}
Variable Vec3(float x, float y, float z) {
    return ConstVec("Vec3", {x, y, z});
}
Variable Vec4(float x, float y, float z, float w) {
    return ConstVec("Vec4", {x, y, z, w});
}

// --------------------------------- Arithmetic --------------------------------
Variable Add(const Variable& x0, const Variable& x1) {
    OpDesc desc;
    desc.name = "Add";
    desc.fwd_code = "{y}={x0}+{x1};";
    desc.bwd = [](const Variables&, const Variable&, const Variable& gy,
                  uint32_t) -> Variable { return gy; };
    desc.cpu = MapCpu2([](float a, float b) { return a + b; });
    return MakeOp(std::move(desc), {x0, x1});
}

Variable Sub(const Variable& x0, const Variable& x1) {
    OpDesc desc;
    desc.name = "Sub";
    desc.fwd_code = "{y}={x0}-{x1};";
    desc.bwd = [](const Variables&, const Variable&, const Variable& gy,
                  uint32_t bwd_idx) -> Variable {
        return bwd_idx == 0 ? gy : Neg(gy);
    };
    desc.cpu = MapCpu2([](float a, float b) { return a - b; });
    return MakeOp(std::move(desc), {x0, x1});
}

Variable Mul(const Variable& x0, const Variable& x1) {
    // GLSL `mat * mat` / `mat * vec` are matrix products, not component-wise
    DRESSI_CHECK(!IsMatrixVType(x0.getVType()) ||
                         x1.getVType() == FLOAT,
                 "Use F::MatMul for matrix products");
    DRESSI_CHECK(!IsMatrixVType(x1.getVType()) ||
                         x0.getVType() == FLOAT,
                 "Use F::MatMul for matrix products");
    OpDesc desc;
    desc.name = "Mul";
    desc.fwd_code = "{y}={x0}*{x1};";
    desc.bwd = [](const Variables& xs, const Variable&, const Variable& gy,
                  uint32_t bwd_idx) -> Variable {
        return bwd_idx == 0 ? Mul(gy, xs[1]) : Mul(gy, xs[0]);
    };
    desc.cpu = MapCpu2([](float a, float b) { return a * b; });
    return MakeOp(std::move(desc), {x0, x1});
}

Variable Div(const Variable& x0, const Variable& x1) {
    DRESSI_CHECK(!IsMatrixVType(x1.getVType()),
                 "Division by a matrix is not supported");
    OpDesc desc;
    desc.name = "Div";
    desc.fwd_code = "{y}={x0}/{x1};";
    desc.bwd = [](const Variables& xs, const Variable&, const Variable& gy,
                  uint32_t bwd_idx) -> Variable {
        if (bwd_idx == 0) {
            return Div(gy, xs[1]);
        }
        return Neg(Div(Mul(gy, xs[0]), Mul(xs[1], xs[1])));
    };
    desc.cpu = MapCpu2([](float a, float b) { return a / b; });
    return MakeOp(std::move(desc), {x0, x1});
}

Variable Neg(const Variable& x) {
    OpDesc desc;
    desc.name = "Neg";
    desc.fwd_code = "{y}=-{x0};";
    desc.bwd = [](const Variables&, const Variable&, const Variable& gy,
                  uint32_t) -> Variable { return Neg(gy); };
    desc.cpu = MapCpu1([](float a) { return -a; });
    return MakeOp(std::move(desc), {x});
}

// ----------------------------------- Math ------------------------------------
namespace {

// Unary math op factory; `bwd` builds the local derivative graph d y / d x
// which is multiplied by gy in the returned BwdFunc.
Variable UnaryMath(const char* name, const char* glsl_fn, float (*cpu)(float),
                   const std::function<Variable(const Variable& x,
                                                const Variable& y)>& dydx,
                   const Variable& x) {
    OpDesc desc;
    desc.name = name;
    desc.fwd_code = std::string("{y}=") + glsl_fn + "({x0});";
    if (dydx) {
        desc.bwd = [dydx](const Variables& xs, const Variable& y,
                          const Variable& gy, uint32_t) -> Variable {
            return Mul(gy, dydx(xs[0], y));
        };
    } else {
        desc.bwd = NullBwd;
    }
    desc.cpu = MapCpu1(cpu);
    return MakeOp(std::move(desc), {x});
}

}  // namespace

Variable Sin(const Variable& x) {
    return UnaryMath("Sin", "sin", [](float v) { return std::sin(v); },
                     [](const Variable& x_, const Variable&) {
                         return Cos(x_);
                     },
                     x);
}

Variable Cos(const Variable& x) {
    return UnaryMath("Cos", "cos", [](float v) { return std::cos(v); },
                     [](const Variable& x_, const Variable&) {
                         return Neg(Sin(x_));
                     },
                     x);
}

Variable Tan(const Variable& x) {
    return UnaryMath("Tan", "tan", [](float v) { return std::tan(v); },
                     [](const Variable&, const Variable& y_) {
                         return Add(F::Float(1.f), Mul(y_, y_));
                     },
                     x);
}

Variable Asin(const Variable& x) {
    return UnaryMath("Asin", "asin", [](float v) { return std::asin(v); },
                     [](const Variable& x_, const Variable&) {
                         return InverseSqrt(
                                 Sub(F::Float(1.f), Mul(x_, x_)));
                     },
                     x);
}

Variable Acos(const Variable& x) {
    return UnaryMath("Acos", "acos", [](float v) { return std::acos(v); },
                     [](const Variable& x_, const Variable&) {
                         return Neg(InverseSqrt(
                                 Sub(F::Float(1.f), Mul(x_, x_))));
                     },
                     x);
}

Variable Atan(const Variable& x) {
    return UnaryMath("Atan", "atan", [](float v) { return std::atan(v); },
                     [](const Variable& x_, const Variable&) {
                         return Div(F::Float(1.f),
                                    Add(F::Float(1.f), Mul(x_, x_)));
                     },
                     x);
}

Variable Atan2(const Variable& y, const Variable& x) {
    OpDesc desc;
    desc.name = "Atan2";
    desc.fwd_code = "{y}=atan({x0},{x1});";
    desc.bwd = [](const Variables& xs, const Variable&, const Variable& gy,
                  uint32_t bwd_idx) -> Variable {
        Variable denom = Add(Mul(xs[0], xs[0]), Mul(xs[1], xs[1]));
        if (bwd_idx == 0) {
            return Div(Mul(gy, xs[1]), denom);
        }
        return Neg(Div(Mul(gy, xs[0]), denom));
    };
    desc.cpu = MapCpu2([](float a, float b) { return std::atan2(a, b); });
    return MakeOp(std::move(desc), {y, x});
}

Variable Exp(const Variable& x) {
    return UnaryMath("Exp", "exp", [](float v) { return std::exp(v); },
                     [](const Variable&, const Variable& y_) { return y_; },
                     x);
}

Variable Exp2(const Variable& x) {
    return UnaryMath("Exp2", "exp2", [](float v) { return std::exp2(v); },
                     [](const Variable&, const Variable& y_) {
                         return Mul(y_, F::Float(0.6931471805599453f));
                     },
                     x);
}

Variable Log(const Variable& x) {
    return UnaryMath("Log", "log", [](float v) { return std::log(v); },
                     [](const Variable& x_, const Variable&) {
                         return Div(F::Float(1.f), x_);
                     },
                     x);
}

Variable Log2(const Variable& x) {
    return UnaryMath("Log2", "log2", [](float v) { return std::log2(v); },
                     [](const Variable& x_, const Variable&) {
                         return Div(F::Float(1.4426950408889634f), x_);
                     },
                     x);
}

Variable Sqrt(const Variable& x) {
    return UnaryMath("Sqrt", "sqrt", [](float v) { return std::sqrt(v); },
                     [](const Variable&, const Variable& y_) {
                         return Div(F::Float(0.5f), y_);
                     },
                     x);
}

Variable InverseSqrt(const Variable& x) {
    return UnaryMath("InverseSqrt", "inversesqrt",
                     [](float v) { return 1.f / std::sqrt(v); },
                     [](const Variable&, const Variable& y_) {
                         return Mul(F::Float(-0.5f), Mul(y_, Mul(y_, y_)));
                     },
                     x);
}

Variable Abs(const Variable& x) {
    return UnaryMath("Abs", "abs", [](float v) { return std::abs(v); },
                     [](const Variable& x_, const Variable&) {
                         return Sign(x_);
                     },
                     x);
}

Variable Sign(const Variable& x) {
    return UnaryMath("Sign", "sign",
                     [](float v) { return float((v > 0.f) - (v < 0.f)); },
                     nullptr, x);
}

Variable Floor(const Variable& x) {
    return UnaryMath("Floor", "floor", [](float v) { return std::floor(v); },
                     nullptr, x);
}

Variable Fract(const Variable& x) {
    return UnaryMath("Fract", "fract",
                     [](float v) { return v - std::floor(v); },
                     [](const Variable&, const Variable&) {
                         return F::Float(1.f);
                     },
                     x);
}

Variable Mod(const Variable& x, const Variable& m) {
    OpDesc desc;
    desc.name = "Mod";
    desc.fwd_code = "{y}=mod({x0},{x1});";
    desc.bwd = [](const Variables& xs, const Variable&, const Variable& gy,
                  uint32_t bwd_idx) -> Variable {
        if (bwd_idx == 0) {
            return gy;
        }
        return Neg(Mul(gy, Floor(Div(xs[0], xs[1]))));
    };
    desc.cpu = MapCpu2(
            [](float a, float b) { return a - b * std::floor(a / b); });
    return MakeOp(std::move(desc), {x, m});
}

Variable Pow(const Variable& x, const Variable& p) {
    OpDesc desc;
    desc.name = "Pow";
    desc.fwd_code = "{y}=pow({x0},{x1});";
    desc.bwd = [](const Variables& xs, const Variable& y, const Variable& gy,
                  uint32_t bwd_idx) -> Variable {
        if (bwd_idx == 0) {
            return Mul(gy, Mul(xs[1], Pow(xs[0], Sub(xs[1], F::Float(1.f)))));
        }
        return Mul(gy, Mul(y, Log(xs[0])));
    };
    desc.cpu = MapCpu2([](float a, float b) { return std::pow(a, b); });
    return MakeOp(std::move(desc), {x, p});
}

Variable Max(const Variable& x0, const Variable& x1) {
    OpDesc desc;
    desc.name = "Max";
    desc.fwd_code = "{y}=max({x0},{x1});";
    desc.bwd = [](const Variables& xs, const Variable&, const Variable& gy,
                  uint32_t bwd_idx) -> Variable {
        Variable mask = Step(xs[1], xs[0]);  // x0 >= x1
        if (bwd_idx == 0) {
            return Mul(gy, mask);
        }
        return Mul(gy, Sub(F::Float(1.f), mask));
    };
    desc.cpu = MapCpu2([](float a, float b) { return std::max(a, b); });
    return MakeOp(std::move(desc), {x0, x1});
}

Variable Min(const Variable& x0, const Variable& x1) {
    OpDesc desc;
    desc.name = "Min";
    desc.fwd_code = "{y}=min({x0},{x1});";
    desc.bwd = [](const Variables& xs, const Variable&, const Variable& gy,
                  uint32_t bwd_idx) -> Variable {
        Variable mask = Step(xs[0], xs[1]);  // x1 >= x0 -> x0 selected
        if (bwd_idx == 0) {
            return Mul(gy, mask);
        }
        return Mul(gy, Sub(F::Float(1.f), mask));
    };
    desc.cpu = MapCpu2([](float a, float b) { return std::min(a, b); });
    return MakeOp(std::move(desc), {x0, x1});
}

Variable Clamp(const Variable& x, const Variable& lo, const Variable& hi) {
    OpDesc desc;
    desc.name = "Clamp";
    desc.fwd_code = "{y}=clamp({x0},{x1},{x2});";
    desc.bwd = [](const Variables& xs, const Variable&, const Variable& gy,
                  uint32_t bwd_idx) -> Variable {
        if (bwd_idx != 0) {
            return nullptr;  // no gradient toward bounds (M1)
        }
        Variable inside = Mul(Step(xs[1], xs[0]), Step(xs[0], xs[2]));
        return Mul(gy, inside);
    };
    desc.cpu = MapCpu3([](float v, float lo_, float hi_) {
        return std::min(std::max(v, lo_), hi_);
    });
    return MakeOp(std::move(desc), {x, lo, hi});
}

Variable Mix(const Variable& x0, const Variable& x1, const Variable& t) {
    OpDesc desc;
    desc.name = "Mix";
    desc.fwd_code = "{y}=mix({x0},{x1},{x2});";
    desc.bwd = [](const Variables& xs, const Variable&, const Variable& gy,
                  uint32_t bwd_idx) -> Variable {
        if (bwd_idx == 0) {
            return Mul(gy, Sub(F::Float(1.f), xs[2]));
        }
        if (bwd_idx == 1) {
            return Mul(gy, xs[2]);
        }
        return Mul(gy, Sub(xs[1], xs[0]));
    };
    desc.cpu = MapCpu3(
            [](float a, float b, float t_) { return a + (b - a) * t_; });
    return MakeOp(std::move(desc), {x0, x1, t});
}

Variable Step(const Variable& edge, const Variable& x) {
    OpDesc desc;
    desc.name = "Step";
    desc.fwd_code = "{y}=step({x0},{x1});";
    desc.bwd = NullBwd;
    desc.cpu = MapCpu2([](float e, float v) { return v < e ? 0.f : 1.f; });
    return MakeOp(std::move(desc), {edge, x});
}

Variable SmoothStep(const Variable& e0, const Variable& e1,
                    const Variable& x) {
    OpDesc desc;
    desc.name = "SmoothStep";
    desc.fwd_code = "{y}=smoothstep({x0},{x1},{x2});";
    desc.bwd = [](const Variables& xs, const Variable&, const Variable& gy,
                  uint32_t bwd_idx) -> Variable {
        if (bwd_idx != 2) {
            return nullptr;  // no gradient toward edges (M1)
        }
        Variable range = Sub(xs[1], xs[0]);
        Variable t = Clamp(Div(Sub(xs[2], xs[0]), range), F::Float(0.f),
                           F::Float(1.f));
        Variable dt = Div(Mul(F::Float(6.f),
                              Mul(t, Sub(F::Float(1.f), t))),
                          range);
        return Mul(gy, dt);
    };
    desc.cpu = MapCpu3([](float a, float b, float v) {
        const float t = std::min(std::max((v - a) / (b - a), 0.f), 1.f);
        return t * t * (3.f - 2.f * t);
    });
    return MakeOp(std::move(desc), {e0, e1, x});
}

// -------------------------- Comparison (scalar, 0/1) --------------------------
Variable Less(const Variable& x0, const Variable& x1) {
    return CompareOp("Less", "<", x0, x1,
                     [](float a, float b) { return float(a < b); });
}
Variable LessEqual(const Variable& x0, const Variable& x1) {
    return CompareOp("LessEqual", "<=", x0, x1,
                     [](float a, float b) { return float(a <= b); });
}
Variable Greater(const Variable& x0, const Variable& x1) {
    return CompareOp("Greater", ">", x0, x1,
                     [](float a, float b) { return float(a > b); });
}
Variable GreaterEqual(const Variable& x0, const Variable& x1) {
    return CompareOp("GreaterEqual", ">=", x0, x1,
                     [](float a, float b) { return float(a >= b); });
}

// ---------------------------------- Vector -----------------------------------
Variable Vec2(const Variable& x, const Variable& y) {
    return Compose({x, y});
}
Variable Vec3(const Variable& x, const Variable& y, const Variable& z) {
    return Compose({x, y, z});
}
Variable Vec3(const Variable& xy, const Variable& z) {
    return Compose({xy, z});
}
Variable Vec4(const Variable& x, const Variable& y, const Variable& z,
              const Variable& w) {
    return Compose({x, y, z, w});
}
Variable Vec4(const Variable& xyz, const Variable& w) {
    return Compose({xyz, w});
}

Variable GetX(const Variable& v) {
    return ExtractComps(v, 0, 1);
}
Variable GetY(const Variable& v) {
    return ExtractComps(v, 1, 1);
}
Variable GetZ(const Variable& v) {
    return ExtractComps(v, 2, 1);
}
Variable GetW(const Variable& v) {
    return ExtractComps(v, 3, 1);
}

Variable Dot(const Variable& a, const Variable& b) {
    OpDesc desc;
    desc.name = "Dot";
    desc.fwd_code = "{y}=dot({x0},{x1});";
    desc.infer = [](const Variables& xs) -> std::pair<VType, ImgSize> {
        DRESSI_CHECK(IsVectorVType(xs[0].getVType()) &&
                             xs[0].getVType() == xs[1].getVType(),
                     "Dot requires two equal vector types");
        return {FLOAT, InferCommonImgSize(xs)};
    };
    desc.bwd = [](const Variables& xs, const Variable&, const Variable& gy,
                  uint32_t bwd_idx) -> Variable {
        return Mul(gy, xs[bwd_idx == 0 ? 1 : 0]);
    };
    desc.cpu = ElementwiseCpu(
            [](const std::vector<const float*>& xs,
               const std::vector<uint32_t>& ns, float* y, uint32_t) {
                float acc = 0.f;
                for (uint32_t c = 0; c < ns[0]; c++) {
                    acc += xs[0][c] * xs[1][c];
                }
                y[0] = acc;
            },
            FLOAT);
    return MakeOp(std::move(desc), {a, b});
}

Variable Cross(const Variable& a, const Variable& b) {
    OpDesc desc;
    desc.name = "Cross";
    desc.fwd_code = "{y}=cross({x0},{x1});";
    desc.infer = [](const Variables& xs) -> std::pair<VType, ImgSize> {
        DRESSI_CHECK(xs[0].getVType() == VEC3 && xs[1].getVType() == VEC3,
                     "Cross requires VEC3 inputs");
        return {VEC3, InferCommonImgSize(xs)};
    };
    desc.bwd = [](const Variables& xs, const Variable&, const Variable& gy,
                  uint32_t bwd_idx) -> Variable {
        if (bwd_idx == 0) {
            return Cross(xs[1], gy);
        }
        return Cross(gy, xs[0]);
    };
    desc.cpu = ElementwiseCpu(
            [](const std::vector<const float*>& xs,
               const std::vector<uint32_t>&, float* y, uint32_t) {
                const float* a_ = xs[0];
                const float* b_ = xs[1];
                y[0] = a_[1] * b_[2] - a_[2] * b_[1];
                y[1] = a_[2] * b_[0] - a_[0] * b_[2];
                y[2] = a_[0] * b_[1] - a_[1] * b_[0];
            },
            VEC3);
    return MakeOp(std::move(desc), {a, b});
}

Variable Length(const Variable& v) {
    OpDesc desc;
    desc.name = "Length";
    desc.fwd_code = "{y}=length({x0});";
    desc.infer = [](const Variables& xs) -> std::pair<VType, ImgSize> {
        DRESSI_CHECK(IsVectorVType(xs[0].getVType()) ||
                             xs[0].getVType() == FLOAT,
                     "Length requires a float vector");
        return {FLOAT, InferCommonImgSize(xs)};
    };
    desc.bwd = [](const Variables& xs, const Variable& y, const Variable& gy,
                  uint32_t) -> Variable {
        return Mul(Div(gy, y), xs[0]);
    };
    desc.cpu = ElementwiseCpu(
            [](const std::vector<const float*>& xs,
               const std::vector<uint32_t>& ns, float* y, uint32_t) {
                float acc = 0.f;
                for (uint32_t c = 0; c < ns[0]; c++) {
                    acc += xs[0][c] * xs[0][c];
                }
                y[0] = std::sqrt(acc);
            },
            FLOAT);
    return MakeOp(std::move(desc), {v});
}

Variable Distance(const Variable& a, const Variable& b) {
    return Length(Sub(a, b));
}

Variable Normalize(const Variable& v) {
    OpDesc desc;
    desc.name = "Normalize";
    desc.fwd_code = "{y}=normalize({x0});";
    desc.bwd = [](const Variables& xs, const Variable& y, const Variable& gy,
                  uint32_t) -> Variable {
        // d/dx (x/|x|) applied to gy: (gy - y*dot(y,gy)) / |x|
        Variable len = Length(xs[0]);
        return Div(Sub(gy, Mul(y, Dot(y, gy))), len);
    };
    desc.cpu = ElementwiseCpu(
            [](const std::vector<const float*>& xs,
               const std::vector<uint32_t>& ns, float* y, uint32_t ny) {
                float acc = 0.f;
                for (uint32_t c = 0; c < ns[0]; c++) {
                    acc += xs[0][c] * xs[0][c];
                }
                const float inv = 1.f / std::sqrt(acc);
                for (uint32_t c = 0; c < ny; c++) {
                    y[c] = xs[0][c] * inv;
                }
            });
    return MakeOp(std::move(desc), {v});
}

Variable Reflect(const Variable& incident, const Variable& normal) {
    // I - 2*dot(N,I)*N as a composite; autodiff falls out of the primitives
    return Sub(incident,
               Mul(Mul(F::Float(2.f), Dot(normal, incident)), normal));
}

Variable CompSum(const Variable& v) {
    const VType vtype = v.getVType();
    if (vtype == FLOAT) {
        return v;
    }
    DRESSI_CHECK(IsVectorVType(vtype) && !IsIntVType(vtype),
                 "CompSum requires a float vector");
    const uint32_t n = NumComponents(vtype);
    std::string code = std::string("{y}=dot({x0},") + GlslTypeName(vtype) +
                       "(1.0));";
    OpDesc desc;
    desc.name = "CompSum";
    desc.fwd_code = code;
    desc.infer = FixedInfer(FLOAT);
    desc.bwd = [n](const Variables&, const Variable&, const Variable& gy,
                   uint32_t) -> Variable { return ComposeReplicate(gy, n); };
    desc.cpu = ElementwiseCpu(
            [](const std::vector<const float*>& xs,
               const std::vector<uint32_t>& ns, float* y, uint32_t) {
                float acc = 0.f;
                for (uint32_t c = 0; c < ns[0]; c++) {
                    acc += xs[0][c];
                }
                y[0] = acc;
            },
            FLOAT);
    return MakeOp(std::move(desc), {v});
}

// ---------------------------------- Matrix -----------------------------------
Variable MatMul(const Variable& a, const Variable& b) {
    const VType ta = a.getVType();
    const VType tb = b.getVType();
    DRESSI_CHECK(IsMatrixVType(ta), "MatMul: lhs must be a matrix");
    const uint32_t dim = (ta == MAT2) ? 2 : (ta == MAT3) ? 3 : 4;
    const bool vec_rhs = IsVectorVType(tb);
    if (vec_rhs) {
        DRESSI_CHECK(NumComponents(tb) == dim, "MatMul: dimension mismatch");
    } else {
        DRESSI_CHECK(tb == ta, "MatMul: rhs must match lhs matrix type");
    }
    const VType out_vtype = vec_rhs ? tb : ta;

    OpDesc desc;
    desc.name = "MatMul";
    desc.fwd_code = "{y}={x0}*{x1};";
    desc.infer = FixedInfer(out_vtype);
    desc.bwd = [vec_rhs](const Variables& xs, const Variable&,
                         const Variable& gy, uint32_t bwd_idx) -> Variable {
        if (vec_rhs) {
            if (bwd_idx == 0) {
                return OuterProduct(gy, xs[1]);
            }
            return MatMul(Transpose(xs[0]), gy);
        }
        if (bwd_idx == 0) {
            return MatMul(gy, Transpose(xs[1]));
        }
        return MatMul(Transpose(xs[0]), gy);
    };
    // Column-major storage: comp[col * dim + row] (GLSL convention)
    desc.cpu = ElementwiseCpu(
            [dim, vec_rhs](const std::vector<const float*>& xs,
                           const std::vector<uint32_t>&, float* y,
                           uint32_t) {
                const float* m = xs[0];
                const float* r = xs[1];
                if (vec_rhs) {
                    for (uint32_t row = 0; row < dim; row++) {
                        float acc = 0.f;
                        for (uint32_t col = 0; col < dim; col++) {
                            acc += m[col * dim + row] * r[col];
                        }
                        y[row] = acc;
                    }
                } else {
                    for (uint32_t col = 0; col < dim; col++) {
                        for (uint32_t row = 0; row < dim; row++) {
                            float acc = 0.f;
                            for (uint32_t k = 0; k < dim; k++) {
                                acc += m[k * dim + row] * r[col * dim + k];
                            }
                            y[col * dim + row] = acc;
                        }
                    }
                }
            },
            out_vtype);
    return MakeOp(std::move(desc), {a, b});
}

Variable Transpose(const Variable& m) {
    const VType t = m.getVType();
    DRESSI_CHECK(IsMatrixVType(t), "Transpose requires a matrix");
    const uint32_t dim = (t == MAT2) ? 2 : (t == MAT3) ? 3 : 4;
    OpDesc desc;
    desc.name = "Transpose";
    desc.fwd_code = "{y}=transpose({x0});";
    desc.infer = FixedInfer(t);
    desc.bwd = [](const Variables&, const Variable&, const Variable& gy,
                  uint32_t) -> Variable { return Transpose(gy); };
    desc.cpu = ElementwiseCpu(
            [dim](const std::vector<const float*>& xs,
                  const std::vector<uint32_t>&, float* y, uint32_t) {
                for (uint32_t col = 0; col < dim; col++) {
                    for (uint32_t row = 0; row < dim; row++) {
                        y[col * dim + row] = xs[0][row * dim + col];
                    }
                }
            },
            t);
    return MakeOp(std::move(desc), {m});
}

Variable OuterProduct(const Variable& col, const Variable& row) {
    const VType tc = col.getVType();
    DRESSI_CHECK(IsVectorVType(tc) && row.getVType() == tc,
                 "OuterProduct requires two equal float vectors");
    const uint32_t dim = NumComponents(tc);
    const VType out_vtype = (dim == 2) ? MAT2 : (dim == 3) ? MAT3 : MAT4;
    OpDesc desc;
    desc.name = "OuterProduct";
    desc.fwd_code = "{y}=outerProduct({x0},{x1});";
    desc.infer = FixedInfer(out_vtype);
    desc.bwd = [](const Variables& xs, const Variable&, const Variable& gy,
                  uint32_t bwd_idx) -> Variable {
        if (bwd_idx == 0) {
            return MatMul(gy, xs[1]);
        }
        return MatMul(Transpose(gy), xs[0]);
    };
    desc.cpu = ElementwiseCpu(
            [dim](const std::vector<const float*>& xs,
                  const std::vector<uint32_t>&, float* y, uint32_t) {
                for (uint32_t c = 0; c < dim; c++) {
                    for (uint32_t r = 0; r < dim; r++) {
                        y[c * dim + r] = xs[0][r] * xs[1][c];
                    }
                }
            },
            out_vtype);
    return MakeOp(std::move(desc), {col, row});
}

// --------------------------------- Reduction ---------------------------------
Variable SumPixelWise(const Variables& xs) {
    DRESSI_CHECK(!xs.empty(), "SumPixelWise requires inputs");
    if (xs.size() == 1) {
        return xs[0];
    }
    std::string code = "{y}=";
    for (size_t i = 0; i < xs.size(); i++) {
        code += (i ? "+" : "");
        code += "{x" + std::to_string(i) + "}";
    }
    code += ";";
    OpDesc desc;
    desc.name = "SumPixelWise";
    desc.fwd_code = code;
    desc.bwd = [](const Variables&, const Variable&, const Variable& gy,
                  uint32_t) -> Variable { return gy; };
    desc.cpu = ElementwiseCpu([](const std::vector<const float*>& xs_,
                                 const std::vector<uint32_t>& ns, float* y,
                                 uint32_t ny) {
        for (uint32_t c = 0; c < ny; c++) {
            float acc = 0.f;
            for (size_t i = 0; i < xs_.size(); i++) {
                acc += xs_[i][ns[i] == 1 ? 0 : c];
            }
            y[c] = acc;
        }
    });
    return MakeOp(std::move(desc), xs);
}

namespace {

ImgSize HalfSize(ImgSize s) {
    return {(s.w + 1) / 2, (s.h + 1) / 2};
}

Variable Reduce2x2Sum(const Variable& x);

// Nearest 2x-upsample of `x` to `target_size` (backward of Reduce2x2Sum)
Variable UpsampleNearest2x(const Variable& x, ImgSize target_size) {
    DRESSI_CHECK(HalfSize(target_size) == x.getImgSize(),
                 "UpsampleNearest2x: size mismatch");
    OpDesc desc;
    desc.name = "UpsampleNearest2x";
    // Special-cased by the shader codegen (reads {s0} at {coord}/2)
    desc.fwd_code = "__upsample_nearest_2x__";
    desc.input_access = {InputAccess::TexelFetch};
    desc.infer = [target_size](const Variables& xs_)
            -> std::pair<VType, ImgSize> {
        return {xs_[0].getVType(), target_size};
    };
    desc.bwd = [](const Variables&, const Variable&, const Variable& gy,
                  uint32_t) -> Variable { return Reduce2x2Sum(gy); };
    desc.cpu = [target_size](const std::vector<CpuTensor>& xs_) -> CpuTensor {
        const CpuTensor& src = xs_[0];
        CpuTensor y;
        y.vtype = src.vtype;
        y.size = target_size;
        const uint32_t nc = src.numComp();
        y.data.resize(size_t(target_size.w) * target_size.h * nc);
        for (uint32_t py = 0; py < target_size.h; py++) {
            for (uint32_t px = 0; px < target_size.w; px++) {
                const size_t src_idx =
                        (size_t(py / 2) * src.size.w + px / 2) * nc;
                const size_t dst_idx =
                        (size_t(py) * target_size.w + px) * nc;
                for (uint32_t c = 0; c < nc; c++) {
                    y.data[dst_idx + c] = src.data[src_idx + c];
                }
            }
        }
        return y;
    };
    return MakeOp(std::move(desc), {x});
}

// Sums each 2x2 block into one output pixel ({ceil(w/2), ceil(h/2)});
// out-of-range texels contribute zero
Variable Reduce2x2Sum(const Variable& x) {
    const ImgSize src_size = x.getImgSize();
    DRESSI_CHECK(!src_size.isUniform(), "Reduce2x2Sum on {1,1} input");
    OpDesc desc;
    desc.name = "Reduce2x2Sum";
    // Special-cased by the shader codegen (gathers 4 texels of {s0})
    desc.fwd_code = "__reduce_2x2_sum__";
    desc.input_access = {InputAccess::TexelFetch};
    desc.infer = [](const Variables& xs_) -> std::pair<VType, ImgSize> {
        return {xs_[0].getVType(), HalfSize(xs_[0].getImgSize())};
    };
    desc.bwd = [src_size](const Variables&, const Variable&,
                          const Variable& gy, uint32_t) -> Variable {
        return UpsampleNearest2x(gy, src_size);
    };
    desc.cpu = [](const std::vector<CpuTensor>& xs_) -> CpuTensor {
        const CpuTensor& src = xs_[0];
        CpuTensor y;
        y.vtype = src.vtype;
        y.size = HalfSize(src.size);
        const uint32_t nc = src.numComp();
        y.data.assign(size_t(y.size.w) * y.size.h * nc, 0.f);
        for (uint32_t py = 0; py < src.size.h; py++) {
            for (uint32_t px = 0; px < src.size.w; px++) {
                const size_t src_idx = (size_t(py) * src.size.w + px) * nc;
                const size_t dst_idx =
                        (size_t(py / 2) * y.size.w + px / 2) * nc;
                for (uint32_t c = 0; c < nc; c++) {
                    y.data[dst_idx + c] += src.data[src_idx + c];
                }
            }
        }
        return y;
    };
    return MakeOp(std::move(desc), {x});
}

}  // namespace

Variable Sum(const Variable& x) {
    Variable cur = x;
    while (!cur.getImgSize().isUniform()) {
        cur = Reduce2x2Sum(cur);
    }
    return cur;
}

Variable Mean(const Variable& x) {
    const ImgSize size = x.getImgSize();
    const uint32_t n_comp = NumComponents(x.getVType());
    const float inv_n = 1.f / (float(size.w) * float(size.h) * float(n_comp));
    return Mul(CompSum(Sum(x)), Float(inv_n));
}

// ------------------------------- DR functions --------------------------------
Variable StopGradient(const Variable& x) {
    OpDesc desc;
    desc.name = "StopGradient";
    desc.fwd_code = "{y}={x0};";
    desc.bwd = NullBwd;
    desc.cpu = MapCpu1([](float v) { return v; });
    return MakeOp(std::move(desc), {x});
}

Variable SetFragDepth(const Variable& depth) {
    OpDesc desc;
    desc.name = "SetFragDepth";
    // Special-cased by the shader codegen (writes gl_FragDepth)
    desc.fwd_code = "{y}={x0};";
    desc.bwd = [](const Variables&, const Variable&, const Variable& gy,
                  uint32_t) -> Variable { return gy; };
    desc.cpu = MapCpu1([](float v) { return v; });
    return MakeOp(std::move(desc), {depth});
}

Variable PeelDepth(const Variable& frag_depth,
                   const Variable& prev_frag_depth) {
    OpDesc desc;
    desc.name = "PeelDepth";
    desc.fwd_code = "if({x0}<={x1})discard; {y}=1.0;";
    desc.infer = FixedInfer(FLOAT);
    desc.bwd = NullBwd;
    return MakeOp(std::move(desc), {frag_depth, prev_frag_depth});
}

Variable Rasterize(const Variable& vtx_clip_pos, const Variable& vtx_attrib,
                   const Variable& faces, ImgSize screen_size) {
    DRESSI_CHECK(vtx_clip_pos.getVType() == VEC4 &&
                         vtx_clip_pos.getImgSize().h == 1,
                 "Rasterize: vtx_clip_pos must be VEC4 {V,1}");
    const VType attr_type = vtx_attrib.getVType();
    DRESSI_CHECK(!IsIntVType(attr_type) && !IsMatrixVType(attr_type),
                 "Rasterize: attribute must be a float scalar/vector");
    DRESSI_CHECK(vtx_attrib.getImgSize() == vtx_clip_pos.getImgSize(),
                 "Rasterize: attribute count must match vertex count");
    DRESSI_CHECK(faces.getVType() == IVEC3 && faces.getImgSize().h == 1,
                 "Rasterize: faces must be IVEC3 {F,1}");

    OpDesc desc;
    desc.name = "Rasterize";
    // Special-cased by the shader codegen (pass-through VS + interpolation)
    desc.fwd_code = "__rasterize__";
    desc.shader_type = RASTER;
    desc.infer = [attr_type, screen_size](const Variables&)
            -> std::pair<VType, ImgSize> { return {attr_type, screen_size}; };
    desc.bwd = NullBwd;  // non-differentiable w.r.t. geometry (M2)
    return MakeOp(std::move(desc), {vtx_clip_pos, vtx_attrib, faces});
}

namespace {

// Gathers the screen-space gradient into texture space through the
// inverse-UV table; texels whose forward sampling disagrees with the table
// entry (occluded in this view) receive zero gradient.
Variable GatherByInverseUV(const Variable& gy_screen, const Variable& inv_uv,
                           const Variable& uv_screen) {
    DRESSI_CHECK(inv_uv.getVType() == VEC4,
                 "GatherByInverseUV: inv_uv must be VEC4");
    OpDesc desc;
    desc.name = "GatherByInverseUV";
    // Special-cased by the shader codegen
    desc.fwd_code = "__gather_inv_uv__";
    desc.input_access = {InputAccess::TexelFetch, InputAccess::TexelFetch,
                         InputAccess::TexelFetch};
    desc.infer = [](const Variables& xs) -> std::pair<VType, ImgSize> {
        return {xs[0].getVType(), xs[1].getImgSize()};
    };
    desc.bwd = NullBwd;
    return MakeOp(std::move(desc), {gy_screen, inv_uv, uv_screen});
}

}  // namespace

Variable Texture(const Variable& tex, const Variable& uv,
                 const Variable& inv_uv) {
    const VType tex_type = tex.getVType();
    DRESSI_CHECK(!IsIntVType(tex_type) && !IsMatrixVType(tex_type),
                 "Texture: texture must be a float image");
    DRESSI_CHECK(!tex.getImgSize().isUniform(),
                 "Texture: texture must be a non-uniform image");
    DRESSI_CHECK(uv.getVType() == VEC2, "Texture: uv must be VEC2");
    DRESSI_CHECK(inv_uv.getVType() == VEC4 &&
                         inv_uv.getImgSize() == tex.getImgSize(),
                 "Texture: inv_uv must be VEC4 of the texture size");

    static const char* kSwizzles[] = {"", ".x", ".xy", ".xyz", ""};
    const uint32_t n = NumComponents(tex_type);

    OpDesc desc;
    desc.name = "Texture";
    desc.fwd_code = std::string("{y}=texture({s0},{x1})") +
                    (n == 4 ? "" : kSwizzles[n]) + ";";
    desc.input_access = {InputAccess::Sampled, InputAccess::SamePixel};
    desc.infer = [tex_type](const Variables& xs)
            -> std::pair<VType, ImgSize> {
        return {tex_type, xs[1].getImgSize()};
    };
    // The backward graph references the inverse-UV table and the forward
    // screen-space UVs through the closure (not forward inputs)
    desc.bwd = [inv_uv](const Variables& xs, const Variable&,
                        const Variable& gy, uint32_t bwd_idx) -> Variable {
        if (bwd_idx != 0) {
            return nullptr;  // not differentiable w.r.t. uv (M2)
        }
        return GatherByInverseUV(gy, inv_uv, xs[1]);
    };
    return MakeOp(std::move(desc), {tex, uv});
}

}  // namespace F

// ----------------------------- Operator overloads -----------------------------
Variable operator+(const Variable& a, const Variable& b) {
    return F::Add(a, b);
}
Variable operator-(const Variable& a, const Variable& b) {
    return F::Sub(a, b);
}
Variable operator*(const Variable& a, const Variable& b) {
    return F::Mul(a, b);
}
Variable operator/(const Variable& a, const Variable& b) {
    return F::Div(a, b);
}
Variable operator+(const Variable& a, float b) {
    return F::Add(a, F::Float(b));
}
Variable operator-(const Variable& a, float b) {
    return F::Sub(a, F::Float(b));
}
Variable operator*(const Variable& a, float b) {
    return F::Mul(a, F::Float(b));
}
Variable operator/(const Variable& a, float b) {
    return F::Div(a, F::Float(b));
}
Variable operator+(float a, const Variable& b) {
    return F::Add(F::Float(a), b);
}
Variable operator-(float a, const Variable& b) {
    return F::Sub(F::Float(a), b);
}
Variable operator*(float a, const Variable& b) {
    return F::Mul(F::Float(a), b);
}
Variable operator/(float a, const Variable& b) {
    return F::Div(F::Float(a), b);
}
Variable operator-(const Variable& x) {
    return F::Neg(x);
}
Variable operator<(const Variable& a, const Variable& b) {
    return F::Less(a, b);
}
Variable operator<(float a, const Variable& b) {
    return F::Less(F::Float(a), b);
}
Variable operator<(const Variable& a, float b) {
    return F::Less(a, F::Float(b));
}
Variable operator<=(const Variable& a, const Variable& b) {
    return F::LessEqual(a, b);
}
Variable operator>(const Variable& a, const Variable& b) {
    return F::Greater(a, b);
}
Variable operator>=(const Variable& a, const Variable& b) {
    return F::GreaterEqual(a, b);
}

}  // namespace dressi
