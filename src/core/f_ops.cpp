#include "dressi/f.h"

#include <cmath>
#include <cstdio>
#include <string>

#include "core/cpu_raster.h"
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

Variable Sigmoid(const Variable& x) {
    OpDesc desc;
    desc.name = "Sigmoid";
    desc.fwd_code = "{y}=1.0/(1.0+exp(-{x0}));";
    desc.bwd = [](const Variables&, const Variable& y, const Variable& gy,
                  uint32_t) -> Variable {
        return Mul(gy, Mul(y, Sub(F::Float(1.f), y)));
    };
    desc.cpu = MapCpu1([](float v) { return 1.f / (1.f + std::exp(-v)); });
    return MakeOp(std::move(desc), {x});
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

Variable Broadcast(const Variable& x, ImgSize size) {
    DRESSI_CHECK(x.getImgSize().isUniform(),
                 "Broadcast: input must be a uniform {1,1} Variable");
    if (size.isUniform()) {
        return x;
    }
    OpDesc desc;
    desc.name = "Broadcast";
    desc.fwd_code = "{y}={x0};";
    desc.input_access = {InputAccess::TexelFetch};  // uniform read at (0,0)
    const VType vtype = x.getVType();
    desc.infer = [vtype, size](const Variables&)
            -> std::pair<VType, ImgSize> { return {vtype, size}; };
    desc.bwd = [](const Variables&, const Variable&, const Variable& gy,
                  uint32_t) -> Variable { return Sum(gy); };
    desc.cpu = [vtype, size](const std::vector<CpuTensor>& xs) {
        CpuTensor y;
        y.vtype = vtype;
        y.size = size;
        const uint32_t n = NumComponents(vtype);
        y.data.resize(size_t(size.w) * size.h * n);
        for (size_t p = 0; p < size_t(size.w) * size.h; p++) {
            for (uint32_t c = 0; c < n; c++) {
                y.data[p * n + c] = xs[0].data[c];
            }
        }
        return y;
    };
    return MakeOp(std::move(desc), {x});
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
    desc.bwd = NullBwd;  // geometry gradients live in RasterizeSoft/AntiAlias
    desc.cpu = [screen_size](const std::vector<CpuTensor>& xs) {
        return RasterizeHardCpu(xs[0], xs[1], xs[2], screen_size);
    };
    return MakeOp(std::move(desc), {vtx_clip_pos, vtx_attrib, faces});
}

namespace {

// Screen->vertex gradient gather for RasterizeSoft's dist channel: for each
// hard vertex, sums gy.x * d dist / d clip over all covered pixels of faces
// containing it (argmin edge + envelope theorem; screen->clip Jacobian with
// the 1/w terms, z gradient is exactly zero).
Variable GatherDistGrad(const Variable& gy_screen, const Variable& raster_out,
                        const Variable& vtx_clip_tex,
                        const Variable& faces_tex,
                        const Variable& vtx_faces_tex, float radius_px) {
    OpDesc desc;
    desc.name = "GatherDistGrad";
    // Special-cased by the shader codegen; the enlargement radius rides on
    // the marker so the shader can bound each vertex's pixel scan by the
    // exact soft-triangle bbox of its incident faces (from vtx_faces_tex)
    desc.fwd_code = "__gather_dist_grad__ r=" + FloatLiteral(radius_px);
    desc.input_access = {InputAccess::TexelFetch, InputAccess::TexelFetch,
                         InputAccess::TexelFetch, InputAccess::TexelFetch,
                         InputAccess::TexelFetch};
    desc.infer = [](const Variables& xs) -> std::pair<VType, ImgSize> {
        return {VEC4, xs[2].getImgSize()};
    };
    desc.bwd = NullBwd;
    desc.cpu = [](const std::vector<CpuTensor>& xs) {
        const CpuTensor& gy = xs[0];
        const CpuTensor& ro = xs[1];
        const CpuTensor& clip = xs[2];
        const CpuTensor& faces = xs[3];
        const ImgSize screen = ro.size;
        CpuTensor out;
        out.vtype = VEC4;
        out.size = clip.size;
        out.data.assign(size_t(clip.size.w) * 4, 0.f);
        const float w_scr = float(screen.w);
        const float h_scr = float(screen.h);
        for (uint32_t y = 0; y < screen.h; y++) {
            for (uint32_t x = 0; x < screen.w; x++) {
                const size_t o = (size_t(y) * screen.w + x) * 4;
                if (ro.data[o + 3] < 0.5f) {
                    continue;  // background
                }
                const float g = gy.data[o + 0];
                if (g == 0.f) {
                    continue;
                }
                const uint32_t f = uint32_t(int64_t(ro.data[o + 1] + 0.5f));
                uint32_t idx[3];
                float s[3][2];
                bool valid = true;
                for (int k = 0; k < 3; k++) {
                    idx[k] = uint32_t(int64_t(
                            faces.data[size_t(f) * 3 + k] + 0.5f));
                    if (!ProjectClipToScreen(&clip.data[size_t(idx[k]) * 4],
                                             screen, s[k], nullptr)) {
                        valid = false;
                        break;
                    }
                }
                if (!valid) {
                    continue;
                }
                const float p[2] = {float(x) + 0.5f, float(y) + 0.5f};
                const TriDist td = SignedDistToTri(p, s);
                float g_s[3][2];
                SignedDistGradToTri(p, s, td, g_s);
                for (int k = 0; k < 3; k++) {
                    const float gsx = g * g_s[k][0];
                    const float gsy = g * g_s[k][1];
                    if (gsx == 0.f && gsy == 0.f) {
                        continue;
                    }
                    const float* c = &clip.data[size_t(idx[k]) * 4];
                    float* dst = &out.data[size_t(idx[k]) * 4];
                    dst[0] += gsx * 0.5f * w_scr / c[3];
                    dst[1] += gsy * 0.5f * h_scr / c[3];
                    dst[3] += -(gsx * 0.5f * w_scr * c[0] +
                                gsy * 0.5f * h_scr * c[1]) /
                              (c[3] * c[3]);
                }
            }
        }
        return out;
    };
    return MakeOp(std::move(desc), {gy_screen, raster_out, vtx_clip_tex,
                                    faces_tex, vtx_faces_tex});
}

}  // namespace

Variable RasterizeSoft(const Variable& vtx_clip_soft, const Variable& face_id,
                       const Variable& faces_soft,
                       const Variable& vtx_clip_hard_tex,
                       const Variable& faces_tex,
                       const Variable& vtx_faces_tex, ImgSize screen_size,
                       float radius_px) {
    DRESSI_CHECK(vtx_clip_soft.getVType() == VEC4 &&
                         vtx_clip_soft.getImgSize().h == 1,
                 "RasterizeSoft: vtx_clip_soft must be VEC4 {3F,1}");
    DRESSI_CHECK(face_id.getVType() == FLOAT &&
                         face_id.getImgSize() == vtx_clip_soft.getImgSize(),
                 "RasterizeSoft: face_id must be FLOAT {3F,1}");
    DRESSI_CHECK(faces_soft.getVType() == IVEC3 &&
                         faces_soft.getImgSize().h == 1,
                 "RasterizeSoft: faces_soft must be IVEC3 {F,1}");
    DRESSI_CHECK(vtx_clip_soft.getImgSize().w ==
                         faces_soft.getImgSize().w * 3,
                 "RasterizeSoft: soft vertices must be unwelded (3 per face)");
    DRESSI_CHECK(vtx_clip_hard_tex.getVType() == VEC4 &&
                         vtx_clip_hard_tex.getImgSize().h == 1,
                 "RasterizeSoft: vtx_clip_hard_tex must be VEC4 {V,1}");
    DRESSI_CHECK(faces_tex.getVType() == VEC3 &&
                         faces_tex.getImgSize() == faces_soft.getImgSize(),
                 "RasterizeSoft: faces_tex must be VEC3 {F,1}");
    DRESSI_CHECK(vtx_faces_tex.getVType() == FLOAT &&
                         vtx_faces_tex.getImgSize().w ==
                                 vtx_clip_hard_tex.getImgSize().w,
                 "RasterizeSoft: vtx_faces_tex must be FLOAT {V,max_deg}");
    DRESSI_CHECK(radius_px > 0.f, "RasterizeSoft: radius_px must be > 0");

    OpDesc desc;
    desc.name = "RasterizeSoft";
    // Special-cased by the shader codegen; the enlargement radius rides on
    // the marker string (it is compiled into the depth-shift constant)
    desc.fwd_code = "__rasterize_soft__ r=" + FloatLiteral(radius_px);
    desc.shader_type = RASTER;
    desc.input_access = {InputAccess::SamePixel, InputAccess::SamePixel,
                         InputAccess::SamePixel, InputAccess::TexelFetch,
                         InputAccess::TexelFetch, InputAccess::TexelFetch};
    desc.infer = [screen_size](const Variables&)
            -> std::pair<VType, ImgSize> { return {VEC4, screen_size}; };
    desc.cpu = [screen_size, radius_px](const std::vector<CpuTensor>& xs) {
        return RasterizeSoftCpu(xs[0], xs[1], xs[2], xs[3], xs[4],
                                screen_size, radius_px);
    };
    // Only the dist channel is differentiable, and only w.r.t. the hard
    // clip positions; coverage/face_id/depth are piecewise constant in them
    desc.bwd = [radius_px](const Variables& xs, const Variable& y,
                           const Variable& gy, uint32_t bwd_idx) -> Variable {
        if (bwd_idx != 3) {
            return nullptr;
        }
        return GatherDistGrad(gy, y, xs[3], xs[4], xs[5], radius_px);
    };
    return MakeOp(std::move(desc),
                  {vtx_clip_soft, face_id, faces_soft, vtx_clip_hard_tex,
                   faces_tex, vtx_faces_tex});
}

Variable RasterizeSoft(const Variable& vtx_clip_soft, const Variable& face_id,
                       const Variable& faces_soft,
                       const Variable& vtx_clip_hard_tex,
                       const Variable& faces_tex,
                       const Variable& vtx_faces_tex,
                       const Variable& prev_shifted_depth,
                       ImgSize screen_size, float radius_px) {
    // Depth-peeling variant (Alg.2's outer K loop): fragments whose Eq.3
    // shifted depth is <= the previous layer's are discarded. Recreate the
    // previous layer's shifted depth from its output channels:
    //   shift = coverage * mix(0.5 + 0.5*clamp(-dist/r), 0.5*clamp(z),
    //                          step(0, dist))
    DRESSI_CHECK(prev_shifted_depth.getVType() == FLOAT &&
                         prev_shifted_depth.getImgSize() == screen_size,
                 "RasterizeSoft: prev_shifted_depth must be FLOAT {S,S}");
    // Same checks as the base overload run inside it; replicate the parts
    // the shared builder needs
    OpDesc desc;
    desc.name = "RasterizeSoftPeel";
    desc.fwd_code =
            "__rasterize_soft__ r=" + FloatLiteral(radius_px) + " peel=1";
    desc.shader_type = RASTER;
    desc.input_access = {InputAccess::SamePixel,  InputAccess::SamePixel,
                         InputAccess::SamePixel,  InputAccess::TexelFetch,
                         InputAccess::TexelFetch, InputAccess::TexelFetch,
                         InputAccess::TexelFetch};
    desc.infer = [screen_size](const Variables&)
            -> std::pair<VType, ImgSize> { return {VEC4, screen_size}; };
    desc.cpu = [screen_size, radius_px](const std::vector<CpuTensor>& xs) {
        return RasterizeSoftCpu(xs[0], xs[1], xs[2], xs[3], xs[4],
                                screen_size, radius_px, &xs[6]);
    };
    desc.bwd = [radius_px](const Variables& xs, const Variable& y,
                           const Variable& gy, uint32_t bwd_idx) -> Variable {
        if (bwd_idx != 3) {
            return nullptr;
        }
        return GatherDistGrad(gy, y, xs[3], xs[4], xs[5], radius_px);
    };
    return MakeOp(std::move(desc),
                  {vtx_clip_soft, face_id, faces_soft, vtx_clip_hard_tex,
                   faces_tex, vtx_faces_tex, prev_shifted_depth});
}

// ---------------- Indexed access (the paper's HardSoftRas core) --------------

namespace {

// Wang hash -> [0,1); identical integer arithmetic in GLSL and C++ so the
// stochastic backward is bit-reproducible across CPU/GPU
inline uint32_t WangHash(uint32_t x) {
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x ^= x >> 4;
    x *= 0x27d4eb2du;
    x ^= x >> 15;
    return x;
}

inline float HashToUnit(uint32_t f, uint32_t s, uint32_t seed,
                        uint32_t salt) {
    const uint32_t h =
            WangHash(f * 9781u ^ s * 6151u ^ seed * 26699u ^ salt * 42589u);
    return float(h & 0xFFFFFFu) / 16777216.f;
}

}  // namespace

Variable ScreenCoord(ImgSize size) {
    DRESSI_CHECK(!size.isUniform(), "ScreenCoord: size must be non-uniform");
    OpDesc desc;
    desc.name = "ScreenCoord";
    desc.fwd_code = "__screen_coord__";
    desc.infer = [size](const Variables&) -> std::pair<VType, ImgSize> {
        return {VEC2, size};
    };
    desc.bwd = NullBwd;
    desc.cpu = [size](const std::vector<CpuTensor>&) {
        CpuTensor y;
        y.vtype = VEC2;
        y.size = size;
        y.data.resize(size_t(size.w) * size.h * 2);
        for (uint32_t py = 0; py < size.h; py++) {
            for (uint32_t px = 0; px < size.w; px++) {
                const size_t o = (size_t(py) * size.w + px) * 2;
                y.data[o + 0] = float(px) + 0.5f;
                y.data[o + 1] = float(py) + 0.5f;
            }
        }
        return y;
    };
    return MakeOp(std::move(desc), {});
}

namespace {

// Exact backward of LookupFaces: each vertex sums the gradients of its
// incident face corners (adjacency texture; corner match checked)
Variable LookupFacesBwd(const Variable& gy, const Variable& faces_tex,
                        const Variable& vtx_faces_tex, uint32_t corner) {
    OpDesc desc;
    desc.name = "LookupFacesBwd";
    desc.fwd_code = "__lookup_faces_bwd__ k=" + std::to_string(corner);
    desc.input_access = {InputAccess::TexelFetch, InputAccess::TexelFetch,
                         InputAccess::TexelFetch};
    desc.infer = [](const Variables& xs) -> std::pair<VType, ImgSize> {
        return {xs[0].getVType(), {xs[2].getImgSize().w, 1}};
    };
    desc.bwd = NullBwd;
    desc.cpu = [corner](const std::vector<CpuTensor>& xs) {
        const CpuTensor& gy_t = xs[0];
        const CpuTensor& faces = xs[1];
        const CpuTensor& vf = xs[2];
        const uint32_t n_verts = vf.size.w;
        const uint32_t max_deg = vf.size.h;
        const uint32_t n = gy_t.numComp();
        CpuTensor out;
        out.vtype = gy_t.vtype;
        out.size = {n_verts, 1};
        out.data.assign(size_t(n_verts) * n, 0.f);
        for (uint32_t v = 0; v < n_verts; v++) {
            for (uint32_t d = 0; d < max_deg; d++) {
                const float fv = vf.data[size_t(d) * n_verts + v];
                if (fv < -0.5f) {
                    continue;
                }
                const uint32_t f = uint32_t(int64_t(fv + 0.5f));
                const uint32_t vi = uint32_t(int64_t(
                        faces.data[size_t(f) * 3 + corner] + 0.5f));
                if (vi != v) {
                    continue;
                }
                for (uint32_t c = 0; c < n; c++) {
                    out.data[size_t(v) * n + c] +=
                            gy_t.data[size_t(f) * n + c];
                }
            }
        }
        return out;
    };
    return MakeOp(std::move(desc), {gy, faces_tex, vtx_faces_tex});
}

}  // namespace

Variable LookupFaces(const Variable& vtx_attr, const Variable& faces_tex,
                     const Variable& vtx_faces_tex, uint32_t corner) {
    DRESSI_CHECK(!IsIntVType(vtx_attr.getVType()) &&
                         !IsMatrixVType(vtx_attr.getVType()) &&
                         vtx_attr.getImgSize().h == 1,
                 "LookupFaces: vtx_attr must be a float {V,1} Variable");
    DRESSI_CHECK(faces_tex.getVType() == VEC3 &&
                         faces_tex.getImgSize().h == 1,
                 "LookupFaces: faces_tex must be VEC3 {F,1}");
    DRESSI_CHECK(vtx_faces_tex.getVType() == FLOAT &&
                         vtx_faces_tex.getImgSize().w ==
                                 vtx_attr.getImgSize().w,
                 "LookupFaces: vtx_faces_tex must be FLOAT {V,max_deg}");
    DRESSI_CHECK(corner < 3, "LookupFaces: corner must be 0, 1 or 2");

    OpDesc desc;
    desc.name = "LookupFaces";
    desc.fwd_code = "__lookup_faces__ k=" + std::to_string(corner);
    desc.input_access = {InputAccess::TexelFetch, InputAccess::TexelFetch,
                         InputAccess::TexelFetch};
    const VType vtype = vtx_attr.getVType();
    const uint32_t n_faces = faces_tex.getImgSize().w;
    desc.infer = [vtype, n_faces](const Variables&)
            -> std::pair<VType, ImgSize> { return {vtype, {n_faces, 1}}; };
    desc.bwd = [corner](const Variables& xs, const Variable&,
                        const Variable& gy, uint32_t bwd_idx) -> Variable {
        if (bwd_idx != 0) {
            return nullptr;
        }
        return LookupFacesBwd(gy, xs[1], xs[2], corner);
    };
    desc.cpu = [corner](const std::vector<CpuTensor>& xs) {
        const CpuTensor& attr = xs[0];
        const CpuTensor& faces = xs[1];
        const uint32_t nf = faces.size.w;
        const uint32_t n = attr.numComp();
        CpuTensor out;
        out.vtype = attr.vtype;
        out.size = {nf, 1};
        out.data.resize(size_t(nf) * n);
        for (uint32_t f = 0; f < nf; f++) {
            const uint32_t vi = uint32_t(int64_t(
                    faces.data[size_t(f) * 3 + corner] + 0.5f));
            for (uint32_t c = 0; c < n; c++) {
                out.data[size_t(f) * n + c] =
                        attr.data[size_t(vi) * n + c];
            }
        }
        return out;
    };
    return MakeOp(std::move(desc), {vtx_attr, faces_tex, vtx_faces_tex});
}

namespace {

// Stochastic backward of FaceFetch (the paper's inverse-UV philosophy,
// Alg.1 adapted): per face, gather gy at n_samples quasi-random points
// along the hard edges (offset within +-radius_px), guarded by the ID
// buffer. Fully parallel O(F); the sampling jitter (`seed`) makes the
// estimate cover the gradient band over iterations.
Variable FaceFetchBwd(const Variable& gy, const Variable& idx_img,
                      const Variable& tri_prj_0, const Variable& tri_prj_1,
                      const Variable& tri_prj_2, const Variable& seed,
                      float radius_px, uint32_t n_samples) {
    OpDesc desc;
    desc.name = "FaceFetchBwd";
    desc.fwd_code = "__face_fetch_bwd__ r=" + FloatLiteral(radius_px) +
                    " n=" + std::to_string(n_samples);
    desc.input_access = {InputAccess::TexelFetch, InputAccess::TexelFetch,
                         InputAccess::TexelFetch, InputAccess::TexelFetch,
                         InputAccess::TexelFetch, InputAccess::TexelFetch};
    desc.infer = [](const Variables& xs) -> std::pair<VType, ImgSize> {
        return {xs[0].getVType(), xs[2].getImgSize()};
    };
    desc.bwd = NullBwd;
    desc.cpu = [radius_px, n_samples](const std::vector<CpuTensor>& xs) {
        const CpuTensor& gy_t = xs[0];
        const CpuTensor& idx = xs[1];
        const CpuTensor* prj[3] = {&xs[2], &xs[3], &xs[4]};
        const uint32_t seed_u = uint32_t(int64_t(xs[5].data[0] + 0.5f));
        const ImgSize screen = idx.size;
        const uint32_t nf = xs[2].size.w;
        const uint32_t n = gy_t.numComp();
        CpuTensor out;
        out.vtype = gy_t.vtype;
        out.size = {nf, 1};
        out.data.assign(size_t(nf) * n, 0.f);
        for (uint32_t f = 0; f < nf; f++) {
            float s[3][2];
            bool ok = true;
            for (int k = 0; k < 3; k++) {
                const float* c = &prj[k]->data[size_t(f) * 4];
                if (c[3] <= 1e-6f) {
                    ok = false;
                    break;
                }
                s[k][0] = (c[0] / c[3] * 0.5f + 0.5f) * float(screen.w);
                s[k][1] = (c[1] / c[3] * 0.5f + 0.5f) * float(screen.h);
            }
            if (!ok) {
                continue;
            }
            float* dst = &out.data[size_t(f) * n];
            for (uint32_t smp = 0; smp < n_samples; smp++) {
                const int e = int(smp % 3);
                const float* a = s[e];
                const float* b = s[(e + 1) % 3];
                const float ex = b[0] - a[0];
                const float ey = b[1] - a[1];
                const float len = std::sqrt(ex * ex + ey * ey);
                if (len < 1e-6f) {
                    continue;
                }
                const float u = HashToUnit(f, smp, seed_u, 0);
                const float o = (HashToUnit(f, smp, seed_u, 1) * 2.f - 1.f) *
                                radius_px;
                const float px = a[0] + u * ex - o * (ey / len);
                const float py = a[1] + u * ey + o * (ex / len);
                const int ix = int(std::floor(px));
                const int iy = int(std::floor(py));
                if (ix < 0 || iy < 0 || ix >= int(screen.w) ||
                    iy >= int(screen.h)) {
                    continue;
                }
                const size_t p = size_t(iy) * screen.w + ix;
                if (uint32_t(int64_t(idx.data[p] + 0.5f)) != f + 1) {
                    continue;  // not visible as this face here
                }
                for (uint32_t c = 0; c < n; c++) {
                    dst[c] += gy_t.data[p * n + c];
                }
            }
        }
        return out;
    };
    return MakeOp(std::move(desc),
                  {gy, idx_img, tri_prj_0, tri_prj_1, tri_prj_2, seed});
}

}  // namespace

Variable FaceFetch(const Variable& tri_attr, const Variable& idx_img,
                   const Variable& tri_prj_0, const Variable& tri_prj_1,
                   const Variable& tri_prj_2, const Variable& seed,
                   float radius_px, uint32_t n_samples) {
    DRESSI_CHECK(!IsIntVType(tri_attr.getVType()) &&
                         !IsMatrixVType(tri_attr.getVType()) &&
                         tri_attr.getImgSize().h == 1,
                 "FaceFetch: tri_attr must be a float {F,1} Variable");
    DRESSI_CHECK(idx_img.getVType() == FLOAT &&
                         !idx_img.getImgSize().isUniform(),
                 "FaceFetch: idx_img must be a FLOAT image");
    for (const Variable* prj : {&tri_prj_0, &tri_prj_1, &tri_prj_2}) {
        DRESSI_CHECK(prj->getVType() == VEC4 &&
                             prj->getImgSize() == tri_attr.getImgSize(),
                     "FaceFetch: tri_prj must be VEC4 {F,1}");
    }
    DRESSI_CHECK(seed.getVType() == FLOAT && seed.getImgSize().isUniform(),
                 "FaceFetch: seed must be FLOAT {1,1}");
    DRESSI_CHECK(radius_px > 0.f && n_samples > 0,
                 "FaceFetch: radius/samples must be positive");

    OpDesc desc;
    desc.name = "FaceFetch";
    desc.fwd_code = "__face_fetch__";
    desc.input_access = {InputAccess::TexelFetch, InputAccess::TexelFetch,
                         InputAccess::TexelFetch, InputAccess::TexelFetch,
                         InputAccess::TexelFetch, InputAccess::TexelFetch};
    const VType vtype = tri_attr.getVType();
    desc.infer = [vtype](const Variables& xs) -> std::pair<VType, ImgSize> {
        return {vtype, xs[1].getImgSize()};
    };
    desc.bwd = [radius_px, n_samples](const Variables& xs, const Variable&,
                                      const Variable& gy,
                                      uint32_t bwd_idx) -> Variable {
        if (bwd_idx != 0) {
            return nullptr;  // ids / sampling geometry / seed: no gradient
        }
        return FaceFetchBwd(gy, xs[1], xs[2], xs[3], xs[4], xs[5],
                            radius_px, n_samples);
    };
    desc.cpu = [](const std::vector<CpuTensor>& xs) {
        const CpuTensor& attr = xs[0];
        const CpuTensor& idx = xs[1];
        const uint32_t n = attr.numComp();
        CpuTensor out;
        out.vtype = attr.vtype;
        out.size = idx.size;
        out.data.assign(idx.data.size() * n, 0.f);
        for (size_t p = 0; p < idx.data.size(); p++) {
            const int f = int(idx.data[p] + 0.5f) - 1;
            if (f < 0 || uint32_t(f) >= attr.size.w) {
                continue;
            }
            for (uint32_t c = 0; c < n; c++) {
                out.data[p * n + c] = attr.data[size_t(f) * n + c];
            }
        }
        return out;
    };
    return MakeOp(std::move(desc),
                  {tri_attr, idx_img, tri_prj_0, tri_prj_1, tri_prj_2,
                   seed});
}

// ---------------------- Mesh utilities (GPU-resident) ------------------------

namespace {

void CrossOfFace(const CpuTensor& pos, const CpuTensor& faces, uint32_t f,
                 uint32_t idx[3], float c[3]) {
    for (int k = 0; k < 3; k++) {
        idx[k] = uint32_t(int64_t(faces.data[size_t(f) * 3 + k] + 0.5f));
    }
    const float* p0 = &pos.data[size_t(idx[0]) * 3];
    const float* p1 = &pos.data[size_t(idx[1]) * 3];
    const float* p2 = &pos.data[size_t(idx[2]) * 3];
    const float e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
    const float e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
    c[0] = e1[1] * e2[2] - e1[2] * e2[1];
    c[1] = e1[2] * e2[0] - e1[0] * e2[2];
    c[2] = e1[0] * e2[1] - e1[1] * e2[0];
}

}  // namespace

Variable SoftClip(const Variable& vtx_clip_hard_tex, const Variable& faces_tex,
                  ImgSize screen_size, float radius_px) {
    DRESSI_CHECK(vtx_clip_hard_tex.getVType() == VEC4 &&
                         vtx_clip_hard_tex.getImgSize().h == 1,
                 "SoftClip: vtx_clip_hard_tex must be VEC4 {V,1}");
    DRESSI_CHECK(faces_tex.getVType() == VEC3 &&
                         faces_tex.getImgSize().h == 1,
                 "SoftClip: faces_tex must be VEC3 {F,1}");
    DRESSI_CHECK(radius_px > 0.f, "SoftClip: radius_px must be > 0");
    const uint32_t n_faces = faces_tex.getImgSize().w;

    OpDesc desc;
    desc.name = "SoftClip";
    // Special-cased by the shader codegen (radius/screen ride the marker)
    desc.fwd_code = "__soft_clip__ r=" + FloatLiteral(radius_px) +
                    " w=" + std::to_string(screen_size.w) +
                    " h=" + std::to_string(screen_size.h);
    desc.input_access = {InputAccess::TexelFetch, InputAccess::TexelFetch};
    desc.infer = [n_faces](const Variables&) -> std::pair<VType, ImgSize> {
        return {VEC4, {n_faces * 3, 1}};
    };
    desc.bwd = NullBwd;  // coverage geometry: piecewise constant by design
    desc.cpu = [screen_size, radius_px](const std::vector<CpuTensor>& xs) {
        const CpuTensor& clip = xs[0];
        const CpuTensor& faces = xs[1];
        const uint32_t nf = faces.size.w;
        CpuTensor out;
        out.vtype = VEC4;
        out.size = {nf * 3, 1};
        out.data.assign(size_t(nf) * 12, 0.f);
        for (uint32_t f = 0; f < nf; f++) {
            uint32_t idx[3];
            float s[3][2];
            const float* c[3];
            bool valid = true;
            for (int k = 0; k < 3; k++) {
                idx[k] = uint32_t(int64_t(
                        faces.data[size_t(f) * 3 + k] + 0.5f));
                c[k] = &clip.data[size_t(idx[k]) * 4];
                if (c[k][3] <= 1e-6f ||
                    !ProjectClipToScreen(c[k], screen_size, s[k], nullptr)) {
                    valid = false;
                }
            }
            float scale = 1.f;
            float cx = 0.f, cy = 0.f;
            if (valid) {
                cx = (s[0][0] + s[1][0] + s[2][0]) / 3.f;
                cy = (s[0][1] + s[1][1] + s[2][1]) / 3.f;
                float d_min = 1e30f;
                for (int k = 0; k < 3; k++) {
                    const int k2 = (k + 1) % 3;
                    const float ex = s[k2][0] - s[k][0];
                    const float ey = s[k2][1] - s[k][1];
                    const float len = std::sqrt(ex * ex + ey * ey);
                    if (len < 1e-6f) {
                        continue;
                    }
                    d_min = std::min(d_min,
                                     std::abs(ex * (cy - s[k][1]) -
                                              ey * (cx - s[k][0])) /
                                             len);
                }
                scale = d_min < 1e29f
                                ? std::min(1.f + radius_px /
                                                         std::max(d_min,
                                                                  1e-3f),
                                           8.f)
                                : 1.f;
            }
            for (int k = 0; k < 3; k++) {
                float* dst = &out.data[(size_t(f) * 3 + size_t(k)) * 4];
                for (int j = 0; j < 4; j++) {
                    dst[j] = c[k][j];
                }
                if (valid) {
                    const float sx = cx + (s[k][0] - cx) * scale;
                    const float sy = cy + (s[k][1] - cy) * scale;
                    dst[0] = (sx / float(screen_size.w) * 2.f - 1.f) *
                             c[k][3];
                    dst[1] = (sy / float(screen_size.h) * 2.f - 1.f) *
                             c[k][3];
                }
            }
        }
        return out;
    };
    return MakeOp(std::move(desc), {vtx_clip_hard_tex, faces_tex});
}

Variable VertexNeighborMean(const Variable& pos_tex,
                            const Variable& vtx_neighbors_tex) {
    DRESSI_CHECK(pos_tex.getVType() == VEC3 && pos_tex.getImgSize().h == 1,
                 "VertexNeighborMean: pos_tex must be VEC3 {V,1}");
    DRESSI_CHECK(vtx_neighbors_tex.getVType() == FLOAT &&
                         vtx_neighbors_tex.getImgSize().w ==
                                 pos_tex.getImgSize().w,
                 "VertexNeighborMean: adjacency must be FLOAT {V,max_deg}");
    OpDesc desc;
    desc.name = "VertexNeighborMean";
    desc.fwd_code = "__vertex_neighbor_mean__";
    desc.input_access = {InputAccess::TexelFetch, InputAccess::TexelFetch};
    desc.infer = [](const Variables& xs) -> std::pair<VType, ImgSize> {
        return {VEC3, xs[0].getImgSize()};
    };
    desc.bwd = NullBwd;
    desc.cpu = [](const std::vector<CpuTensor>& xs) {
        const CpuTensor& pos = xs[0];
        const CpuTensor& adj = xs[1];
        const uint32_t n_verts = pos.size.w;
        const uint32_t max_deg = adj.size.h;
        CpuTensor out = pos;
        for (uint32_t v = 0; v < n_verts; v++) {
            float mean[3] = {0.f, 0.f, 0.f};
            uint32_t count = 0;
            for (uint32_t d = 0; d < max_deg; d++) {
                const float nv = adj.data[size_t(d) * n_verts + v];
                if (nv < -0.5f) {
                    continue;
                }
                const size_t n = size_t(int64_t(nv + 0.5f));
                for (int c = 0; c < 3; c++) {
                    mean[c] += pos.data[n * 3 + size_t(c)];
                }
                count++;
            }
            if (count > 0) {
                for (int c = 0; c < 3; c++) {
                    out.data[size_t(v) * 3 + size_t(c)] =
                            mean[c] / float(count);
                }
            }
        }
        return out;
    };
    return MakeOp(std::move(desc), {pos_tex, vtx_neighbors_tex});
}

Variable NormalConsistencyFaceTerm(const Variable& pos_tex,
                                   const Variable& faces_tex,
                                   const Variable& face_adj_tex) {
    DRESSI_CHECK(pos_tex.getVType() == VEC3 && pos_tex.getImgSize().h == 1,
                 "NormalConsistencyFaceTerm: pos_tex must be VEC3 {V,1}");
    DRESSI_CHECK(faces_tex.getVType() == VEC3 &&
                         face_adj_tex.getVType() == VEC3 &&
                         face_adj_tex.getImgSize() == faces_tex.getImgSize(),
                 "NormalConsistencyFaceTerm: faces/face_adj must be VEC3 "
                 "{F,1}");
    OpDesc desc;
    desc.name = "NormalConsistencyFaceTerm";
    desc.fwd_code = "__nc_face_term__";
    desc.input_access = {InputAccess::TexelFetch, InputAccess::TexelFetch,
                         InputAccess::TexelFetch};
    desc.infer = [](const Variables& xs) -> std::pair<VType, ImgSize> {
        return {VEC3, xs[1].getImgSize()};
    };
    desc.bwd = NullBwd;
    desc.cpu = [](const std::vector<CpuTensor>& xs) {
        const CpuTensor& pos = xs[0];
        const CpuTensor& faces = xs[1];
        const CpuTensor& fadj = xs[2];
        const uint32_t nf = faces.size.w;
        CpuTensor out;
        out.vtype = VEC3;
        out.size = {nf, 1};
        out.data.assign(size_t(nf) * 3, 0.f);
        for (uint32_t f = 0; f < nf; f++) {
            uint32_t ia[3];
            float ca[3];
            CrossOfFace(pos, faces, f, ia, ca);
            const float la = std::sqrt(ca[0] * ca[0] + ca[1] * ca[1] +
                                       ca[2] * ca[2]);
            if (la < 1e-12f) {
                continue;
            }
            const float na[3] = {ca[0] / la, ca[1] / la, ca[2] / la};
            float g[3] = {0.f, 0.f, 0.f};
            for (int j = 0; j < 3; j++) {
                const float gv = fadj.data[size_t(f) * 3 + j];
                if (gv < -0.5f) {
                    continue;
                }
                uint32_t ib[3];
                float cb[3];
                CrossOfFace(pos, faces, uint32_t(int64_t(gv + 0.5f)), ib,
                            cb);
                const float lb = std::sqrt(cb[0] * cb[0] + cb[1] * cb[1] +
                                           cb[2] * cb[2]);
                if (lb < 1e-12f) {
                    continue;
                }
                const float nb[3] = {cb[0] / lb, cb[1] / lb, cb[2] / lb};
                const float d = na[0] * nb[0] + na[1] * nb[1] +
                                na[2] * nb[2];
                for (int c = 0; c < 3; c++) {
                    g[c] += -(nb[c] - d * na[c]) / la;
                }
            }
            for (int c = 0; c < 3; c++) {
                out.data[size_t(f) * 3 + size_t(c)] = g[c];
            }
        }
        return out;
    };
    return MakeOp(std::move(desc), {pos_tex, faces_tex, face_adj_tex});
}

Variable NormalConsistencyVertexGrad(const Variable& face_term,
                                     const Variable& pos_tex,
                                     const Variable& faces_tex,
                                     const Variable& vtx_faces_tex) {
    DRESSI_CHECK(face_term.getVType() == VEC3 &&
                         face_term.getImgSize() == faces_tex.getImgSize(),
                 "NormalConsistencyVertexGrad: face_term must be VEC3 {F,1}");
    DRESSI_CHECK(vtx_faces_tex.getVType() == FLOAT &&
                         vtx_faces_tex.getImgSize().w ==
                                 pos_tex.getImgSize().w,
                 "NormalConsistencyVertexGrad: vtx_faces_tex must be FLOAT "
                 "{V,max_deg}");
    OpDesc desc;
    desc.name = "NormalConsistencyVertexGrad";
    desc.fwd_code = "__nc_vertex_grad__";
    desc.input_access = {InputAccess::TexelFetch, InputAccess::TexelFetch,
                         InputAccess::TexelFetch, InputAccess::TexelFetch};
    desc.infer = [](const Variables& xs) -> std::pair<VType, ImgSize> {
        return {VEC3, xs[1].getImgSize()};
    };
    desc.bwd = NullBwd;
    desc.cpu = [](const std::vector<CpuTensor>& xs) {
        const CpuTensor& term = xs[0];
        const CpuTensor& pos = xs[1];
        const CpuTensor& faces = xs[2];
        const CpuTensor& vf = xs[3];
        const uint32_t n_verts = pos.size.w;
        const uint32_t max_deg = vf.size.h;
        CpuTensor out;
        out.vtype = VEC3;
        out.size = {n_verts, 1};
        out.data.assign(size_t(n_verts) * 3, 0.f);
        for (uint32_t v = 0; v < n_verts; v++) {
            float acc[3] = {0.f, 0.f, 0.f};
            for (uint32_t d = 0; d < max_deg; d++) {
                const float fv = vf.data[size_t(d) * n_verts + v];
                if (fv < -0.5f) {
                    continue;
                }
                const uint32_t f = uint32_t(int64_t(fv + 0.5f));
                uint32_t idx[3];
                for (int k = 0; k < 3; k++) {
                    idx[k] = uint32_t(int64_t(
                            faces.data[size_t(f) * 3 + k] + 0.5f));
                }
                const float* p0 = &pos.data[size_t(idx[0]) * 3];
                const float* p1 = &pos.data[size_t(idx[1]) * 3];
                const float* p2 = &pos.data[size_t(idx[2]) * 3];
                const float e1[3] = {p1[0] - p0[0], p1[1] - p0[1],
                                     p1[2] - p0[2]};
                const float e2[3] = {p2[0] - p0[0], p2[1] - p0[1],
                                     p2[2] - p0[2]};
                const float* g = &term.data[size_t(f) * 3];
                // c = e1 x e2: grad_p1 = e2 x g, grad_p2 = g x e1,
                // grad_p0 = -(both)
                const float g1[3] = {e2[1] * g[2] - e2[2] * g[1],
                                     e2[2] * g[0] - e2[0] * g[2],
                                     e2[0] * g[1] - e2[1] * g[0]};
                const float g2[3] = {g[1] * e1[2] - g[2] * e1[1],
                                     g[2] * e1[0] - g[0] * e1[2],
                                     g[0] * e1[1] - g[1] * e1[0]};
                for (int c = 0; c < 3; c++) {
                    if (idx[1] == v) {
                        acc[c] += g1[c];
                    }
                    if (idx[2] == v) {
                        acc[c] += g2[c];
                    }
                    if (idx[0] == v) {
                        acc[c] -= g1[c] + g2[c];
                    }
                }
            }
            for (int c = 0; c < 3; c++) {
                out.data[size_t(v) * 3 + size_t(c)] = acc[c];
            }
        }
        return out;
    };
    return MakeOp(std::move(desc),
                  {face_term, pos_tex, faces_tex, vtx_faces_tex});
}

Variable RasterizeFaceId(const Variable& vtx_clip_pos,
                         const Variable& vtx_attrib_dummy,
                         const Variable& faces, ImgSize screen_size) {
    DRESSI_CHECK(vtx_clip_pos.getVType() == VEC4 &&
                         vtx_clip_pos.getImgSize().h == 1,
                 "RasterizeFaceId: vtx_clip_pos must be VEC4 {V,1}");
    const VType attr_type = vtx_attrib_dummy.getVType();
    DRESSI_CHECK(!IsIntVType(attr_type) && !IsMatrixVType(attr_type),
                 "RasterizeFaceId: attribute must be a float scalar/vector");
    DRESSI_CHECK(vtx_attrib_dummy.getImgSize() == vtx_clip_pos.getImgSize(),
                 "RasterizeFaceId: attribute count must match vertex count");
    DRESSI_CHECK(faces.getVType() == IVEC3 && faces.getImgSize().h == 1,
                 "RasterizeFaceId: faces must be IVEC3 {F,1}");

    OpDesc desc;
    desc.name = "RasterizeFaceId";
    // Special-cased by the shader codegen (writes gl_PrimitiveID + 1)
    desc.fwd_code = "__rasterize_face_id__";
    desc.shader_type = RASTER;
    desc.infer = [screen_size](const Variables&)
            -> std::pair<VType, ImgSize> { return {FLOAT, screen_size}; };
    desc.bwd = NullBwd;
    desc.cpu = [screen_size](const std::vector<CpuTensor>& xs) {
        return RasterizeFaceIdCpu(xs[0], xs[2], screen_size);
    };
    return MakeOp(std::move(desc), {vtx_clip_pos, vtx_attrib_dummy, faces});
}

namespace {

constexpr int kAaOffs[8][2] = {{-1, -1}, {0, -1}, {1, -1}, {-1, 0},
                               {1, 0},   {-1, 1}, {0, 1},  {1, 1}};

// d c_aa / d img: c_aa is linear in the image given fixed geometry.
// g(x) = (1/9) [ gy(x) (1 + sum_k alpha_k) + sum_k beta_k gy(n_k) ] with
// alpha_k = r(x, n_k) for active pairs else 1, beta_k = 1 - r(n_k, x) for
// active swapped pairs else 0 (r is asymmetric).
Variable AntiAliasBwdImg(const Variable& gy, const Variable& tri_id,
                         const Variable& vtx_clip, const Variable& faces) {
    OpDesc desc;
    desc.name = "AntiAliasBwdImg";
    desc.fwd_code = "__antialias_bwd_img__";
    desc.input_access = {InputAccess::TexelFetch, InputAccess::TexelFetch,
                         InputAccess::TexelFetch, InputAccess::TexelFetch};
    desc.infer = [](const Variables& xs) -> std::pair<VType, ImgSize> {
        return {xs[0].getVType(), xs[0].getImgSize()};
    };
    desc.bwd = NullBwd;
    desc.cpu = [](const std::vector<CpuTensor>& xs) {
        const CpuTensor& gy = xs[0];
        const CpuTensor& tri = xs[1];
        const CpuTensor& clip = xs[2];
        const CpuTensor& faces_t = xs[3];
        const ImgSize screen = gy.size;
        const uint32_t n = gy.numComp();
        CpuTensor out;
        out.vtype = gy.vtype;
        out.size = screen;
        out.data.assign(gy.data.size(), 0.f);
        for (int y = 0; y < int(screen.h); y++) {
            for (int x = 0; x < int(screen.w); x++) {
                const size_t so = (size_t(y) * screen.w + x) * n;
                float alpha_sum = 0.f;
                float* dst = &out.data[so];
                for (int k = 0; k < 8; k++) {
                    const int nx = x + kAaOffs[k][0];
                    const int ny = y + kAaOffs[k][1];
                    if (nx < 0 || ny < 0 || nx >= int(screen.w) ||
                        ny >= int(screen.h)) {
                        alpha_sum += 1.f;
                        continue;
                    }
                    AaPair pr;
                    alpha_sum += EvalAaPair(tri, clip, faces_t, screen, x, y,
                                            nx, ny, &pr)
                                         ? pr.r
                                         : 1.f;
                    AaPair pr2;
                    if (EvalAaPair(tri, clip, faces_t, screen, nx, ny, x, y,
                                   &pr2)) {
                        const size_t no = (size_t(ny) * screen.w + nx) * n;
                        for (uint32_t c = 0; c < n; c++) {
                            dst[c] += (1.f - pr2.r) * gy.data[no + c];
                        }
                    }
                }
                for (uint32_t c = 0; c < n; c++) {
                    dst[c] += gy.data[so + c] * (1.f + alpha_sum);
                    dst[c] /= 9.f;
                }
            }
        }
        return out;
    };
    return MakeOp(std::move(desc), {gy, tri_id, vtx_clip, faces});
}

// d c_aa / d vtx_clip, gathered per vertex: for every active boundary pair
// whose chosen edge touches this vertex, accumulate
// (1/9) dot(gy(s), c(s) - c(n)) * d r / d clip (perpendicular-distance
// derivative + screen->clip Jacobian; zero when r clamps at 1).
Variable AntiAliasBwdVtx(const Variable& gy, const Variable& img,
                         const Variable& tri_id, const Variable& vtx_clip,
                         const Variable& faces,
                         const Variable& vtx_faces_tex, const Variable& seed,
                         uint32_t n_samples) {
    OpDesc desc;
    desc.name = "AntiAliasBwdVtx";
    // n=0: exact scan of the incident faces' pixel bboxes; n>0: the
    // paper-pattern stochastic backward (n jittered samples along each
    // incident face's edges per iteration)
    desc.fwd_code =
            "__antialias_bwd_vtx__ n=" + std::to_string(n_samples);
    desc.input_access = {InputAccess::TexelFetch, InputAccess::TexelFetch,
                         InputAccess::TexelFetch, InputAccess::TexelFetch,
                         InputAccess::TexelFetch, InputAccess::TexelFetch,
                         InputAccess::TexelFetch};
    desc.infer = [](const Variables& xs) -> std::pair<VType, ImgSize> {
        return {VEC4, xs[3].getImgSize()};
    };
    desc.bwd = NullBwd;
    desc.cpu = [n_samples](const std::vector<CpuTensor>& xs) {
        const CpuTensor& gy = xs[0];
        const CpuTensor& img = xs[1];
        const CpuTensor& tri = xs[2];
        const CpuTensor& clip = xs[3];
        const CpuTensor& faces_t = xs[4];
        const ImgSize screen = gy.size;
        const uint32_t n = gy.numComp();
        const float w_scr = float(screen.w);
        const float h_scr = float(screen.h);
        CpuTensor out;
        out.vtype = VEC4;
        out.size = clip.size;
        out.data.assign(size_t(clip.size.w) * 4, 0.f);
        // Accumulates one active pair's contribution for vertex `v` only
        const auto accumulate = [&](uint32_t v, int x, int y, int nx,
                                    int ny, const AaPair& pr,
                                    bool only_v) {
            const size_t so = (size_t(y) * screen.w + x) * n;
            const size_t no = (size_t(ny) * screen.w + nx) * n;
            float gr = 0.f;
            for (uint32_t c = 0; c < n; c++) {
                gr += gy.data[so + c] *
                      (img.data[so + c] - img.data[no + c]);
            }
            gr /= 9.f;
            if (gr == 0.f) {
                return;
            }
            const float ps[2] = {float(x) + 0.5f, float(y) + 0.5f};
            const float sgn = pr.es >= 0.f ? 1.f : -1.f;
            const float inv_l = 1.f / pr.len;
            const float l3 = std::abs(pr.es) * inv_l * inv_l * inv_l;
            const float ex = pr.qb[0] - pr.qa[0];
            const float ey = pr.qb[1] - pr.qa[1];
            const float dqa[2] = {
                    sgn * inv_l * (pr.qb[1] - ps[1]) + l3 * ex,
                    sgn * inv_l * (ps[0] - pr.qb[0]) + l3 * ey};
            const float dqb[2] = {
                    sgn * inv_l * (ps[1] - pr.qa[1]) - l3 * ex,
                    sgn * inv_l * (pr.qa[0] - ps[0]) - l3 * ey};
            const uint32_t ids[2] = {pr.ia, pr.ib};
            const float* dqs[2] = {dqa, dqb};
            for (int e = 0; e < 2; e++) {
                if (only_v && ids[e] != v) {
                    continue;
                }
                const float gqx = gr * dqs[e][0];
                const float gqy = gr * dqs[e][1];
                const float* c = &clip.data[size_t(ids[e]) * 4];
                float* dst = &out.data[size_t(ids[e]) * 4];
                dst[0] += gqx * 0.5f * w_scr / c[3];
                dst[1] += gqy * 0.5f * h_scr / c[3];
                dst[3] += -(gqx * 0.5f * w_scr * c[0] +
                            gqy * 0.5f * h_scr * c[1]) /
                          (c[3] * c[3]);
            }
        };
        if (n_samples == 0) {
            // Exact: scatter over every boundary pixel pair
            for (int y = 0; y < int(screen.h); y++) {
                for (int x = 0; x < int(screen.w); x++) {
                    for (int k = 0; k < 8; k++) {
                        const int nx = x + kAaOffs[k][0];
                        const int ny = y + kAaOffs[k][1];
                        if (nx < 0 || ny < 0 || nx >= int(screen.w) ||
                            ny >= int(screen.h)) {
                            continue;
                        }
                        AaPair pr;
                        if (!EvalAaPair(tri, clip, faces_t, screen, x, y,
                                        nx, ny, &pr) ||
                            pr.r >= 1.f) {
                            continue;
                        }
                        accumulate(0, x, y, nx, ny, pr, false);
                    }
                }
            }
            return out;
        }
        // Stochastic (paper pattern): per vertex, jittered samples along
        // its incident faces' edges (shared per-face sample positions)
        const CpuTensor& vf = xs[5];
        const uint32_t seed_u = uint32_t(int64_t(xs[6].data[0] + 0.5f));
        const uint32_t n_verts = clip.size.w;
        const uint32_t max_deg = vf.size.h;
        for (uint32_t v = 0; v < n_verts; v++) {
            for (uint32_t d = 0; d < max_deg; d++) {
                const float fv = vf.data[size_t(d) * n_verts + v];
                if (fv < -0.5f) {
                    continue;
                }
                const uint32_t f = uint32_t(int64_t(fv + 0.5f));
                float s3[3][2];
                bool ok = true;
                for (int k = 0; k < 3; k++) {
                    const uint32_t vi = uint32_t(int64_t(
                            faces_t.data[size_t(f) * 3 + k] + 0.5f));
                    const float* c = &clip.data[size_t(vi) * 4];
                    if (c[3] <= 1e-6f) {
                        ok = false;
                        break;
                    }
                    s3[k][0] = (c[0] / c[3] * 0.5f + 0.5f) * w_scr;
                    s3[k][1] = (c[1] / c[3] * 0.5f + 0.5f) * h_scr;
                }
                if (!ok) {
                    continue;
                }
                for (uint32_t smp = 0; smp < n_samples; smp++) {
                    const int e = int(smp % 3);
                    const float* a = s3[e];
                    const float* b = s3[(e + 1) % 3];
                    const float ex = b[0] - a[0];
                    const float ey = b[1] - a[1];
                    const float len = std::sqrt(ex * ex + ey * ey);
                    if (len < 1e-6f) {
                        continue;
                    }
                    const float u = HashToUnit(f, smp, seed_u, 2);
                    const float o = (HashToUnit(f, smp, seed_u, 3) * 2.f -
                                     1.f) *
                                    2.f;
                    const int x = int(std::floor(a[0] + u * ex -
                                                 o * (ey / len)));
                    const int y = int(std::floor(a[1] + u * ey +
                                                 o * (ex / len)));
                    if (x < 0 || y < 0 || x >= int(screen.w) ||
                        y >= int(screen.h)) {
                        continue;
                    }
                    for (int k = 0; k < 8; k++) {
                        const int nx = x + kAaOffs[k][0];
                        const int ny = y + kAaOffs[k][1];
                        if (nx < 0 || ny < 0 || nx >= int(screen.w) ||
                            ny >= int(screen.h)) {
                            continue;
                        }
                        AaPair pr;
                        if (!EvalAaPair(tri, clip, faces_t, screen, x, y,
                                        nx, ny, &pr) ||
                            pr.r >= 1.f) {
                            continue;
                        }
                        if (pr.ia != v && pr.ib != v) {
                            continue;
                        }
                        accumulate(v, x, y, nx, ny, pr, true);
                    }
                }
            }
        }
        return out;
    };
    return MakeOp(std::move(desc),
                  {gy, img, tri_id, vtx_clip, faces, vtx_faces_tex, seed});
}

}  // namespace

Variable AntiAlias(const Variable& img, const Variable& tri_id,
                   const Variable& vtx_clip, const Variable& faces,
                   const Variable& vtx_faces_tex, const Variable& seed,
                   uint32_t n_samples) {
    const VType img_type = img.getVType();
    DRESSI_CHECK(seed.getVType() == FLOAT && seed.getImgSize().isUniform(),
                 "AntiAlias: seed must be FLOAT {1,1}");
    DRESSI_CHECK(!IsIntVType(img_type) && !IsMatrixVType(img_type),
                 "AntiAlias: img must be a float image");
    DRESSI_CHECK(!img.getImgSize().isUniform(),
                 "AntiAlias: img must be a non-uniform image");
    DRESSI_CHECK(tri_id.getVType() == FLOAT &&
                         tri_id.getImgSize() == img.getImgSize(),
                 "AntiAlias: tri_id must be a FLOAT image of img's size");
    DRESSI_CHECK(vtx_clip.getVType() == VEC4 &&
                         vtx_clip.getImgSize().h == 1,
                 "AntiAlias: vtx_clip must be VEC4 {V,1}");
    DRESSI_CHECK(faces.getVType() == VEC3 && faces.getImgSize().h == 1,
                 "AntiAlias: faces must be VEC3 {F,1}");
    DRESSI_CHECK(vtx_faces_tex.getVType() == FLOAT &&
                         vtx_faces_tex.getImgSize().w ==
                                 vtx_clip.getImgSize().w,
                 "AntiAlias: vtx_faces_tex must be FLOAT {V,max_deg}");

    OpDesc desc;
    desc.name = "AntiAlias";
    desc.fwd_code = "__antialias__";
    desc.input_access = {InputAccess::TexelFetch, InputAccess::TexelFetch,
                         InputAccess::TexelFetch, InputAccess::TexelFetch,
                         InputAccess::TexelFetch, InputAccess::TexelFetch};
    desc.infer = [img_type](const Variables& xs)
            -> std::pair<VType, ImgSize> {
        return {img_type, xs[0].getImgSize()};
    };
    desc.bwd = [n_samples](const Variables& xs, const Variable&,
                           const Variable& gy, uint32_t bwd_idx) -> Variable {
        if (bwd_idx == 0) {
            return AntiAliasBwdImg(gy, xs[1], xs[2], xs[3]);
        }
        if (bwd_idx == 2) {
            return AntiAliasBwdVtx(gy, xs[0], xs[1], xs[2], xs[3], xs[4],
                                   xs[5], n_samples);
        }
        return nullptr;  // tri_id / faces / adjacency / seed: no gradient
    };
    desc.cpu = [](const std::vector<CpuTensor>& xs) {
        const CpuTensor& img_t = xs[0];
        const CpuTensor& tri = xs[1];
        const CpuTensor& clip = xs[2];
        const CpuTensor& faces_t = xs[3];
        const ImgSize screen = img_t.size;
        const uint32_t n = img_t.numComp();
        CpuTensor out;
        out.vtype = img_t.vtype;
        out.size = screen;
        out.data.assign(img_t.data.size(), 0.f);
        for (int y = 0; y < int(screen.h); y++) {
            for (int x = 0; x < int(screen.w); x++) {
                const size_t so = (size_t(y) * screen.w + x) * n;
                const float* cs = &img_t.data[so];
                float* dst = &out.data[so];
                for (uint32_t c = 0; c < n; c++) {
                    dst[c] = cs[c];
                }
                for (int k = 0; k < 8; k++) {
                    const int nx = x + kAaOffs[k][0];
                    const int ny = y + kAaOffs[k][1];
                    AaPair pr;
                    if (nx < 0 || ny < 0 || nx >= int(screen.w) ||
                        ny >= int(screen.h) ||
                        !EvalAaPair(tri, clip, faces_t, screen, x, y, nx, ny,
                                    &pr)) {
                        for (uint32_t c = 0; c < n; c++) {
                            dst[c] += cs[c];
                        }
                        continue;
                    }
                    const float* cn =
                            &img_t.data[(size_t(ny) * screen.w + nx) * n];
                    for (uint32_t c = 0; c < n; c++) {
                        dst[c] += pr.r * cs[c] + (1.f - pr.r) * cn[c];
                    }
                }
                for (uint32_t c = 0; c < n; c++) {
                    dst[c] /= 9.f;
                }
            }
        }
        return out;
    };
    return MakeOp(std::move(desc),
                  {img, tri_id, vtx_clip, faces, vtx_faces_tex, seed});
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
