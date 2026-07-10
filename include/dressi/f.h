#ifndef DRESSI_F_H
#define DRESSI_F_H

#include "dressi/function.h"
#include "dressi/types.h"
#include "dressi/variable.h"

namespace dressi {
namespace F {

// --------------------------------- Constants ---------------------------------
Variable Float(float fv);
Variable Vec2(float x, float y);
Variable Vec3(float x, float y, float z);
Variable Vec4(float x, float y, float z, float w);

// --------------------------------- Arithmetic --------------------------------
Variable Add(const Variable& x0, const Variable& x1);
Variable Sub(const Variable& x0, const Variable& x1);
Variable Mul(const Variable& x0, const Variable& x1);
Variable Div(const Variable& x0, const Variable& x1);
Variable Neg(const Variable& x);

// ----------------------------------- Math ------------------------------------
Variable Sin(const Variable& x);
Variable Cos(const Variable& x);
Variable Tan(const Variable& x);
Variable Asin(const Variable& x);
Variable Acos(const Variable& x);
Variable Atan(const Variable& x);
Variable Atan2(const Variable& y, const Variable& x);
Variable Exp(const Variable& x);
Variable Exp2(const Variable& x);
Variable Log(const Variable& x);
Variable Log2(const Variable& x);
Variable Sqrt(const Variable& x);
Variable InverseSqrt(const Variable& x);
Variable Abs(const Variable& x);
Variable Sign(const Variable& x);
Variable Floor(const Variable& x);
Variable Fract(const Variable& x);
Variable Mod(const Variable& x, const Variable& m);
Variable Pow(const Variable& x, const Variable& p);
Variable Max(const Variable& x0, const Variable& x1);
Variable Min(const Variable& x0, const Variable& x1);
Variable Clamp(const Variable& x, const Variable& lo, const Variable& hi);
Variable Mix(const Variable& x0, const Variable& x1, const Variable& t);
Variable Step(const Variable& edge, const Variable& x);
Variable SmoothStep(const Variable& e0, const Variable& e1, const Variable& x);
Variable Sigmoid(const Variable& x);

// -------------------------- Comparison (scalar, 0/1) --------------------------
Variable Less(const Variable& x0, const Variable& x1);
Variable LessEqual(const Variable& x0, const Variable& x1);
Variable Greater(const Variable& x0, const Variable& x1);
Variable GreaterEqual(const Variable& x0, const Variable& x1);

// ---------------------------------- Vector -----------------------------------
Variable Vec2(const Variable& x, const Variable& y);
Variable Vec3(const Variable& x, const Variable& y, const Variable& z);
Variable Vec3(const Variable& xy, const Variable& z);   // vec2 + scalar
Variable Vec4(const Variable& x, const Variable& y, const Variable& z,
              const Variable& w);
Variable Vec4(const Variable& xyz, const Variable& w);  // vec3 + scalar
Variable GetX(const Variable& v);
Variable GetY(const Variable& v);
Variable GetZ(const Variable& v);
Variable GetW(const Variable& v);
Variable Dot(const Variable& a, const Variable& b);
Variable Cross(const Variable& a, const Variable& b);
Variable Length(const Variable& v);
Variable Distance(const Variable& a, const Variable& b);
Variable Normalize(const Variable& v);
Variable Reflect(const Variable& incident, const Variable& normal);
Variable CompSum(const Variable& v);  // sum of components -> FLOAT

// ---------------------------------- Matrix -----------------------------------
Variable MatMul(const Variable& a, const Variable& b);  // MAT*MAT / MAT*VEC
Variable Transpose(const Variable& m);
Variable OuterProduct(const Variable& col, const Variable& row);

// --------------------------------- Reduction ---------------------------------
// Pixel-wise sum of a list of images (single n-ary node)
Variable SumPixelWise(const Variables& xs);
// Spatial reductions over the whole image
Variable Sum(const Variable& x);   // -> same vtype, {1,1}
Variable Mean(const Variable& x);  // -> FLOAT, {1,1} (component average)

// ------------------------------- DR functions --------------------------------
Variable StopGradient(const Variable& x);
Variable SetFragDepth(const Variable& depth);
Variable PeelDepth(const Variable& frag_depth, const Variable& prev_frag_depth);

// Rasterizes indexed triangles and interpolates a vertex attribute into a
// screen-space image (depth-tested; background = 0). Non-differentiable
// with respect to geometry in this milestone.
//   vtx_clip_pos: VEC4 {V,1} clip-space positions
//   vtx_attrib:   float scalar/vector {V,1} attribute to interpolate
//   faces:        IVEC3 {F,1} triangle vertex indices
Variable Rasterize(const Variable& vtx_clip_pos, const Variable& vtx_attrib,
                   const Variable& faces, ImgSize screen_size);

// Samples `tex` (nearest) at per-pixel `uv`. Differentiable with respect to
// the texture through the inverse-UV lookup table `inv_uv`: a VEC4
// {tex_size} image of (screen_x, screen_y, valid, 0) built by rasterizing
// the mesh in UV space (see BuildInverseUV in the examples). The gradient
// gathers from the screen-space gradient at inv_uv, rejecting texels whose
// forward sampling disagrees (occlusion guard). Not differentiable with
// respect to `uv`.
Variable Texture(const Variable& tex, const Variable& uv,
                 const Variable& inv_uv);

}  // namespace F

// ----------------------------- Operator overloads -----------------------------
Variable operator+(const Variable& a, const Variable& b);
Variable operator-(const Variable& a, const Variable& b);
Variable operator*(const Variable& a, const Variable& b);
Variable operator/(const Variable& a, const Variable& b);
Variable operator+(const Variable& a, float b);
Variable operator-(const Variable& a, float b);
Variable operator*(const Variable& a, float b);
Variable operator/(const Variable& a, float b);
Variable operator+(float a, const Variable& b);
Variable operator-(float a, const Variable& b);
Variable operator*(float a, const Variable& b);
Variable operator/(float a, const Variable& b);
Variable operator-(const Variable& x);
Variable operator<(const Variable& a, const Variable& b);
Variable operator<(float a, const Variable& b);
Variable operator<(const Variable& a, float b);
Variable operator<=(const Variable& a, const Variable& b);
Variable operator>(const Variable& a, const Variable& b);
Variable operator>=(const Variable& a, const Variable& b);

}  // namespace dressi

#endif  // DRESSI_F_H
