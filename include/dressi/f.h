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

// -------------------------------- Broadcast ----------------------------------
// Replicates a uniform {1,1} value into an image of `size` (backward sums
// back to {1,1}); inline constants compile to a literal
Variable Broadcast(const Variable& x, ImgSize size);

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

// HardSoftRas soft rasterization (arXiv:2204.01386 Alg.2, K=1): rasterizes
// CPU-enlarged ("soft") triangles and outputs per pixel
//   VEC4 (signed_dist_px, face_id, hard_depth_ndc, coverage)
// where signed_dist_px is the screen-space distance to the ORIGINAL (hard)
// triangle (>= 0 inside) and gl_FragDepth applies the paper's Eq.3 depth
// shift (hard region in [0, 0.5], soft rim in [0.5, 1]). Background stays
// (0,0,0,0); gate downstream sigmoids with the coverage channel.
// Differentiable w.r.t. vtx_clip_hard_tex only (through the dist channel).
//   vtx_clip_soft:     VEC4 {3F,1} enlarged unwelded clip positions (leaf)
//   face_id:           FLOAT {3F,1} face index vertex attribute
//   faces_soft:        IVEC3 {F,1} sequential (3i,3i+1,3i+2) faces (leaf)
//   vtx_clip_hard_tex: VEC4 {V,1} original clip positions as an image leaf
//   faces_tex:         VEC3 {F,1} float-valued vertex indices as an image
//   vtx_faces_tex:     FLOAT {V,max_deg} vertex -> incident-face ids
//                      (-1 padding; static topology, bounds the backward's
//                      per-vertex pixel scan)
// (see BuildSoftGeometry / VertexFacesTex in the examples)
Variable RasterizeSoft(const Variable& vtx_clip_soft, const Variable& face_id,
                       const Variable& faces_soft,
                       const Variable& vtx_clip_hard_tex,
                       const Variable& faces_tex,
                       const Variable& vtx_faces_tex, ImgSize screen_size,
                       float radius_px);

// Depth-peeling variant (Alg.2, peel k > 0): fragments whose Eq.3-shifted
// depth is <= `prev_shifted_depth` (+1e-4) are discarded, exposing the
// next layer. Reconstruct the previous layer's shifted depth from its
// output channels with elementwise ops (see the silhouette example).
Variable RasterizeSoft(const Variable& vtx_clip_soft, const Variable& face_id,
                       const Variable& faces_soft,
                       const Variable& vtx_clip_hard_tex,
                       const Variable& faces_tex,
                       const Variable& vtx_faces_tex,
                       const Variable& prev_shifted_depth,
                       ImgSize screen_size, float radius_px);

// Rasterizes the triangle-ID buffer: float(face_index) + 1 per covered
// pixel (depth-tested), 0 for background. `vtx_attrib_dummy` is rasterized
// vertex-format-wise but ignored (keeps the {pos, attrib, faces} executor
// contract); pass any per-vertex float attribute. Not differentiable.
Variable RasterizeFaceId(const Variable& vtx_clip_pos,
                         const Variable& vtx_attrib_dummy,
                         const Variable& faces, ImgSize screen_size);

// Screen-space anti-aliasing (arXiv:2403.17496 Eq.3-5): blends each pixel
// with its 8 neighbors across triangle-ID boundaries using the distance
// from the pixel center to the silhouette edge of the triangle covering
// the neighbor (owner preference {tri(n), tri(s)} approximates
// closest-depth selection without a depth buffer). Differentiable w.r.t.
// `img` and w.r.t. `vtx_clip` (dense boundary gradients through the edge
// distance). Pass the SAME clip-position leaf used for rasterization so
// the gradient reaches it.
//   img:           float image to anti-alias (the rasterized mask/color)
//   tri_id:        FLOAT image from RasterizeFaceId (same size as img)
//   vtx_clip:      VEC4 {V,1} clip positions as an image leaf
//   faces:         VEC3 {F,1} float-valued vertex indices as an image leaf
//   vtx_faces_tex: FLOAT {V,max_deg} vertex -> incident-face ids (-1
//                  padding)
//   seed:          FLOAT {1,1} jitter seed for the stochastic backward
//   n_samples:     vertex-backward mode -- 0 = exact scan of the incident
//                  faces' pixel bboxes; > 0 = the paper-pattern stochastic
//                  backward (n jittered samples along each incident
//                  face's edges per iteration, fully parallel)
Variable AntiAlias(const Variable& img, const Variable& tri_id,
                   const Variable& vtx_clip, const Variable& faces,
                   const Variable& vtx_faces_tex, const Variable& seed,
                   uint32_t n_samples = 8);

// ---------------- Indexed access (the paper's HardSoftRas core) --------------

// Pixel-center screen coordinates as a VEC2 image (x+0.5, y+0.5); no
// inputs, not differentiable. Feeds image-space geometric math.
Variable ScreenCoord(ImgSize size);

// Per-face corner attribute: out{F,1} = vtx_attr[faces[f][corner]]
// (the paper's LookupFaces). Differentiable w.r.t. vtx_attr with an EXACT
// backward (each vertex sums the gradients of its incident face corners
// through the vertex->face adjacency texture).
Variable LookupFaces(const Variable& vtx_attr, const Variable& faces_tex,
                     const Variable& vtx_faces_tex, uint32_t corner);

// Per-pixel indexed fetch: out(p) = tri_attr[idx(p) - 1], 0 where
// idx(p) == 0 (background). The backward w.r.t. tri_attr follows the
// paper's inverse-UV philosophy (Alg.1): instead of an exact sum over all
// covered pixels (which a graphics pipeline cannot scatter without float
// atomics), each face samples `n_samples` quasi-random points along its
// hard edges (band offset in [-radius_px, radius_px], jittered by `seed`,
// e.g. the in-graph iteration counter) and gathers gy where the ID buffer
// confirms visibility -- a stochastic, fully parallel O(F) gather.
//   tri_attr:      {F,1} float per-face values (a LookupFaces output)
//   idx_img:       FLOAT {S,S} face ids from the rasterizer (0=background)
//   tri_prj_0/1/2: VEC4 {F,1} corner clip positions (sampling geometry)
//   seed:          FLOAT {1,1} jitter seed, changes every iteration
Variable FaceFetch(const Variable& tri_attr, const Variable& idx_img,
                   const Variable& tri_prj_0, const Variable& tri_prj_1,
                   const Variable& tri_prj_2, const Variable& seed,
                   float radius_px, uint32_t n_samples = 4);

// ---------------------- Mesh utilities (GPU-resident) ------------------------
// Forward-only helpers so geometry optimization runs without per-frame
// CPU round trips. All inputs are texelFetch images; none are
// differentiable (regularizer terms enter the OPTIMIZER update graph,
// which is never differentiated).

// HardSoftRas soft geometry on the GPU: per-face unwelded clip positions
// enlarged about the screen-space centroid (the in-graph counterpart of
// the examples' BuildSoftGeometry). Output VEC4 {3F,1}; behind-camera
// faces pass through un-enlarged.
Variable SoftClip(const Variable& vtx_clip_hard_tex, const Variable& faces_tex,
                  ImgSize screen_size, float radius_px);

// Mean of each vertex's neighbor positions: pos_tex VEC3 {V,1},
// vtx_neighbors_tex FLOAT {V,max_deg} (-1 padding). Isolated vertices
// return their own position (zero Laplacian).
Variable VertexNeighborMean(const Variable& pos_tex,
                            const Variable& vtx_neighbors_tex);

// Normal-consistency gradient of sum over adjacent face pairs of
// (1 - n_f . n_g), split in two stages:
//   face term: d E / d cross(f) summed over the <= 3 edge neighbors
//              (face_adj_tex VEC3 {F,1}, -1 padding) -> VEC3 {F,1}
//   vertex grad: chain through the cross product to VEC3 {V,1} using the
//                vertex -> incident-face adjacency
Variable NormalConsistencyFaceTerm(const Variable& pos_tex,
                                   const Variable& faces_tex,
                                   const Variable& face_adj_tex);
Variable NormalConsistencyVertexGrad(const Variable& face_term,
                                     const Variable& pos_tex,
                                     const Variable& faces_tex,
                                     const Variable& vtx_faces_tex);

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
