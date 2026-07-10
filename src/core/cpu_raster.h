#ifndef DRESSI_CORE_CPU_RASTER_H
#define DRESSI_CORE_CPU_RASTER_H

#include "core/node.h"

namespace dressi {

// Screen-space helpers shared by the CPU rasterizers, the HardSoftRas
// distance kernels and the AA kernels. Conventions (matching the Vulkan
// executor): NDC y is down after PerspectiveVk's y-flip, so
// screen = (clip.xy / clip.w * 0.5 + 0.5) * (W, H) with row 0 at the top;
// pixel centers at +0.5; depth = clip.z / clip.w in [0, 1]; LessOrEqual
// depth test against a clear value of 1; background = 0.

// Point-to-triangle signed distance in screen space (pixels).
// dist >= 0 inside (winding-agnostic), < 0 outside. `edge` is the argmin
// edge k = (s[k], s[(k+1)%3]) and `t` the closest-point parameter on it.
struct TriDist {
    float dist = 0.f;
    int edge = 0;
    float t = 0.f;
};
TriDist SignedDistToTri(const float p[2], const float s[3][2]);

// Per-vertex screen-space gradient of SignedDistToTri (envelope theorem:
// the argmin edge and clamp state are held fixed). g_s[i] accumulates
// d dist / d s[i]; entries for vertices not on the argmin edge are zero.
void SignedDistGradToTri(const float p[2], const float s[3][2],
                         const TriDist& td, float g_s[3][2]);

// 2D screen-space barycentric coordinates of p in triangle s (unclamped).
// Returns false for degenerate triangles.
bool ScreenBarycentric(const float p[2], const float s[3][2], float lambda[3]);

// CPU reference rasterizer for F::Rasterize: depth-tested, perspective-
// correct attribute interpolation. Faces with any |w| <= eps are skipped
// (no near-plane clipping).
CpuTensor RasterizeHardCpu(const CpuTensor& clip, const CpuTensor& attrib,
                           const CpuTensor& faces, ImgSize screen);

// CPU reference for F::RasterizeFaceId: float(face_index) + 1, background 0.
CpuTensor RasterizeFaceIdCpu(const CpuTensor& clip, const CpuTensor& faces,
                             ImgSize screen);

// CPU reference for F::RasterizeSoft: rasterizes the enlarged (soft)
// triangles, evaluates the signed distance to the hard face and depth-tests
// with the Eq.3 depth shift. Output VEC4 (dist_px, face_id, hard_z, coverage).
// prev_shifted_depth (optional, {S,S} FLOAT): depth-peeling input -- a
// fragment whose Eq.3-shifted depth is <= prev + 1e-4 is discarded.
CpuTensor RasterizeSoftCpu(const CpuTensor& soft_clip,
                           const CpuTensor& face_id,
                           const CpuTensor& faces_soft,
                           const CpuTensor& hard_clip,
                           const CpuTensor& faces_tex, ImgSize screen,
                           float radius_px,
                           const CpuTensor* prev_shifted_depth = nullptr);

// Projects clip vertex `v` (VEC4) to screen pixels; z = NDC depth [0,1].
// Returns false when |w| <= eps or w < 0 (behind the camera).
bool ProjectClipToScreen(const float clip[4], ImgSize screen, float s_xy[2],
                         float* z_ndc);

// Screen-space AA pixel pair (Dr.Hair Eq.4-5): the silhouette edge chosen
// for the (s, n) neighbor pair and the blend weight r = min(|es|/len, 1)
// (perpendicular distance from s's pixel center to the edge line, pixels).
// Owner preference {tri(n), tri(s)} approximates closest-depth selection.
// NOTE: r is asymmetric -- EvalAaPair(s, n) != EvalAaPair(n, s).
struct AaPair {
    float r = 1.f;
    uint32_t ia = 0, ib = 0;  // global vertex ids of the chosen edge
    float qa[2] = {0.f, 0.f}, qb[2] = {0.f, 0.f};  // screen endpoints
    float es = 0.f;   // signed edge function at s's pixel center
    float len = 0.f;  // edge length
};
// tri: FLOAT screen image of float(face)+1 (0 = background); clip: VEC4
// {V,1}; faces: float VEC3 {F,1}. Returns false when the pair is inactive
// (same tri id, out of range, or no separating silhouette edge).
bool EvalAaPair(const CpuTensor& tri, const CpuTensor& clip,
                const CpuTensor& faces, ImgSize screen, int sx, int sy,
                int nx, int ny, AaPair* out);

}  // namespace dressi

#endif  // DRESSI_CORE_CPU_RASTER_H
