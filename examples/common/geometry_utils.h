#ifndef DRESSI_EXAMPLES_GEOMETRY_UTILS_H
#define DRESSI_EXAMPLES_GEOMETRY_UTILS_H

#include <string>
#include <vector>

#include "asset_utils.h"

namespace dressi_examples {

// Welded unit-radius icosphere (uv left zero): level 0 = icosahedron
// (12 v / 20 f), each level quadruples faces (level 3 = 642 v / 1280 f).
Mesh GenerateIcosphere(uint32_t level);

// Writes positions/faces as a Wavefront OBJ (uv omitted)
void SaveObjMesh(const std::string& path, const Mesh& mesh);

// Vertex -> neighbor-vertex adjacency from the face list
std::vector<std::vector<uint32_t>> BuildVertexAdjacency(
        const dressi::CpuImage& faces, uint32_t n_verts);

// Uniform-Laplacian regularization gradient {V,1,3}:
// lambda * (pos - mean(neighbors)); isolated vertices get zero
dressi::CpuImage UniformLaplacianGrad(
        const dressi::CpuImage& pos,
        const std::vector<std::vector<uint32_t>>& adj, float lambda);

// Adjacent face pairs sharing an edge (each unordered pair once)
std::vector<std::array<uint32_t, 2>> BuildFaceAdjacency(
        const dressi::CpuImage& faces);

// Normal-consistency regularization gradient {V,1,3} of
// lambda * sum_{adjacent faces (f,g)} (1 - n_f . n_g)
// (analytic through the normalized cross products; penalizes creases and
// flipped faces that a positional Laplacian cannot see)
dressi::CpuImage NormalConsistencyGrad(
        const dressi::CpuImage& pos, const dressi::CpuImage& faces,
        const std::vector<std::array<uint32_t, 2>>& face_adj, float lambda);

// Number of adjacent face pairs whose normals disagree (n_f . n_g < 0) --
// a direct count of flipped/creased faces for diagnostics
uint32_t CountFlippedFacePairs(
        const dressi::CpuImage& pos, const dressi::CpuImage& faces,
        const std::vector<std::array<uint32_t, 2>>& face_adj);

// Per-face unwelded soft geometry for HardSoftRas: triangles enlarged by
// `radius_px` in screen space (scaled about the centroid), face index as a
// per-vertex attribute, and sequential faces (3i, 3i+1, 3i+2).
// Behind-camera faces (w <= eps) are passed through un-enlarged.
struct SoftGeometry {
    dressi::CpuImage clip;     // {3F,1,4}
    dressi::CpuImage face_id;  // {3F,1,1}
    dressi::CpuImage faces;    // {F,1,3}
};
SoftGeometry BuildSoftGeometry(const dressi::CpuImage& hard_clip,
                               const dressi::CpuImage& faces,
                               dressi::ImgSize screen, float radius_px);

// Chain rule through the (fixed) projection: g_pos {V,1,3} from the
// clip-position gradient {V,1,4} and the column-major MVP
dressi::CpuImage ChainRuleClipToPos(const dressi::CpuImage& g_clip,
                                    const Mat4& mvp);

// Adam optimizer state over a flat float parameter vector
struct AdamState {
    std::vector<float> m, v;
    int t = 0;
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
};
void AdamStep(AdamState& state, std::vector<float>& params,
              const std::vector<float>& grad, float lr);

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_GEOMETRY_UTILS_H
