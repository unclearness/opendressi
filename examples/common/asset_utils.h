#ifndef DRESSI_EXAMPLES_ASSET_UTILS_H
#define DRESSI_EXAMPLES_ASSET_UTILS_H

#include <array>
#include <string>
#include <vector>

#include "dressi/mesh_utils.h"
#include "dressi/types.h"

namespace dressi_examples {

// Unwelded triangle mesh: one vertex per unique (position, uv) pair.
// Layouts match Dressi-AD leaves: pos {V,1,3}, uv {V,1,2}, faces {F,1,3}.
struct Mesh {
    dressi::CpuImage pos;
    dressi::CpuImage uv;
    dressi::CpuImage faces;
    uint32_t numVertices() const { return pos.width; }
    uint32_t numFaces() const { return faces.width; }
};

// Loads an OBJ (triangulated) and normalizes positions so the mesh is
// centered at the origin with maximum extent 1.
Mesh LoadObjMesh(const std::string& path);

// Loads the first primitive of the first mesh from a glTF 2.0 file
// (.gltf/.glb via tinygltf): POSITION + indices (+ TEXCOORD_0 when
// present), normalized like LoadObjMesh.
Mesh LoadGltfMesh(const std::string& path);

// First mesh-bearing node's first primitive plus its pbrMetallicRoughness
// material, for the PBR examples. Textures are decoded to float [0,1]
// (width == 0 when a material slot is absent); base_color/emissive stay
// sRGB-encoded (convert with SrgbToLinear before shading in linear space).
// Unlike LoadGltfMesh, uv keeps the raw glTF V (top-left origin — the 1-v
// flip there is an OBJ convention that would mirror glTF textures).
struct GltfScene {
    Mesh mesh;
    dressi::CpuImage normal;              // {V,1,3}, unit, node-transformed
    dressi::CpuImage tangent;             // {V,1,4}, xyz + w handedness
    dressi::CpuImage base_color;          // sRGB-encoded
    dressi::CpuImage metallic_roughness;  // linear; G=roughness, B=metallic
    dressi::CpuImage normal_map;          // linear, tangent space
    dressi::CpuImage occlusion;           // linear; R = AO
    dressi::CpuImage emissive;            // sRGB-encoded
};

// NORMAL is computed (area-weighted) and TANGENT generated (Lengyel) when
// the file lacks them; the node's local transform is pre-applied on the
// CPU. `normalize` recenters/rescales positions like LoadGltfMesh.
GltfScene LoadGltfScene(const std::string& path, bool normalize = true);

// Loads an RGB image as float [0,1]
dressi::CpuImage LoadImageRgb(const std::string& path);

// Loads an EXR (e.g. an equirectangular HDR env map) as 3-channel float
dressi::CpuImage LoadImageExr(const std::string& path);

// Gamma-decodes an sRGB-encoded [0,1] image to linear (pow 2.2)
dressi::CpuImage SrgbToLinear(const dressi::CpuImage& img);

// 2x2 box-downsample; odd extents round up (edge rows/cols clamp)
dressi::CpuImage DownsampleHalf(const dressi::CpuImage& img);

// Saves a 1- or 3-channel float image (clamped to [0,1]) as PNG
void SaveImagePng(const std::string& path, const dressi::CpuImage& img);

// Tiles same-sized 1- or 3-channel images into a 3-channel grid
// (row-major, `cols` per row; gray sources replicate)
dressi::CpuImage TileImages(const std::vector<dressi::CpuImage>& imgs,
                            uint32_t cols);

// Column-major 4x4 matrices (GLSL convention: m[col*4+row])
using Mat4 = std::array<float, 16>;
Mat4 Mul(const Mat4& a, const Mat4& b);
Mat4 LookAt(const std::array<float, 3>& eye,
            const std::array<float, 3>& center,
            const std::array<float, 3>& up);
// Vulkan-style perspective: y flipped, depth in [0,1]
Mat4 PerspectiveVk(float fovy_rad, float aspect, float z_near, float z_far);

// CPU vertex transform: clip {V,1,4} = mvp * (pos, 1)
dressi::CpuImage TransformToClip(const dressi::CpuImage& pos,
                                 const Mat4& mvp);

// Per-vertex inverse-UV inputs now live in the library (dressi/mesh_utils.h)
using dressi::ScreenAttr;
using dressi::UvAsClip;

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_ASSET_UTILS_H
