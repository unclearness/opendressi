#ifndef DRESSI_EXAMPLES_ASSET_UTILS_H
#define DRESSI_EXAMPLES_ASSET_UTILS_H

#include <array>
#include <string>

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

// Loads an RGB image as float [0,1]
dressi::CpuImage LoadImageRgb(const std::string& path);

// Saves a 1- or 3-channel float image (clamped to [0,1]) as PNG
void SaveImagePng(const std::string& path, const dressi::CpuImage& img);

// Tiles same-sized 3-channel images into a grid (row-major, `cols` per row)
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

// Per-vertex inverse-UV inputs:
//   UvAsClip:   {V,1,4} = (u*2-1, v*2-1, 0.5, 1)   (rasterize in UV space)
//   ScreenAttr: {V,1,4} = (screen_x, screen_y, 1, 0) from clip positions
dressi::CpuImage UvAsClip(const dressi::CpuImage& uv);
dressi::CpuImage ScreenAttr(const dressi::CpuImage& clip,
                            dressi::ImgSize screen);

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_ASSET_UTILS_H
