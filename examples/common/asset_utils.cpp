#include "asset_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace dressi_examples {

using dressi::CpuImage;
using dressi::ImgSize;

Mesh LoadObjMesh(const std::string& path) {
    tinyobj::ObjReader reader;
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;
    if (!reader.ParseFromFile(path, config)) {
        throw std::runtime_error("Failed to load OBJ: " + path + " " +
                                 reader.Error());
    }
    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();

    // Unweld: one vertex per unique (position index, texcoord index)
    std::map<std::pair<int, int>, uint32_t> unique;
    std::vector<float> positions;
    std::vector<float> uvs;
    std::vector<uint32_t> indices;
    for (const auto& shape : shapes) {
        for (const auto& idx : shape.mesh.indices) {
            const auto key = std::make_pair(idx.vertex_index,
                                            idx.texcoord_index);
            auto it = unique.find(key);
            if (it == unique.end()) {
                const uint32_t new_idx = uint32_t(positions.size() / 3);
                it = unique.emplace(key, new_idx).first;
                for (int c = 0; c < 3; c++) {
                    positions.push_back(
                            attrib.vertices[3 * idx.vertex_index + c]);
                }
                if (idx.texcoord_index >= 0) {
                    uvs.push_back(
                            attrib.texcoords[2 * idx.texcoord_index + 0]);
                    uvs.push_back(
                            attrib.texcoords[2 * idx.texcoord_index + 1]);
                } else {
                    uvs.push_back(0.f);
                    uvs.push_back(0.f);
                }
            }
            indices.push_back(it->second);
        }
    }
    if (indices.empty() || indices.size() % 3 != 0) {
        throw std::runtime_error("OBJ has no triangles: " + path);
    }

    // Normalize: center at origin, maximum extent 1
    float mins[3] = {1e30f, 1e30f, 1e30f};
    float maxs[3] = {-1e30f, -1e30f, -1e30f};
    for (size_t i = 0; i < positions.size(); i += 3) {
        for (int c = 0; c < 3; c++) {
            mins[c] = std::min(mins[c], positions[i + c]);
            maxs[c] = std::max(maxs[c], positions[i + c]);
        }
    }
    const float extent = std::max(
            {maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2]});
    const float scale = 1.f / extent;
    for (size_t i = 0; i < positions.size(); i += 3) {
        for (int c = 0; c < 3; c++) {
            positions[i + c] = (positions[i + c] -
                                0.5f * (mins[c] + maxs[c])) *
                               scale;
        }
    }

    Mesh mesh;
    const uint32_t n_verts = uint32_t(positions.size() / 3);
    const uint32_t n_faces = uint32_t(indices.size() / 3);
    mesh.pos = CpuImage(n_verts, 1, 3);
    mesh.pos.data = std::move(positions);
    mesh.uv = CpuImage(n_verts, 1, 2);
    mesh.uv.data = std::move(uvs);
    mesh.faces = CpuImage(n_faces, 1, 3);
    for (size_t i = 0; i < indices.size(); i++) {
        mesh.faces.data[i] = float(indices[i]);
    }
    return mesh;
}

CpuImage LoadImageRgb(const std::string& path) {
    int w = 0, h = 0, comps = 0;
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &comps, 3);
    if (!pixels) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    CpuImage img(uint32_t(w), uint32_t(h), 3);
    for (size_t i = 0; i < img.data.size(); i++) {
        img.data[i] = float(pixels[i]) / 255.f;
    }
    stbi_image_free(pixels);
    return img;
}

void SaveImagePng(const std::string& path, const CpuImage& img) {
    std::vector<unsigned char> bytes(size_t(img.width) * img.height * 3);
    for (size_t p = 0; p < size_t(img.width) * img.height; p++) {
        for (uint32_t c = 0; c < 3; c++) {
            const float v =
                    img.data[p * img.channels + (img.channels == 1 ? 0 : c)];
            bytes[p * 3 + c] = (unsigned char)(std::clamp(v, 0.f, 1.f) *
                                                       255.f +
                                               0.5f);
        }
    }
    if (!stbi_write_png(path.c_str(), int(img.width), int(img.height), 3,
                        bytes.data(), int(img.width) * 3)) {
        throw std::runtime_error("Failed to save image: " + path);
    }
}

Mat4 Mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
            float acc = 0.f;
            for (int k = 0; k < 4; k++) {
                acc += a[k * 4 + row] * b[col * 4 + k];
            }
            r[col * 4 + row] = acc;
        }
    }
    return r;
}

namespace {

std::array<float, 3> Normalize3(const std::array<float, 3>& v) {
    const float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return {v[0] / len, v[1] / len, v[2] / len};
}

std::array<float, 3> Cross3(const std::array<float, 3>& a,
                            const std::array<float, 3>& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

float Dot3(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

}  // namespace

Mat4 LookAt(const std::array<float, 3>& eye,
            const std::array<float, 3>& center,
            const std::array<float, 3>& up) {
    const auto f = Normalize3(
            {center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]});
    const auto s = Normalize3(Cross3(f, up));
    const auto u = Cross3(s, f);
    Mat4 m{};
    m[0] = s[0];
    m[4] = s[1];
    m[8] = s[2];
    m[1] = u[0];
    m[5] = u[1];
    m[9] = u[2];
    m[2] = -f[0];
    m[6] = -f[1];
    m[10] = -f[2];
    m[12] = -Dot3(s, eye);
    m[13] = -Dot3(u, eye);
    m[14] = Dot3(f, eye);
    m[15] = 1.f;
    return m;
}

Mat4 PerspectiveVk(float fovy_rad, float aspect, float z_near, float z_far) {
    const float t = 1.f / std::tan(fovy_rad * 0.5f);
    Mat4 m{};
    m[0] = t / aspect;
    m[5] = -t;  // Vulkan clip-space y points down
    m[10] = z_far / (z_near - z_far);
    m[11] = -1.f;
    m[14] = z_near * z_far / (z_near - z_far);
    return m;
}

CpuImage TransformToClip(const CpuImage& pos, const Mat4& mvp) {
    CpuImage clip(pos.width, 1, 4);
    for (uint32_t i = 0; i < pos.width; i++) {
        const float p[4] = {pos.at(i, 0, 0), pos.at(i, 0, 1),
                            pos.at(i, 0, 2), 1.f};
        for (int row = 0; row < 4; row++) {
            float acc = 0.f;
            for (int k = 0; k < 4; k++) {
                acc += mvp[k * 4 + row] * p[k];
            }
            clip.at(i, 0, uint32_t(row)) = acc;
        }
    }
    return clip;
}

CpuImage UvAsClip(const CpuImage& uv) {
    CpuImage clip(uv.width, 1, 4);
    for (uint32_t i = 0; i < uv.width; i++) {
        clip.at(i, 0, 0) = uv.at(i, 0, 0) * 2.f - 1.f;
        clip.at(i, 0, 1) = uv.at(i, 0, 1) * 2.f - 1.f;
        clip.at(i, 0, 2) = 0.5f;
        clip.at(i, 0, 3) = 1.f;
    }
    return clip;
}

CpuImage ScreenAttr(const CpuImage& clip, ImgSize screen) {
    CpuImage attr(clip.width, 1, 4);
    for (uint32_t i = 0; i < clip.width; i++) {
        const float w = clip.at(i, 0, 3);
        const float ndc_x = clip.at(i, 0, 0) / w;
        const float ndc_y = clip.at(i, 0, 1) / w;
        attr.at(i, 0, 0) = (ndc_x * 0.5f + 0.5f) * float(screen.w);
        attr.at(i, 0, 1) = (ndc_y * 0.5f + 0.5f) * float(screen.h);
        attr.at(i, 0, 2) = 1.f;
        attr.at(i, 0, 3) = 0.f;
    }
    return attr;
}

}  // namespace dressi_examples
