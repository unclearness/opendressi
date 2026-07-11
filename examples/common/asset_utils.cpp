#include "asset_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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

// tinygltf reuses the stb implementations compiled above
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#include <tiny_gltf.h>

// tinyexr too (stb_image's zlib inflate + stb_image_write's deflate)
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

namespace dressi_examples {

using dressi::CpuImage;
using dressi::ImgSize;

namespace {

// Normalizes positions to center origin / max extent 1 (shared with the
// OBJ loader's convention)
void NormalizeMesh(Mesh& mesh) {
    float mins[3] = {1e30f, 1e30f, 1e30f};
    float maxs[3] = {-1e30f, -1e30f, -1e30f};
    for (uint32_t v = 0; v < mesh.numVertices(); v++) {
        for (uint32_t c = 0; c < 3; c++) {
            mins[c] = std::min(mins[c], mesh.pos.at(v, 0, c));
            maxs[c] = std::max(maxs[c], mesh.pos.at(v, 0, c));
        }
    }
    float extent = 1e-12f;
    for (uint32_t c = 0; c < 3; c++) {
        extent = std::max(extent, maxs[c] - mins[c]);
    }
    for (uint32_t v = 0; v < mesh.numVertices(); v++) {
        for (uint32_t c = 0; c < 3; c++) {
            mesh.pos.at(v, 0, c) =
                    (mesh.pos.at(v, 0, c) - 0.5f * (mins[c] + maxs[c])) /
                    extent;
        }
    }
}

// Reads accessor element `i`, component `c` as float (handles bufferView
// byte stride; float and normalized/unnormalized integer components)
float ReadAccessorFloat(const tinygltf::Model& model,
                        const tinygltf::Accessor& acc, size_t i,
                        uint32_t c) {
    const auto& bv = model.bufferViews[size_t(acc.bufferView)];
    const auto& buf = model.buffers[size_t(bv.buffer)];
    const int n_comp = tinygltf::GetNumComponentsInType(uint32_t(acc.type));
    const int comp_size =
            tinygltf::GetComponentSizeInBytes(uint32_t(acc.componentType));
    const size_t stride = bv.byteStride
                                  ? bv.byteStride
                                  : size_t(n_comp) * size_t(comp_size);
    const uint8_t* p = buf.data.data() + bv.byteOffset + acc.byteOffset +
                       i * stride + size_t(c) * size_t(comp_size);
    switch (acc.componentType) {
        case TINYGLTF_COMPONENT_TYPE_FLOAT: {
            float f;
            std::memcpy(&f, p, 4);
            return f;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
            uint32_t u;
            std::memcpy(&u, p, 4);
            return float(u);
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
            uint16_t u;
            std::memcpy(&u, p, 2);
            return float(u);
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            return float(*p);
        default:
            throw std::runtime_error("glTF: unsupported component type");
    }
}

}  // namespace

Mesh LoadGltfMesh(const std::string& path) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    const bool binary = path.size() > 4 &&
                        path.compare(path.size() - 4, 4, ".glb") == 0;
    const bool ok = binary
                            ? loader.LoadBinaryFromFile(&model, &err, &warn,
                                                        path)
                            : loader.LoadASCIIFromFile(&model, &err, &warn,
                                                       path);
    if (!ok) {
        throw std::runtime_error("Failed to load glTF: " + path + " " +
                                 err);
    }
    if (model.meshes.empty() || model.meshes[0].primitives.empty()) {
        throw std::runtime_error("glTF has no mesh primitives: " + path);
    }
    const auto& prim = model.meshes[0].primitives[0];
    const auto pos_it = prim.attributes.find("POSITION");
    if (pos_it == prim.attributes.end() || prim.indices < 0) {
        throw std::runtime_error("glTF primitive needs POSITION + indices");
    }
    const auto& pos_acc = model.accessors[size_t(pos_it->second)];
    const auto uv_it = prim.attributes.find("TEXCOORD_0");
    const auto& idx_acc = model.accessors[size_t(prim.indices)];
    const uint32_t n_verts = uint32_t(pos_acc.count);
    const uint32_t n_faces = uint32_t(idx_acc.count / 3);

    Mesh mesh;
    mesh.pos = CpuImage(n_verts, 1, 3);
    mesh.uv = CpuImage(n_verts, 1, 2, 0.f);
    mesh.faces = CpuImage(n_faces, 1, 3);
    for (uint32_t v = 0; v < n_verts; v++) {
        for (uint32_t c = 0; c < 3; c++) {
            mesh.pos.at(v, 0, c) = ReadAccessorFloat(model, pos_acc, v, c);
        }
    }
    if (uv_it != prim.attributes.end()) {
        const auto& uv_acc = model.accessors[size_t(uv_it->second)];
        for (uint32_t v = 0; v < n_verts; v++) {
            mesh.uv.at(v, 0, 0) = ReadAccessorFloat(model, uv_acc, v, 0);
            mesh.uv.at(v, 0, 1) =
                    1.f - ReadAccessorFloat(model, uv_acc, v, 1);
        }
    }
    for (uint32_t f = 0; f < n_faces; f++) {
        for (uint32_t k = 0; k < 3; k++) {
            mesh.faces.at(f, 0, k) =
                    ReadAccessorFloat(model, idx_acc, size_t(f) * 3 + k, 0);
        }
    }
    NormalizeMesh(mesh);
    return mesh;
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

// Column-major node-local matrix (node.matrix or T*R*S)
Mat4 NodeLocalMatrix(const tinygltf::Node& node) {
    if (node.matrix.size() == 16) {
        Mat4 m;
        for (int i = 0; i < 16; i++) {
            m[size_t(i)] = float(node.matrix[size_t(i)]);
        }
        return m;
    }
    Mat4 m{};
    m[0] = m[5] = m[10] = m[15] = 1.f;
    if (node.rotation.size() == 4) {  // quaternion (x, y, z, w)
        const float x = float(node.rotation[0]);
        const float y = float(node.rotation[1]);
        const float z = float(node.rotation[2]);
        const float w = float(node.rotation[3]);
        m[0] = 1.f - 2.f * (y * y + z * z);
        m[1] = 2.f * (x * y + z * w);
        m[2] = 2.f * (x * z - y * w);
        m[4] = 2.f * (x * y - z * w);
        m[5] = 1.f - 2.f * (x * x + z * z);
        m[6] = 2.f * (y * z + x * w);
        m[8] = 2.f * (x * z + y * w);
        m[9] = 2.f * (y * z - x * w);
        m[10] = 1.f - 2.f * (x * x + y * y);
    }
    if (node.scale.size() == 3) {
        for (int col = 0; col < 3; col++) {
            for (int row = 0; row < 3; row++) {
                m[size_t(col * 4 + row)] *= float(node.scale[size_t(col)]);
            }
        }
    }
    if (node.translation.size() == 3) {
        m[12] = float(node.translation[0]);
        m[13] = float(node.translation[1]);
        m[14] = float(node.translation[2]);
    }
    return m;
}

// Inverse-transpose of the upper-left 3x3 (normal matrix), column-major
std::array<float, 9> NormalMatrix(const Mat4& m) {
    const float a = m[0], b = m[4], c = m[8];
    const float d = m[1], e = m[5], f = m[9];
    const float g = m[2], h = m[6], i = m[10];
    const float det = a * (e * i - f * h) - b * (d * i - f * g) +
                      c * (d * h - e * g);
    const float inv_det = 1.f / (det == 0.f ? 1e-12f : det);
    // inverse (cofactor/det), already laid out transposed below
    return {(e * i - f * h) * inv_det, (c * h - b * i) * inv_det,
            (b * f - c * e) * inv_det, (f * g - d * i) * inv_det,
            (a * i - c * g) * inv_det, (c * d - a * f) * inv_det,
            (d * h - e * g) * inv_det, (b * g - a * h) * inv_det,
            (a * e - b * d) * inv_det};
}

// Decodes a glTF texture slot to a 3-channel float [0,1] image (empty
// image when the slot / source is absent)
CpuImage DecodeGltfTexture(const tinygltf::Model& model, int tex_idx) {
    if (tex_idx < 0) {
        return CpuImage();
    }
    const auto& tex = model.textures[size_t(tex_idx)];
    if (tex.source < 0) {
        return CpuImage();
    }
    const auto& im = model.images[size_t(tex.source)];
    if (im.image.empty()) {
        return CpuImage();
    }
    if (im.bits != 8) {
        throw std::runtime_error("glTF: only 8-bit textures supported");
    }
    const uint32_t comp = uint32_t(im.component);
    CpuImage out(uint32_t(im.width), uint32_t(im.height), 3);
    for (size_t p = 0; p < size_t(im.width) * size_t(im.height); p++) {
        for (uint32_t c = 0; c < 3; c++) {
            const uint32_t src_c = std::min(c, comp - 1);
            out.data[p * 3 + c] =
                    float(im.image[p * comp + src_c]) / 255.f;
        }
    }
    return out;
}

}  // namespace

GltfScene LoadGltfScene(const std::string& path, bool normalize) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    const bool binary = path.size() > 4 &&
                        path.compare(path.size() - 4, 4, ".glb") == 0;
    const bool ok = binary
                            ? loader.LoadBinaryFromFile(&model, &err, &warn,
                                                        path)
                            : loader.LoadASCIIFromFile(&model, &err, &warn,
                                                       path);
    if (!ok) {
        throw std::runtime_error("Failed to load glTF: " + path + " " +
                                 err);
    }

    // First mesh-bearing node (hierarchy transforms are not composed; the
    // sample models used here keep the mesh on a root node)
    int mesh_idx = -1;
    Mat4 xform{};
    xform[0] = xform[5] = xform[10] = xform[15] = 1.f;
    for (const auto& node : model.nodes) {
        if (node.mesh >= 0) {
            mesh_idx = node.mesh;
            xform = NodeLocalMatrix(node);
            break;
        }
    }
    if (mesh_idx < 0 && !model.meshes.empty()) {
        mesh_idx = 0;
    }
    if (mesh_idx < 0 || model.meshes[size_t(mesh_idx)].primitives.empty()) {
        throw std::runtime_error("glTF has no mesh primitives: " + path);
    }
    const auto& prim = model.meshes[size_t(mesh_idx)].primitives[0];
    const auto pos_it = prim.attributes.find("POSITION");
    if (pos_it == prim.attributes.end() || prim.indices < 0) {
        throw std::runtime_error("glTF primitive needs POSITION + indices");
    }
    const auto& pos_acc = model.accessors[size_t(pos_it->second)];
    const auto& idx_acc = model.accessors[size_t(prim.indices)];
    const uint32_t n_verts = uint32_t(pos_acc.count);
    const uint32_t n_faces = uint32_t(idx_acc.count / 3);

    GltfScene scene;
    scene.mesh.pos = CpuImage(n_verts, 1, 3);
    scene.mesh.uv = CpuImage(n_verts, 1, 2, 0.f);
    scene.mesh.faces = CpuImage(n_faces, 1, 3);
    scene.normal = CpuImage(n_verts, 1, 3, 0.f);
    scene.tangent = CpuImage(n_verts, 1, 4, 0.f);

    for (uint32_t v = 0; v < n_verts; v++) {
        for (uint32_t c = 0; c < 3; c++) {
            scene.mesh.pos.at(v, 0, c) =
                    ReadAccessorFloat(model, pos_acc, v, c);
        }
    }
    const auto uv_it = prim.attributes.find("TEXCOORD_0");
    if (uv_it != prim.attributes.end()) {
        const auto& uv_acc = model.accessors[size_t(uv_it->second)];
        for (uint32_t v = 0; v < n_verts; v++) {
            // glTF UV origin is top-left, matching image row 0 — no flip
            scene.mesh.uv.at(v, 0, 0) =
                    ReadAccessorFloat(model, uv_acc, v, 0);
            scene.mesh.uv.at(v, 0, 1) =
                    ReadAccessorFloat(model, uv_acc, v, 1);
        }
    }
    for (uint32_t f = 0; f < n_faces; f++) {
        for (uint32_t k = 0; k < 3; k++) {
            scene.mesh.faces.at(f, 0, k) = ReadAccessorFloat(
                    model, idx_acc, size_t(f) * 3 + k, 0);
        }
    }

    const auto n_it = prim.attributes.find("NORMAL");
    if (n_it != prim.attributes.end()) {
        const auto& n_acc = model.accessors[size_t(n_it->second)];
        for (uint32_t v = 0; v < n_verts; v++) {
            for (uint32_t c = 0; c < 3; c++) {
                scene.normal.at(v, 0, c) =
                        ReadAccessorFloat(model, n_acc, v, c);
            }
        }
    } else {
        // Area-weighted face-normal accumulation
        for (uint32_t f = 0; f < n_faces; f++) {
            uint32_t vi[3];
            std::array<float, 3> p[3];
            for (uint32_t k = 0; k < 3; k++) {
                vi[k] = uint32_t(scene.mesh.faces.at(f, 0, k));
                for (uint32_t c = 0; c < 3; c++) {
                    p[k][c] = scene.mesh.pos.at(vi[k], 0, c);
                }
            }
            const std::array<float, 3> e1 = {p[1][0] - p[0][0],
                                             p[1][1] - p[0][1],
                                             p[1][2] - p[0][2]};
            const std::array<float, 3> e2 = {p[2][0] - p[0][0],
                                             p[2][1] - p[0][1],
                                             p[2][2] - p[0][2]};
            const auto fn = Cross3(e1, e2);
            for (uint32_t k = 0; k < 3; k++) {
                for (uint32_t c = 0; c < 3; c++) {
                    scene.normal.at(vi[k], 0, c) += fn[c];
                }
            }
        }
    }

    const auto t_it = prim.attributes.find("TANGENT");
    if (t_it != prim.attributes.end()) {
        const auto& t_acc = model.accessors[size_t(t_it->second)];
        for (uint32_t v = 0; v < n_verts; v++) {
            for (uint32_t c = 0; c < 4; c++) {
                scene.tangent.at(v, 0, c) =
                        ReadAccessorFloat(model, t_acc, v, c);
            }
        }
    } else {
        // Lengyel: per-face tangent/bitangent accumulation from UVs,
        // then Gram-Schmidt against the vertex normal + handedness w
        std::vector<float> tan(size_t(n_verts) * 3, 0.f);
        std::vector<float> bitan(size_t(n_verts) * 3, 0.f);
        for (uint32_t f = 0; f < n_faces; f++) {
            uint32_t vi[3];
            std::array<float, 3> p[3];
            std::array<float, 2> t[3];
            for (uint32_t k = 0; k < 3; k++) {
                vi[k] = uint32_t(scene.mesh.faces.at(f, 0, k));
                for (uint32_t c = 0; c < 3; c++) {
                    p[k][c] = scene.mesh.pos.at(vi[k], 0, c);
                }
                t[k] = {scene.mesh.uv.at(vi[k], 0, 0),
                        scene.mesh.uv.at(vi[k], 0, 1)};
            }
            const float du1 = t[1][0] - t[0][0], dv1 = t[1][1] - t[0][1];
            const float du2 = t[2][0] - t[0][0], dv2 = t[2][1] - t[0][1];
            const float det = du1 * dv2 - du2 * dv1;
            if (std::abs(det) < 1e-12f) {
                continue;
            }
            const float r = 1.f / det;
            for (uint32_t c = 0; c < 3; c++) {
                const float e1 = p[1][c] - p[0][c];
                const float e2 = p[2][c] - p[0][c];
                const float tc = (e1 * dv2 - e2 * dv1) * r;
                const float bc = (e2 * du1 - e1 * du2) * r;
                for (uint32_t k = 0; k < 3; k++) {
                    tan[size_t(vi[k]) * 3 + c] += tc;
                    bitan[size_t(vi[k]) * 3 + c] += bc;
                }
            }
        }
        for (uint32_t v = 0; v < n_verts; v++) {
            const std::array<float, 3> n = {scene.normal.at(v, 0, 0),
                                            scene.normal.at(v, 0, 1),
                                            scene.normal.at(v, 0, 2)};
            std::array<float, 3> tv = {tan[size_t(v) * 3 + 0],
                                       tan[size_t(v) * 3 + 1],
                                       tan[size_t(v) * 3 + 2]};
            const float ndt = Dot3(n, tv);
            for (uint32_t c = 0; c < 3; c++) {
                tv[c] -= n[c] * ndt;
            }
            const float len = std::sqrt(Dot3(tv, tv));
            if (len < 1e-12f) {
                // Degenerate UVs: any unit vector perpendicular to n
                tv = std::abs(n[0]) < 0.9f
                             ? Normalize3(Cross3(n, {1.f, 0.f, 0.f}))
                             : Normalize3(Cross3(n, {0.f, 1.f, 0.f}));
            } else {
                for (uint32_t c = 0; c < 3; c++) {
                    tv[c] /= len;
                }
            }
            const std::array<float, 3> bv = {bitan[size_t(v) * 3 + 0],
                                             bitan[size_t(v) * 3 + 1],
                                             bitan[size_t(v) * 3 + 2]};
            for (uint32_t c = 0; c < 3; c++) {
                scene.tangent.at(v, 0, c) = tv[c];
            }
            scene.tangent.at(v, 0, 3) =
                    Dot3(Cross3(n, tv), bv) < 0.f ? -1.f : 1.f;
        }
    }

    // Pre-apply the node transform (positions by M; normals by the
    // inverse-transpose; tangent.xyz by M like a direction)
    const auto nrm_m = NormalMatrix(xform);
    for (uint32_t v = 0; v < n_verts; v++) {
        const float p[3] = {scene.mesh.pos.at(v, 0, 0),
                            scene.mesh.pos.at(v, 0, 1),
                            scene.mesh.pos.at(v, 0, 2)};
        const float n[3] = {scene.normal.at(v, 0, 0),
                            scene.normal.at(v, 0, 1),
                            scene.normal.at(v, 0, 2)};
        const float t[3] = {scene.tangent.at(v, 0, 0),
                            scene.tangent.at(v, 0, 1),
                            scene.tangent.at(v, 0, 2)};
        std::array<float, 3> np{}, nn{}, nt{};
        for (uint32_t r = 0; r < 3; r++) {
            np[r] = xform[0 * 4 + r] * p[0] + xform[1 * 4 + r] * p[1] +
                    xform[2 * 4 + r] * p[2] + xform[3 * 4 + r];
            nn[r] = nrm_m[0 * 3 + r] * n[0] + nrm_m[1 * 3 + r] * n[1] +
                    nrm_m[2 * 3 + r] * n[2];
            nt[r] = xform[0 * 4 + r] * t[0] + xform[1 * 4 + r] * t[1] +
                    xform[2 * 4 + r] * t[2];
        }
        nn = Normalize3(nn);
        nt = Normalize3(nt);
        for (uint32_t c = 0; c < 3; c++) {
            scene.mesh.pos.at(v, 0, c) = np[c];
            scene.normal.at(v, 0, c) = nn[c];
            scene.tangent.at(v, 0, c) = nt[c];
        }
    }
    if (normalize) {
        NormalizeMesh(scene.mesh);
    }

    if (prim.material >= 0) {
        const auto& mat = model.materials[size_t(prim.material)];
        scene.base_color = DecodeGltfTexture(
                model, mat.pbrMetallicRoughness.baseColorTexture.index);
        scene.metallic_roughness = DecodeGltfTexture(
                model,
                mat.pbrMetallicRoughness.metallicRoughnessTexture.index);
        scene.normal_map =
                DecodeGltfTexture(model, mat.normalTexture.index);
        scene.occlusion =
                DecodeGltfTexture(model, mat.occlusionTexture.index);
        scene.emissive =
                DecodeGltfTexture(model, mat.emissiveTexture.index);
    }
    return scene;
}

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
                    // OBJ vt origin is bottom-left; image row 0 is the top
                    uvs.push_back(
                            1.f - attrib.texcoords[2 * idx.texcoord_index + 1]);
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

CpuImage LoadImageExr(const std::string& path) {
    float* rgba = nullptr;
    int w = 0, h = 0;
    const char* err = nullptr;
    if (LoadEXR(&rgba, &w, &h, path.c_str(), &err) != TINYEXR_SUCCESS) {
        std::string msg = err ? err : "unknown error";
        FreeEXRErrorMessage(err);
        throw std::runtime_error("Failed to load EXR: " + path + " " + msg);
    }
    CpuImage img(uint32_t(w), uint32_t(h), 3);
    for (size_t p = 0; p < size_t(w) * size_t(h); p++) {
        for (uint32_t c = 0; c < 3; c++) {
            img.data[p * 3 + c] = rgba[p * 4 + c];
        }
    }
    free(rgba);
    return img;
}

CpuImage SrgbToLinear(const CpuImage& img) {
    CpuImage out = img;
    for (float& v : out.data) {
        v = std::pow(std::max(v, 0.f), 2.2f);
    }
    return out;
}

CpuImage DownsampleHalf(const CpuImage& img) {
    const uint32_t w = std::max(1u, (img.width + 1) / 2);
    const uint32_t h = std::max(1u, (img.height + 1) / 2);
    CpuImage out(w, h, img.channels);
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            for (uint32_t c = 0; c < img.channels; c++) {
                float acc = 0.f;
                for (uint32_t dy = 0; dy < 2; dy++) {
                    for (uint32_t dx = 0; dx < 2; dx++) {
                        const uint32_t sx =
                                std::min(2 * x + dx, img.width - 1);
                        const uint32_t sy =
                                std::min(2 * y + dy, img.height - 1);
                        acc += img.at(sx, sy, c);
                    }
                }
                out.at(x, y, c) = acc * 0.25f;
            }
        }
    }
    return out;
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

CpuImage TileImages(const std::vector<CpuImage>& imgs, uint32_t cols) {
    if (imgs.empty()) {
        return CpuImage();
    }
    const uint32_t w = imgs[0].width;
    const uint32_t h = imgs[0].height;
    const uint32_t rows = (uint32_t(imgs.size()) + cols - 1) / cols;
    CpuImage tile(w * cols, h * rows, 3);
    for (size_t i = 0; i < imgs.size(); i++) {
        const uint32_t ox = uint32_t(i % cols) * w;
        const uint32_t oy = uint32_t(i / cols) * h;
        // 1-channel sources replicate to gray (indexing c > 0 would read
        // the neighboring pixels' values)
        const bool gray = imgs[i].channels == 1;
        for (uint32_t y = 0; y < h; y++) {
            for (uint32_t x = 0; x < w; x++) {
                for (uint32_t c = 0; c < 3; c++) {
                    tile.at(ox + x, oy + y, c) =
                            imgs[i].at(x, y, gray ? 0 : c);
                }
            }
        }
    }
    return tile;
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

}  // namespace dressi_examples
