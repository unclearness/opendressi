#include "geometry_utils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <set>

namespace dressi_examples {

namespace {

struct V3 {
    float x, y, z;
};

V3 Normalized(const V3& v) {
    const float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return {v.x / len, v.y / len, v.z / len};
}

}  // namespace

Mesh GenerateIcosphere(uint32_t level) {
    // Icosahedron from three orthogonal golden rectangles
    const float t = (1.f + std::sqrt(5.f)) / 2.f;
    std::vector<V3> verts = {
            {-1, t, 0}, {1, t, 0},   {-1, -t, 0}, {1, -t, 0},
            {0, -1, t}, {0, 1, t},   {0, -1, -t}, {0, 1, -t},
            {t, 0, -1}, {t, 0, 1},   {-t, 0, -1}, {-t, 0, 1}};
    for (auto& v : verts) {
        v = Normalized(v);
    }
    std::vector<std::array<uint32_t, 3>> faces = {
            {0, 11, 5}, {0, 5, 1},   {0, 1, 7},   {0, 7, 10}, {0, 10, 11},
            {1, 5, 9},  {5, 11, 4},  {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4},  {3, 4, 2},   {3, 2, 6},   {3, 6, 8},  {3, 8, 9},
            {4, 9, 5},  {2, 4, 11},  {6, 2, 10},  {8, 6, 7},  {9, 8, 1}};

    for (uint32_t l = 0; l < level; l++) {
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> midpoint;
        const auto mid = [&](uint32_t a, uint32_t b) -> uint32_t {
            const auto key = std::minmax(a, b);
            if (auto it = midpoint.find(key); it != midpoint.end()) {
                return it->second;
            }
            const V3 m = Normalized({(verts[a].x + verts[b].x) * 0.5f,
                                     (verts[a].y + verts[b].y) * 0.5f,
                                     (verts[a].z + verts[b].z) * 0.5f});
            verts.push_back(m);
            const uint32_t idx = uint32_t(verts.size() - 1);
            midpoint[key] = idx;
            return idx;
        };
        std::vector<std::array<uint32_t, 3>> next;
        next.reserve(faces.size() * 4);
        for (const auto& f : faces) {
            const uint32_t a = mid(f[0], f[1]);
            const uint32_t b = mid(f[1], f[2]);
            const uint32_t c = mid(f[2], f[0]);
            next.push_back({f[0], a, c});
            next.push_back({f[1], b, a});
            next.push_back({f[2], c, b});
            next.push_back({a, b, c});
        }
        faces = std::move(next);
    }

    Mesh mesh;
    mesh.pos = dressi::CpuImage(uint32_t(verts.size()), 1, 3);
    mesh.uv = dressi::CpuImage(uint32_t(verts.size()), 1, 2, 0.f);
    mesh.faces = dressi::CpuImage(uint32_t(faces.size()), 1, 3);
    for (uint32_t i = 0; i < verts.size(); i++) {
        mesh.pos.at(i, 0, 0) = verts[i].x;
        mesh.pos.at(i, 0, 1) = verts[i].y;
        mesh.pos.at(i, 0, 2) = verts[i].z;
    }
    for (uint32_t i = 0; i < faces.size(); i++) {
        for (uint32_t c = 0; c < 3; c++) {
            mesh.faces.at(i, 0, c) = float(faces[i][c]);
        }
    }
    return mesh;
}

void SaveObjMesh(const std::string& path, const Mesh& mesh) {
    std::ofstream ofs(path);
    if (!ofs) {
        return;
    }
    for (uint32_t i = 0; i < mesh.numVertices(); i++) {
        ofs << "v " << mesh.pos.at(i, 0, 0) << ' ' << mesh.pos.at(i, 0, 1)
            << ' ' << mesh.pos.at(i, 0, 2) << '\n';
    }
    for (uint32_t i = 0; i < mesh.numFaces(); i++) {
        ofs << "f " << int(mesh.faces.at(i, 0, 0)) + 1 << ' '
            << int(mesh.faces.at(i, 0, 1)) + 1 << ' '
            << int(mesh.faces.at(i, 0, 2)) + 1 << '\n';
    }
}

std::vector<std::vector<uint32_t>> BuildVertexAdjacency(
        const dressi::CpuImage& faces, uint32_t n_verts) {
    std::vector<std::set<uint32_t>> adj_sets(n_verts);
    for (uint32_t f = 0; f < faces.width; f++) {
        uint32_t idx[3];
        for (int k = 0; k < 3; k++) {
            idx[k] = uint32_t(faces.at(f, 0, uint32_t(k)));
        }
        for (int k = 0; k < 3; k++) {
            adj_sets[idx[k]].insert(idx[(k + 1) % 3]);
            adj_sets[idx[k]].insert(idx[(k + 2) % 3]);
        }
    }
    std::vector<std::vector<uint32_t>> adj(n_verts);
    for (uint32_t v = 0; v < n_verts; v++) {
        adj[v].assign(adj_sets[v].begin(), adj_sets[v].end());
    }
    return adj;
}

dressi::CpuImage UniformLaplacianGrad(
        const dressi::CpuImage& pos,
        const std::vector<std::vector<uint32_t>>& adj, float lambda) {
    dressi::CpuImage grad(pos.width, 1, 3, 0.f);
    for (uint32_t v = 0; v < pos.width; v++) {
        if (adj[v].empty()) {
            continue;
        }
        float mean[3] = {0.f, 0.f, 0.f};
        for (uint32_t n : adj[v]) {
            for (uint32_t c = 0; c < 3; c++) {
                mean[c] += pos.at(n, 0, c);
            }
        }
        const float inv_n = 1.f / float(adj[v].size());
        for (uint32_t c = 0; c < 3; c++) {
            grad.at(v, 0, c) = lambda * (pos.at(v, 0, c) - mean[c] * inv_n);
        }
    }
    return grad;
}

namespace {

V3 FaceCross(const dressi::CpuImage& pos, const dressi::CpuImage& faces,
             uint32_t f, uint32_t idx[3]) {
    for (int k = 0; k < 3; k++) {
        idx[k] = uint32_t(faces.at(f, 0, uint32_t(k)));
    }
    const auto p = [&](uint32_t v, uint32_t c) { return pos.at(v, 0, c); };
    const V3 e1 = {p(idx[1], 0) - p(idx[0], 0), p(idx[1], 1) - p(idx[0], 1),
                   p(idx[1], 2) - p(idx[0], 2)};
    const V3 e2 = {p(idx[2], 0) - p(idx[0], 0), p(idx[2], 1) - p(idx[0], 1),
                   p(idx[2], 2) - p(idx[0], 2)};
    return {e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z,
            e1.x * e2.y - e1.y * e2.x};
}

}  // namespace

std::vector<std::array<uint32_t, 2>> BuildFaceAdjacency(
        const dressi::CpuImage& faces) {
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> edge_faces;
    for (uint32_t f = 0; f < faces.width; f++) {
        for (int k = 0; k < 3; k++) {
            const uint32_t a = uint32_t(faces.at(f, 0, uint32_t(k)));
            const uint32_t b =
                    uint32_t(faces.at(f, 0, uint32_t((k + 1) % 3)));
            edge_faces[std::minmax(a, b)].push_back(f);
        }
    }
    std::vector<std::array<uint32_t, 2>> pairs;
    for (const auto& [edge, fs] : edge_faces) {
        (void)edge;
        for (size_t i = 0; i < fs.size(); i++) {
            for (size_t j = i + 1; j < fs.size(); j++) {
                pairs.push_back({fs[i], fs[j]});
            }
        }
    }
    return pairs;
}

dressi::CpuImage NormalConsistencyGrad(
        const dressi::CpuImage& pos, const dressi::CpuImage& faces,
        const std::vector<std::array<uint32_t, 2>>& face_adj, float lambda) {
    dressi::CpuImage grad(pos.width, 1, 3, 0.f);
    const auto add_face_grad = [&](uint32_t f, const uint32_t idx[3],
                                   const V3& g) {
        // c = e1 x e2 with e1 = p1 - p0, e2 = p2 - p0:
        // dE via g . dc  =>  grad_p1 += e2 x g, grad_p2 += g x e1,
        // grad_p0 -= both (translation invariance)
        (void)f;
        const auto p = [&](uint32_t v, uint32_t c) {
            return pos.at(v, 0, c);
        };
        const V3 e1 = {p(idx[1], 0) - p(idx[0], 0),
                       p(idx[1], 1) - p(idx[0], 1),
                       p(idx[1], 2) - p(idx[0], 2)};
        const V3 e2 = {p(idx[2], 0) - p(idx[0], 0),
                       p(idx[2], 1) - p(idx[0], 1),
                       p(idx[2], 2) - p(idx[0], 2)};
        const V3 g1 = {e2.y * g.z - e2.z * g.y, e2.z * g.x - e2.x * g.z,
                       e2.x * g.y - e2.y * g.x};  // e2 x g
        const V3 g2 = {g.y * e1.z - g.z * e1.y, g.z * e1.x - g.x * e1.z,
                       g.x * e1.y - g.y * e1.x};  // g x e1
        grad.at(idx[1], 0, 0) += g1.x;
        grad.at(idx[1], 0, 1) += g1.y;
        grad.at(idx[1], 0, 2) += g1.z;
        grad.at(idx[2], 0, 0) += g2.x;
        grad.at(idx[2], 0, 1) += g2.y;
        grad.at(idx[2], 0, 2) += g2.z;
        grad.at(idx[0], 0, 0) -= g1.x + g2.x;
        grad.at(idx[0], 0, 1) -= g1.y + g2.y;
        grad.at(idx[0], 0, 2) -= g1.z + g2.z;
    };

    for (const auto& pair : face_adj) {
        uint32_t ia[3], ib[3];
        const V3 ca = FaceCross(pos, faces, pair[0], ia);
        const V3 cb = FaceCross(pos, faces, pair[1], ib);
        const float la =
                std::sqrt(ca.x * ca.x + ca.y * ca.y + ca.z * ca.z);
        const float lb =
                std::sqrt(cb.x * cb.x + cb.y * cb.y + cb.z * cb.z);
        if (la < 1e-12f || lb < 1e-12f) {
            continue;  // degenerate face: no defined normal
        }
        const V3 na = {ca.x / la, ca.y / la, ca.z / la};
        const V3 nb = {cb.x / lb, cb.y / lb, cb.z / lb};
        const float d = na.x * nb.x + na.y * nb.y + na.z * nb.z;
        // E = lambda (1 - n_a . n_b);
        // dE/dc_a = -lambda (n_b - d n_a) / |c_a|   (and symmetrically)
        const V3 ga = {-lambda * (nb.x - d * na.x) / la,
                       -lambda * (nb.y - d * na.y) / la,
                       -lambda * (nb.z - d * na.z) / la};
        const V3 gb = {-lambda * (na.x - d * nb.x) / lb,
                       -lambda * (na.y - d * nb.y) / lb,
                       -lambda * (na.z - d * nb.z) / lb};
        add_face_grad(pair[0], ia, ga);
        add_face_grad(pair[1], ib, gb);
    }
    return grad;
}

uint32_t CountFlippedFacePairs(
        const dressi::CpuImage& pos, const dressi::CpuImage& faces,
        const std::vector<std::array<uint32_t, 2>>& face_adj) {
    uint32_t flipped = 0;
    for (const auto& pair : face_adj) {
        uint32_t ia[3], ib[3];
        const V3 ca = FaceCross(pos, faces, pair[0], ia);
        const V3 cb = FaceCross(pos, faces, pair[1], ib);
        if (ca.x * cb.x + ca.y * cb.y + ca.z * cb.z < 0.f) {
            flipped++;
        }
    }
    return flipped;
}

SoftGeometry BuildSoftGeometry(const dressi::CpuImage& hard_clip,
                               const dressi::CpuImage& faces,
                               dressi::ImgSize screen, float radius_px) {
    const uint32_t n_faces = faces.width;
    SoftGeometry soft;
    soft.clip = dressi::CpuImage(n_faces * 3, 1, 4);
    soft.face_id = dressi::CpuImage(n_faces * 3, 1, 1);
    soft.faces = dressi::CpuImage(n_faces, 1, 3);

    constexpr float kWEps = 1e-6f;
    constexpr float kMaxScale = 8.f;

    for (uint32_t f = 0; f < n_faces; f++) {
        uint32_t idx[3];
        float clip[3][4];
        float sx[3], sy[3];
        bool valid = true;
        for (int k = 0; k < 3; k++) {
            idx[k] = uint32_t(faces.at(f, 0, uint32_t(k)));
            for (uint32_t c = 0; c < 4; c++) {
                clip[k][c] = hard_clip.at(idx[k], 0, c);
            }
            if (clip[k][3] <= kWEps) {
                valid = false;
            } else {
                sx[k] = (clip[k][0] / clip[k][3] * 0.5f + 0.5f) *
                        float(screen.w);
                sy[k] = (clip[k][1] / clip[k][3] * 0.5f + 0.5f) *
                        float(screen.h);
            }
        }

        float scale = 1.f;
        if (valid) {
            // Min distance from the centroid to an edge; scaling about the
            // centroid by 1 + r/d_min dilates every edge outward by >= r
            const float cx = (sx[0] + sx[1] + sx[2]) / 3.f;
            const float cy = (sy[0] + sy[1] + sy[2]) / 3.f;
            float d_min = 1e30f;
            for (int k = 0; k < 3; k++) {
                const int k2 = (k + 1) % 3;
                const float ex = sx[k2] - sx[k];
                const float ey = sy[k2] - sy[k];
                const float len = std::sqrt(ex * ex + ey * ey);
                if (len < 1e-6f) {
                    continue;
                }
                const float d = std::abs(ex * (cy - sy[k]) -
                                         ey * (cx - sx[k])) /
                                len;
                d_min = std::min(d_min, d);
            }
            if (d_min < 1e29f) {
                scale = std::min(1.f + radius_px / std::max(d_min, 1e-3f),
                                 kMaxScale);
            }
            for (int k = 0; k < 3; k++) {
                const float ex = cx + (sx[k] - cx) * scale;
                const float ey = cy + (sy[k] - cy) * scale;
                // Unproject keeping z and w fixed
                clip[k][0] = (ex / float(screen.w) * 2.f - 1.f) * clip[k][3];
                clip[k][1] = (ey / float(screen.h) * 2.f - 1.f) * clip[k][3];
            }
        }

        for (int k = 0; k < 3; k++) {
            const uint32_t v = f * 3 + uint32_t(k);
            for (uint32_t c = 0; c < 4; c++) {
                soft.clip.at(v, 0, c) = clip[k][c];
            }
            soft.face_id.at(v, 0, 0) = float(f);
            soft.faces.at(f, 0, uint32_t(k)) = float(v);
        }
    }
    return soft;
}

dressi::CpuImage ChainRuleClipToPos(const dressi::CpuImage& g_clip,
                                    const Mat4& mvp) {
    // clip_c = sum_j mvp[j*4+c] * pos_j + mvp[12+c]  (column-major)
    dressi::CpuImage g_pos(g_clip.width, 1, 3, 0.f);
    for (uint32_t v = 0; v < g_clip.width; v++) {
        for (uint32_t j = 0; j < 3; j++) {
            float g = 0.f;
            for (uint32_t c = 0; c < 4; c++) {
                g += g_clip.at(v, 0, c) * mvp[j * 4 + c];
            }
            g_pos.at(v, 0, j) = g;
        }
    }
    return g_pos;
}

void AdamStep(AdamState& state, std::vector<float>& params,
              const std::vector<float>& grad, float lr) {
    if (state.m.size() != params.size()) {
        state.m.assign(params.size(), 0.f);
        state.v.assign(params.size(), 0.f);
        state.t = 0;
    }
    state.t++;
    const float bc1 = 1.f - std::pow(state.beta1, float(state.t));
    const float bc2 = 1.f - std::pow(state.beta2, float(state.t));
    for (size_t i = 0; i < params.size(); i++) {
        state.m[i] = state.beta1 * state.m[i] + (1.f - state.beta1) * grad[i];
        state.v[i] = state.beta2 * state.v[i] +
                     (1.f - state.beta2) * grad[i] * grad[i];
        const float m_hat = state.m[i] / bc1;
        const float v_hat = state.v[i] / bc2;
        params[i] -= lr * m_hat / (std::sqrt(v_hat) + state.eps);
    }
}

}  // namespace dressi_examples
