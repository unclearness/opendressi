#ifndef DRESSI_CORE_IBL_MATH_H
#define DRESSI_CORE_IBL_MATH_H

// CPU mirrors of the GLSL IBL helpers emitted by the shader codegen
// (dressi_equirect_uv / dressi_dir_from_equirect / dressi_bilerp_wrap /
// dressi_bilerp_clamp / dressi_radical_inverse / dressi_hammersley /
// dressi_ggx_sample / dressi_g1_ibl in glsl_codegen.cpp). The CPU evaluator
// is the test oracle, so every formula here must match its GLSL twin:
// integer bit ops exactly, transcendentals within ULPs.

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace dressi {
namespace ibl {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 6.28318530717958647692f;
constexpr float kInvPi = 0.31830988618379067154f;
constexpr float kInvTwoPi = 0.15915494309189533577f;

struct Vec3 {
    float x = 0.f, y = 0.f, z = 0.f;
};

inline Vec3 Cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

inline float Dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 Normalize(const Vec3& v) {
    const float len = std::sqrt(Dot(v, v));
    return {v.x / len, v.y / len, v.z / len};
}

// Unit direction -> equirectangular uv in [0,1]^2 (u wraps at the atan2
// +-pi seam, v = 0 at the +Y pole)
inline void EquirectUvFromDir(const Vec3& d, float* u, float* v) {
    *u = std::atan2(d.z, d.x) * kInvTwoPi + 0.5f;
    *v = std::acos(std::clamp(d.y, -1.f, 1.f)) * kInvPi;
}

// Equirectangular uv -> unit direction (inverse of EquirectUvFromDir)
inline Vec3 DirFromEquirectUv(float u, float v) {
    const float phi = (u - 0.5f) * kTwoPi;
    const float theta = v * kPi;
    const float st = std::sin(theta);
    return {st * std::cos(phi), std::cos(theta), st * std::sin(phi)};
}

// 4-tap bilinear over a tightly packed {w,h,n} tensor; u wraps
// (equirect seam), v clamps (poles). Accumulates into out[0..n).
inline void BilerpWrap(const float* data, uint32_t w, uint32_t h, uint32_t n,
                       float u, float v, float* out) {
    const float sx = u * float(w) - 0.5f;
    const float sy = v * float(h) - 0.5f;
    const int ix = int(std::floor(sx));
    const int iy = int(std::floor(sy));
    const float fx = sx - float(ix);
    const float fy = sy - float(iy);
    for (uint32_t c = 0; c < n; c++) {
        out[c] = 0.f;
    }
    for (int dy = 0; dy <= 1; dy++) {
        for (int dx = 0; dx <= 1; dx++) {
            int x = ix + dx;
            if (x < 0) {
                x += int(w);
            }
            if (x >= int(w)) {
                x -= int(w);
            }
            const int y = std::clamp(iy + dy, 0, int(h) - 1);
            const float wgt = (dx == 0 ? 1.f - fx : fx) *
                              (dy == 0 ? 1.f - fy : fy);
            const float* p =
                    data + (size_t(y) * w + size_t(x)) * n;
            for (uint32_t c = 0; c < n; c++) {
                out[c] += wgt * p[c];
            }
        }
    }
}

// 4-tap bilinear, both axes clamped (plain texture / LUT lookup)
inline void BilerpClamp(const float* data, uint32_t w, uint32_t h, uint32_t n,
                        float u, float v, float* out) {
    const float sx = u * float(w) - 0.5f;
    const float sy = v * float(h) - 0.5f;
    const int ix = int(std::floor(sx));
    const int iy = int(std::floor(sy));
    const float fx = sx - float(ix);
    const float fy = sy - float(iy);
    for (uint32_t c = 0; c < n; c++) {
        out[c] = 0.f;
    }
    for (int dy = 0; dy <= 1; dy++) {
        for (int dx = 0; dx <= 1; dx++) {
            const int x = std::clamp(ix + dx, 0, int(w) - 1);
            const int y = std::clamp(iy + dy, 0, int(h) - 1);
            const float wgt = (dx == 0 ? 1.f - fx : fx) *
                              (dy == 0 ? 1.f - fy : fy);
            const float* p =
                    data + (size_t(y) * w + size_t(x)) * n;
            for (uint32_t c = 0; c < n; c++) {
                out[c] += wgt * p[c];
            }
        }
    }
}

// Van der Corput radical inverse (manual bit reversal — identical on the
// GLSL side, no bitfieldReverse)
inline float RadicalInverse(uint32_t bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f;  // 1 / 2^32
}

inline void Hammersley(uint32_t i, uint32_t n, float* xi0, float* xi1) {
    *xi0 = float(i) / float(n);
    *xi1 = RadicalInverse(i);
}

// GGX importance sample about normal `n` (learnopengl convention:
// alpha = roughness^2, tangent frame built from the smaller axis)
inline Vec3 ImportanceSampleGGX(float xi0, float xi1, float roughness,
                                const Vec3& n) {
    const float a = roughness * roughness;
    const float phi = kTwoPi * xi0;
    const float ct = std::sqrt((1.f - xi1) / (1.f + (a * a - 1.f) * xi1));
    const float st = std::sqrt(std::max(1.f - ct * ct, 0.f));
    const Vec3 h_t = {std::cos(phi) * st, std::sin(phi) * st, ct};
    const Vec3 up = std::abs(n.z) < 0.999f ? Vec3{0.f, 0.f, 1.f}
                                           : Vec3{1.f, 0.f, 0.f};
    const Vec3 tx = Normalize(Cross(up, n));
    const Vec3 ty = Cross(n, tx);
    return Normalize({tx.x * h_t.x + ty.x * h_t.y + n.x * h_t.z,
                      tx.y * h_t.x + ty.y * h_t.y + n.y * h_t.z,
                      tx.z * h_t.x + ty.z * h_t.y + n.z * h_t.z});
}

// Schlick-GGX single-direction shadowing (IBL k = roughness^2 / 2)
inline float G1Ibl(float n_dot_x, float k) {
    return n_dot_x / (n_dot_x * (1.f - k) + k);
}

}  // namespace ibl
}  // namespace dressi

#endif  // DRESSI_CORE_IBL_MATH_H
