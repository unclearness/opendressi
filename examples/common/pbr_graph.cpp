#include "pbr_graph.h"

#include "dressi/ibl.h"

namespace dressi_examples {

using namespace dressi;

namespace {

constexpr float kPi = 3.14159265358979324f;

// normalize() that returns 0 for the zero vector instead of NaN — the
// rasterizer writes 0 to background pixels and a NaN would survive the
// coverage-mask composite (0 * NaN == NaN)
Variable SafeNormalize(const Variable& v) {
    return v * F::InverseSqrt(F::Dot(v, v) + 1e-12f);
}

}  // namespace

PbrIblMaps BuildPbrIblMaps(const Variable& env) {
    PbrIblMaps maps;
    maps.env = env;
    maps.irr = BuildIrradianceSample(env);
    maps.pref = BuildPrefEnvironmentSample(env);
    maps.lut = BuildBrdfIntegrationMap();
    return maps;
}

PbrOutputs BuildPbrForward(const PbrSceneLeaves& scene,
                           const PbrFrameLeaves& frame, const PbrIblMaps& ibl,
                           const Variable& inv_uv, ImgSize screen,
                           bool normal_mapping) {
    // ----------------------------- G-buffer -----------------------------
    Variable g_uvh = F::Rasterize(scene.clip, scene.uvh, scene.faces, screen);
    Variable g_n = F::Rasterize(scene.clip, scene.normal, scene.faces, screen);
    Variable g_t =
            F::Rasterize(scene.clip, scene.tangent, scene.faces, screen);
    Variable g_p = F::Rasterize(scene.clip, scene.wpos, scene.faces, screen);
    Variable uv = F::Vec2(F::GetX(g_uvh), F::GetY(g_uvh));
    Variable mask = F::GetZ(g_uvh);
    // Interpolated handedness stays near +-1 inside a face; +1 at background
    Variable handed = F::Sign(F::GetW(g_uvh) + 1e-6f);

    // ------------------------- Material fetches -------------------------
    Variable albedo = F::TextureBilinear(scene.albedo, uv, inv_uv);
    Variable mr = F::TextureBilinear(scene.mr, uv, inv_uv);
    Variable rough = F::Clamp(F::GetY(mr), F::Float(0.045f), F::Float(1.f));
    Variable metal = F::GetZ(mr);
    Variable ao = F::GetX(F::TextureBilinear(scene.ao, uv, inv_uv));
    Variable emissive = F::TextureBilinear(scene.emissive, uv, inv_uv);

    // --------------------------- Shading normal --------------------------
    Variable n0 = SafeNormalize(g_n);
    Variable n = n0;
    if (normal_mapping) {
        Variable t0 = SafeNormalize(g_t - n0 * F::Dot(n0, g_t));
        Variable b0 = F::Cross(n0, t0) * handed;
        Variable nm = F::TextureBilinear(scene.normal_map, uv, inv_uv) * 2.f -
                      1.f;
        n = SafeNormalize(t0 * F::GetX(nm) + b0 * F::GetY(nm) +
                          n0 * F::GetZ(nm));
    }

    // ------------------------ Common shading terms -----------------------
    Variable v = SafeNormalize(frame.cam_pos - g_p);
    Variable n_dot_v = F::Max(F::Dot(n, v), F::Float(1e-4f));
    Variable f0 = F::Mix(F::Vec3(0.04f, 0.04f, 0.04f), albedo, metal);

    // ---------------------- Point light Cook-Torrance --------------------
    Variable to_light = frame.light_pos - g_p;
    Variable dist2 = F::Max(F::Dot(to_light, to_light), F::Float(1e-4f));
    Variable l = SafeNormalize(to_light);
    Variable h = SafeNormalize(v + l);
    Variable n_dot_l = F::Max(F::Dot(n, l), F::Float(0.f));
    Variable n_dot_h = F::Max(F::Dot(n, h), F::Float(0.f));
    Variable v_dot_h = F::Max(F::Dot(v, h), F::Float(0.f));

    Variable alpha = rough * rough;
    Variable alpha2 = alpha * alpha;
    Variable denom_d = n_dot_h * n_dot_h * (alpha2 - 1.f) + 1.f;
    Variable dist_d = alpha2 / (kPi * denom_d * denom_d);
    Variable k_direct = (rough + 1.f) * (rough + 1.f) / 8.f;
    Variable g_v = n_dot_v / (n_dot_v * (1.f - k_direct) + k_direct);
    Variable g_l = F::Max(n_dot_l, F::Float(1e-4f)) /
                   (F::Max(n_dot_l, F::Float(1e-4f)) * (1.f - k_direct) +
                    k_direct);
    Variable fresnel =
            f0 + (1.f - f0) * F::Pow(1.f - v_dot_h, F::Float(5.f));
    Variable spec_d = fresnel * (dist_d * g_v * g_l /
                                 (4.f * n_dot_v * n_dot_l + 1e-4f));
    Variable kd_direct = (1.f - fresnel) * (1.f - metal);
    Variable radiance = frame.light_col / dist2;
    Variable lo =
            (kd_direct * albedo / kPi + spec_d) * radiance * n_dot_l;

    // ------------------------------- IBL ---------------------------------
    // Roughness-aware Fresnel (learnopengl's fresnelSchlickRoughness)
    Variable f_ibl = f0 + (F::Max(1.f - rough, f0) - f0) *
                                  F::Pow(1.f - n_dot_v, F::Float(5.f));
    Variable kd_ibl = (1.f - f_ibl) * (1.f - metal);
    Variable irr_col = SampleIrradiance(ibl.irr, n);
    Variable refl = F::Reflect(F::Neg(v), n);
    Variable pref_col = SamplePrefEnvironment(ibl.pref, refl, rough);
    Variable ab = SampleBrdfLut(ibl.lut, n_dot_v, rough);
    Variable spec_ibl = pref_col * (f0 * F::GetX(ab) + F::GetY(ab));
    Variable ambient = (kd_ibl * irr_col * albedo + spec_ibl) * ao;

    // --------------------- Background (env by view ray) -------------------
    Variable sc = F::ScreenCoord(screen);
    Variable ndc_x =
            F::GetX(sc) * (2.f / float(screen.w)) - 1.f;
    Variable ndc_y =
            F::GetY(sc) * (2.f / float(screen.h)) - 1.f;
    // Screen y points down; cam_up is the world up axis (pre-scaled)
    Variable ray = frame.cam_fwd + frame.cam_right * ndc_x -
                   frame.cam_up * ndc_y;
    Variable bg = F::EquirectSample(ibl.env, ray);

    // ------------------------ Composite + tonemap ------------------------
    PbrOutputs out;
    Variable shaded = lo + ambient + emissive;
    out.hdr = mask * shaded + (1.f - mask) * bg;
    Variable mapped = out.hdr / (out.hdr + 1.f);
    out.ldr = F::Pow(mapped, F::Float(1.f / 2.2f));
    out.mask = mask;
    out.uv = uv;
    out.albedo = albedo;
    out.normal = n;
    return out;
}

}  // namespace dressi_examples
