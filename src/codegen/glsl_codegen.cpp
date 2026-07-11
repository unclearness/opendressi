#include "codegen/glsl_codegen.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>
#include <string>

#include "core/node.h"

namespace dressi {

namespace {

std::string FloatLit(float fv) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.9g", double(fv));
    std::string s(buf);
    if (s.find_first_of(".eEnif") == std::string::npos) {
        s += ".0";
    }
    return s;
}

// GLSL expression for an inline constant variable
std::string ConstExpr(const Variable& var) {
    const auto& node = NodeAccess::Node(var);
    const auto& vals = *node->const_val;
    if (vals.size() == 1) {
        return FloatLit(vals[0]);
    }
    std::string s = std::string(GlslTypeName(var.getVType())) + "(";
    for (size_t i = 0; i < vals.size(); i++) {
        s += (i ? "," : "") + FloatLit(vals[i]);
    }
    return s + ")";
}

const char* SwizzleOf(uint32_t n_comps) {
    switch (n_comps) {
        case 1: return ".x";
        case 2: return ".xy";
        case 3: return ".xyz";
        default: return "";
    }
}

std::string ReplaceAll(std::string s, const std::string& from,
                       const std::string& to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

}  // namespace

uint32_t PhysChannels(VType vtype) {
    const uint32_t n = NumComponents(vtype);
    DRESSI_CHECK(n <= 4, "Matrix images are not supported on GPU yet (M1)");
    return n == 3 ? 4 : n;
}

std::string FullscreenVertShader() {
    return R"(#version 450
void main() {
    vec2 pos = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
)";
}

namespace {

std::string GenerateShaderImpl(const SubStage& ss, bool comp) {
    // FRAG substages, raster-headed substages (a RASTER function at the
    // top followed by fused FRAG functions -- the paper's "same shader
    // types except for top rasterization"), or COMP substages (all inputs
    // texelFetch/UBO, outputs storage images).
    const bool raster_headed = ss.shader_type == RASTER;
    DRESSI_CHECK(comp ? ss.shader_type == COMP
                      : (ss.shader_type == FRAG || raster_headed),
                 "GenerateShaderImpl: substage/shader type mismatch");
    if (raster_headed) {
        DRESSI_CHECK(ss.inp_vars.empty() && ss.tex_vars.empty(),
                     "Raster-headed substages support slt/uif inputs only");
    } else if (comp) {
        DRESSI_CHECK(ss.tex_vars.empty() && ss.vtx_vars.empty(),
                     "COMP substages support inp/slt/uif inputs only");
    } else {
        DRESSI_CHECK(ss.vtx_vars.empty(),
                     "Codegen supports inp/tex/slt inputs only");
    }

    // Substage-local normalized names: inputs first, then generated vars,
    // deterministically ordered for stable golden tests and shader caching.
    std::map<Variable, std::string> names;
    uint32_t next_local = 0;
    const auto local_name = [&](const Variable& v) {
        auto it = names.find(v);
        if (it != names.end()) {
            return it->second;
        }
        const std::string n = "v" + std::to_string(next_local++);
        names.emplace(v, n);
        return n;
    };

    std::ostringstream head;
    std::ostringstream body;
    head << "#version 450\n";
    if (comp) {
        head << "layout(local_size_x = 64, local_size_y = 1, "
                "local_size_z = 1) in;\n";
    }

    // Binding convention shared with the executor: inp first, then slt.
    // COMP has no input attachments; inp inputs are samplers read by
    // texelFetch at the invocation coordinate.
    for (size_t i = 0; i < ss.inp_vars.size(); i++) {
        const Variable& v = ss.inp_vars[i];
        DRESSI_CHECK(!IsIntVType(v.getVType()),
                     "Int images are not supported on GPU yet (M1)");
        if (comp) {
            head << "layout(set=0, binding=" << i
                 << ") uniform sampler2D u_inp" << i << ";\n";
        } else {
            head << "layout(input_attachment_index=" << i
                 << ", set=0, binding=" << i << ") uniform subpassInput u_inp"
                 << i << ";\n";
        }
    }
    // tex inputs: samplers accessed with UV coordinates via texture()
    const size_t tex_binding_ofs = ss.inp_vars.size();
    for (size_t i = 0; i < ss.tex_vars.size(); i++) {
        head << "layout(set=0, binding=" << (tex_binding_ofs + i)
             << ") uniform sampler2D u_tex" << i << ";\n";
    }

    const size_t slt_binding_ofs = tex_binding_ofs + ss.tex_vars.size();

    // slt inputs: combined image samplers read via texelFetch
    for (size_t i = 0; i < ss.slt_vars.size(); i++) {
        const Variable& v = ss.slt_vars[i];
        DRESSI_CHECK(!IsIntVType(v.getVType()),
                     "Int images are not supported on GPU yet (M1)");
        head << "layout(set=0, binding=" << (slt_binding_ofs + i)
             << ") uniform sampler2D u_slt" << i << ";\n";
    }

    // uif inputs: {1,1} values as real uniforms (one vec4 UBO per value,
    // filled by a tiny image->buffer copy after the producing stage)
    const size_t uif_binding_ofs = slt_binding_ofs + ss.slt_vars.size();
    for (size_t i = 0; i < ss.uif_vars.size(); i++) {
        const Variable& v = ss.uif_vars[i];
        DRESSI_CHECK(!IsIntVType(v.getVType()),
                     "Int images are not supported on GPU yet (M1)");
        head << "layout(set=0, binding=" << (uif_binding_ofs + i)
             << ") uniform UifB" << i << " { vec4 u_uif" << i << "; };\n";
    }

    // Raster-headed: the interpolated vertex attribute arrives as v_attr
    if (raster_headed) {
        const uint32_t attr_n = NumComponents(ss.vtx_vars[1].getVType());
        head << "layout(location=0) in "
             << (attr_n == 1 ? "float" : ("vec" + std::to_string(attr_n)))
             << " v_attr;\n";
    }

    // Outputs: padded float attachments (FRAG) or storage images (COMP,
    // bound after the uif UBOs)
    for (size_t i = 0; i < ss.out_vars.size(); i++) {
        const uint32_t phys = PhysChannels(ss.out_vars[i].getVType());
        DRESSI_CHECK(ss.out_vars[i].getImgSize() == ss.img_size,
                     "Substage outputs must share the image size");
        if (comp) {
            const char* fmt = phys == 1 ? "r32f"
                              : phys == 2 ? "rg32f"
                                          : "rgba32f";
            head << "layout(set=0, binding="
                 << (uif_binding_ofs + ss.uif_vars.size() + i) << ", " << fmt
                 << ") uniform writeonly image2D u_out" << i << ";\n";
        } else {
            head << "layout(location=" << i << ") out "
                 << (phys == 1 ? "float" : ("vec" + std::to_string(phys)))
                 << " o_" << i << ";\n";
        }
    }

    // Shared helper for the screen-space AA ops (Dr.Hair Eq.4-5): the
    // silhouette edge separating a pixel pair and the blend weight
    // r = min(|edge_fn(p_s)| / edge_len, 1). Owner preference
    // {tri(n), tri(s)} approximates closest-depth selection.
    bool needs_aa_helper = false;
    bool needs_face_cross = false;
    bool needs_hash = false;
    // IBL helpers (CPU twins in core/ibl_math.h — keep formulas identical)
    bool needs_equirect_uv = false;
    bool needs_dir_from_equirect = false;
    bool needs_bilerp_wrap = false;
    bool needs_bilerp_clamp = false;
    bool needs_hammersley = false;
    bool needs_ggx_sample = false;
    bool needs_g1_ibl = false;
    for (const auto& f : ss.funcs) {
        const std::string& c = NodeAccess::Node(f)->fwd_code;
        if (c == "__antialias__" || c == "__antialias_bwd_img__" ||
            c.rfind("__antialias_bwd_vtx__", 0) == 0) {
            needs_aa_helper = true;
        }
        if (c == "__nc_face_term__") {
            needs_face_cross = true;
        }
        if (c.rfind("__face_fetch_bwd__", 0) == 0 ||
            c.rfind("__antialias_bwd_vtx__", 0) == 0) {
            needs_hash = true;
        }
        if (c == "__equirect_sample__") {
            needs_equirect_uv = needs_bilerp_wrap = true;
        }
        if (c == "__sample_bilinear__") {
            needs_bilerp_clamp = true;
        }
        if (c == "__irradiance_conv__") {
            needs_dir_from_equirect = true;
        }
        if (c.rfind("__prefilter_env__", 0) == 0) {
            needs_dir_from_equirect = needs_equirect_uv = needs_bilerp_wrap =
                    needs_hammersley = needs_ggx_sample = true;
        }
        if (c.rfind("__brdf_lut__", 0) == 0) {
            needs_hammersley = needs_ggx_sample = needs_g1_ibl = true;
        }
    }
    if (needs_hash) {
        // Wang hash -> [0,1); integer arithmetic identical to the CPU
        // kernel so the stochastic backward is reproducible
        head << R"(float dressi_hash(uint f, uint s, uint seed, uint salt) {
    uint x = f * 9781u ^ s * 6151u ^ seed * 26699u ^ salt * 42589u;
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x ^= x >> 4;
    x *= 0x27d4eb2du;
    x ^= x >> 15;
    return float(x & 0xFFFFFFu) / 16777216.0;
}
)";
    }
    if (needs_face_cross) {
        head << R"(vec3 dressi_face_cross(sampler2D vpos, sampler2D faces, int f) {
    ivec3 vi = ivec3(texelFetch(faces, ivec2(f, 0), 0).xyz + 0.5);
    vec3 p0 = texelFetch(vpos, ivec2(vi.x, 0), 0).xyz;
    vec3 p1 = texelFetch(vpos, ivec2(vi.y, 0), 0).xyz;
    vec3 p2 = texelFetch(vpos, ivec2(vi.z, 0), 0).xyz;
    return cross(p1 - p0, p2 - p0);
}
)";
    }
    if (needs_aa_helper) {
        head << R"(const ivec2 dressi_aa_offs[8] = ivec2[8](
    ivec2(-1,-1), ivec2(0,-1), ivec2(1,-1), ivec2(-1,0),
    ivec2(1,0), ivec2(-1,1), ivec2(0,1), ivec2(1,1));
struct DressiAaPair {
    float r; int ia; int ib; vec2 qa; vec2 qb; float es; float len;
};
bool dressi_aa_pair(sampler2D tri, sampler2D vclip, sampler2D faces,
                    ivec2 sc, ivec2 nc, vec2 wh, out DressiAaPair pr) {
    int its = int(texelFetch(tri, sc, 0).x + 0.5);
    int itn = int(texelFetch(tri, nc, 0).x + 0.5);
    if (its == itn) return false;
    vec2 ps = vec2(sc) + 0.5;
    vec2 pn = vec2(nc) + 0.5;
    for (int o = 0; o < 2; o++) {
        int owner = o == 0 ? itn : its;
        if (owner == 0) continue;
        vec3 fidx = texelFetch(faces, ivec2(owner - 1, 0), 0).xyz;
        ivec3 vi = ivec3(fidx + 0.5);
        vec2 q[3];
        bool ok = true;
        for (int k = 0; k < 3; k++) {
            vec4 c = texelFetch(vclip, ivec2(vi[k], 0), 0);
            if (abs(c.w) < 1e-6) { ok = false; break; }
            q[k] = (c.xy / c.w * 0.5 + 0.5) * wh;
        }
        if (!ok) continue;
        int bj = -1; float bt = 0.0;
        for (int j = 0; j < 3; j++) {
            vec2 a = q[j]; vec2 b = q[(j + 1) % 3];
            float es = (b.x - a.x) * (ps.y - a.y) - (b.y - a.y) * (ps.x - a.x);
            float en = (b.x - a.x) * (pn.y - a.y) - (b.y - a.y) * (pn.x - a.x);
            if (es * en >= 0.0) continue;
            float t = es / (es - en);
            if (bj < 0 || (o == 0 ? t > bt : t < bt)) { bt = t; bj = j; }
        }
        if (bj < 0) continue;
        vec2 a = q[bj]; vec2 b = q[(bj + 1) % 3];
        float len = length(b - a);
        if (len < 1e-6) continue;
        float es = (b.x - a.x) * (ps.y - a.y) - (b.y - a.y) * (ps.x - a.x);
        pr.r = min(abs(es) / len, 1.0);
        pr.ia = vi[bj]; pr.ib = vi[(bj + 1) % 3];
        pr.qa = a; pr.qb = b; pr.es = es; pr.len = len;
        return true;
    }
    return false;
}
)";
    }

    if (needs_equirect_uv) {
        head << R"(vec2 dressi_equirect_uv(vec3 d) {
    return vec2(atan(d.z, d.x) * 0.159154943091895336 + 0.5,
                acos(clamp(d.y, -1.0, 1.0)) * 0.318309886183790672);
}
)";
    }
    if (needs_dir_from_equirect) {
        head << R"(vec3 dressi_dir_from_equirect(vec2 uv) {
    float phi = (uv.x - 0.5) * 6.28318530717958648;
    float theta = uv.y * 3.14159265358979324;
    float st = sin(theta);
    return vec3(st * cos(phi), cos(theta), st * sin(phi));
}
)";
    }
    if (needs_bilerp_wrap) {
        head << R"(vec4 dressi_bilerp_wrap(sampler2D t, ivec2 sz, vec2 uv) {
    vec2 st = uv * vec2(sz) - 0.5;
    ivec2 i0 = ivec2(floor(st));
    vec2 f = st - vec2(i0);
    vec4 acc = vec4(0.0);
    for (int dy = 0; dy <= 1; dy++)
    for (int dx = 0; dx <= 1; dx++) {
        int x = i0.x + dx;
        if (x < 0) x += sz.x;
        if (x >= sz.x) x -= sz.x;
        int y = clamp(i0.y + dy, 0, sz.y - 1);
        float w = (dx == 0 ? 1.0 - f.x : f.x) * (dy == 0 ? 1.0 - f.y : f.y);
        acc += w * texelFetch(t, ivec2(x, y), 0);
    }
    return acc;
}
)";
    }
    if (needs_bilerp_clamp) {
        head << R"(vec4 dressi_bilerp_clamp(sampler2D t, ivec2 sz, vec2 uv) {
    vec2 st = uv * vec2(sz) - 0.5;
    ivec2 i0 = ivec2(floor(st));
    vec2 f = st - vec2(i0);
    vec4 acc = vec4(0.0);
    for (int dy = 0; dy <= 1; dy++)
    for (int dx = 0; dx <= 1; dx++) {
        int x = clamp(i0.x + dx, 0, sz.x - 1);
        int y = clamp(i0.y + dy, 0, sz.y - 1);
        float w = (dx == 0 ? 1.0 - f.x : f.x) * (dy == 0 ? 1.0 - f.y : f.y);
        acc += w * texelFetch(t, ivec2(x, y), 0);
    }
    return acc;
}
)";
    }
    if (needs_hammersley) {
        // Manual bit reversal (no bitfieldReverse) — integer arithmetic
        // identical to the CPU kernel
        head << R"(float dressi_radical_inverse(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}
vec2 dressi_hammersley(uint i, uint n) {
    return vec2(float(i) / float(n), dressi_radical_inverse(i));
}
)";
    }
    if (needs_ggx_sample) {
        head << R"(vec3 dressi_ggx_sample(vec2 xi, float rough, vec3 n) {
    float a = rough * rough;
    float phi = 6.28318530717958648 * xi.x;
    float ct = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    float st = sqrt(max(1.0 - ct * ct, 0.0));
    vec3 h = vec3(cos(phi) * st, sin(phi) * st, ct);
    vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tx = normalize(cross(up, n));
    vec3 ty = cross(n, tx);
    return normalize(tx * h.x + ty * h.y + n * h.z);
}
)";
    }
    if (needs_g1_ibl) {
        head << R"(float dressi_g1_ibl(float ndx, float k) {
    return ndx / (ndx * (1.0 - k) + k);
}
)";
    }

    body << "void main() {\n";
    if (comp) {
        body << "    ivec2 dressi_coord = ivec2(gl_GlobalInvocationID.xy);\n";
        body << "    if (dressi_coord.x >= " << ss.img_size.w
             << " || dressi_coord.y >= " << ss.img_size.h << ") return;\n";
    } else {
        body << "    ivec2 dressi_coord = ivec2(gl_FragCoord.xy);\n";
    }

    // Which slt inputs are consumed by coordinate-generic loads (special ops
    // fetch their input themselves)
    const auto is_special = [](const Function& f) {
        const std::string& code = NodeAccess::Node(f)->fwd_code;
        return code == "__reduce_2x2_sum__" ||
               code == "__upsample_nearest_2x__" ||
               code == "__gather_inv_uv__" ||
               code.rfind("__gather_dist_grad__", 0) == 0 ||
               code == "__antialias__" ||
               code == "__antialias_bwd_img__" ||
               code.rfind("__antialias_bwd_vtx__", 0) == 0 ||
               code == "__col_sum__" ||
               code.rfind("__soft_clip__", 0) == 0 ||
               code == "__vertex_neighbor_mean__" ||
               code == "__nc_face_term__" ||
               code == "__nc_vertex_grad__" ||
               code == "__screen_coord__" ||
               code == "__sum_all__" ||
               code == "__sum_partial__" ||
               code.rfind("__pixel_fetch__", 0) == 0 ||
               code.rfind("__tile_fetch__", 0) == 0 ||
               code.rfind("__lookup_faces__", 0) == 0 ||
               code.rfind("__lookup_faces_bwd__", 0) == 0 ||
               code == "__face_fetch__" ||
               code.rfind("__face_fetch_bwd__", 0) == 0 ||
               code == "__rasterize__" ||
               code == "__rasterize_face_id__" ||
               code.rfind("__rasterize_soft__", 0) == 0 ||
               code == "__equirect_sample__" ||
               code == "__sample_bilinear__" ||
               code == "__gather_inv_uv_bilinear__" ||
               code == "__irradiance_conv__" ||
               code.rfind("__prefilter_env__", 0) == 0 ||
               code.rfind("__brdf_lut__", 0) == 0;
    };
    std::map<Variable, size_t> slt_binding;
    for (size_t i = 0; i < ss.slt_vars.size(); i++) {
        slt_binding[ss.slt_vars[i]] = i;
    }
    std::map<Variable, size_t> tex_binding;
    for (size_t i = 0; i < ss.tex_vars.size(); i++) {
        tex_binding[ss.tex_vars[i]] = i;
    }
    // tex/sampled inputs are referenced through {sN} markers, never loaded
    std::map<Variable, bool> needs_generic_load;
    for (const auto& f : ss.funcs) {
        const bool special = is_special(f);
        const std::string& code = NodeAccess::Node(f)->fwd_code;
        const Variables fxs = f.getInputVars();
        for (size_t i = 0; i < fxs.size(); i++) {
            const Variable& x = fxs[i];
            // The dir/uv operand of the IBL sampling ops is same-pixel:
            // when the packer routed it through slt (Naive mode), give it
            // the generic coordinate load (the map operand is fetched by
            // the emission block itself)
            const bool same_pixel_operand =
                    (code == "__equirect_sample__" ||
                     code == "__sample_bilinear__") &&
                    i == 1;
            if (slt_binding.count(x) && (!special || same_pixel_operand) &&
                !tex_binding.count(x)) {
                needs_generic_load[x] = true;
            }
        }
    }

    // Loads for input attachments (same-pixel, same-size by construction)
    for (size_t i = 0; i < ss.inp_vars.size(); i++) {
        const Variable& v = ss.inp_vars[i];
        const uint32_t n = NumComponents(v.getVType());
        body << "    " << GlslTypeName(v.getVType()) << " " << local_name(v)
             << (comp ? " = texelFetch(u_inp" : " = subpassLoad(u_inp") << i
             << (comp ? ", dressi_coord, 0)" : ")") << SwizzleOf(n) << ";\n";
    }

    // Generic loads for slt inputs (skipped when the same variable is
    // already loaded as an input attachment: same-pixel consumers read the
    // subpassLoad value, texelFetch consumers use the slt binding directly)
    for (size_t i = 0; i < ss.slt_vars.size(); i++) {
        const Variable& v = ss.slt_vars[i];
        if (!needs_generic_load.count(v) || names.count(v)) {
            continue;
        }
        const uint32_t n = NumComponents(v.getVType());
        const std::string coord = v.getImgSize().isUniform()
                                          ? "ivec2(0,0)"
                                          : (v.getImgSize() == ss.img_size
                                                     ? "dressi_coord"
                                                     : "");
        DRESSI_CHECK(!coord.empty(),
                     "Non-matching input size requires a special op");
        body << "    " << GlslTypeName(v.getVType()) << " " << local_name(v)
             << " = texelFetch(u_slt" << i << ", " << coord << ", 0)"
             << SwizzleOf(n) << ";\n";
    }

    // Loads for uniform inputs (always generic: special ops take their
    // dynamic scalars through TexelFetch access, i.e. slt)
    for (size_t i = 0; i < ss.uif_vars.size(); i++) {
        const Variable& v = ss.uif_vars[i];
        if (names.count(v)) {
            continue;
        }
        const uint32_t n = NumComponents(v.getVType());
        body << "    " << GlslTypeName(v.getVType()) << " " << local_name(v)
             << " = u_uif" << i << SwizzleOf(n) << ";\n";
    }

    // Expression for a function input: local name, or inline constant
    const auto input_expr = [&](const Variable& x) -> std::string {
        if (IsInlineConst(x)) {
            return ConstExpr(x);
        }
        auto it = names.find(x);
        DRESSI_CHECK(it != names.end(),
                     "Function input is neither loaded nor generated");
        return it->second;
    };

    // Function bodies in topological (creation id) order
    Functions funcs = ss.funcs;
    std::sort(funcs.begin(), funcs.end(),
              [](const Function& a, const Function& b) {
                  return a.id() < b.id();
              });
    for (const auto& f : funcs) {
        const auto& node = NodeAccess::Node(f);
        const Variable y = f.getOutputVar();
        const Variables xs = f.getInputVars();
        const std::string y_name = local_name(y);
        const char* y_type = GlslTypeName(y.getVType());

        if (node->fwd_code == "__reduce_2x2_sum__") {
            const size_t bind = slt_binding.at(xs[0]);
            const ImgSize src = xs[0].getImgSize();
            const uint32_t n = NumComponents(y.getVType());
            body << "    " << y_type << " " << y_name << " = " << y_type
                 << "(0.0);\n";
            body << "    for (int dy = 0; dy < 2; dy++)"
                 << " for (int dx = 0; dx < 2; dx++) {\n";
            body << "        ivec2 p = dressi_coord * 2 + ivec2(dx, dy);\n";
            body << "        if (p.x < " << src.w << " && p.y < " << src.h
                 << ") { " << y_name << " += texelFetch(u_slt" << bind
                 << ", p, 0)" << SwizzleOf(n) << "; }\n";
            body << "    }\n";
            continue;
        }
        if (node->fwd_code == "__upsample_nearest_2x__") {
            const size_t bind = slt_binding.at(xs[0]);
            const uint32_t n = NumComponents(y.getVType());
            body << "    " << y_type << " " << y_name << " = texelFetch(u_slt"
                 << bind << ", dressi_coord / 2, 0)" << SwizzleOf(n) << ";\n";
            continue;
        }
        if (node->fwd_code == "__gather_inv_uv__") {
            // xs = {gy_screen, inv_uv, uv_screen}; output lives in texture
            // space. The inverse-UV entry is quantized and affinely
            // interpolated, so search its neighborhood for a screen pixel
            // that actually samples this texel (occlusion guard included:
            // occluded texels find no matching pixel).
            const size_t gy_bind = slt_binding.at(xs[0]);
            const size_t iv_bind = slt_binding.at(xs[1]);
            const size_t uv_bind = slt_binding.at(xs[2]);
            const ImgSize tex_size = y.getImgSize();
            const ImgSize scr_size = xs[0].getImgSize();
            const uint32_t n = NumComponents(y.getVType());
            body << "    " << y_type << " " << y_name << " = " << y_type
                 << "(0.0);\n";
            body << "    {\n";
            // Table entry of this texel, or of a 3x3 neighbor (on-the-fly
            // dilation: chart-boundary texels are sampled by the forward
            // pass but their centers fall outside the UV-space triangles)
            body << "        vec4 iv = texelFetch(u_slt" << iv_bind
                 << ", dressi_coord, 0);\n";
            // Dilation must reach at least as far as the sampling jitter
            // pushes reads beyond the UV charts
            body << "        for (int ny = -4; ny <= 4 && iv.z < 0.5; ny++)"
                 << " for (int nx = -4; nx <= 4; nx++) {\n";
            body << "            ivec2 tp = dressi_coord + ivec2(nx, ny);\n";
            body << "            if (tp.x < 0 || tp.y < 0 || tp.x >= "
                 << tex_size.w << " || tp.y >= " << tex_size.h
                 << ") continue;\n";
            body << "            iv = texelFetch(u_slt" << iv_bind
                 << ", tp, 0);\n";
            body << "            if (iv.z > 0.5) break;\n";
            body << "        }\n";
            body << "        if (iv.z > 0.5) {\n";
            body << "            ivec2 sc = ivec2(iv.xy);\n";
            body << "            for (int dy = -3; dy <= 3; dy++)"
                 << " for (int dx = -3; dx <= 3; dx++) {\n";
            body << "                ivec2 sp = sc + ivec2(dx, dy);\n";
            body << "                if (sp.x < 0 || sp.y < 0 || sp.x >= "
                 << scr_size.w << " || sp.y >= " << scr_size.h
                 << ") continue;\n";
            body << "                vec2 suv = texelFetch(u_slt" << uv_bind
                 << ", sp, 0).xy;\n";
            body << "                ivec2 st = ivec2(suv * vec2("
                 << tex_size.w << ".0, " << tex_size.h << ".0));\n";
            body << "                if (st == dressi_coord) { " << y_name
                 << " = texelFetch(u_slt" << gy_bind << ", sp, 0)"
                 << SwizzleOf(n) << "; dy = 4; break; }\n";
            body << "            }\n";
            body << "        }\n";
            body << "    }\n";
            continue;
        }

        if (node->fwd_code == "__screen_coord__") {
            body << "    vec2 " << y_name
                 << " = vec2(dressi_coord) + 0.5;\n";
            continue;
        }
        if (node->fwd_code.rfind("__pixel_fetch__", 0) == 0) {
            const int px_x = std::stoi(node->fwd_code.substr(
                    node->fwd_code.find("x=") + 2));
            const size_t tx_bind = slt_binding.at(xs[0]);
            const size_t id_bind = slt_binding.at(xs[1]);
            const uint32_t n = NumComponents(y.getVType());
            body << "    " << y_type << " " << y_name
                 << " = texelFetch(u_slt" << tx_bind << ", ivec2(" << px_x
                 << ", int(texelFetch(u_slt" << id_bind
                 << ", ivec2(0, 0), 0).x + 0.5)), 0)" << SwizzleOf(n)
                 << ";\n";
            continue;
        }
        if (node->fwd_code.rfind("__tile_fetch__", 0) == 0) {
            const int tile_h = std::stoi(node->fwd_code.substr(
                    node->fwd_code.find("h=") + 2));
            const size_t at_bind = slt_binding.at(xs[0]);
            const size_t id_bind = slt_binding.at(xs[1]);
            const uint32_t n = NumComponents(y.getVType());
            body << "    " << y_type << " " << y_name
                 << " = texelFetch(u_slt" << at_bind
                 << ", ivec2(dressi_coord.x, dressi_coord.y"
                    " + int(texelFetch(u_slt"
                 << id_bind << ", ivec2(0, 0), 0).x + 0.5) * " << tile_h
                 << "), 0)" << SwizzleOf(n) << ";\n";
            continue;
        }
        if (node->fwd_code == "__sum_partial__") {
            // Strided partial sums over the flattened image: fragment p of
            // the {parts,1} output sums linear pixels p, p+parts, ...
            const size_t in_bind = slt_binding.at(xs[0]);
            const ImgSize src = xs[0].getImgSize();
            const uint32_t parts = y.getImgSize().w;
            const uint32_t n = NumComponents(xs[0].getVType());
            body << "    float " << y_name << " = 0.0;\n";
            body << "    for (int li = dressi_coord.x; li < "
                 << (size_t(src.w) * src.h) << "; li += " << parts << ") {\n";
            body << "        " << (n == 1 ? "float" : GlslTypeName(xs[0].getVType()))
                 << " v = texelFetch(u_slt" << in_bind << ", ivec2(li % "
                 << src.w << ", li / " << src.w << "), 0)" << SwizzleOf(n)
                 << ";\n";
            body << "        " << y_name << " += "
                 << (n == 1 ? "v"
                     : n == 2 ? "v.x + v.y"
                     : n == 3 ? "v.x + v.y + v.z"
                              : "v.x + v.y + v.z + v.w")
                 << ";\n";
            body << "    }\n";
            continue;
        }
        if (node->fwd_code == "__sum_all__") {
            // Single-thread whole-image sum (small inputs only)
            const size_t in_bind = slt_binding.at(xs[0]);
            const ImgSize src = xs[0].getImgSize();
            const uint32_t n = NumComponents(xs[0].getVType());
            body << "    float " << y_name << " = 0.0;\n";
            body << "    for (int sy = 0; sy < " << src.h << "; sy++)\n";
            body << "    for (int sx = 0; sx < " << src.w << "; sx++) {\n";
            if (n == 1) {
                body << "        " << y_name << " += texelFetch(u_slt"
                     << in_bind << ", ivec2(sx, sy), 0).x;\n";
            } else {
                body << "        " << GlslTypeName(xs[0].getVType())
                     << " v = texelFetch(u_slt" << in_bind
                     << ", ivec2(sx, sy), 0)" << SwizzleOf(n) << ";\n";
                body << "        " << y_name << " += "
                     << (n == 2 ? "v.x + v.y"
                                : n == 3 ? "v.x + v.y + v.z"
                                         : "v.x + v.y + v.z + v.w")
                     << ";\n";
            }
            body << "    }\n";
            continue;
        }
        if (node->fwd_code == "__equirect_sample__") {
            // xs = {map (slt), dir (same-pixel)}; zero-length directions
            // (rasterizer background) return 0
            const size_t map_bind = slt_binding.at(xs[0]);
            const ImgSize map_size = xs[0].getImgSize();
            const uint32_t n = NumComponents(y.getVType());
            body << "    " << y_type << " " << y_name << " = " << y_type
                 << "(0.0);\n";
            body << "    {\n";
            body << "        vec3 d = " << input_expr(xs[1]) << ";\n";
            body << "        float dlen = length(d);\n";
            body << "        if (dlen >= 1e-8) {\n";
            body << "            d /= dlen;\n";
            body << "            " << y_name << " = dressi_bilerp_wrap(u_slt"
                 << map_bind << ", ivec2(" << map_size.w << ", " << map_size.h
                 << "), dressi_equirect_uv(d))" << SwizzleOf(n) << ";\n";
            body << "        }\n";
            body << "    }\n";
            continue;
        }
        if (node->fwd_code == "__sample_bilinear__") {
            // xs = {tex (slt), uv (same-pixel)}
            const size_t tex_bind = slt_binding.at(xs[0]);
            const ImgSize tex_size = xs[0].getImgSize();
            const uint32_t n = NumComponents(y.getVType());
            body << "    " << y_type << " " << y_name
                 << " = dressi_bilerp_clamp(u_slt" << tex_bind << ", ivec2("
                 << tex_size.w << ", " << tex_size.h << "), "
                 << input_expr(xs[1]) << ")" << SwizzleOf(n) << ";\n";
            continue;
        }
        if (node->fwd_code == "__gather_inv_uv_bilinear__") {
            // xs = {gy_screen, inv_uv, uv_screen}; bilinear-weighted
            // variant of __gather_inv_uv__: anchor at the (dilated)
            // inverse-UV entry, then ACCUMULATE every nearby screen
            // pixel's gradient with the forward's tent weight
            const size_t gy_bind = slt_binding.at(xs[0]);
            const size_t iv_bind = slt_binding.at(xs[1]);
            const size_t uv_bind = slt_binding.at(xs[2]);
            const ImgSize tex_size = y.getImgSize();
            const ImgSize scr_size = xs[0].getImgSize();
            const uint32_t n = NumComponents(y.getVType());
            body << "    " << y_type << " " << y_name << " = " << y_type
                 << "(0.0);\n";
            body << "    {\n";
            body << "        vec4 iv = texelFetch(u_slt" << iv_bind
                 << ", dressi_coord, 0);\n";
            body << "        for (int ny = -4; ny <= 4 && iv.z < 0.5; ny++)"
                 << " for (int nx = -4; nx <= 4; nx++) {\n";
            body << "            ivec2 tp = dressi_coord + ivec2(nx, ny);\n";
            body << "            if (tp.x < 0 || tp.y < 0 || tp.x >= "
                 << tex_size.w << " || tp.y >= " << tex_size.h
                 << ") continue;\n";
            body << "            iv = texelFetch(u_slt" << iv_bind
                 << ", tp, 0);\n";
            body << "            if (iv.z > 0.5) break;\n";
            body << "        }\n";
            body << "        if (iv.z > 0.5) {\n";
            body << "            ivec2 sc = ivec2(iv.xy);\n";
            body << "            for (int dy = -3; dy <= 3; dy++)"
                 << " for (int dx = -3; dx <= 3; dx++) {\n";
            body << "                ivec2 sp = sc + ivec2(dx, dy);\n";
            body << "                if (sp.x < 0 || sp.y < 0 || sp.x >= "
                 << scr_size.w << " || sp.y >= " << scr_size.h
                 << ") continue;\n";
            body << "                vec2 st = texelFetch(u_slt" << uv_bind
                 << ", sp, 0).xy * vec2(" << tex_size.w << ".0, "
                 << tex_size.h << ".0) - 0.5;\n";
            body << "                float w = max(0.0, 1.0 - abs(st.x - "
                    "float(dressi_coord.x))) * max(0.0, 1.0 - abs(st.y - "
                    "float(dressi_coord.y)));\n";
            body << "                if (w > 0.0) { " << y_name
                 << " += texelFetch(u_slt" << gy_bind << ", sp, 0)"
                 << SwizzleOf(n) << " * w; }\n";
            body << "            }\n";
            body << "        }\n";
            body << "    }\n";
            continue;
        }
        if (node->fwd_code == "__irradiance_conv__") {
            // Deterministic cos-weighted direct sum over every source
            // texel; stores E(N)/pi (CPU twin in F::IrradianceConv)
            const size_t env_bind = slt_binding.at(xs[0]);
            const ImgSize src = xs[0].getImgSize();
            const ImgSize out = y.getImgSize();
            const uint32_t n = NumComponents(y.getVType());
            const float scale = 6.28318530717958648f /
                                (float(src.w) * float(src.h));
            body << "    " << y_type << " " << y_name << " = " << y_type
                 << "(0.0);\n";
            body << "    {\n";
            body << "        vec3 N = dressi_dir_from_equirect("
                 << "(vec2(dressi_coord) + 0.5) / vec2(" << out.w << ".0, "
                 << out.h << ".0));\n";
            body << "        for (int sy = 0; sy < " << src.h
                 << "; sy++) {\n";
            body << "            float sv = (float(sy) + 0.5) / " << src.h
                 << ".0;\n";
            body << "            float sin_th = sin(sv * "
                    "3.14159265358979324);\n";
            body << "            for (int sx = 0; sx < " << src.w
                 << "; sx++) {\n";
            body << "                vec3 d = dressi_dir_from_equirect("
                 << "vec2((float(sx) + 0.5) / " << src.w
                 << ".0, sv));\n";
            body << "                " << y_name << " += texelFetch(u_slt"
                 << env_bind << ", ivec2(sx, sy), 0)" << SwizzleOf(n)
                 << " * (max(dot(N, d), 0.0) * sin_th);\n";
            body << "            }\n";
            body << "        }\n";
            body << "        " << y_name << " *= " << FloatLit(scale)
                 << ";\n";
            body << "    }\n";
            continue;
        }
        if (node->fwd_code.rfind("__prefilter_env__", 0) == 0) {
            // GGX-importance-sampled prefilter (split-sum first term,
            // V=N=R; CPU twin in F::PrefilterEnv)
            const float rough = std::stof(node->fwd_code.substr(
                    node->fwd_code.find("r=") + 2));
            const int n_samples = std::stoi(node->fwd_code.substr(
                    node->fwd_code.find("n=") + 2));
            const size_t env_bind = slt_binding.at(xs[0]);
            const ImgSize src = xs[0].getImgSize();
            const ImgSize out = y.getImgSize();
            const uint32_t n = NumComponents(y.getVType());
            body << "    " << y_type << " " << y_name << " = " << y_type
                 << "(0.0);\n";
            body << "    {\n";
            body << "        vec3 N = dressi_dir_from_equirect("
                 << "(vec2(dressi_coord) + 0.5) / vec2(" << out.w << ".0, "
                 << out.h << ".0));\n";
            body << "        float wsum = 0.0;\n";
            body << "        for (int i = 0; i < " << n_samples
                 << "; i++) {\n";
            body << "            vec2 xi = dressi_hammersley(uint(i), "
                 << n_samples << "u);\n";
            body << "            vec3 h = dressi_ggx_sample(xi, "
                 << FloatLit(rough) << ", N);\n";
            body << "            vec3 L = normalize(2.0 * dot(N, h) * h - "
                    "N);\n";
            body << "            float nol = dot(N, L);\n";
            body << "            if (nol > 0.0) {\n";
            body << "                " << y_name
                 << " += dressi_bilerp_wrap(u_slt" << env_bind << ", ivec2("
                 << src.w << ", " << src.h
                 << "), dressi_equirect_uv(L))" << SwizzleOf(n)
                 << " * nol;\n";
            body << "                wsum += nol;\n";
            body << "            }\n";
            body << "        }\n";
            body << "        " << y_name << " /= max(wsum, 1e-4);\n";
            body << "    }\n";
            continue;
        }
        if (node->fwd_code.rfind("__brdf_lut__", 0) == 0) {
            // Split-sum BRDF integration LUT (scale A, bias B); zero
            // inputs (CPU twin in F::BrdfIntegrationLut)
            const int n_samples = std::stoi(node->fwd_code.substr(
                    node->fwd_code.find("n=") + 2));
            const ImgSize out = y.getImgSize();
            body << "    vec2 " << y_name << " = vec2(0.0);\n";
            body << "    {\n";
            body << "        float ndv = (float(dressi_coord.x) + 0.5) / "
                 << out.w << ".0;\n";
            body << "        float rough = (float(dressi_coord.y) + 0.5) / "
                 << out.h << ".0;\n";
            body << "        float k = rough * rough / 2.0;\n";
            body << "        vec3 V = vec3(sqrt(max(1.0 - ndv * ndv, 0.0)), "
                    "0.0, ndv);\n";
            body << "        for (int i = 0; i < " << n_samples
                 << "; i++) {\n";
            body << "            vec2 xi = dressi_hammersley(uint(i), "
                 << n_samples << "u);\n";
            body << "            vec3 h = dressi_ggx_sample(xi, rough, "
                    "vec3(0.0, 0.0, 1.0));\n";
            body << "            vec3 L = normalize(2.0 * dot(V, h) * h - "
                    "V);\n";
            body << "            float nol = max(L.z, 0.0);\n";
            body << "            float noh = max(h.z, 0.0);\n";
            body << "            float voh = max(dot(V, h), 0.0);\n";
            body << "            if (nol > 0.0) {\n";
            body << "                float g = dressi_g1_ibl(ndv, k) * "
                    "dressi_g1_ibl(nol, k);\n";
            body << "                float gv = g * voh / max(noh * ndv, "
                    "1e-8);\n";
            body << "                float fc = pow(1.0 - voh, 5.0);\n";
            body << "                " << y_name
                 << " += vec2((1.0 - fc) * gv, fc * gv);\n";
            body << "            }\n";
            body << "        }\n";
            body << "        " << y_name << " /= " << n_samples << ".0;\n";
            body << "    }\n";
            continue;
        }
        if (node->fwd_code == "__rasterize__") {
            // Top rasterization of the substage: the hardware interpolated
            // the vertex attribute into v_attr
            body << "    " << y_type << " " << y_name << " = v_attr;\n";
            continue;
        }
        if (node->fwd_code == "__rasterize_face_id__") {
            body << "    float " << y_name
                 << " = float(gl_PrimitiveID) + 1.0;\n";
            continue;
        }
        if (node->fwd_code.rfind("__rasterize_soft__", 0) == 0) {
            // HardSoftRas forward (top rasterization): v_attr carries the
            // hard face index; fetch the hard triangle from the clip/faces
            // textures, output VEC4 (dist, face_id, hard_z, coverage) and
            // apply the Eq.3 depth shift via gl_FragDepth
            const float radius_px = std::stof(node->fwd_code.substr(
                    std::strlen("__rasterize_soft__ r=")));
            const size_t cl_bind = slt_binding.at(xs[3]);
            const size_t fc_bind = slt_binding.at(xs[4]);
            const ImgSize scr = y.getImgSize();
            body << "    vec4 " << y_name << ";\n";
            body << "    {\n";
            body << "        int f = int(v_attr + 0.5);\n";
            body << "        ivec3 vi = ivec3(texelFetch(u_slt" << fc_bind
                 << ", ivec2(f, 0), 0).xyz + 0.5);\n";
            body << "        vec2 s[3]; float z[3];\n";
            body << "        for (int k = 0; k < 3; k++) {\n";
            body << "            vec4 c = texelFetch(u_slt" << cl_bind
                 << ", ivec2(vi[k], 0), 0);\n";
            body << "            s[k] = (c.xy / c.w * 0.5 + 0.5) * vec2("
                 << scr.w << ".0, " << scr.h << ".0);\n";
            body << "            z[k] = c.z / c.w;\n";
            body << "        }\n";
            body << "        vec2 p = gl_FragCoord.xy;\n";
            body << "        float best_d2 = -1.0;"
                    " int npos = 0; int nneg = 0;\n";
            body << "        for (int k = 0; k < 3; k++) {\n";
            body << "            vec2 a = s[k]; vec2 b = s[(k + 1) % 3];\n";
            body << "            vec2 e = b - a; float len2 = dot(e, e);\n";
            body << "            float t = len2 > 1e-12 ?"
                    " clamp(dot(p - a, e) / len2, 0.0, 1.0) : 0.0;\n";
            body << "            vec2 q = a + t * e;"
                    " float d2 = dot(p - q, p - q);\n";
            body << "            if (best_d2 < 0.0 || d2 < best_d2)"
                    " best_d2 = d2;\n";
            body << "            float cr = e.x * (p.y - a.y)"
                    " - e.y * (p.x - a.x);\n";
            body << "            if (cr >= 0.0) npos++;\n";
            body << "            if (cr <= 0.0) nneg++;\n";
            body << "        }\n";
            body << "        float sgn ="
                    " (npos == 3 || nneg == 3) ? 1.0 : -1.0;\n";
            body << "        float dist = sgn * sqrt(max(best_d2, 0.0));\n";
            body << "        float area = (s[1].x - s[0].x)"
                    " * (s[2].y - s[0].y)"
                    " - (s[1].y - s[0].y) * (s[2].x - s[0].x);\n";
            body << "        float hard_z = 0.5;\n";
            body << "        if (abs(area) > 1e-9) {\n";
            body << "            float l0 = ((s[1].x - p.x) * (s[2].y - p.y)"
                    " - (s[1].y - p.y) * (s[2].x - p.x)) / area;\n";
            body << "            float l1 = ((s[2].x - p.x) * (s[0].y - p.y)"
                    " - (s[2].y - p.y) * (s[0].x - p.x)) / area;\n";
            body << "            hard_z = l0 * z[0] + l1 * z[1]"
                    " + (1.0 - l0 - l1) * z[2];\n";
            body << "        }\n";
            body << "        float dsh = dist >= 0.0"
                    " ? 0.5 * clamp(hard_z, 0.0, 1.0)"
                    " : 0.5 + 0.5 * clamp(-dist / "
                 << FloatLit(radius_px) << ", 0.0, 1.0);\n";
            if (node->fwd_code.find("peel=1") != std::string::npos) {
                // Depth peeling: only fragments behind the previous layer
                const size_t pv_bind = slt_binding.at(xs[6]);
                body << "        if (dsh <= texelFetch(u_slt" << pv_bind
                     << ", dressi_coord, 0).x + 1e-4) discard;\n";
            }
            body << "        gl_FragDepth = dsh;\n";
            body << "        " << y_name
                 << " = vec4(dist, v_attr, hard_z, 1.0);\n";
            body << "    }\n";
            continue;
        }
        if (node->fwd_code.rfind("__lookup_faces__", 0) == 0) {
            // xs = {vtx_attr, faces_tex, vtx_faces_tex}
            const int corner = std::stoi(node->fwd_code.substr(
                    node->fwd_code.find("k=") + 2));
            const size_t at_bind = slt_binding.at(xs[0]);
            const size_t fc_bind = slt_binding.at(xs[1]);
            const uint32_t n = NumComponents(y.getVType());
            body << "    " << y_type << " " << y_name
                 << " = texelFetch(u_slt" << at_bind
                 << ", ivec2(int(texelFetch(u_slt" << fc_bind
                 << ", ivec2(dressi_coord.x, 0), 0)[" << corner
                 << "] + 0.5), 0), 0)" << SwizzleOf(n) << ";\n";
            continue;
        }
        if (node->fwd_code.rfind("__lookup_faces_bwd__", 0) == 0) {
            // xs = {gy {F,1}, faces_tex, vtx_faces_tex}; output {V,1}
            const int corner = std::stoi(node->fwd_code.substr(
                    node->fwd_code.find("k=") + 2));
            const size_t gy_bind = slt_binding.at(xs[0]);
            const size_t fc_bind = slt_binding.at(xs[1]);
            const size_t vf_bind = slt_binding.at(xs[2]);
            const uint32_t max_deg = xs[2].getImgSize().h;
            const uint32_t n = NumComponents(y.getVType());
            body << "    " << y_type << " " << y_name << " = " << y_type
                 << "(0.0);\n";
            body << "    {\n";
            body << "        int vid = dressi_coord.x;\n";
            body << "        for (int d = 0; d < " << max_deg
                 << "; d++) {\n";
            body << "            float fv = texelFetch(u_slt" << vf_bind
                 << ", ivec2(vid, d), 0).x;\n";
            body << "            if (fv < -0.5) continue;\n";
            body << "            int f = int(fv + 0.5);\n";
            body << "            if (int(texelFetch(u_slt" << fc_bind
                 << ", ivec2(f, 0), 0)[" << corner
                 << "] + 0.5) != vid) continue;\n";
            body << "            " << y_name << " += texelFetch(u_slt"
                 << gy_bind << ", ivec2(f, 0), 0)" << SwizzleOf(n) << ";\n";
            body << "        }\n";
            body << "    }\n";
            continue;
        }
        if (node->fwd_code == "__face_fetch__") {
            // xs = {tri_attr, idx_img, tri_prj_0..2, seed}; only the first
            // two are read in the forward
            const size_t at_bind = slt_binding.at(xs[0]);
            const size_t id_bind = slt_binding.at(xs[1]);
            const uint32_t n = NumComponents(y.getVType());
            body << "    " << y_type << " " << y_name << " = " << y_type
                 << "(0.0);\n";
            body << "    {\n";
            body << "        int f = int(texelFetch(u_slt" << id_bind
                 << ", dressi_coord, 0).x + 0.5) - 1;\n";
            body << "        if (f >= 0) " << y_name
                 << " = texelFetch(u_slt" << at_bind << ", ivec2(f, 0), 0)"
                 << SwizzleOf(n) << ";\n";
            body << "    }\n";
            continue;
        }
        if (node->fwd_code.rfind("__face_fetch_bwd__", 0) == 0) {
            // xs = {gy, idx_img, tri_prj_0..2, seed}; output {F,1}. Per
            // face: gather gy at n quasi-random points along the hard
            // edges (band +-r px), guarded by the ID buffer (the paper's
            // Alg.1 inverse-table philosophy without a scatter pass).
            const std::string& code2 = node->fwd_code;
            const float radius_px =
                    std::stof(code2.substr(code2.find("r=") + 2));
            const int n_samples =
                    std::stoi(code2.substr(code2.find("n=") + 2));
            const size_t gy_bind = slt_binding.at(xs[0]);
            const size_t id_bind = slt_binding.at(xs[1]);
            const size_t p0_bind = slt_binding.at(xs[2]);
            const size_t p1_bind = slt_binding.at(xs[3]);
            const size_t p2_bind = slt_binding.at(xs[4]);
            const size_t sd_bind = slt_binding.at(xs[5]);
            const ImgSize scr = xs[1].getImgSize();
            const uint32_t n = NumComponents(y.getVType());
            const char* swz = SwizzleOf(n);
            body << "    " << y_type << " " << y_name << " = " << y_type
                 << "(0.0);\n";
            body << "    {\n";
            body << "        int f = dressi_coord.x;\n";
            body << "        uint seed = uint(texelFetch(u_slt" << sd_bind
                 << ", ivec2(0, 0), 0).x + 0.5);\n";
            body << "        vec2 s[3]; bool ok = true;\n";
            body << "        {\n";
            for (int k = 0; k < 3; k++) {
                const size_t b = k == 0 ? p0_bind : (k == 1 ? p1_bind
                                                            : p2_bind);
                body << "            vec4 c" << k << " = texelFetch(u_slt"
                     << b << ", ivec2(f, 0), 0);\n";
                body << "            if (c" << k
                     << ".w <= 1e-6) ok = false;\n";
                body << "            else s[" << k << "] = (c" << k
                     << ".xy / c" << k << ".w * 0.5 + 0.5) * vec2(" << scr.w
                     << ".0, " << scr.h << ".0);\n";
            }
            body << "        }\n";
            if (n_samples == 0) {
                // Exact mode: accumulate gy over every pixel of the face's
                // screen bbox that the ID buffer attributes to the face
                // (the true VJP; interior support for general interpolation)
                body << "        if (ok) {\n";
                body << "            vec2 lo = min(min(s[0], s[1]), s[2]);\n";
                body << "            vec2 hi = max(max(s[0], s[1]), s[2]);\n";
                body << "            int x0 = max(int(floor(lo.x - 1.0)),"
                        " 0);\n";
                body << "            int x1 = min(int(ceil(hi.x + 1.0)), "
                     << (scr.w - 1) << ");\n";
                body << "            int y0 = max(int(floor(lo.y - 1.0)),"
                        " 0);\n";
                body << "            int y1 = min(int(ceil(hi.y + 1.0)), "
                     << (scr.h - 1) << ");\n";
                body << "            for (int py = y0; py <= y1; py++) {\n";
                body << "                for (int px = x0; px <= x1;"
                        " px++) {\n";
                body << "                    ivec2 pi = ivec2(px, py);\n";
                body << "                    if (int(texelFetch(u_slt"
                     << id_bind << ", pi, 0).x + 0.5) != f + 1) continue;\n";
                body << "                    " << y_name
                     << " += texelFetch(u_slt" << gy_bind << ", pi, 0)" << swz
                     << ";\n";
                body << "                }\n";
                body << "            }\n";
                body << "        }\n";
                body << "    }\n";
                continue;
            }
            body << "        if (ok) {\n";
            body << "            for (int smp = 0; smp < " << n_samples
                 << "; smp++) {\n";
            body << "                int e = smp % 3;\n";
            body << "                vec2 a = s[e];"
                    " vec2 b = s[(e + 1) % 3];\n";
            body << "                vec2 ev = b - a;"
                    " float len = length(ev);\n";
            body << "                if (len < 1e-6) continue;\n";
            body << "                float u = dressi_hash(uint(f),"
                    " uint(smp), seed, 0u);\n";
            body << "                float o = (dressi_hash(uint(f),"
                    " uint(smp), seed, 1u) * 2.0 - 1.0) * "
                 << FloatLit(radius_px) << ";\n";
            body << "                vec2 p = a + u * ev"
                    " + o * vec2(-ev.y, ev.x) / len;\n";
            body << "                ivec2 pi = ivec2(floor(p));\n";
            body << "                if (pi.x < 0 || pi.y < 0 || pi.x >= "
                 << scr.w << " || pi.y >= " << scr.h << ") continue;\n";
            body << "                if (int(texelFetch(u_slt" << id_bind
                 << ", pi, 0).x + 0.5) != f + 1) continue;\n";
            body << "                " << y_name << " += texelFetch(u_slt"
                 << gy_bind << ", pi, 0)" << swz << ";\n";
            body << "            }\n";
            body << "        }\n";
            body << "    }\n";
            continue;
        }
        if (node->fwd_code.rfind("__soft_clip__", 0) == 0) {
            // xs = {vtx_clip_hard_tex, faces_tex}; output texel = soft
            // vertex (face f = i/3, corner i%3): centroid-scaled clip
            // position (GPU counterpart of BuildSoftGeometry)
            const std::string& code2 = node->fwd_code;
            const float radius_px =
                    std::stof(code2.substr(code2.find("r=") + 2));
            const int scr_w = std::stoi(code2.substr(code2.find("w=") + 2));
            const int scr_h = std::stoi(code2.substr(code2.find("h=") + 2));
            const size_t cl_bind = slt_binding.at(xs[0]);
            const size_t fc_bind = slt_binding.at(xs[1]);
            body << "    vec4 " << y_name << ";\n";
            body << "    {\n";
            body << "        int i = dressi_coord.x;\n";
            body << "        int f = i / 3;\n";
            body << "        int kc = i - f * 3;\n";
            body << "        ivec3 vi = ivec3(texelFetch(u_slt" << fc_bind
                 << ", ivec2(f, 0), 0).xyz + 0.5);\n";
            body << "        vec4 cc[3]; vec2 s[3]; bool ok = true;\n";
            body << "        for (int k = 0; k < 3; k++) {\n";
            body << "            cc[k] = texelFetch(u_slt" << cl_bind
                 << ", ivec2(vi[k], 0), 0);\n";
            body << "            if (cc[k].w <= 1e-6) { ok = false; }\n";
            body << "            else s[k] = (cc[k].xy / cc[k].w * 0.5"
                    " + 0.5) * vec2("
                 << scr_w << ".0, " << scr_h << ".0);\n";
            body << "        }\n";
            body << "        " << y_name << " = cc[kc];\n";
            body << "        if (ok) {\n";
            body << "            vec2 ce = (s[0] + s[1] + s[2]) / 3.0;\n";
            body << "            float dmin = 1e30;\n";
            body << "            for (int k = 0; k < 3; k++) {\n";
            body << "                vec2 a = s[k];"
                    " vec2 b = s[(k + 1) % 3];\n";
            body << "                vec2 e = b - a;"
                    " float len = length(e);\n";
            body << "                if (len < 1e-6) continue;\n";
            body << "                dmin = min(dmin, abs(e.x * (ce.y - a.y)"
                    " - e.y * (ce.x - a.x)) / len);\n";
            body << "            }\n";
            body << "            float kk = dmin < 1e29 ? min(1.0 + "
                 << FloatLit(radius_px)
                 << " / max(dmin, 1e-3), 8.0) : 1.0;\n";
            body << "            vec2 sv = ce + kk * (s[kc] - ce);\n";
            body << "            " << y_name << ".x = (sv.x / " << scr_w
                 << ".0 * 2.0 - 1.0) * cc[kc].w;\n";
            body << "            " << y_name << ".y = (sv.y / " << scr_h
                 << ".0 * 2.0 - 1.0) * cc[kc].w;\n";
            body << "        }\n";
            body << "    }\n";
            continue;
        }
        if (node->fwd_code == "__vertex_neighbor_mean__") {
            // xs = {pos_tex, vtx_neighbors_tex}
            const size_t ps_bind = slt_binding.at(xs[0]);
            const size_t ad_bind = slt_binding.at(xs[1]);
            const uint32_t max_deg = xs[1].getImgSize().h;
            body << "    vec3 " << y_name << ";\n";
            body << "    {\n";
            body << "        int vid = dressi_coord.x;\n";
            body << "        vec3 mean = vec3(0.0); float cnt = 0.0;\n";
            body << "        for (int d = 0; d < " << max_deg
                 << "; d++) {\n";
            body << "            float nv = texelFetch(u_slt" << ad_bind
                 << ", ivec2(vid, d), 0).x;\n";
            body << "            if (nv < -0.5) continue;\n";
            body << "            mean += texelFetch(u_slt" << ps_bind
                 << ", ivec2(int(nv + 0.5), 0), 0).xyz;\n";
            body << "            cnt += 1.0;\n";
            body << "        }\n";
            body << "        " << y_name << " = cnt > 0.5 ? mean / cnt"
                    " : texelFetch(u_slt"
                 << ps_bind << ", ivec2(vid, 0), 0).xyz;\n";
            body << "    }\n";
            continue;
        }
        if (node->fwd_code == "__nc_face_term__") {
            // xs = {pos_tex, faces_tex, face_adj_tex}: d E / d cross(f) of
            // sum over adjacent face pairs of (1 - n_f . n_g)
            const size_t ps_bind = slt_binding.at(xs[0]);
            const size_t fc_bind = slt_binding.at(xs[1]);
            const size_t fa_bind = slt_binding.at(xs[2]);
            body << "    vec3 " << y_name << " = vec3(0.0);\n";
            body << "    {\n";
            body << "        int f = dressi_coord.x;\n";
            body << "        vec3 ca = dressi_face_cross(u_slt" << ps_bind
                 << ", u_slt" << fc_bind << ", f);\n";
            body << "        float la = length(ca);\n";
            body << "        if (la >= 1e-12) {\n";
            body << "            vec3 na = ca / la;\n";
            body << "            vec3 adjf = texelFetch(u_slt" << fa_bind
                 << ", ivec2(f, 0), 0).xyz;\n";
            body << "            for (int j = 0; j < 3; j++) {\n";
            body << "                if (adjf[j] < -0.5) continue;\n";
            body << "                vec3 cb = dressi_face_cross(u_slt"
                 << ps_bind << ", u_slt" << fc_bind
                 << ", int(adjf[j] + 0.5));\n";
            body << "                float lb = length(cb);\n";
            body << "                if (lb < 1e-12) continue;\n";
            body << "                vec3 nb = cb / lb;\n";
            body << "                " << y_name
                 << " += -(nb - dot(na, nb) * na) / la;\n";
            body << "            }\n";
            body << "        }\n";
            body << "    }\n";
            continue;
        }
        if (node->fwd_code == "__nc_vertex_grad__") {
            // xs = {face_term, pos_tex, faces_tex, vtx_faces_tex}: chain
            // the per-face cross-product gradient to the vertices
            const size_t tm_bind = slt_binding.at(xs[0]);
            const size_t ps_bind = slt_binding.at(xs[1]);
            const size_t fc_bind = slt_binding.at(xs[2]);
            const size_t vf_bind = slt_binding.at(xs[3]);
            const uint32_t max_deg = xs[3].getImgSize().h;
            body << "    vec3 " << y_name << " = vec3(0.0);\n";
            body << "    {\n";
            body << "        int vid = dressi_coord.x;\n";
            body << "        for (int d = 0; d < " << max_deg
                 << "; d++) {\n";
            body << "            float fv = texelFetch(u_slt" << vf_bind
                 << ", ivec2(vid, d), 0).x;\n";
            body << "            if (fv < -0.5) continue;\n";
            body << "            int f = int(fv + 0.5);\n";
            body << "            ivec3 vi = ivec3(texelFetch(u_slt"
                 << fc_bind << ", ivec2(f, 0), 0).xyz + 0.5);\n";
            body << "            vec3 p0 = texelFetch(u_slt" << ps_bind
                 << ", ivec2(vi.x, 0), 0).xyz;\n";
            body << "            vec3 e1 = texelFetch(u_slt" << ps_bind
                 << ", ivec2(vi.y, 0), 0).xyz - p0;\n";
            body << "            vec3 e2 = texelFetch(u_slt" << ps_bind
                 << ", ivec2(vi.z, 0), 0).xyz - p0;\n";
            body << "            vec3 g = texelFetch(u_slt" << tm_bind
                 << ", ivec2(f, 0), 0).xyz;\n";
            body << "            vec3 g1 = cross(e2, g);\n";
            body << "            vec3 g2 = cross(g, e1);\n";
            body << "            if (vi.y == vid) " << y_name
                 << " += g1;\n";
            body << "            if (vi.z == vid) " << y_name
                 << " += g2;\n";
            body << "            if (vi.x == vid) " << y_name
                 << " -= g1 + g2;\n";
            body << "        }\n";
            body << "    }\n";
            continue;
        }
        if (node->fwd_code == "__antialias__") {
            // xs = {img, tri_id, vtx_clip, faces}: Eq.3-5 forward blend
            const size_t img_b = slt_binding.at(xs[0]);
            const size_t tri_b = slt_binding.at(xs[1]);
            const size_t cl_b = slt_binding.at(xs[2]);
            const size_t fc_b = slt_binding.at(xs[3]);
            const ImgSize scr = xs[0].getImgSize();
            const uint32_t n = NumComponents(y.getVType());
            const char* swz = SwizzleOf(n);
            const std::string wh = "vec2(" + std::to_string(scr.w) + ".0, " +
                                   std::to_string(scr.h) + ".0)";
            body << "    " << y_type << " " << y_name << ";\n";
            body << "    {\n";
            body << "        " << y_type << " cs = texelFetch(u_slt" << img_b
                 << ", dressi_coord, 0)" << swz << ";\n";
            body << "        " << y_type << " acc = cs;\n";
            body << "        for (int k = 0; k < 8; k++) {\n";
            body << "            ivec2 nc = dressi_coord"
                    " + dressi_aa_offs[k];\n";
            body << "            " << y_type << " cb = cs;\n";
            body << "            if (nc.x >= 0 && nc.y >= 0 && nc.x < "
                 << scr.w << " && nc.y < " << scr.h << ") {\n";
            body << "                DressiAaPair pr;\n";
            body << "                if (dressi_aa_pair(u_slt" << tri_b
                 << ", u_slt" << cl_b << ", u_slt" << fc_b
                 << ", dressi_coord, nc, " << wh << ", pr))\n";
            body << "                    cb = pr.r * cs + (1.0 - pr.r)"
                    " * texelFetch(u_slt"
                 << img_b << ", nc, 0)" << swz << ";\n";
            body << "            }\n";
            body << "            acc += cb;\n";
            body << "        }\n";
            body << "        " << y_name << " = acc * (1.0 / 9.0);\n";
            body << "    }\n";
            continue;
        }
        if (node->fwd_code == "__antialias_bwd_img__") {
            // xs = {gy, tri_id, vtx_clip, faces}: c_aa is linear in the
            // image; alpha from (s,n) pairs, beta from swapped (n,s) pairs
            const size_t gy_b = slt_binding.at(xs[0]);
            const size_t tri_b = slt_binding.at(xs[1]);
            const size_t cl_b = slt_binding.at(xs[2]);
            const size_t fc_b = slt_binding.at(xs[3]);
            const ImgSize scr = xs[0].getImgSize();
            const uint32_t n = NumComponents(y.getVType());
            const char* swz = SwizzleOf(n);
            const std::string wh = "vec2(" + std::to_string(scr.w) + ".0, " +
                                   std::to_string(scr.h) + ".0)";
            const std::string pair_args = "u_slt" + std::to_string(tri_b) +
                                          ", u_slt" + std::to_string(cl_b) +
                                          ", u_slt" + std::to_string(fc_b);
            body << "    " << y_type << " " << y_name << ";\n";
            body << "    {\n";
            body << "        " << y_type << " g = texelFetch(u_slt" << gy_b
                 << ", dressi_coord, 0)" << swz << ";\n";
            body << "        " << y_type << " acc = g;\n";
            body << "        float asum = 0.0;\n";
            body << "        for (int k = 0; k < 8; k++) {\n";
            body << "            ivec2 nc = dressi_coord"
                    " + dressi_aa_offs[k];\n";
            body << "            if (nc.x < 0 || nc.y < 0 || nc.x >= "
                 << scr.w << " || nc.y >= " << scr.h
                 << ") { asum += 1.0; continue; }\n";
            body << "            DressiAaPair pr;\n";
            body << "            asum += dressi_aa_pair(" << pair_args
                 << ", dressi_coord, nc, " << wh << ", pr) ? pr.r : 1.0;\n";
            body << "            DressiAaPair pr2;\n";
            body << "            if (dressi_aa_pair(" << pair_args
                 << ", nc, dressi_coord, " << wh << ", pr2))\n";
            body << "                acc += (1.0 - pr2.r)"
                    " * texelFetch(u_slt"
                 << gy_b << ", nc, 0)" << swz << ";\n";
            body << "        }\n";
            body << "        " << y_name
                 << " = (acc + asum * g) * (1.0 / 9.0);\n";
            body << "    }\n";
            continue;
        }
        if (node->fwd_code.rfind("__antialias_bwd_vtx__", 0) == 0) {
            // xs = {gy, img, tri_id, vtx_clip, faces, vtx_faces, seed};
            // output texel = vertex. Accumulates
            // (1/9) dot(gy, cs - cn) * d r / d clip over active boundary
            // pairs whose edge touches vid -- exactly (n=0: incident-face
            // bbox scan) or stochastically (n>0: jittered edge samples,
            // the paper's inverse-table philosophy).
            const int aa_samples = std::stoi(node->fwd_code.substr(
                    node->fwd_code.find("n=") + 2));
            const bool aa_wide =
                    node->fwd_code.find("wide=1") != std::string::npos;
            const size_t gy_b = slt_binding.at(xs[0]);
            const size_t img_b = slt_binding.at(xs[1]);
            const size_t tri_b = slt_binding.at(xs[2]);
            const size_t cl_b = slt_binding.at(xs[3]);
            const size_t fc_b = slt_binding.at(xs[4]);
            const ImgSize scr = xs[0].getImgSize();
            const uint32_t n = NumComponents(xs[0].getVType());
            const char* swz = SwizzleOf(n);
            const char* img_type = GlslTypeName(xs[0].getVType());
            const std::string wh = "vec2(" + std::to_string(scr.w) + ".0, " +
                                   std::to_string(scr.h) + ".0)";
            const std::string pair_args = "u_slt" + std::to_string(tri_b) +
                                          ", u_slt" + std::to_string(cl_b) +
                                          ", u_slt" + std::to_string(fc_b);
            const size_t vf_b = slt_binding.at(xs[5]);
            const uint32_t max_deg = xs[5].getImgSize().h;
            body << "    vec4 " << y_name << " = vec4(0.0);\n";
            body << "    {\n";
            body << "        int vid = dressi_coord.x;\n";
            body << "        vec4 vc = texelFetch(u_slt" << cl_b
                 << ", ivec2(vid, 0), 0);\n";
            if (aa_samples == 0) {
                // Exact: scan the incident faces' hard bboxes padded by
                // the 8-neighborhood reach
                body << "        vec2 bbmin = vec2(1e30);"
                        " vec2 bbmax = vec2(-1e30);\n";
                body << "        for (int d = 0; d < " << max_deg
                     << "; d++) {\n";
                body << "            float fv = texelFetch(u_slt" << vf_b
                     << ", ivec2(vid, d), 0).x;\n";
                body << "            if (fv < -0.5) continue;\n";
                body << "            ivec3 bvi = ivec3(texelFetch(u_slt"
                     << fc_b << ", ivec2(int(fv + 0.5), 0), 0).xyz"
                        " + 0.5);\n";
                body << "            for (int k = 0; k < 3; k++) {\n";
                body << "                vec4 c = texelFetch(u_slt" << cl_b
                     << ", ivec2(bvi[k], 0), 0);\n";
                body << "                if (abs(c.w) < 1e-6) continue;\n";
                body << "                vec2 sv = (c.xy / c.w * 0.5"
                        " + 0.5) * "
                     << wh << ";\n";
                body << "                bbmin = min(bbmin, sv);"
                        " bbmax = max(bbmax, sv);\n";
                body << "            }\n";
                body << "        }\n";
                body << "        int px0 = clamp(int(floor(bbmin.x - 3.0)),"
                        " 0, "
                     << (scr.w - 1) << ");\n";
                body << "        int py0 = clamp(int(floor(bbmin.y - 3.0)),"
                        " 0, "
                     << (scr.h - 1) << ");\n";
                body << "        int px1 = clamp(int(ceil(bbmax.x + 2.0)),"
                        " 0, "
                     << (scr.w - 1) << ");\n";
                body << "        int py1 = clamp(int(ceil(bbmax.y + 2.0)),"
                        " 0, "
                     << (scr.h - 1) << ");\n";
                body << "        if (bbmin.x > bbmax.x) px1 = -1;\n";
                body << "        for (int py = py0; py <= py1; py++)\n";
                body << "        for (int px = px0; px <= px1; px++) {\n";
                body << "            ivec2 sc = ivec2(px, py);\n";
            } else {
                // Stochastic: jittered samples along the incident faces'
                // edges (per-face sample positions shared across vertices;
                // flattened to a single loop to keep the shared tail).
                // Wide mode handles ONE incident face per fragment (row =
                // face slot) so the thread count scales with max_deg.
                const size_t sd_b = slt_binding.at(xs[6]);
                body << "        uint seed = uint(texelFetch(u_slt" << sd_b
                     << ", ivec2(0, 0), 0).x + 0.5);\n";
                if (aa_wide) {
                    body << "        int d = dressi_coord.y;\n";
                    body << "        for (int smp = 0; smp < " << aa_samples
                         << "; smp++) {\n";
                } else {
                    body << "        for (int it = 0; it < "
                         << (max_deg * uint32_t(aa_samples)) << "; it++) {\n";
                    body << "            int d = it / " << aa_samples
                         << ";\n";
                    body << "            int smp = it - d * " << aa_samples
                         << ";\n";
                }
                body << "            float fv = texelFetch(u_slt" << vf_b
                     << ", ivec2(vid, d), 0).x;\n";
                body << "            if (fv < -0.5) continue;\n";
                body << "            int f = int(fv + 0.5);\n";
                body << "            ivec3 bvi = ivec3(texelFetch(u_slt"
                     << fc_b << ", ivec2(f, 0), 0).xyz + 0.5);\n";
                body << "            vec2 s3[3]; bool okf = true;\n";
                body << "            for (int k = 0; k < 3; k++) {\n";
                body << "                vec4 c = texelFetch(u_slt" << cl_b
                     << ", ivec2(bvi[k], 0), 0);\n";
                body << "                if (c.w <= 1e-6)"
                        " { okf = false; }\n";
                body << "                else s3[k] = (c.xy / c.w * 0.5"
                        " + 0.5) * "
                     << wh << ";\n";
                body << "            }\n";
                body << "            if (!okf) continue;\n";
                body << "            int e = smp % 3;\n";
                body << "            vec2 a = s3[e];"
                        " vec2 b = s3[(e + 1) % 3];\n";
                body << "            vec2 ev = b - a;"
                        " float len = length(ev);\n";
                body << "            if (len < 1e-6) continue;\n";
                body << "            float u = dressi_hash(uint(f),"
                        " uint(smp), seed, 2u);\n";
                body << "            float o = (dressi_hash(uint(f),"
                        " uint(smp), seed, 3u) * 2.0 - 1.0) * 2.0;\n";
                body << "            ivec2 sc = ivec2(floor(a + u * ev"
                        " + o * vec2(-ev.y, ev.x) / len));\n";
                body << "            if (sc.x < 0 || sc.y < 0 || sc.x >= "
                     << scr.w << " || sc.y >= " << scr.h << ") continue;\n";
            }
            body << "            for (int k = 0; k < 8; k++) {\n";
            body << "                ivec2 nc = sc + dressi_aa_offs[k];\n";
            body << "                if (nc.x < 0 || nc.y < 0 || nc.x >= "
                 << scr.w << " || nc.y >= " << scr.h << ") continue;\n";
            body << "                DressiAaPair pr;\n";
            body << "                if (!dressi_aa_pair(" << pair_args
                 << ", sc, nc, " << wh << ", pr)) continue;\n";
            body << "                if (pr.ia != vid && pr.ib != vid)"
                    " continue;\n";
            body << "                if (pr.r >= 1.0) continue;\n";
            body << "                " << img_type
                 << " cs = texelFetch(u_slt" << img_b << ", sc, 0)" << swz
                 << ";\n";
            body << "                " << img_type
                 << " cn = texelFetch(u_slt" << img_b << ", nc, 0)" << swz
                 << ";\n";
            body << "                " << img_type
                 << " g = texelFetch(u_slt" << gy_b << ", sc, 0)" << swz
                 << ";\n";
            if (n == 1) {
                body << "                float gr = g * (cs - cn) / 9.0;\n";
            } else {
                body << "                float gr = dot(g, cs - cn)"
                        " / 9.0;\n";
            }
            body << "                if (gr == 0.0) continue;\n";
            body << "                vec2 ps = vec2(sc) + 0.5;\n";
            body << "                float sgn ="
                    " pr.es >= 0.0 ? 1.0 : -1.0;\n";
            body << "                float invl = 1.0 / pr.len;\n";
            body << "                float l3 = abs(pr.es)"
                    " * invl * invl * invl;\n";
            body << "                vec2 e = pr.qb - pr.qa;\n";
            body << "                vec2 dqa = sgn * invl"
                    " * vec2(pr.qb.y - ps.y, ps.x - pr.qb.x) + l3 * e;\n";
            body << "                vec2 dqb = sgn * invl"
                    " * vec2(ps.y - pr.qa.y, pr.qa.x - ps.x) - l3 * e;\n";
            body << "                for (int e2 = 0; e2 < 2; e2++) {\n";
            body << "                    int id2 = e2 == 0 ? pr.ia"
                    " : pr.ib;\n";
            body << "                    if (id2 != vid) continue;\n";
            body << "                    vec2 gq = gr"
                    " * (e2 == 0 ? dqa : dqb);\n";
            body << "                    " << y_name << ".x += gq.x * 0.5 * "
                 << scr.w << ".0 / vc.w;\n";
            body << "                    " << y_name << ".y += gq.y * 0.5 * "
                 << scr.h << ".0 / vc.w;\n";
            body << "                    " << y_name
                 << ".w += -(gq.x * 0.5 * " << scr.w
                 << ".0 * vc.x + gq.y * 0.5 * " << scr.h
                 << ".0 * vc.y) / (vc.w * vc.w);\n";
            body << "                }\n";
            body << "            }\n";
            body << "        }\n";
            body << "    }\n";
            continue;
        }
        if (node->fwd_code == "__col_sum__") {
            // Per-column sum over the input's rows: {W,H} -> {W,1}
            const size_t x_b = slt_binding.at(xs[0]);
            const uint32_t rows = xs[0].getImgSize().h;
            const uint32_t n = NumComponents(y.getVType());
            const char* swz = SwizzleOf(n);
            body << "    " << y_type << " " << y_name << " = " << y_type
                 << "(0.0);\n";
            body << "    for (int r = 0; r < " << rows << "; r++) {\n";
            body << "        " << y_name << " += texelFetch(u_slt" << x_b
                 << ", ivec2(dressi_coord.x, r), 0)" << swz << ";\n";
            body << "    }\n";
            continue;
        }
        if (node->fwd_code.rfind("__gather_dist_grad__", 0) == 0) {
            // xs = {gy_screen, raster_out, vtx_clip_tex, faces_tex,
            // vtx_faces_tex}; output texel = hard vertex. Gathers
            // gy.x * d dist / d clip over every covered pixel of faces
            // containing this vertex (the scatter-free formulation of the
            // HardSoftRas backward). The pixel scan is bounded by the exact
            // soft-triangle bbox of the vertex's incident faces (from the
            // precomputed adjacency texture; same centroid-scaling formula
            // as the CPU-side BuildSoftGeometry, radius from the marker),
            // padded 1 px.
            const float radius_px = std::stof(node->fwd_code.substr(
                    std::strlen("__gather_dist_grad__ r=")));
            const bool gd_wide =
                    node->fwd_code.find("wide=1") != std::string::npos;
            const size_t gy_bind = slt_binding.at(xs[0]);
            const size_t ro_bind = slt_binding.at(xs[1]);
            const size_t cl_bind = slt_binding.at(xs[2]);
            const size_t fc_bind = slt_binding.at(xs[3]);
            const size_t vf_bind = slt_binding.at(xs[4]);
            const ImgSize scr = xs[1].getImgSize();
            const uint32_t max_deg = xs[4].getImgSize().h;
            const std::string wh = "vec2(" + std::to_string(scr.w) + ".0, " +
                                   std::to_string(scr.h) + ".0)";
            body << "    vec4 " << y_name << " = vec4(0.0);\n";
            body << "    {\n";
            body << "        int vid = dressi_coord.x;\n";
            body << "        vec4 vc = texelFetch(u_slt" << cl_bind
                 << ", ivec2(vid, 0), 0);\n";
            // Per-vertex scan bbox over the soft triangles of incident
            // faces; wide mode handles ONE incident face per fragment
            // (row = adjacency slot) with an exact face-id pixel guard
            body << "        vec2 bbmin = vec2(1e30);"
                    " vec2 bbmax = vec2(-1e30);\n";
            if (gd_wide) {
                body << "        int gface = -1;\n";
                body << "        { int d = dressi_coord.y; {\n";
            } else {
                body << "        for (int d = 0; d < " << max_deg
                     << "; d++) {\n";
            }
            body << "            float fv = texelFetch(u_slt" << vf_bind
                 << ", ivec2(vid, d), 0).x;\n";
            if (gd_wide) {
                body << "            if (fv >= -0.5) {\n";
                body << "            gface = int(fv + 0.5);\n";
            } else {
                body << "            if (fv < -0.5) continue;\n";
            }
            body << "            ivec3 bvi = ivec3(texelFetch(u_slt"
                 << fc_bind << ", ivec2(int(fv + 0.5), 0), 0).xyz + 0.5);\n";
            body << "            vec2 bs[3]; bool bok = true;\n";
            body << "            for (int k = 0; k < 3; k++) {\n";
            body << "                vec4 c = texelFetch(u_slt" << cl_bind
                 << ", ivec2(bvi[k], 0), 0);\n";
            body << "                if (c.w < 1e-6)"
                    " { bok = false; break; }\n";
            body << "                bs[k] = (c.xy / c.w * 0.5 + 0.5) * "
                 << wh << ";\n";
            body << "            }\n";
            if (gd_wide) {
                body << "            if (bok) {\n";
            } else {
                body << "            if (!bok) continue;\n";
            }
            body << "            vec2 ce = (bs[0] + bs[1] + bs[2]) / 3.0;\n";
            body << "            float dmin = 1e30;\n";
            body << "            for (int k = 0; k < 3; k++) {\n";
            body << "                vec2 a = bs[k];"
                    " vec2 b = bs[(k + 1) % 3];\n";
            body << "                vec2 e = b - a;"
                    " float len = length(e);\n";
            body << "                if (len < 1e-6) continue;\n";
            body << "                dmin = min(dmin, abs(e.x * (ce.y - a.y)"
                    " - e.y * (ce.x - a.x)) / len);\n";
            body << "            }\n";
            body << "            float kk = dmin < 1e29 ? min(1.0 + "
                 << FloatLit(radius_px)
                 << " / max(dmin, 1e-3), 8.0) : 1.0;\n";
            body << "            for (int k = 0; k < 3; k++) {\n";
            body << "                vec2 sv = ce + kk * (bs[k] - ce);\n";
            body << "                bbmin = min(bbmin, sv);"
                    " bbmax = max(bbmax, sv);\n";
            body << "            }\n";
            if (gd_wide) {
                body << "            } } } }\n";  // bok / fv / d scopes
            } else {
                body << "        }\n";
            }
            body << "        int px0 = clamp(int(floor(bbmin.x - 1.5)), 0, "
                 << (scr.w - 1) << ");\n";
            body << "        int py0 = clamp(int(floor(bbmin.y - 1.5)), 0, "
                 << (scr.h - 1) << ");\n";
            body << "        int px1 = clamp(int(ceil(bbmax.x + 0.5)), 0, "
                 << (scr.w - 1) << ");\n";
            body << "        int py1 = clamp(int(ceil(bbmax.y + 0.5)), 0, "
                 << (scr.h - 1) << ");\n";
            body << "        if (bbmin.x > bbmax.x) px1 = -1;\n";
            body << "        for (int py = py0; py <= py1; py++)\n";
            body << "        for (int px = px0; px <= px1; px++) {\n";
            body << "            ivec2 sp = ivec2(px, py);\n";
            body << "            vec4 ro = texelFetch(u_slt" << ro_bind
                 << ", sp, 0);\n";
            body << "            if (ro.w < 0.5) continue;\n";
            if (gd_wide) {
                // Exact split: the pixel belongs to exactly one face, so
                // only the (vid, d)-slot holding that face accumulates it
                body << "            if (int(ro.y + 0.5) != gface)"
                        " continue;\n";
            }
            body << "            float g = texelFetch(u_slt" << gy_bind
                 << ", sp, 0).x;\n";
            body << "            if (g == 0.0) continue;\n";
            body << "            vec3 fidx = texelFetch(u_slt" << fc_bind
                 << ", ivec2(int(ro.y + 0.5), 0), 0).xyz;\n";
            body << "            ivec3 vi = ivec3(fidx + 0.5);\n";
            body << "            if (vid != vi.x && vid != vi.y"
                    " && vid != vi.z) continue;\n";
            body << "            vec2 s[3];\n";
            body << "            for (int k = 0; k < 3; k++) {\n";
            body << "                vec4 c = texelFetch(u_slt" << cl_bind
                 << ", ivec2(vi[k], 0), 0);\n";
            body << "                s[k] = (c.xy / c.w * 0.5 + 0.5) * vec2("
                 << scr.w << ".0, " << scr.h << ".0);\n";
            body << "            }\n";
            body << "            vec2 p = vec2(sp) + 0.5;\n";
            // Argmin edge of the signed distance (same as the forward)
            body << "            float best_d2 = -1.0; int be = 0;"
                    " float bt = 0.0; int npos = 0; int nneg = 0;\n";
            body << "            for (int k = 0; k < 3; k++) {\n";
            body << "                vec2 a = s[k];"
                    " vec2 b = s[(k + 1) % 3];\n";
            body << "                vec2 e = b - a;"
                    " float len2 = dot(e, e);\n";
            body << "                float t = len2 > 1e-12 ?"
                    " clamp(dot(p - a, e) / len2, 0.0, 1.0) : 0.0;\n";
            body << "                vec2 q = a + t * e;"
                    " float d2 = dot(p - q, p - q);\n";
            body << "                if (best_d2 < 0.0 || d2 < best_d2)"
                    " { best_d2 = d2; be = k; bt = t; }\n";
            body << "                float cr = e.x * (p.y - a.y)"
                    " - e.y * (p.x - a.x);\n";
            body << "                if (cr >= 0.0) npos++;\n";
            body << "                if (cr <= 0.0) nneg++;\n";
            body << "            }\n";
            body << "            float d = sqrt(max(best_d2, 0.0));\n";
            body << "            if (d < 1e-6) continue;\n";
            body << "            float sgn ="
                    " (npos == 3 || nneg == 3) ? 1.0 : -1.0;\n";
            body << "            int ia = be; int ib = (be + 1) % 3;\n";
            body << "            vec2 q = s[ia]"
                    " + bt * (s[ib] - s[ia]);\n";
            body << "            vec2 n = (p - q) / d;\n";
            // d dist / d s[ia] = -sgn (1-t) n, d dist / d s[ib] = -sgn t n
            body << "            for (int e2 = 0; e2 < 2; e2++) {\n";
            body << "                int k = e2 == 0 ? ia : ib;\n";
            body << "                if (vi[k] != vid) continue;\n";
            body << "                float wgt = e2 == 0"
                    " ? (1.0 - bt) : bt;\n";
            body << "                vec2 gs = g * (-sgn * wgt) * n;\n";
            body << "                " << y_name << ".x += gs.x * 0.5 * "
                 << scr.w << ".0 / vc.w;\n";
            body << "                " << y_name << ".y += gs.y * 0.5 * "
                 << scr.h << ".0 / vc.w;\n";
            body << "                " << y_name << ".w += -(gs.x * 0.5 * "
                 << scr.w << ".0 * vc.x + gs.y * 0.5 * " << scr.h
                 << ".0 * vc.y) / (vc.w * vc.w);\n";
            body << "            }\n";
            body << "        }\n";
            body << "    }\n";
            continue;
        }

        std::string snippet = node->fwd_code;
        snippet = ReplaceAll(snippet, "{y}", y_name);
        // Component-wise ops (default inference) accept scalar inputs by
        // broadcasting, but GLSL builtins like pow() do not; promote scalar
        // inputs to the output vector type explicitly.
        const bool promote = !node->infer && NumComponents(y.getVType()) > 1;
        for (size_t i = 0; i < xs.size(); i++) {
            // {sN}: the sampler bound for a Sampled input
            if (auto it = tex_binding.find(xs[i]); it != tex_binding.end()) {
                snippet = ReplaceAll(snippet, "{s" + std::to_string(i) + "}",
                                     "u_tex" + std::to_string(it->second));
                continue;
            }
            std::string expr = input_expr(xs[i]);
            if (promote && NumComponents(xs[i].getVType()) == 1) {
                expr = std::string(y_type) + "(" + expr + ")";
            }
            snippet = ReplaceAll(snippet, "{x" + std::to_string(i) + "}",
                                 expr);
        }
        body << "    " << y_type << " " << y_name << "; " << snippet << "\n";
    }

    // Padded output writes
    for (size_t i = 0; i < ss.out_vars.size(); i++) {
        const Variable& v = ss.out_vars[i];
        const uint32_t n = NumComponents(v.getVType());
        const uint32_t phys = PhysChannels(v.getVType());
        const std::string src = names.at(v);
        if (comp) {
            // imageStore always takes a vec4; extra components are ignored
            // by narrower formats
            body << "    imageStore(u_out" << i << ", dressi_coord, ";
            if (n == 1) {
                body << "vec4(" << src << ", 0.0, 0.0, 0.0)";
            } else if (n == 4) {
                body << src;
            } else {
                body << "vec4(" << src << (n == 2 ? ", 0.0, 0.0)" : ", 0.0)");
            }
            body << ");\n";
        } else if (n == phys) {
            body << "    o_" << i << " = " << src << ";\n";
        } else {  // VEC3 -> vec4 padding
            body << "    o_" << i << " = vec4(" << src << ", 0.0);\n";
        }
    }
    body << "}\n";

    return head.str() + body.str();
}

}  // namespace

std::string GenerateFragShader(const SubStage& ss) {
    return GenerateShaderImpl(ss, /*comp=*/false);
}

std::string GenerateCompShader(const SubStage& ss) {
    return GenerateShaderImpl(ss, /*comp=*/true);
}

RasterShaders GenerateRasterShaders(const SubStage& ss) {
    // A raster substage holds one RASTER function (its earliest, by
    // creation id) optionally followed by fused FRAG functions (the
    // paper's raster-headed packing). The vertex shader is a passthrough;
    // the fragment shader is generated by the common FRAG codegen, which
    // special-cases the rasterization markers.
    DRESSI_CHECK(ss.shader_type == RASTER && !ss.funcs.empty(),
                 "GenerateRasterShaders: RASTER substage expected");
    DRESSI_CHECK(ss.vtx_vars.size() == 3,
                 "RASTER substage expects {pos, attrib, faces}");
    const uint32_t attr_n = NumComponents(ss.vtx_vars[1].getVType());
    const std::string attr_type =
            attr_n == 1 ? "float" : ("vec" + std::to_string(attr_n));

    RasterShaders shaders;
    {
        std::ostringstream vs;
        vs << "#version 450\n";
        vs << "layout(location=0) in vec4 a_pos;\n";
        vs << "layout(location=1) in " << attr_type << " a_attr;\n";
        vs << "layout(location=0) out " << attr_type << " v_attr;\n";
        vs << "void main() {\n";
        vs << "    gl_Position = a_pos;\n";
        vs << "    v_attr = a_attr;\n";
        vs << "}\n";
        shaders.vert = vs.str();
    }
    shaders.frag = GenerateFragShader(ss);
    return shaders;
}

}  // namespace dressi
