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

std::string GenerateFragShader(const SubStage& ss) {
    DRESSI_CHECK(ss.shader_type == FRAG, "GenerateFragShader: FRAG only");
    DRESSI_CHECK(ss.uif_vars.empty() && ss.vtx_vars.empty(),
                 "Codegen supports inp/tex/slt inputs only");

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

    // Binding convention shared with the executor: inp first, then slt
    for (size_t i = 0; i < ss.inp_vars.size(); i++) {
        const Variable& v = ss.inp_vars[i];
        DRESSI_CHECK(!IsIntVType(v.getVType()),
                     "Int images are not supported on GPU yet (M1)");
        head << "layout(input_attachment_index=" << i << ", set=0, binding="
             << i << ") uniform subpassInput u_inp" << i << ";\n";
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

    // Outputs: padded float attachments
    for (size_t i = 0; i < ss.out_vars.size(); i++) {
        const uint32_t phys = PhysChannels(ss.out_vars[i].getVType());
        DRESSI_CHECK(ss.out_vars[i].getImgSize() == ss.img_size,
                     "Substage outputs must share the image size");
        head << "layout(location=" << i << ") out "
             << (phys == 1 ? "float" : ("vec" + std::to_string(phys)))
             << " o_" << i << ";\n";
    }

    body << "void main() {\n";
    body << "    ivec2 dressi_coord = ivec2(gl_FragCoord.xy);\n";

    // Which slt inputs are consumed by coordinate-generic loads (special ops
    // fetch their input themselves)
    const auto is_special = [](const Function& f) {
        const std::string& code = NodeAccess::Node(f)->fwd_code;
        return code == "__reduce_2x2_sum__" ||
               code == "__upsample_nearest_2x__" ||
               code == "__gather_inv_uv__" ||
               code == "__gather_dist_grad__";
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
        for (const auto& x : f.getInputVars()) {
            if (slt_binding.count(x) && !special && !tex_binding.count(x)) {
                needs_generic_load[x] = true;
            }
        }
    }

    // Loads for input attachments (same-pixel, same-size by construction)
    for (size_t i = 0; i < ss.inp_vars.size(); i++) {
        const Variable& v = ss.inp_vars[i];
        const uint32_t n = NumComponents(v.getVType());
        body << "    " << GlslTypeName(v.getVType()) << " " << local_name(v)
             << " = subpassLoad(u_inp" << i << ")" << SwizzleOf(n) << ";\n";
    }

    // Generic loads for slt inputs
    for (size_t i = 0; i < ss.slt_vars.size(); i++) {
        const Variable& v = ss.slt_vars[i];
        if (!needs_generic_load.count(v)) {
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

        if (node->fwd_code == "__gather_dist_grad__") {
            // xs = {gy_screen, raster_out, vtx_clip_tex, faces_tex}; output
            // texel = hard vertex. Gathers gy.x * d dist / d clip over every
            // covered pixel of faces containing this vertex (the scatter-
            // free formulation of the HardSoftRas backward).
            const size_t gy_bind = slt_binding.at(xs[0]);
            const size_t ro_bind = slt_binding.at(xs[1]);
            const size_t cl_bind = slt_binding.at(xs[2]);
            const size_t fc_bind = slt_binding.at(xs[3]);
            const ImgSize scr = xs[1].getImgSize();
            body << "    vec4 " << y_name << " = vec4(0.0);\n";
            body << "    {\n";
            body << "        int vid = dressi_coord.x;\n";
            body << "        vec4 vc = texelFetch(u_slt" << cl_bind
                 << ", ivec2(vid, 0), 0);\n";
            body << "        for (int py = 0; py < " << scr.h
                 << "; py++)\n";
            body << "        for (int px = 0; px < " << scr.w
                 << "; px++) {\n";
            body << "            ivec2 sp = ivec2(px, py);\n";
            body << "            vec4 ro = texelFetch(u_slt" << ro_bind
                 << ", sp, 0);\n";
            body << "            if (ro.w < 0.5) continue;\n";
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
        if (n == phys) {
            body << "    o_" << i << " = " << src << ";\n";
        } else {  // VEC3 -> vec4 padding
            body << "    o_" << i << " = vec4(" << src << ", 0.0);\n";
        }
    }
    body << "}\n";

    return head.str() + body.str();
}

RasterShaders GenerateRasterShaders(const SubStage& ss) {
    DRESSI_CHECK(ss.shader_type == RASTER && ss.funcs.size() == 1,
                 "RASTER substage must hold exactly one Rasterize function");
    const Function& func = ss.funcs[0];
    const Variable y = func.getOutputVar();
    const std::string& code = NodeAccess::Node(func)->fwd_code;
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

    if (code.rfind("__rasterize_soft__", 0) == 0) {
        // HardSoftRas forward: v_attr carries the hard face index; the hard
        // triangle is fetched from the clip/faces textures, the signed
        // screen-space distance to it is the differentiable output and the
        // Eq.3 depth shift goes to gl_FragDepth.
        constexpr const char* kPrefix = "__rasterize_soft__ r=";
        const float radius_px = std::stof(code.substr(std::strlen(kPrefix)));
        DRESSI_CHECK(radius_px > 0.f,
                     "__rasterize_soft__ marker must carry the radius");
        const Variables xs = func.getInputVars();
        std::map<Variable, size_t> slt_binding;
        for (size_t i = 0; i < ss.slt_vars.size(); i++) {
            slt_binding[ss.slt_vars[i]] = i;
        }
        const size_t clip_bind = slt_binding.at(xs[3]);
        const size_t faces_bind = slt_binding.at(xs[4]);
        const ImgSize screen = y.getImgSize();

        std::ostringstream fs;
        fs << "#version 450\n";
        fs << "layout(location=0) in float v_attr;\n";
        fs << "layout(location=0) out vec4 o_0;\n";
        for (size_t i = 0; i < ss.slt_vars.size(); i++) {
            fs << "layout(set=0, binding=" << i
               << ") uniform sampler2D u_slt" << i << ";\n";
        }
        fs << "void main() {\n";
        fs << "    int f = int(v_attr + 0.5);\n";
        fs << "    vec3 fidx = texelFetch(u_slt" << faces_bind
           << ", ivec2(f, 0), 0).xyz;\n";
        fs << "    vec2 s[3]; float z[3];\n";
        fs << "    for (int k = 0; k < 3; k++) {\n";
        fs << "        vec4 c = texelFetch(u_slt" << clip_bind
           << ", ivec2(int(fidx[k] + 0.5), 0), 0);\n";
        fs << "        s[k] = (c.xy / c.w * 0.5 + 0.5) * vec2(" << screen.w
           << ".0, " << screen.h << ".0);\n";
        fs << "        z[k] = c.z / c.w;\n";
        fs << "    }\n";
        fs << "    vec2 p = gl_FragCoord.xy;\n";
        // Signed distance to the hard triangle (same algorithm as
        // SignedDistToTri in cpu_raster.cpp)
        fs << "    float best_d2 = -1.0; int npos = 0; int nneg = 0;\n";
        fs << "    for (int k = 0; k < 3; k++) {\n";
        fs << "        vec2 a = s[k]; vec2 b = s[(k + 1) % 3];\n";
        fs << "        vec2 e = b - a; float len2 = dot(e, e);\n";
        fs << "        float t = len2 > 1e-12 ?"
              " clamp(dot(p - a, e) / len2, 0.0, 1.0) : 0.0;\n";
        fs << "        vec2 q = a + t * e; float d2 = dot(p - q, p - q);\n";
        fs << "        if (best_d2 < 0.0 || d2 < best_d2) best_d2 = d2;\n";
        fs << "        float cr = e.x * (p.y - a.y) - e.y * (p.x - a.x);\n";
        fs << "        if (cr >= 0.0) npos++;\n";
        fs << "        if (cr <= 0.0) nneg++;\n";
        fs << "    }\n";
        fs << "    float sgn = (npos == 3 || nneg == 3) ? 1.0 : -1.0;\n";
        fs << "    float dist = sgn * sqrt(max(best_d2, 0.0));\n";
        // Hard depth via (unclamped) screen barycentrics: NDC z is affine
        fs << "    float area = (s[1].x - s[0].x) * (s[2].y - s[0].y)"
              " - (s[1].y - s[0].y) * (s[2].x - s[0].x);\n";
        fs << "    float hard_z = 0.5;\n";
        fs << "    if (abs(area) > 1e-9) {\n";
        fs << "        float l0 = ((s[1].x - p.x) * (s[2].y - p.y)"
              " - (s[1].y - p.y) * (s[2].x - p.x)) / area;\n";
        fs << "        float l1 = ((s[2].x - p.x) * (s[0].y - p.y)"
              " - (s[2].y - p.y) * (s[0].x - p.x)) / area;\n";
        fs << "        hard_z = l0 * z[0] + l1 * z[1]"
              " + (1.0 - l0 - l1) * z[2];\n";
        fs << "    }\n";
        fs << "    gl_FragDepth = dist >= 0.0"
              " ? 0.5 * clamp(hard_z, 0.0, 1.0)"
              " : 0.5 + 0.5 * clamp(-dist / "
           << FloatLit(radius_px) << ", 0.0, 1.0);\n";
        fs << "    o_0 = vec4(dist, v_attr, hard_z, 1.0);\n";
        fs << "}\n";
        shaders.frag = fs.str();
        return shaders;
    }

    DRESSI_CHECK(code == "__rasterize__",
                 "Unknown RASTER marker: " + code);
    {
        const uint32_t n = NumComponents(y.getVType());
        const uint32_t phys = PhysChannels(y.getVType());
        const std::string out_type =
                phys == 1 ? "float" : ("vec" + std::to_string(phys));
        std::ostringstream fs;
        fs << "#version 450\n";
        fs << "layout(location=0) in " << attr_type << " v_attr;\n";
        fs << "layout(location=0) out " << out_type << " o_0;\n";
        fs << "void main() {\n";
        if (n == phys) {
            fs << "    o_0 = v_attr;\n";
        } else {  // VEC3 -> vec4 padding
            fs << "    o_0 = vec4(v_attr, 0.0);\n";
        }
        fs << "}\n";
        shaders.frag = fs.str();
    }
    return shaders;
}

}  // namespace dressi
