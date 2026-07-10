#include <algorithm>

#include "core/node.h"
#include "pack/substage.h"

namespace dressi {

bool IsInlineConst(const Variable& var) {
    return var && NodeAccess::Node(var)->const_val.has_value();
}

SubStages TrivialPackSubStages(const Functions& all_funcs) {
    SubStages substages;
    for (const auto& func : all_funcs) {
        Variable y = func.getOutputVar();
        if (IsInlineConst(y)) {
            continue;  // constants are inlined into consumer shaders
        }
        const ShaderType type = func.getShaderType();
        DRESSI_CHECK(type == FRAG || type == RASTER,
                     "COMP functions are not executable yet");

        SubStage ss;
        ss.funcs = {func};
        ss.out_vars = {y};
        ss.gen_vars = {y};
        ss.shader_type = type;
        ss.img_size = y.getImgSize();
        const Variables xs = func.getInputVars();
        const auto& access = NodeAccess::Node(func)->input_access;
        for (size_t i = 0; i < xs.size(); i++) {
            const Variable& x = xs[i];
            if (IsInlineConst(x)) {
                continue;
            }
            if (type == RASTER) {
                if (!access.empty() && access[i] == InputAccess::TexelFetch) {
                    if (std::find(ss.slt_vars.begin(), ss.slt_vars.end(),
                                  x) == ss.slt_vars.end()) {
                        ss.slt_vars.push_back(x);  // raster FS texelFetch
                    }
                } else {
                    ss.vtx_vars.push_back(x);  // ordered: pos, attrib, faces
                }
            } else if (!access.empty() &&
                       access[i] == InputAccess::Sampled) {
                if (std::find(ss.tex_vars.begin(), ss.tex_vars.end(), x) ==
                    ss.tex_vars.end()) {
                    ss.tex_vars.push_back(x);
                }
            } else if (std::find(ss.slt_vars.begin(), ss.slt_vars.end(),
                                 x) == ss.slt_vars.end()) {
                ss.slt_vars.push_back(x);
            }
        }
        substages.push_back(std::move(ss));
    }
    return substages;
}

Stages WrapSubStagesIntoStages(SubStages substages) {
    Stages stages;
    for (auto& ss : substages) {
        Stage stage;
        stage.vtx_vars = ss.vtx_vars;
        stage.inp_vars = ss.inp_vars;
        stage.tex_vars = ss.tex_vars;
        stage.slt_vars = ss.slt_vars;
        stage.out_vars = ss.out_vars;
        stage.shader_type = ss.shader_type;
        stage.img_size = ss.img_size;
        stage.substages = {std::move(ss)};
        stages.push_back(std::move(stage));
    }
    return stages;
}

}  // namespace dressi
