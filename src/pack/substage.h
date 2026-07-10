#ifndef DRESSI_PACK_SUBSTAGE_H
#define DRESSI_PACK_SUBSTAGE_H

#include <string>
#include <vector>

#include "dressi/function.h"
#include "dressi/variable.h"

namespace dressi {

// A SubStage is one shader (one subpass): a set of fused functions with
// classified I/O (Appendix A of the Dressi paper).
struct SubStage {
    Variables vtx_vars;  // vertex buffer inputs (RASTER; unused in M1)
    Variables inp_vars;  // input attachments (same-pixel reads within a stage)
    Variables tex_vars;  // texture sampler inputs (UV sampling; unused in M1)
    Variables slt_vars;  // sampler-less texture inputs (texelFetch)
    Variables uif_vars;  // uniform inputs ({1,1} leaves)
    Variables out_vars;  // color attachments (outputs)
    Variables gen_vars;  // all variables generated inside the shader
    Functions funcs;     // fused functions, topologically ordered
    std::string shader_code;       // fragment shader
    std::string vert_shader_code;  // RASTER only (pass-through VS)

    ShaderType shader_type = FRAG;
    ImgSize img_size = {1, 1};  // common output size
};
using SubStages = std::vector<SubStage>;

// A Stage maps to one VkRenderPass; its substages become subpasses.
struct Stage {
    Variables vtx_vars, inp_vars, tex_vars, slt_vars, uif_vars, out_vars;
    SubStages substages;
    ShaderType shader_type = FRAG;
    ImgSize img_size = {1, 1};
};
using Stages = std::vector<Stage>;

enum class PackingMode {
    Naive,  // 1 function = 1 substage = 1 stage (correctness baseline)
    RSP,    // greedy reactive shader packing
};

// Trivial packing: every non-constant function becomes its own substage;
// all non-constant inputs are read via texelFetch (slt).
SubStages TrivialPackSubStages(const Functions& all_funcs);

// 1:1 substage -> stage wrapping (naive mode)
Stages WrapSubStagesIntoStages(SubStages substages);

// True when the variable's value is a compile-time constant to inline into
// shaders (no image, no substage for its creator).
bool IsInlineConst(const Variable& var);

}  // namespace dressi

#endif  // DRESSI_PACK_SUBSTAGE_H
