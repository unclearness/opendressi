#ifndef DRESSI_EXAMPLES_PBR_GRAPH_H
#define DRESSI_EXAMPLES_PBR_GRAPH_H

// The one Cook-Torrance + split-sum IBL forward graph shared by the
// real-time viewer and the material / envmap optimization examples: the
// optimizers differentiate exactly the shader the viewer displays.

#include <vector>

#include "dressi/dressi.h"

namespace dressi_examples {

// Static scene leaves (upload once; `clip` is re-sent per frame/view)
struct PbrSceneLeaves {
    dressi::Variable clip{nullptr};        // VEC4 {V,1} clip positions
    dressi::Variable uvh{nullptr};         // VEC4 {V,1} (u, v, 1, tangent_w)
    dressi::Variable normal{nullptr};      // VEC3 {V,1} world normals
    dressi::Variable tangent{nullptr};     // VEC3 {V,1} world tangents
    dressi::Variable wpos{nullptr};        // VEC3 {V,1} world positions
    dressi::Variable faces{nullptr};       // IVEC3 {F,1}
    dressi::Variable albedo{nullptr};      // VEC3 texture (linear)
    dressi::Variable mr{nullptr};          // VEC3 texture (G=rough, B=metal)
    dressi::Variable normal_map{nullptr};  // VEC3 texture (tangent space)
    dressi::Variable ao{nullptr};          // VEC3 texture (R = AO)
    dressi::Variable emissive{nullptr};    // VEC3 texture (linear)
};

// Per-frame {1,1} uniforms (send the SAME set every frame so the reactive
// cache sees a stable dirty pattern)
struct PbrFrameLeaves {
    dressi::Variable cam_pos{nullptr};    // VEC3 world eye
    dressi::Variable cam_fwd{nullptr};    // VEC3 unit forward
    dressi::Variable cam_right{nullptr};  // VEC3 right * tan(fov/2) * aspect
    dressi::Variable cam_up{nullptr};     // VEC3 up * tan(fov/2)
    dressi::Variable light_pos{nullptr};  // VEC3 world light position
    dressi::Variable light_col{nullptr};  // VEC3 radiant intensity
};

// IBL precompute wired off the env leaf: static once env is uploaded, so
// the reactive cache runs it once and prunes it from the steady frame
struct PbrIblMaps {
    dressi::Variable env{nullptr};  // VEC3 equirect HDR leaf (linear)
    dressi::Variable irr{nullptr};
    std::vector<dressi::Variable> pref;
    dressi::Variable lut{nullptr};
};

PbrIblMaps BuildPbrIblMaps(const dressi::Variable& env);

struct PbrOutputs {
    dressi::Variable ldr{nullptr};     // VEC3 screen: Reinhard + gamma
    dressi::Variable hdr{nullptr};     // VEC3 screen: pre-tonemap composite
    dressi::Variable mask{nullptr};    // FLOAT screen coverage
    dressi::Variable uv{nullptr};      // VEC2 screen G-buffer UV
    dressi::Variable albedo{nullptr};  // VEC3 screen sampled albedo (debug)
    dressi::Variable normal{nullptr};  // VEC3 screen shading normal (debug)
};

// `inv_uv` is the texture-sized VEC4 inverse-UV table consumed by the
// differentiable texture fetches' backward (rasterize the mesh in UV
// space per view); the viewer passes a never-uploaded dummy Variable.
PbrOutputs BuildPbrForward(const PbrSceneLeaves& scene,
                           const PbrFrameLeaves& frame,
                           const PbrIblMaps& ibl,
                           const dressi::Variable& inv_uv,
                           dressi::ImgSize screen,
                           bool normal_mapping = true);

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_PBR_GRAPH_H
