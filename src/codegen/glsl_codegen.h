#ifndef DRESSI_CODEGEN_GLSL_CODEGEN_H
#define DRESSI_CODEGEN_GLSL_CODEGEN_H

#include <string>

#include "pack/substage.h"

namespace dressi {

// The shared pass-through vertex shader rasterizing a fullscreen triangle
// (gl_VertexIndex trick; draw with 3 vertices, no vertex buffer).
std::string FullscreenVertShader();

// Generates the fragment shader for a substage: joins the GLSL snippets of
// its functions with substage-local normalized names (v0..vN), declares the
// classified inputs, and writes the padded color attachment outputs.
// Binding convention (shared with the executor), set=0:
//   [0, n_inp)                       input attachments
//   [n_inp, n_inp+n_tex)             texture samplers (UV sampling)
//   [n_inp+n_tex, ...+n_slt)         sampler-less textures (texelFetch)
std::string GenerateFragShader(const SubStage& substage);

// Generates the pass-through vertex shader and the interpolation fragment
// shader for a RASTER substage (one F::Rasterize function).
struct RasterShaders {
    std::string vert;
    std::string frag;
};
RasterShaders GenerateRasterShaders(const SubStage& substage);

// Physical channel count of an image for a VType (VEC3 pads to 4)
uint32_t PhysChannels(VType vtype);

}  // namespace dressi

#endif  // DRESSI_CODEGEN_GLSL_CODEGEN_H
