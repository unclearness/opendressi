#ifndef DRESSI_IBL_H
#define DRESSI_IBL_H

// Split-sum IBL builders shaped after the paper appendix's PBS sample
// (BuildIrradianceSample / BuildPrefEnvironmentSample /
// BuildBrdfIntegrationMap). All maps are equirectangular 2D images
// (documented deviation: no cubemaps / hardware mips in this IR, so the
// prefiltered environment is a vector of per-roughness-level Variables
// instead of one mip-mapped image). Every builder output depends only on
// the env leaf: send the env once and the reactive cache executes the
// whole precompute exactly once, then prunes it from the steady-state
// frame while its cached images keep feeding the shading pass.

#include <vector>

#include "dressi/types.h"
#include "dressi/variable.h"

namespace dressi {

// Diffuse irradiance map (stores E(N)/pi): pools the env down to at most
// 64x32 with AvgPool2x2, then runs the exact cos-weighted convolution.
// diffuse = kD * albedo * SampleIrradiance(irrad, n).
Variable BuildIrradianceSample(const Variable& env_img,
                               ImgSize out_size = {32, 16});

// Prefiltered specular chain: `levels` equirect images from `top` halving
// each level, roughness = level / (levels - 1), GGX importance sampled
// with `n_samples` taps. Sample with SamplePrefEnvironment.
std::vector<Variable> BuildPrefEnvironmentSample(const Variable& env_img,
                                                 uint32_t levels = 5,
                                                 ImgSize top = {128, 64},
                                                 uint32_t n_samples = 256);

// Split-sum BRDF integration LUT: VEC2 (scale A, bias B) over
// (NdotV, roughness). No inputs.
Variable BuildBrdfIntegrationMap(ImgSize size = {256, 256},
                                 uint32_t n_samples = 512);

// -------- Runtime lookups (compose F:: ops; screen-size passes) --------

// Irradiance at the per-pixel (world) normal
Variable SampleIrradiance(const Variable& irrad, const Variable& normal);

// Prefiltered radiance at the reflection direction with fractional-LOD
// blending across the level images: lod = roughness * (levels-1), tent
// weights clamp(1 - |lod - i|, 0, 1) (at most two are nonzero)
Variable SamplePrefEnvironment(const std::vector<Variable>& levels,
                               const Variable& refl_dir,
                               const Variable& roughness);

// BRDF LUT at (NdotV, roughness); returns VEC2 (A, B)
Variable SampleBrdfLut(const Variable& lut, const Variable& n_dot_v,
                       const Variable& roughness);

}  // namespace dressi

#endif  // DRESSI_IBL_H
