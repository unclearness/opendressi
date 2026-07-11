#include "dressi/ibl.h"

#include <string>

#include "dressi/f.h"

namespace dressi {

namespace {

// Pool an equirect env down until it fits {max_w, max_h}. Halving keeps
// the equirect parameterization; power-of-two inputs stay exact.
Variable PoolTo(Variable env, uint32_t max_w, uint32_t max_h) {
    while (env.getImgSize().w > max_w || env.getImgSize().h > max_h) {
        env = F::AvgPool2x2(env);
    }
    return env;
}

}  // namespace

Variable BuildIrradianceSample(const Variable& env_img, ImgSize out_size) {
    return F::IrradianceConv(PoolTo(env_img, 64, 32), out_size);
}

std::vector<Variable> BuildPrefEnvironmentSample(const Variable& env_img,
                                                 uint32_t levels, ImgSize top,
                                                 uint32_t n_samples) {
    DRESSI_CHECK(levels >= 2, "BuildPrefEnvironmentSample: need >= 2 levels");
    // Importance sampling reads the env at point taps: a moderately pooled
    // source (2x the top level) bounds the tap cost without visible loss
    const Variable src = PoolTo(env_img, top.w * 2, top.h * 2);
    std::vector<Variable> out;
    ImgSize size = top;
    for (uint32_t i = 0; i < levels; i++) {
        const float rough = float(i) / float(levels - 1);
        out.push_back(F::PrefilterEnv(src, size, rough, n_samples));
        size = {size.w > 1 ? size.w / 2 : 1, size.h > 1 ? size.h / 2 : 1};
    }
    return out;
}

Variable BuildBrdfIntegrationMap(ImgSize size, uint32_t n_samples) {
    return F::BrdfIntegrationLut(size, n_samples);
}

Variable SampleIrradiance(const Variable& irrad, const Variable& normal) {
    return F::EquirectSample(irrad, normal);
}

Variable SamplePrefEnvironment(const std::vector<Variable>& levels,
                               const Variable& refl_dir,
                               const Variable& roughness) {
    DRESSI_CHECK(levels.size() >= 2,
                 "SamplePrefEnvironment: need >= 2 levels");
    const Variable lod = roughness * float(levels.size() - 1);
    Variables terms;
    for (size_t i = 0; i < levels.size(); i++) {
        const Variable w = F::Clamp(
                1.f - F::Abs(lod - float(i)), F::Float(0.f), F::Float(1.f));
        terms.push_back(F::EquirectSample(levels[i], refl_dir) * w);
    }
    return F::SumPixelWise(terms);
}

Variable SampleBrdfLut(const Variable& lut, const Variable& n_dot_v,
                       const Variable& roughness) {
    return F::TextureBilinear(lut, F::Vec2(n_dot_v, roughness),
                              Variable(VEC4, lut.getImgSize()));
}

}  // namespace dressi
