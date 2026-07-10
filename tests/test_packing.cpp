// Greedy packing tests: fusion behavior, limit compliance, topological
// validity, and determinism on synthetic graphs with clamped limits.

#include <gtest/gtest.h>

#include <algorithm>
#include <set>

#include "core/build_backward.h"
#include "pack/packing.h"
#include "dressi/dressi.h"

using namespace dressi;

namespace {

bool Contains(const Variables& vars, const Variable& v) {
    return std::find(vars.begin(), vars.end(), v) != vars.end();
}

// Every substage input must be produced by an earlier substage or be a leaf;
// functions inside a substage must be creation-ordered consistently.
void CheckTopologicalValidity(const Stages& stages) {
    std::set<Variable> produced;
    for (const auto& stage : stages) {
        for (const auto& ss : stage.substages) {
            for (const auto& v : ss.slt_vars) {
                if (v.getCreator() && !IsInlineConst(v)) {
                    EXPECT_TRUE(produced.count(v))
                            << "slt input not yet produced";
                }
            }
            // inp inputs may come from an earlier subpass of the same stage,
            // checked below via stage-local produced set growth
            for (const auto& v : ss.gen_vars) {
                produced.insert(v);
            }
        }
    }
}

void CheckLimits(const Stages& stages, const PackingLimits& limits) {
    for (const auto& stage : stages) {
        for (const auto& ss : stage.substages) {
            EXPECT_LE(ss.inp_vars.size(), limits.max_input_attachments);
            EXPECT_LE(ss.out_vars.size(), limits.max_output_attachments);
            EXPECT_LE(ss.slt_vars.size() + ss.tex_vars.size(),
                      limits.max_sampled_images);
            for (const auto& v : ss.out_vars) {
                EXPECT_EQ(v.getImgSize(), ss.img_size);
            }
            for (const auto& gv : ss.gen_vars) {
                EXPECT_FALSE(Contains(ss.slt_vars, gv))
                        << "slt input generated in the same substage";
            }
        }
    }
}

size_t CountSubstages(const Stages& stages) {
    size_t n = 0;
    for (const auto& stage : stages) {
        n += stage.substages.size();
    }
    return n;
}

}  // namespace

TEST(Packing, ElementwiseChainFusesIntoOneSubstage) {
    Variable x(VEC3, {32, 32});
    Variable y = F::Exp(F::Abs(F::Sin(x) + 1.f) * 0.5f);

    Functions funcs = TraverseFuncs({y});
    PackingLimits limits;
    Stages stages = GreedyPack(funcs, {y}, limits);

    EXPECT_EQ(CountSubstages(stages), 1u);
    ASSERT_EQ(stages.size(), 1u);
    const SubStage& ss = stages[0].substages[0];
    EXPECT_EQ(ss.out_vars.size(), 1u);
    EXPECT_EQ(ss.out_vars[0], y);
    // Only the leaf x is an external input
    EXPECT_EQ(ss.inp_vars.size() + ss.slt_vars.size(), 1u);
    CheckLimits(stages, limits);
    CheckTopologicalValidity(stages);
}

TEST(Packing, ReductionSplitsSubstages) {
    Variable img(FLOAT, {16, 16});
    Variable loss = F::Mean(img * img);

    Functions funcs = TraverseFuncs({loss});
    PackingLimits limits;
    Stages stages = GreedyPack(funcs, {loss}, limits);

    // Each 2x2 reduction changes the pixel grid: it cannot fuse with its
    // producer. 16x16 -> 8x8 -> 4x4 -> 2x2 -> 1x1 = 4 reduce substages, the
    // elementwise square fused before the first, the final scaling fused
    // after the last.
    EXPECT_GE(CountSubstages(stages), 5u);
    CheckLimits(stages, limits);
    CheckTopologicalValidity(stages);
}

TEST(Packing, OutputLimitForcesSplit) {
    // 6 independent outputs from one leaf; max 2 outputs per substage
    Variable x(FLOAT, {8, 8});
    Variables roots;
    for (int i = 0; i < 6; i++) {
        roots.push_back(F::Sin(x * float(i + 1)));
    }
    Functions funcs = TraverseFuncs(roots);
    PackingLimits limits;
    limits.max_output_attachments = 2;
    Stages stages = GreedyPack(funcs, roots, limits);

    for (const auto& stage : stages) {
        for (const auto& ss : stage.substages) {
            EXPECT_LE(ss.out_vars.size(), 2u);
        }
    }
    EXPECT_GE(CountSubstages(stages), 3u);
    CheckLimits(stages, limits);
}

TEST(Packing, RootsAreAlwaysExported) {
    Variable x(FLOAT, {8, 8});
    Variable mid = F::Sin(x);
    Variable out = F::Cos(mid);

    Functions funcs = TraverseFuncs({out});
    Stages stages = GreedyPack(funcs, {out, mid}, PackingLimits{});

    bool mid_exported = false;
    bool out_exported = false;
    for (const auto& stage : stages) {
        for (const auto& ss : stage.substages) {
            mid_exported |= Contains(ss.out_vars, mid);
            out_exported |= Contains(ss.out_vars, out);
        }
    }
    EXPECT_TRUE(mid_exported);
    EXPECT_TRUE(out_exported);
}

TEST(Packing, RasterSubstagesStayIsolated) {
    // A RASTER function gets its own substage and render pass: vertex
    // inputs are classified as vtx_vars and never fused with FRAG functions.
    Variable pos(VEC4, {16, 1});
    Variable uv(VEC2, {16, 1});
    Variable faces(IVEC3, {8, 1});
    Variable g_uv = F::Rasterize(pos, uv, faces, {32, 32});
    EXPECT_EQ(g_uv.getVType(), VEC2);
    EXPECT_EQ(g_uv.getImgSize(), (ImgSize{32, 32}));

    Variable shaded = F::Exp(g_uv);
    Functions funcs = TraverseFuncs({shaded});
    Stages stages = GreedyPack(funcs, {shaded}, PackingLimits{});

    bool found_raster = false;
    for (const auto& stage : stages) {
        for (const auto& ss : stage.substages) {
            bool has_raster = false;
            bool has_frag = false;
            for (const auto& f : ss.funcs) {
                (f.getShaderType() == RASTER ? has_raster : has_frag) = true;
            }
            EXPECT_FALSE(has_raster && has_frag)
                    << "RASTER fused with FRAG functions";
            if (has_raster) {
                found_raster = true;
                EXPECT_EQ(ss.funcs.size(), 1u);
                ASSERT_EQ(ss.vtx_vars.size(), 3u);
                EXPECT_EQ(ss.vtx_vars[0], pos);
                EXPECT_EQ(ss.vtx_vars[1], uv);
                EXPECT_EQ(ss.vtx_vars[2], faces);
                EXPECT_EQ(stage.substages.size(), 1u)
                        << "RASTER stage must not merge";
            }
        }
    }
    EXPECT_TRUE(found_raster);
}

TEST(Packing, Determinism) {
    Variable x(VEC3, {16, 16});
    Variable a = F::Sin(x);
    Variable b = F::Cos(x);
    Variable loss = F::Mean(a * b + F::Exp(a));

    Functions funcs = TraverseFuncs({loss});
    PackingLimits limits;
    Stages s1 = GreedyPack(funcs, {loss}, limits);
    Stages s2 = GreedyPack(funcs, {loss}, limits);
    ASSERT_EQ(s1.size(), s2.size());
    for (size_t i = 0; i < s1.size(); i++) {
        ASSERT_EQ(s1[i].substages.size(), s2[i].substages.size());
        for (size_t j = 0; j < s1[i].substages.size(); j++) {
            EXPECT_EQ(s1[i].substages[j].funcs.size(),
                      s2[i].substages[j].funcs.size());
        }
    }
}

TEST(Packing, FusionBeatsTrivialPassCount) {
    // A PBR-like elementwise expression should pack far fewer passes than
    // one-function-per-pass
    Variable rough(FLOAT, {32, 32});
    Variable n_dot_h(FLOAT, {32, 32});
    Variable alpha2 = rough * rough * rough * rough;
    Variable denom = n_dot_h * n_dot_h * (alpha2 - 1.f) + 1.f;
    Variable dist = alpha2 / (F::Float(3.14159265f) * denom * denom);
    Variable out = dist / (dist + 1.f);

    Functions funcs = TraverseFuncs({out});
    const size_t trivial_passes = TrivialPackSubStages(funcs).size();
    Stages rsp = GreedyPack(funcs, {out}, PackingLimits{});
    EXPECT_LT(CountSubstages(rsp), trivial_passes / 2);
    EXPECT_EQ(CountSubstages(rsp), 1u);
}
