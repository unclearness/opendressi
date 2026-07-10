// Graph core tests: handle identity, wiring, ownership, recursive flags.

#include <gtest/gtest.h>

#include <map>

#include "dressi/dressi.h"

using namespace dressi;

TEST(Graph, VariableBasics) {
    Variable v(VEC3, {64, 32});
    EXPECT_EQ(v.getVType(), VEC3);
    EXPECT_EQ(v.getImgSize(), (ImgSize{64, 32}));
    EXPECT_FALSE(v.getRequiresGrad());
    EXPECT_TRUE(v.IsDirty());
    EXPECT_FALSE(v.getCreator());

    Variable null_var(nullptr);
    EXPECT_FALSE(null_var);
    EXPECT_TRUE(v);
}

TEST(Graph, IdOrderingAndMapKey) {
    Variable a;
    Variable b;
    EXPECT_LT(a.id(), b.id());

    std::map<Variable, int> m;
    m[a] = 1;
    m[b] = 2;
    EXPECT_EQ(m.at(a), 1);
    EXPECT_EQ(m.at(b), 2);
}

TEST(Graph, CreatorAndUsersWiring) {
    Variable x0(FLOAT, {4, 4});
    Variable x1(FLOAT, {4, 4});
    Variable y = F::Add(x0, x1);

    Function creator = y.getCreator();
    ASSERT_TRUE(creator);
    EXPECT_EQ(creator.getFwdCode(), "{y}={x0}+{x1};");
    EXPECT_EQ(creator.getShaderType(), FRAG);
    ASSERT_EQ(creator.getInputVars().size(), 2u);
    EXPECT_EQ(creator.getInputVars()[0], x0);
    EXPECT_EQ(creator.getInputVars()[1], x1);
    EXPECT_EQ(creator.getOutputVar(), y);

    ASSERT_EQ(x0.getUsers().size(), 1u);
    EXPECT_EQ(x0.getUsers()[0], creator);
}

TEST(Graph, OwnershipHoldingOutputKeepsGraphAlive) {
    Variable y(nullptr);
    {
        Variable x0(FLOAT, {4, 4});
        Variable x1(FLOAT, {4, 4});
        y = F::Mul(F::Add(x0, x1), x1);
    }
    // x0/x1 handles are gone, but the graph must be reachable from y
    Function creator = y.getCreator();
    ASSERT_TRUE(creator);
    Variables xs = creator.getInputVars();
    ASSERT_EQ(xs.size(), 2u);
    Function add = xs[0].getCreator();
    ASSERT_TRUE(add);
    EXPECT_EQ(add.getFwdCode(), "{y}={x0}+{x1};");
}

TEST(Graph, RequiresGradRecursivePropagatesDownstream) {
    Variable x(FLOAT, {4, 4});
    Variable a = F::Sin(x);
    Variable b = F::Add(a, F::Float(1.f));
    EXPECT_FALSE(b.getRequiresGrad());

    x.setRequiresGradRecursively();
    EXPECT_TRUE(x.getRequiresGrad());
    EXPECT_TRUE(a.getRequiresGrad());
    EXPECT_TRUE(b.getRequiresGrad());
}

TEST(Graph, DirtyRecursive) {
    Variable x(FLOAT, {4, 4});
    Variable y = F::Exp(x);
    x.setDirtyRecursively(false);
    EXPECT_FALSE(x.IsDirty());
    EXPECT_FALSE(y.IsDirty());
    x.setDirtyRecursively(true);
    EXPECT_TRUE(y.IsDirty());
}

TEST(Graph, AppendixStyleFunctionCtor) {
    // The appendix-faithful raw construction path
    Variable x0(FLOAT, {8, 8});
    Variable x1(FLOAT, {8, 8});
    Function func("{y}={x0}+{x1};", FRAG,
                  [](const Variables&, const Variable&, const Variable& gy,
                     uint32_t) -> Variable { return gy; });
    Variable y = func.buildFwd({x0, x1});
    EXPECT_EQ(y.getVType(), FLOAT);
    EXPECT_EQ(y.getImgSize(), (ImgSize{8, 8}));

    Variable gy(FLOAT, {8, 8});
    Variable gx = func.buildBwd(gy, 0);
    EXPECT_EQ(gx, gy);
}

TEST(Graph, RasterizeValidatesInputs) {
    Variable pos(VEC4, {16, 1});
    Variable attrib(VEC3, {16, 1});
    Variable faces(IVEC3, {8, 1});
    Variable g = F::Rasterize(pos, attrib, faces, {64, 64});
    EXPECT_EQ(g.getVType(), VEC3);
    EXPECT_EQ(g.getImgSize(), (ImgSize{64, 64}));
    EXPECT_EQ(g.getCreator().getShaderType(), RASTER);

    Variable bad_pos(VEC3, {16, 1});
    EXPECT_THROW(F::Rasterize(bad_pos, attrib, faces, {64, 64}),
                 DressiError);
    Variable bad_faces(VEC3, {8, 1});
    EXPECT_THROW(F::Rasterize(pos, attrib, bad_faces, {64, 64}),
                 DressiError);
    Variable mismatched(VEC3, {8, 1});
    EXPECT_THROW(F::Rasterize(pos, mismatched, faces, {64, 64}),
                 DressiError);
}
