// Type/size inference tests: broadcasting, vector widening, error cases.

#include <gtest/gtest.h>

#include "dressi/dressi.h"

using namespace dressi;

TEST(Infer, UniformBroadcastsAgainstImage) {
    Variable img(FLOAT, {32, 16});
    Variable uni(FLOAT, {1, 1});
    EXPECT_EQ(F::Add(img, uni).getImgSize(), (ImgSize{32, 16}));
    EXPECT_EQ(F::Add(uni, img).getImgSize(), (ImgSize{32, 16}));
    EXPECT_EQ(F::Add(uni, uni).getImgSize(), (ImgSize{1, 1}));
}

TEST(Infer, SizeMismatchThrows) {
    Variable a(FLOAT, {32, 16});
    Variable b(FLOAT, {16, 32});
    EXPECT_THROW(F::Add(a, b), DressiError);
}

TEST(Infer, ScalarWidensToVector) {
    Variable s(FLOAT, {8, 8});
    Variable v(VEC3, {8, 8});
    EXPECT_EQ(F::Mul(s, v).getVType(), VEC3);
    EXPECT_EQ(F::Mul(v, s).getVType(), VEC3);
    EXPECT_EQ(F::Mul(s, s).getVType(), FLOAT);
}

TEST(Infer, VectorTypeMismatchThrows) {
    Variable a(VEC2, {8, 8});
    Variable b(VEC3, {8, 8});
    EXPECT_THROW(F::Add(a, b), DressiError);
}

TEST(Infer, IntFloatMixThrows) {
    Variable a(FLOAT, {8, 8});
    Variable b(INT, {8, 8});
    EXPECT_THROW(F::Add(a, b), DressiError);
}

TEST(Infer, VectorOps) {
    Variable a(VEC3, {8, 8});
    Variable b(VEC3, {8, 8});
    EXPECT_EQ(F::Dot(a, b).getVType(), FLOAT);
    EXPECT_EQ(F::Cross(a, b).getVType(), VEC3);
    EXPECT_EQ(F::Length(a).getVType(), FLOAT);
    EXPECT_EQ(F::Normalize(a).getVType(), VEC3);
    EXPECT_EQ(F::CompSum(a).getVType(), FLOAT);
    EXPECT_EQ(F::GetZ(a).getVType(), FLOAT);
    EXPECT_THROW(F::GetW(a), DressiError);  // VEC3 has no w

    Variable v2(VEC2, {8, 8});
    EXPECT_THROW(F::Cross(v2, v2), DressiError);
}

TEST(Infer, ComposeAndExtract) {
    Variable x(FLOAT, {8, 8});
    Variable y(FLOAT, {8, 8});
    Variable z(FLOAT, {8, 8});
    Variable v3 = F::Vec3(x, y, z);
    EXPECT_EQ(v3.getVType(), VEC3);
    EXPECT_EQ(v3.getImgSize(), (ImgSize{8, 8}));

    Variable v4 = F::Vec4(v3, x);
    EXPECT_EQ(v4.getVType(), VEC4);

    Variable c = F::Vec3(1.f, 2.f, 3.f);
    EXPECT_EQ(c.getVType(), VEC3);
    EXPECT_EQ(c.getImgSize(), (ImgSize{1, 1}));
}

TEST(Infer, MatrixOps) {
    Variable m(MAT4, {1, 1});
    Variable v(VEC4, {8, 8});
    EXPECT_EQ(F::MatMul(m, v).getVType(), VEC4);
    EXPECT_EQ(F::MatMul(m, v).getImgSize(), (ImgSize{8, 8}));
    EXPECT_EQ(F::MatMul(m, m).getVType(), MAT4);
    EXPECT_EQ(F::Transpose(m).getVType(), MAT4);

    Variable u(VEC3, {1, 1});
    EXPECT_EQ(F::OuterProduct(u, u).getVType(), MAT3);
    EXPECT_THROW(F::MatMul(m, u), DressiError);
}

TEST(Infer, ComparisonRequiresScalar) {
    Variable a(VEC3, {8, 8});
    Variable b(VEC3, {8, 8});
    EXPECT_THROW(F::Less(a, b), DressiError);

    Variable s(FLOAT, {8, 8});
    Variable cmp = (0.f < s);
    EXPECT_EQ(cmp.getVType(), FLOAT);
}

TEST(Infer, SpatialReductions) {
    Variable img(VEC3, {37, 23});
    Variable s = F::Sum(img);
    EXPECT_EQ(s.getVType(), VEC3);
    EXPECT_EQ(s.getImgSize(), (ImgSize{1, 1}));

    Variable m = F::Mean(img);
    EXPECT_EQ(m.getVType(), FLOAT);
    EXPECT_EQ(m.getImgSize(), (ImgSize{1, 1}));
}

TEST(Infer, IblOps) {
    Variable env(VEC3, {16, 8});
    Variable dir(VEC3, {32, 32});
    Variable uv(VEC2, {32, 32});
    Variable inv_uv(VEC4, {16, 8});

    EXPECT_EQ(F::EquirectSample(env, dir).getVType(), VEC3);
    EXPECT_EQ(F::EquirectSample(env, dir).getImgSize(), (ImgSize{32, 32}));
    EXPECT_THROW(F::EquirectSample(env, uv), DressiError);  // dir not VEC3
    Variable uni(VEC3, {1, 1});
    EXPECT_THROW(F::EquirectSample(uni, dir), DressiError);  // {1,1} map

    EXPECT_EQ(F::TextureBilinear(env, uv, inv_uv).getVType(), VEC3);
    EXPECT_EQ(F::TextureBilinear(env, uv, inv_uv).getImgSize(),
              (ImgSize{32, 32}));
    EXPECT_THROW(F::TextureBilinear(env, dir, inv_uv), DressiError);
    Variable bad_inv(VEC4, {8, 8});
    EXPECT_THROW(F::TextureBilinear(env, uv, bad_inv), DressiError);

    EXPECT_EQ(F::IrradianceConv(env, {8, 4}).getVType(), VEC3);
    EXPECT_EQ(F::IrradianceConv(env, {8, 4}).getImgSize(), (ImgSize{8, 4}));
    EXPECT_EQ(F::PrefilterEnv(env, {8, 4}, 0.5f, 32).getVType(), VEC3);
    EXPECT_EQ(F::PrefilterEnv(env, {8, 4}, 0.5f, 32).getImgSize(),
              (ImgSize{8, 4}));
    EXPECT_EQ(F::BrdfIntegrationLut({16, 16}, 32).getVType(), VEC2);
    EXPECT_EQ(F::BrdfIntegrationLut({16, 16}, 32).getImgSize(),
              (ImgSize{16, 16}));
    EXPECT_EQ(F::AvgPool2x2(env).getImgSize(), (ImgSize{8, 4}));
    EXPECT_EQ(F::AvgPool2x2(env).getVType(), VEC3);
}
