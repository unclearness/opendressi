// Rasterization + texture sampling tests on a fullscreen quad: deferred
// G-buffer parity, textured forward rendering, and texture recovery through
// the inverse-UV backward.

#include <gtest/gtest.h>

#include <cmath>

#include "dressi/dressi.h"

using namespace dressi;

namespace {

// A quad covering clip space exactly: 4 vertices, 2 triangles, UV in [0,1]
struct QuadMesh {
    Variable pos{VEC4, {4, 1}};
    Variable uv{VEC2, {4, 1}};
    Variable faces{IVEC3, {2, 1}};
    CpuImage pos_img{4, 1, 4};
    CpuImage uv_img{4, 1, 2};
    CpuImage faces_img{2, 1, 3};

    QuadMesh() {
        const float p[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
        for (int i = 0; i < 4; i++) {
            pos_img.at(i, 0, 0) = p[i][0];
            pos_img.at(i, 0, 1) = p[i][1];
            pos_img.at(i, 0, 2) = 0.5f;
            pos_img.at(i, 0, 3) = 1.f;
            uv_img.at(i, 0, 0) = (p[i][0] + 1.f) * 0.5f;
            uv_img.at(i, 0, 1) = (p[i][1] + 1.f) * 0.5f;
        }
        const float f[2][3] = {{0, 1, 2}, {0, 2, 3}};
        for (int i = 0; i < 2; i++) {
            for (int c = 0; c < 3; c++) {
                faces_img.at(i, 0, c) = f[i][c];
            }
        }
    }

    void send(DressiAD& ad) const {
        ad.sendImg(pos, pos_img);
        ad.sendImg(uv, uv_img);
        ad.sendImg(faces, faces_img);
    }
};

// Inverse-UV inputs for the fullscreen quad: UV space == screen space up to
// scale, so the screen position of texel (tx,ty) is uv * screen_size.
struct QuadInverseUV {
    Variable uv_clip{VEC4, {4, 1}};
    Variable screen_attr{VEC4, {4, 1}};
    CpuImage uv_clip_img{4, 1, 4};
    CpuImage screen_attr_img{4, 1, 4};

    QuadInverseUV(const QuadMesh& mesh, ImgSize screen) {
        for (int i = 0; i < 4; i++) {
            const float u = mesh.uv_img.at(i, 0, 0);
            const float v = mesh.uv_img.at(i, 0, 1);
            uv_clip_img.at(i, 0, 0) = u * 2.f - 1.f;
            uv_clip_img.at(i, 0, 1) = v * 2.f - 1.f;
            uv_clip_img.at(i, 0, 2) = 0.5f;
            uv_clip_img.at(i, 0, 3) = 1.f;
            screen_attr_img.at(i, 0, 0) = u * float(screen.w);
            screen_attr_img.at(i, 0, 1) = v * float(screen.h);
            screen_attr_img.at(i, 0, 2) = 1.f;
            screen_attr_img.at(i, 0, 3) = 0.f;
        }
    }

    void send(DressiAD& ad) const {
        ad.sendImg(uv_clip, uv_clip_img);
        ad.sendImg(screen_attr, screen_attr_img);
    }
};

}  // namespace

TEST(Raster, QuadGBufferUvAndMask) {
    const ImgSize screen = {16, 16};
    QuadMesh mesh;
    Variable ones(FLOAT, {4, 1});
    CpuImage ones_img(4, 1, 1, 1.f);

    Variable g_uv = F::Rasterize(mesh.pos, mesh.uv, mesh.faces, screen);
    Variable g_mask = F::Rasterize(mesh.pos, ones, mesh.faces, screen);
    // Loss depends on both G-buffer channels so both are traversed
    Variable loss = F::Mean(g_mask) + F::Mean(g_uv);

    DressiAD ad;
    ad.setPackingMode(DressiAD::PackingMode::Naive);
    ad.setLossVar(loss);
    mesh.send(ad);
    ad.sendImg(ones, ones_img);
    ad.execStep();

    const CpuImage uv_out = ad.recvImg(g_uv);
    const CpuImage mask_out = ad.recvImg(g_mask);
    for (uint32_t y = 0; y < screen.h; y++) {
        for (uint32_t x = 0; x < screen.w; x++) {
            ASSERT_FLOAT_EQ(mask_out.at(x, y, 0), 1.f)
                    << "pixel " << x << "," << y;
            // Interpolated UV at the pixel center
            const float expect_u = (float(x) + 0.5f) / float(screen.w);
            const float expect_v = (float(y) + 0.5f) / float(screen.h);
            ASSERT_NEAR(uv_out.at(x, y, 0), expect_u, 1e-5f);
            ASSERT_NEAR(uv_out.at(x, y, 1), expect_v, 1e-5f);
        }
    }
}

TEST(Raster, PartialCoverageAndBackground) {
    // A quad covering only the left half of clip space
    const ImgSize screen = {16, 16};
    QuadMesh mesh;
    for (int i = 0; i < 4; i++) {
        if (mesh.pos_img.at(i, 0, 0) > 0.f) {
            mesh.pos_img.at(i, 0, 0) = 0.f;
        }
    }
    Variable ones(FLOAT, {4, 1});
    CpuImage ones_img(4, 1, 1, 1.f);
    Variable g_mask = F::Rasterize(mesh.pos, ones, mesh.faces, screen);
    Variable loss = F::Mean(g_mask);

    DressiAD ad;
    ad.setPackingMode(DressiAD::PackingMode::Naive);
    ad.setLossVar(loss);
    mesh.send(ad);
    ad.sendImg(ones, ones_img);
    ad.execStep();

    const CpuImage mask_out = ad.recvImg(g_mask);
    for (uint32_t y = 0; y < screen.h; y++) {
        EXPECT_FLOAT_EQ(mask_out.at(2, y, 0), 1.f);   // covered
        EXPECT_FLOAT_EQ(mask_out.at(12, y, 0), 0.f);  // background
    }
    EXPECT_NEAR(ad.recvImg(loss).data[0], 0.5f, 0.05f);
}

TEST(Raster, TexturedQuadForward) {
    const ImgSize screen = {16, 16};
    const ImgSize tex_size = {4, 4};
    QuadMesh mesh;
    QuadInverseUV inv(mesh, screen);

    Variable tex(VEC3, tex_size);
    CpuImage tex_img(tex_size.w, tex_size.h, 3);
    for (uint32_t ty = 0; ty < tex_size.h; ty++) {
        for (uint32_t tx = 0; tx < tex_size.w; tx++) {
            tex_img.at(tx, ty, 0) = float(tx) / 4.f;
            tex_img.at(tx, ty, 1) = float(ty) / 4.f;
            tex_img.at(tx, ty, 2) = float(tx + ty) / 8.f;
        }
    }

    Variable g_uv = F::Rasterize(mesh.pos, mesh.uv, mesh.faces, screen);
    Variable inv_uv =
            F::Rasterize(inv.uv_clip, inv.screen_attr, mesh.faces, tex_size);
    Variable color = F::Texture(tex, g_uv, inv_uv);
    Variable loss = F::Mean(color);

    DressiAD ad;
    ad.setPackingMode(DressiAD::PackingMode::Naive);
    ad.setLossVar(loss);
    mesh.send(ad);
    inv.send(ad);
    ad.sendImg(tex, tex_img);
    ad.execStep();

    // Each 4x4 screen block reads one texel (nearest sampling)
    const CpuImage out = ad.recvImg(color);
    for (uint32_t y = 0; y < screen.h; y++) {
        for (uint32_t x = 0; x < screen.w; x++) {
            const uint32_t tx = x / 4;
            const uint32_t ty = y / 4;
            for (uint32_t c = 0; c < 3; c++) {
                ASSERT_NEAR(out.at(x, y, c), tex_img.at(tx, ty, c), 1e-5f)
                        << "pixel " << x << "," << y << " ch " << c;
            }
        }
    }
}

TEST(Raster, TextureRecoveryThroughInverseUV) {
    // Deferred unlit pipeline on a fullscreen quad: recover the GT texture
    // from a black initialization using the rendered MSE loss.
    const ImgSize screen = {32, 32};
    const ImgSize tex_size = {8, 8};
    QuadMesh mesh;
    QuadInverseUV inv(mesh, screen);

    CpuImage gt_tex(tex_size.w, tex_size.h, 3);
    for (size_t i = 0; i < gt_tex.data.size(); i++) {
        gt_tex.data[i] = 0.1f + 0.8f * float((i * 7) % 11) / 11.f;
    }

    Variable tex(VEC3, tex_size);
    Variable target(VEC3, screen);
    Variable g_uv = F::Rasterize(mesh.pos, mesh.uv, mesh.faces, screen);
    Variable inv_uv =
            F::Rasterize(inv.uv_clip, inv.screen_attr, mesh.faces, tex_size);
    Variable color = F::Texture(tex, g_uv, inv_uv);
    Variable diff = target - color;
    Variable loss = F::Mean(diff * diff);
    Variable tex_mut = tex;
    tex_mut.setRequiresGradRecursively();

    // GT render first (same graph, GT texture, no optimizer step effects)
    DressiAD gt_ad;
    gt_ad.setPackingMode(DressiAD::PackingMode::Naive);
    gt_ad.setLossVar(loss);
    mesh.send(gt_ad);
    inv.send(gt_ad);
    gt_ad.sendImg(tex, gt_tex);
    gt_ad.sendImg(target, CpuImage(screen.w, screen.h, 3, 0.f));
    gt_ad.execStep();
    const CpuImage gt_render = gt_ad.recvImg(color);

    // Optimization from black
    const float lr = 0.8f * float(screen.w * screen.h * 3) / 2.f;
    DressiAD ad;
    ad.setLossVar(loss);
    ad.setOptimizer([lr](Variables xs, Variables gxs) {
        Variables updated;
        for (size_t i = 0; i < xs.size(); i++) {
            updated.push_back(xs[i] - gxs[i] * lr);
        }
        return updated;
    });
    mesh.send(ad);
    inv.send(ad);
    ad.sendImg(tex, CpuImage(tex_size.w, tex_size.h, 3, 0.f));
    ad.sendImg(target, gt_render);

    for (int iter = 0; iter < 100; iter++) {
        ad.execStep();
    }
    EXPECT_LT(ad.recvImg(loss).data[0], 1e-5f);

    const CpuImage recovered = ad.recvImg(tex);
    float max_err = 0.f;
    for (size_t i = 0; i < recovered.data.size(); i++) {
        max_err = std::max(max_err,
                           std::abs(recovered.data[i] - gt_tex.data[i]));
    }
    EXPECT_LT(max_err, 0.02f);
}
