// Multi-view texture optimization on the bunny mesh with deferred unlit
// shading. G-buffer channels (screen-space UV + coverage mask) come from
// non-differentiable rasterization; shading samples the texture, which is
// differentiable through the inverse-UV backward. Starting from a black
// texture, MSE against GT renders (rendered with the GT texture atlas from
// the same pipeline) recovers the atlas on all texels visible in any view.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../common/asset_utils.h"
#include "dressi/dressi.h"

using namespace dressi;
using namespace dressi_examples;

namespace {

struct View {
    Variable clip{nullptr};         // {V,1} VEC4 clip positions
    Variable inv_uv_attr{nullptr};  // {V,1} VEC4 inverse-UV screen attr
    Variable target{nullptr};       // {screen} VEC3 GT render
    CpuImage clip_img;
    CpuImage inv_uv_attr_img;
};

}  // namespace

int main(int argc, char* argv[]) try {
    const std::string data_dir = argc > 1 ? argv[1] : "data/bunny";
    const ImgSize screen = {256, 256};
    const uint32_t n_views = 6;
    const int n_iters = 400;

    // ------------------------- Load assets (CPU) -------------------------
    const Mesh mesh = LoadObjMesh(data_dir + "/bunny.obj");
    const CpuImage gt_tex_img = LoadImageRgb(data_dir + "/bunny-atlas.jpg");
    const ImgSize tex_size = {gt_tex_img.width, gt_tex_img.height};
    std::printf("mesh: %u verts, %u faces; atlas: %ux%u\n",
                mesh.numVertices(), mesh.numFaces(), tex_size.w, tex_size.h);

    // ------------------------ Graph leaf variables ------------------------
    Variable vtx_uv(VEC2, {mesh.numVertices(), 1});
    Variable faces(IVEC3, {mesh.numFaces(), 1});
    Variable uv_clip(VEC4, {mesh.numVertices(), 1});  // UV-space positions
    Variable ones(FLOAT, {mesh.numVertices(), 1});
    Variable tex(VEC3, tex_size);

    const CpuImage uv_clip_img = UvAsClip(mesh.uv);
    const CpuImage ones_img(mesh.numVertices(), 1, 1, 1.f);

    // Cameras on a circle around the object, slightly elevated
    std::vector<View> views(n_views);
    const Mat4 proj = PerspectiveVk(45.f * 3.14159265f / 180.f, 1.f, 0.1f,
                                    10.f);
    for (uint32_t v = 0; v < n_views; v++) {
        const float ang = 2.f * 3.14159265f * float(v) / float(n_views);
        const std::array<float, 3> eye = {1.6f * std::sin(ang), 0.5f,
                                          1.6f * std::cos(ang)};
        const Mat4 mvp = Mul(proj, LookAt(eye, {0, 0, 0}, {0, 1, 0}));
        views[v].clip = Variable(VEC4, {mesh.numVertices(), 1});
        views[v].inv_uv_attr = Variable(VEC4, {mesh.numVertices(), 1});
        views[v].target = Variable(VEC3, screen);
        views[v].clip_img = TransformToClip(mesh.pos, mvp);
        views[v].inv_uv_attr_img = ScreenAttr(views[v].clip_img, screen);
    }

    // ------------------- Deferred unlit rendering graph -------------------
    // G-buffer per view: screen-space UV + coverage mask; shading pass:
    // color = mask * texture(tex, uv)
    Variables colors;
    Variables view_losses;
    for (auto& view : views) {
        Variable g_uv = F::Rasterize(view.clip, vtx_uv, faces, screen);
        Variable g_mask = F::Rasterize(view.clip, ones, faces, screen);
        Variable inv_uv =
                F::Rasterize(uv_clip, view.inv_uv_attr, faces, tex_size);
        Variable color = g_mask * F::Texture(tex, g_uv, inv_uv);
        Variable diff = view.target - color;
        colors.push_back(color);
        view_losses.push_back(F::Mean(diff * diff));
    }
    Variable loss = F::SumPixelWise(view_losses);

    const auto send_shared = [&](DressiAD& ad) {
        ad.sendImg(vtx_uv, mesh.uv);
        ad.sendImg(faces, mesh.faces);
        ad.sendImg(uv_clip, uv_clip_img);
        ad.sendImg(ones, ones_img);
        for (auto& view : views) {
            ad.sendImg(view.clip, view.clip_img);
            ad.sendImg(view.inv_uv_attr, view.inv_uv_attr_img);
        }
    };

    // ----------------------- GT renders (GT texture) ----------------------
    std::vector<CpuImage> gt_renders;
    {
        DressiAD gt_ad;
        gt_ad.setPackingMode(DressiAD::PackingMode::Naive);
        gt_ad.setLossVar(loss);
        send_shared(gt_ad);
        gt_ad.sendImg(tex, gt_tex_img);
        for (auto& view : views) {
            gt_ad.sendImg(view.target,
                          CpuImage(screen.w, screen.h, 3, 0.f));
        }
        gt_ad.execStep();
        for (uint32_t v = 0; v < n_views; v++) {
            gt_renders.push_back(gt_ad.recvImg(colors[v]));
        }
        SaveImagePng("gt_view0.png", gt_renders[0]);
    }

    // --------------- Optimization: black texture -> GT atlas ---------------
    Variable tex_mut = tex;
    tex_mut.setRequiresGradRecursively();

    // MSE mean divides by N; with up to n_views gradient contributions per
    // texel, keep the effective per-texel step below 1
    const float lr = 0.4f * float(screen.w * screen.h * 3) / (2.f * n_views);

    DressiAD ad;
    ad.setLossVar(loss);
    ad.setOptimizer([lr](Variables xs, Variables gxs) {
        Variables updated;
        for (size_t i = 0; i < xs.size(); i++) {
            updated.push_back(xs[i] - gxs[i] * lr);
        }
        return updated;
    });
    send_shared(ad);
    ad.sendImg(tex, CpuImage(tex_size.w, tex_size.h, 3, 0.f));  // black
    for (uint32_t v = 0; v < n_views; v++) {
        ad.sendImg(views[v].target, gt_renders[v]);
    }

    for (int iter = 0; iter < n_iters; iter++) {
        ad.execStep();
        if (iter % 50 == 0 || iter == n_iters - 1) {
            std::printf("iter %4d  loss %.8f  (stages %zu)\n", iter,
                        ad.recvImg(loss).data[0], ad.getStageCount());
        }
    }

    // ------------------------------ Validate ------------------------------
    const float final_loss = ad.recvImg(loss).data[0];
    const CpuImage recovered = ad.recvImg(tex);
    SaveImagePng("recovered_texture.png", recovered);

    // Texels recovered to the GT atlas (only texels visible in some view
    // receive gradients; the rest stay black)
    size_t visible = 0;
    size_t recovered_ok = 0;
    for (size_t p = 0; p < size_t(tex_size.w) * tex_size.h; p++) {
        float rec_mag = 0.f;
        float err = 0.f;
        for (uint32_t c = 0; c < 3; c++) {
            rec_mag += recovered.data[p * 3 + c];
            err = std::max(err, std::abs(recovered.data[p * 3 + c] -
                                         gt_tex_img.data[p * 3 + c]));
        }
        if (rec_mag > 0.01f) {
            visible++;
            if (err < 0.1f) {
                recovered_ok++;
            }
        }
    }
    std::printf("final loss %.8f\n", final_loss);
    std::printf("recovered texels: %zu / %zu visible (%.1f%% accurate)\n",
                recovered_ok, visible,
                visible ? 100.f * float(recovered_ok) / float(visible) : 0.f);
    std::printf("wrote gt_view0.png, recovered_texture.png\n");

    const bool ok = final_loss < 1e-4f &&
                    visible > 0 &&
                    float(recovered_ok) / float(visible) > 0.95f;
    return ok ? 0 : 1;
} catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 2;
}
