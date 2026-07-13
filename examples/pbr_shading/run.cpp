// Real-time PBR viewer: DamagedHelmet with Cook-Torrance direct lighting
// (one point light) + split-sum IBL from an equirectangular EXR env map,
// entirely as one Dressi graph (the same forward shader the optimization
// examples differentiate).
//
// The IBL precompute (irradiance / prefiltered env / BRDF LUT) reads only
// the static env leaf: the reactive cache executes it during warmup and
// prunes it, so the steady-state frame pays only rasterization + one fused
// shading pass. Per frame the same leaf set {clip, camera, light} is
// uploaded and the LDR image read back in a single queue submit.

#include "run.h"

#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "../common/asset_utils.h"
#include "../common/bench.h"
#include "../common/pbr_graph.h"
#include "dressi/dressi.h"

using namespace dressi;
using namespace dressi_examples;

namespace {

CpuImage Vec3Img(float x, float y, float z) {
    CpuImage img(1, 1, 3);
    img.data = {x, y, z};
    return img;
}

CpuImage FallbackTex(float r, float g, float b) {
    CpuImage img(2, 2, 3);
    for (uint32_t p = 0; p < 4; p++) {
        img.data[p * 3 + 0] = r;
        img.data[p * 3 + 1] = g;
        img.data[p * 3 + 2] = b;
    }
    return img;
}

}  // namespace

int dressi_examples::RunPbrShading(const std::vector<std::string>& args,
                                   ExampleHost& host) try {
    spdlog::cfg::load_env_levels();

    std::string mesh_path = "data/DamagedHelmet/glTF/DamagedHelmet.gltf";
    std::string env_path = "data/suburban_garden_4k.exr";
    std::string out_dir = "pbr_out";
    uint32_t size = 512;
    int max_frames = 0;  // 0 = until the window is closed
    int view_interval = 1;  // live-viewer refresh cadence (frames); 0 = off
    for (const std::string& arg : args) {
        if (arg.rfind("--mesh=", 0) == 0) {
            mesh_path = arg.substr(7);
        } else if (arg.rfind("--env=", 0) == 0) {
            env_path = arg.substr(6);
        } else if (arg.rfind("--out-dir=", 0) == 0) {
            out_dir = arg.substr(10);
        } else if (arg.rfind("--size=", 0) == 0) {
            size = uint32_t(std::stoi(arg.substr(7)));
        } else if (arg.rfind("--frames=", 0) == 0) {
            max_frames = std::stoi(arg.substr(9));
        } else if (arg.rfind("--view-interval=", 0) == 0) {
            view_interval = std::stoi(arg.substr(16));
        } else {
            mesh_path = arg;
        }
    }
    const ImgSize screen = {size, size};
    std::filesystem::create_directories(out_dir);

    // ------------------------- Load assets (CPU) -------------------------
    const GltfScene gltf = LoadGltfScene(mesh_path);
    spdlog::info("mesh: {} verts, {} faces; textures: albedo {}x{}",
                 gltf.mesh.numVertices(), gltf.mesh.numFaces(),
                 gltf.base_color.width, gltf.base_color.height);

    CpuImage env_img = LoadImageExr(env_path);
    while (env_img.width > 512) {
        env_img = DownsampleHalf(env_img);
    }
    spdlog::info("env map: {} -> {}x{}", env_path, env_img.width,
                 env_img.height);

    const uint32_t n_verts = gltf.mesh.numVertices();
    CpuImage uvh_img(n_verts, 1, 4);
    CpuImage tangent_img(n_verts, 1, 3);
    for (uint32_t v = 0; v < n_verts; v++) {
        uvh_img.at(v, 0, 0) = gltf.mesh.uv.at(v, 0, 0);
        uvh_img.at(v, 0, 1) = gltf.mesh.uv.at(v, 0, 1);
        uvh_img.at(v, 0, 2) = 1.f;
        uvh_img.at(v, 0, 3) = gltf.tangent.at(v, 0, 3);
        for (uint32_t c = 0; c < 3; c++) {
            tangent_img.at(v, 0, c) = gltf.tangent.at(v, 0, c);
        }
    }
    const CpuImage albedo_img = gltf.base_color.width
                                        ? SrgbToLinear(gltf.base_color)
                                        : FallbackTex(0.8f, 0.8f, 0.8f);
    const CpuImage mr_img = gltf.metallic_roughness.width
                                    ? gltf.metallic_roughness
                                    : FallbackTex(0.f, 0.6f, 0.f);
    const CpuImage nrm_img = gltf.normal_map.width ? gltf.normal_map
                                                   : FallbackTex(0.5f, 0.5f, 1.f);
    const CpuImage ao_img =
            gltf.occlusion.width ? gltf.occlusion : FallbackTex(1.f, 1.f, 1.f);
    const CpuImage emis_img = gltf.emissive.width ? SrgbToLinear(gltf.emissive)
                                                  : FallbackTex(0.f, 0.f, 0.f);

    // ------------------------------ Graph --------------------------------
    PbrGeometryLeaves geom;
    geom.clip = Variable(VEC4, {n_verts, 1});
    geom.uvh = Variable(VEC4, {n_verts, 1});
    geom.normal = Variable(VEC3, {n_verts, 1});
    geom.tangent = Variable(VEC3, {n_verts, 1});
    geom.wpos = Variable(VEC3, {n_verts, 1});
    geom.faces = Variable(IVEC3, {gltf.mesh.numFaces(), 1});
    PbrMaterialLeaves mat;
    mat.albedo = Variable(VEC3, albedo_img.getImgSize());
    mat.mr = Variable(VEC3, mr_img.getImgSize());
    mat.normal_map = Variable(VEC3, nrm_img.getImgSize());
    mat.ao = Variable(VEC3, ao_img.getImgSize());
    mat.emissive = Variable(VEC3, emis_img.getImgSize());

    PbrFrameLeaves frame;
    frame.cam_pos = Variable(VEC3, {1, 1});
    frame.cam_fwd = Variable(VEC3, {1, 1});
    frame.cam_right = Variable(VEC3, {1, 1});
    frame.cam_up = Variable(VEC3, {1, 1});
    frame.light_pos = Variable(VEC3, {1, 1});
    frame.light_col = Variable(VEC3, {1, 1});

    Variable env(VEC3, env_img.getImgSize());
    const PbrIblMaps ibl = BuildPbrIblMaps(env);
    // Viewer: no gradients -> the inverse-UV table is a never-uploaded dummy
    Variable dummy_inv_uv(VEC4, albedo_img.getImgSize());
    const PbrOutputs out =
            BuildPbrForward(geom, mat, frame, ibl, dummy_inv_uv, screen);

    DressiAD ad;
    ad.setLossVar(out.ldr);  // forward-only: the image itself is the "loss"
    ad.markOutput(out.mask);  // RSP would fuse it away (recvImg gotcha)
    ad.markOutput(out.albedo);
    ad.markOutput(out.normal);

    ad.sendImg(geom.uvh, uvh_img);
    ad.sendImg(geom.normal, gltf.normal);
    ad.sendImg(geom.tangent, tangent_img);
    ad.sendImg(geom.wpos, gltf.mesh.pos);
    ad.sendImg(geom.faces, gltf.mesh.faces);
    ad.sendImg(mat.albedo, albedo_img);
    ad.sendImg(mat.mr, mr_img);
    ad.sendImg(mat.normal_map, nrm_img);
    ad.sendImg(mat.ao, ao_img);
    ad.sendImg(mat.emissive, emis_img);
    ad.sendImg(env, env_img);

    // --------------------------- Render loop -----------------------------
    const float fov = 45.f * 3.14159265f / 180.f;
    const float tan_half = std::tan(fov * 0.5f);
    const Mat4 proj = PerspectiveVk(fov, 1.f, 0.1f, 10.f);
    const std::array<float, 3> light_pos = {2.f, 2.f, 2.f};
    const CpuImage light_pos_img =
            Vec3Img(light_pos[0], light_pos[1], light_pos[2]);
    const CpuImage light_col_img = Vec3Img(20.f, 20.f, 20.f);

    auto viewer = host.makeViewer(screen.w, screen.h, "pbr_shading (Dressi)");
    bool viewer_open = viewer->valid();
    if (!viewer_open && max_frames == 0) {
        max_frames = 60;
        spdlog::warn("live viewer unavailable; rendering {} frames headless",
                     max_frames);
    }

    using Clock = std::chrono::steady_clock;
    const auto t_start = Clock::now();
    std::vector<double> exec_ms;  // steady-state samples (>= warmup)
    std::vector<double> warmup_samples;  // the excluded build/rebuild frames
    const int warmup = 20;
    CpuImage frame_img;
    int frame_idx = 0;
    for (;; frame_idx++) {
        if (max_frames > 0 && frame_idx >= max_frames) {
            break;
        }
        if (host.cancelled()) {
            break;
        }
        const float t = std::chrono::duration<float>(Clock::now() - t_start)
                                .count();
        // Interactive: time-based orbit. Headless: one full orbit over the
        // frame budget (deterministic PNGs at the quarter points).
        const float az = max_frames > 0 ? 2.f * 3.14159265f * float(frame_idx) /
                                                  float(max_frames)
                                        : 0.5f * t;
        const std::array<float, 3> eye = {2.2f * std::sin(az) * 0.94f,
                                          0.75f,
                                          2.2f * std::cos(az) * 0.94f};
        const Mat4 view = LookAt(eye, {0, 0, 0}, {0, 1, 0});
        const CpuImage clip_img =
                TransformToClip(gltf.mesh.pos, Mul(proj, view));
        // Camera basis from the view matrix rows (s / u / -f)
        const CpuImage cam_pos_img = Vec3Img(eye[0], eye[1], eye[2]);
        const CpuImage cam_fwd_img =
                Vec3Img(-view[2], -view[6], -view[10]);
        const CpuImage cam_right_img = Vec3Img(view[0] * tan_half,
                                               view[4] * tan_half,
                                               view[8] * tan_half);
        const CpuImage cam_up_img = Vec3Img(view[1] * tan_half,
                                            view[5] * tan_half,
                                            view[9] * tan_half);

        const auto t0 = Clock::now();
        frame_img = ad.execStepWithSendsAndRecvImgsStacked(
                {{geom.clip, CpuImageView(clip_img)},
                 {frame.cam_pos, CpuImageView(cam_pos_img)},
                 {frame.cam_fwd, CpuImageView(cam_fwd_img)},
                 {frame.cam_right, CpuImageView(cam_right_img)},
                 {frame.cam_up, CpuImageView(cam_up_img)},
                 {frame.light_pos, CpuImageView(light_pos_img)},
                 {frame.light_col, CpuImageView(light_col_img)}},
                {out.ldr});
        const double ms = std::chrono::duration<double, std::milli>(
                                  Clock::now() - t0)
                                  .count();
        if (frame_idx >= warmup) {
            exec_ms.push_back(ms);
        } else {
            warmup_samples.push_back(ms);
        }

        if (viewer_open && view_interval > 0 &&
            frame_idx % view_interval == 0) {
            viewer_open = viewer->update(frame_img);
            if (!viewer_open && max_frames == 0) {
                break;
            }
        }
        if (max_frames >= 4 && frame_idx % (max_frames / 4) == 0) {
            SaveImagePng(fmt::format("{}/frame_{:03d}.png", out_dir,
                                     frame_idx),
                         frame_img);
        }
        if (frame_idx % 100 == 0 && !exec_ms.empty()) {
            std::vector<double> s = exec_ms;
            std::nth_element(s.begin(), s.begin() + s.size() / 2, s.end());
            const double med = s[s.size() / 2];
            spdlog::info(
                    "frame {:4d}  exec+readback median {:.3f} ms ({:.0f} FPS"
                    " excl. display)  stages {}",
                    frame_idx, med, 1000.0 / med, ad.getStageCount());
        }
    }

    if (!exec_ms.empty()) {
        const double med = MedianMs(exec_ms);
        const double warmup_ms = WarmupMs(warmup_samples, med);
        const double first_build_ms =
                warmup_samples.empty() ? 0.0 : warmup_samples.front();
        spdlog::info(
                "steady state ({} frames after {}-frame warmup): median "
                "{:.3f} ms/frame = {:.0f} FPS (excl. display); one-time "
                "build/warmup {:.1f} ms (first build {:.1f} ms)",
                exec_ms.size(), warmup, med, 1000.0 / med, warmup_ms,
                first_build_ms);
        // Benchmark record for scripts/bench_summary.py (a viewer, not an
        // optimization — the metric is exec+readback per frame)
        BenchRecord rec("pbr_shading", ad.getDeviceName());
        rec.addPacking(ad.getFuncCount(), ad.getSubStageCount(),
                       ad.getStageCount());
        rec.add("screen", int64_t(screen.w));
        rec.add("frames", int64_t(frame_idx));
        rec.add("median_ms_per_iter", med);
        rec.add("warmup_excluded", int64_t(warmup));
        rec.add("warmup_ms", warmup_ms, 1);
        rec.add("first_build_ms", first_build_ms, 1);
        rec.add("fps", 1000.0 / med, 1);
        rec.save(out_dir + "/bench.json");
    }
    SaveImagePng(out_dir + "/frame_final.png", frame_img);
    SaveImagePng(out_dir + "/debug_albedo.png", ad.recvImg(out.albedo));
    {
        CpuImage n_img = ad.recvImg(out.normal);
        for (float& v : n_img.data) {
            v = v * 0.5f + 0.5f;
        }
        SaveImagePng(out_dir + "/debug_normal.png", n_img);
    }
    spdlog::info("saved {}/frame_final.png", out_dir);

    // Sanity: covered pixels exist and the image is finite
    size_t covered = 0;
    bool finite = true;
    for (float px : frame_img.data) {
        finite = finite && std::isfinite(px);
    }
    const CpuImage mask_img = ad.recvImg(out.mask);
    for (float m : mask_img.data) {
        covered += m > 0.5f ? 1 : 0;
    }
    spdlog::info("coverage {} px, finite {}", covered, finite);
    return (covered > 0 && finite) ? 0 : 1;
} catch (const std::exception& e) {
    spdlog::error("error: {}", e.what());
    return 2;
}
