# dressi

A C++/Vulkan unofficial re-implementation of the **Dressi-AD** layer from the
paper *"Dressi: A Hardware-Agnostic Differentiable Renderer with Reactive
Shader Packing and Soft Rasterization"* (Takimoto et al., Eurographics 2022,
[arXiv:2204.01386](https://arxiv.org/abs/2204.01386)).

<!-- GIFs are hosted on the project wiki (demo/ in the wiki repo), not in
     this repository. Regenerate with scripts/make_readme_gifs.sh. -->
<table>
  <tr>
    <td align="center" width="25%">
      <img src="https://raw.githubusercontent.com/wiki/unclearness/dressi/demo/image_fitting.gif" alt="image fitting" width="100%"><br>
      <sub><a href="examples/image_fitting">image_fitting</a></sub>
    </td>
    <td align="center" width="25%">
      <img src="https://raw.githubusercontent.com/wiki/unclearness/dressi/demo/texture_optimization.gif" alt="texture optimization" width="100%"><br>
      <sub><a href="examples/texture_optimization">texture_optimization</a></sub>
    </td>
    <td align="center" width="25%">
      <img src="https://raw.githubusercontent.com/wiki/unclearness/dressi/demo/silhouette_hardsoftras.gif" alt="silhouette optimization (HardSoftRas)" width="100%"><br>
      <sub><a href="examples/silhouette_optimization">silhouette</a> (HardSoftRas)</sub>
    </td>
    <td align="center" width="25%">
      <img src="https://raw.githubusercontent.com/wiki/unclearness/dressi/demo/silhouette_aa.gif" alt="silhouette optimization (screen-space AA)" width="100%"><br>
      <sub><a href="examples/silhouette_optimization">silhouette</a> (AA)</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="25%">
      <img src="https://raw.githubusercontent.com/wiki/unclearness/dressi/demo/shape_texture_optimization.gif" alt="shape then texture optimization" width="100%"><br>
      <sub><a href="examples/shape_texture_optimization">shape_texture_optimization</a></sub>
    </td>
    <td align="center" width="25%">
      <img src="https://raw.githubusercontent.com/wiki/unclearness/dressi/demo/pbr_shading.gif" alt="PBR shading orbit" width="100%"><br>
      <sub><a href="examples/pbr_shading">pbr_shading</a></sub>
    </td>
    <td align="center" width="25%">
      <img src="https://raw.githubusercontent.com/wiki/unclearness/dressi/demo/pbr_material_optimization.gif" alt="PBR material optimization" width="100%"><br>
      <sub><a href="examples/pbr_material_optimization">pbr_material_optimization</a></sub>
    </td>
    <td align="center" width="25%">
      <img src="https://raw.githubusercontent.com/wiki/unclearness/dressi/demo/pbr_envmap_optimization.gif" alt="PBR environment-map optimization" width="100%"><br>
      <sub><a href="examples/pbr_envmap_optimization">pbr_envmap_optimization</a></sub>
    </td>
  </tr>
</table>

All sequences are live optimization / rendering progress on the Vulkan
engine (every example also runs unmodified on Android — see below):

📹 **shape_texture_optimization on a phone** — camera capture of the
on-device run (Xiaomi 17T Pro, Dimensity 9500 / Mali-G1-Ultra), showing the
target / prediction / recovered-atlas streams tiled on one screen
([full-quality video](https://raw.githubusercontent.com/wiki/unclearness/dressi/demo/20260718_shape_texture_xiaomi_17tpro_dimensity9500_mali.mp4)):

https://github.com/user-attachments/assets/3107391a-ded4-4245-8d2c-35b3ea4d7f65

Dressi-AD is a define-and-run reverse-mode automatic-differentiation framework
specialized for differentiable rendering: computational graphs of 2D images
are compiled to GLSL fragment shaders, packed into a minimal number of Vulkan
render passes (**Reactive Shader Packing**), and executed on any Vulkan-capable
GPU. Because the backend is portable Vulkan — not CUDA — the *same* graph runs
on a workstation GPU, an integrated GPU, a pure-CPU software rasterizer, and a
phone, to the same numerical quality. The public C++ API follows Appendix A of
the paper.

Vertex-position gradients come from two techniques: **HardSoftRas** from the
Dressi paper, and the differentiable **screen-space anti-aliasing** from
*"Dr.Hair"* (Takimoto et al., CVPR 2024,
[arXiv:2403.17496](https://arxiv.org/abs/2403.17496)). A split-sum **PBR / IBL**
layer (physically based shading, material and environment-map optimization)
builds on the same engine.

### Highlights

- **Hardware-agnostic.** One Vulkan engine, verified across 8 devices — three
  NVIDIA discrete GPUs, an Intel iGPU, two phones (Dimensity 9500 / Mali-G1-Ultra,
  Xiaomi 17T Pro and Dimensity 9200+ / Mali-G715, Xiaomi 13T Pro), a tablet
  (Snapdragon 8 Gen 2 / Adreno 740, Xiaomi Pad 6S), and CPU software Vulkan
  (llvmpipe).
- **Faster than the baselines below 2048².** Beats nvdiffrast (GL & CUDA) and
  DRTK at every resolution on the paper's silhouette task, by baking the whole
  forward + backward + optimizer into pre-recorded command buffers with zero
  per-iteration host traffic.
- **Runs on phones, unmodified.** All six examples run on Android with the same
  parameters and results as desktop (see [README-android.md](README-android.md)).
- **PyTorch drop-in.** `dressi.torch` is an nvdiffrast-compatible API with no
  CUDA requirement.

## Contents

- [Building](#building) — [Windows](#windows), [Linux](#linux),
  [Android](#android), [PyTorch bindings](#pytorch-bindings-dressitorch)
- [Examples](#examples)
- [Performance](#performance) · full tables in [docs/benchmarks.md](docs/benchmarks.md)
- [How it works](#how-it-works) · details in [docs/algorithm.md](docs/algorithm.md)
- [References](#references) · [License](#license) · [Layout](#layout)

## Building

Common requirements: a Vulkan 1.1+ GPU, CMake ≥ 3.28, and the LunarG Vulkan
SDK (used for glslang; auto-detected — see below). The engine core is headless
(`VkContext`); GLFW/glad drive an optional live viewer only.

Build verified with **Vulkan SDK 1.4.350.0** (desktop; the Android build
FetchContent-pins glslang / SPIRV-Tools / Vulkan-Headers to the matching
`vulkan-sdk-1.4.350.0` tag). The engine creates its instance at API version
**Vulkan 1.1**, so it runs on any 1.1+ driver — see
[docs/benchmarks.md](docs/benchmarks.md) for the runtime versions verified
across devices.

### Windows

Visual Studio 2022, x64. The Vulkan SDK is auto-detected under `C:\VulkanSDK\*`
(no `VULKAN_SDK` env var required).

```sh
git clone --recursive <this repo>
cmake --preset msvc                # also downloads the example datasets
cmake --build --preset release
ctest --preset release             # all tests (labels: cpu / gpu)
ctest --preset release -LE gpu     # CPU-only tests (no GPU required)
./build/examples/Release/image_fitting.exe
```

### Linux

Tested on Ubuntu 24.04 (gcc 14 / Ninja). Install the LunarG SDK tarball (no
root needed) under `~/vulkan/<version>/` — it is auto-detected (globs
`~/vulkan/*`, `~/VulkanSDK/*`, `/opt/vulkan/*`; `VULKAN_SDK` still wins). A
system `libvulkan` loader + your GPU's ICD provide the runtime.

```sh
git clone --recursive <this repo>
cmake --preset linux
cmake --build --preset linux-release
ctest --preset linux-release
./build/examples/image_fitting
```

### Android

`android/` is an Android Studio project bundling all six examples into one APK
(Kotlin UI + JNI, live view on a SurfaceView, on-device GLSL→SPIR-V). Every
example completes with the same parameters and results as desktop. Build the
**release** variant for meaningful timings (the debug native lib is
unoptimized). Full instructions and on-device notes:
**[README-android.md](README-android.md)**.

```powershell
cd android
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
.\gradlew.bat :app:assembleRelease
adb install -r app\build\outputs\apk\release\app-release.apk
```

### PyTorch bindings (`dressi.torch`)

Python deps are managed with [uv](https://docs.astral.sh/uv/) (`uv sync`
creates `.venv` with CUDA torch; `--extra nvdiffrast` adds the comparison
backend). The bindings themselves need **no** CUDA — they run on any Vulkan GPU.

```sh
uv sync
cmake --preset msvc -DDRESSI_BUILD_PYTHON=ON -DPython_EXECUTABLE=.venv/Scripts/python.exe
cmake --build --preset release --target dressi_py
uv run pytest tests/python
```

### Datasets

`data/` is git-ignored (large / licensed) and populated at CMake **configure**
time by [`cmake/FetchAssets.cmake`](cmake/FetchAssets.cmake) (hash-verified,
skipped if present). Pass `-DDRESSI_FETCH_DATA=OFF` to provision `data/`
yourself for offline builds.

| Path | Source | Size |
| --- | --- | --- |
| `data/bunny/` (obj + mtl + atlas) | [kunzhou.net](http://www.kunzhou.net/tex-models/bunny.zip) | ~3 MB |
| `data/Avocado/glTF/` (paper Table 4 mesh) | [glTF-Sample-Models](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/Avocado) | ~8 MB |
| `data/suburban_garden_4k.exr` (PBS env map) | [Poly Haven](https://polyhaven.com/a/suburban_garden) | ~97 MB |

## Examples

Six examples ship as native executables (and a Python port of two of them):

```sh
# --- differentiable rendering core ---
./build/examples/Release/image_fitting.exe                       # smoke test
./build/examples/Release/texture_optimization.exe data/bunny     # inverse-UV texture grad
./build/examples/Release/silhouette_optimization.exe data/bunny --technique=hardsoftras
./build/examples/Release/silhouette_optimization.exe data/bunny --technique=aa
./build/examples/Release/silhouette_optimization.exe --mesh=data/Avocado/glTF/Avocado.gltf

# --- physically based shading / IBL ---
./build/examples/Release/pbr_shading.exe                         # DamagedHelmet viewer
./build/examples/Release/pbr_material_optimization.exe           # albedo recovery
./build/examples/Release/pbr_envmap_optimization.exe             # environment recovery

# --- PyTorch (nvdiffrast drop-in; same code, either backend) ---
uv run python examples/python/silhouette_optimization.py                       # dressi (Vulkan)
uv run python examples/python/silhouette_optimization.py --backend nvdiffrast  # CUDA
```

A minimal C++ optimization loop:

```cpp
Variable param(VEC3, {64, 64});
Variable target(VEC3, {64, 64});
Variable rendered = param / (param + 1.f);                        // graph of F:: ops
Variable loss = F::Mean((target - rendered) * (target - rendered));
param.setRequiresGradRecursively();

DressiAD ad;
ad.setLossVar(loss);
ad.setOptimizer([](Variables xs, Variables gxs) {                 // SGD
    Variables updated;
    for (size_t i = 0; i < xs.size(); i++)
        updated.push_back(xs[i] - gxs[i] * 0.01f);
    return updated;
});
ad.sendImg(param, init_img);
ad.sendImg(target, target_img);
for (int i = 0; i < 300; i++) ad.execStep();
CpuImage optimized = ad.recvImg(param);
```

## Performance

All figures are the **median steady-state `execStep` wall time** with a 20-iter
warmup excluded (the first steps pay a one-time pack + GLSL-compile + Vulkan
build). At ≤512² the wall is dominated by host submit/fence latency, not GPU
work (GPU timestamps total ~90 µs for HSR at 256² while the wall is ~0.34 ms),
so median (not mean) is the stable metric.

### On-device / cross-device

The identical backward graph runs from a workstation GPU, through an Intel
integrated GPU, down to phones — to the **same quality**. Median ms/iter
(default parameters):

| example (params) | RTX PRO 6000 | Iris Xe (iGPU) | Adreno 740 (tablet) | Mali-G1-Ultra (phone) | Mali-G715 (phone) | quality |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| image_fitting 64px | 0.056 | 0.32 | 0.23 | 1.01 | 2.04 | max err 0.0005 |
| texture_optimization 256px | 0.96 | 15.7 | 48.4 | 23.6 | 24.3 | 98.7% |
| silhouette (aa) 128px | 1.21 | 3.34 | 21.6 | 11.5 | 21.2 | IoU 0.95 |
| silhouette (hardsoftras) 128px | 2.16 | 9.96 | 23.7 | 32.5 | 36.6 | IoU 0.95 |
| pbr_material 256px | 2.73 | 17.9 | 55.2 | 40.9 | 47.0 | PSNR 20.6 dB |
| pbr_envmap 192px | 8.53 | 105 | 304 | 181 | 200 | PSNR 18.2 dB |
| pbr_shading 512px | 677 FPS | 116 FPS | 56 FPS | 19 FPS | 42 FPS | — |

Quality is device-independent (validated across all 8 devices). Verified on
three discrete NVIDIA GPUs, the Intel iGPU, two phones (Dimensity 9500 /
Mali-G1-Ultra, Xiaomi 17T Pro and Dimensity 9200+ / Mali-G715, Xiaomi 13T Pro),
a tablet (Snapdragon 8 Gen 2 / Adreno 740, Xiaomi Pad 6S), and CPU software
Vulkan (llvmpipe) — the full 8-device table, warmup and packing columns are in
**[docs/benchmarks.md](docs/benchmarks.md)**.

### vs the paper, nvdiffrast, and DRTK

On the paper's one-view Avocado silhouette task, both techniques **beat every
baseline at every resolution** — at 256² HardSoftRas K=1 is 0.15 ms vs
nvdiffrast CUDA 1.11 ms and DRTK 2.12 ms; at 4096² AA is 0.78 ms vs 3.28 / ~220.
dressi bakes the whole forward + backward + optimizer into pre-recorded command
buffers with zero per-iteration host traffic, so below 2048² — where the PyTorch
baselines are host-launch bound — it runs at the cost of the GPU work itself,
and rematerialization keeps it ahead at 4096² too. The scaling shape matches the
paper (fixed per-pass overhead at low resolution, pixel work above 1024²). Full
tables (vs the paper's Table 4, vs nvdiffrast GL/CUDA and DRTK) and the
discussion: **[docs/benchmarks.md](docs/benchmarks.md)**.

## How it works

Each `DressiAD::execStep()` walks the paper's InitStatus ladder — build backward
graph → wire optimizer → traverse → pack substages → pack stages → build Vulkan
objects → submit — and reruns each step only when its inputs changed (a reactive
cache), so a steady loop re-executes pre-recorded command buffers with nothing
crossing PCIe. In brief:

- **Dressi-AD core + Reactive Shader Packing.** Image-op graphs (`Variable` /
  `F::`) compile to GLSL; a greedy packer fuses same-pixel chains into minimal
  Vulkan render passes under each device's limits, then rematerializes cheap
  producers and drops dead outputs. Every op has a CPU kernel checked by finite
  differences. Losses stay images; the optimizer writes updates in-place
  (zero-copy aliasing).
- **Deferred rasterization + texture optimization.** `F::Rasterize` writes
  G-buffer channels; `F::Texture` is differentiable w.r.t. the texture via the
  paper's inverse-UV lookup (with Sobol-jittered sampling for full coverage).
- **Vertex-position gradients.** `F::RasterizeSoft` (HardSoftRas, `--peels=K`)
  and `F::AntiAlias` (Dr.Hair screen-space AA) flow gradients to 3D positions;
  the fully GPU-resident silhouette loop reaches IoU ~0.98 with zero
  per-iteration CPU traffic.
- **PyTorch bindings.** `dressi.torch` is an nvdiffrast drop-in backed by the
  Vulkan engine (no CUDA), reconciling eager ⇄ define-and-run with a
  shape-signature engine cache; a native-fused path matches C++ speed from
  Python.
- **PBR / IBL.** Split-sum Cook-Torrance shading with material and
  environment-map optimization; all equirectangular, with exact-transpose
  backwards for the environment ops.

Full details — every op, the packing / rematerialization passes, the WIDE gather
backwards, the PyTorch engine cache, and known deviations from the paper — are
in **[docs/algorithm.md](docs/algorithm.md)**.

## References

This repository re-implements techniques from the following papers. **If you
use this code in academic work, please cite them** — the Dressi paper for the
renderer as a whole (Dressi-AD, Reactive Shader Packing, HardSoftRas), and
additionally the Dr.Hair paper when you use the screen-space anti-aliasing
vertex gradients (`F::AntiAlias`).

```bibtex
@article{takimoto2022dressi,
  title   = {Dressi: A Hardware-Agnostic Differentiable Renderer with
             Reactive Shader Packing and Soft Rasterization},
  author  = {Takimoto, Yusuke and Sato, Hiroyuki and Takehara, Hikari and
             Uragaki, Keishiro and Tawara, Takehiro and Liang, Xiao and
             Oku, Kentaro and Kishimoto, Wataru and Zheng, Bo},
  journal = {Computer Graphics Forum},
  volume  = {41},
  number  = {2},
  pages   = {13--27},
  year    = {2022},
  doi     = {10.1111/cgf.14455}
}

@inproceedings{takimoto2024drhair,
  title     = {{Dr.Hair}: Reconstructing Scalp-Connected Hair Strands
               without Pre-training via Differentiable Rendering of Line
               Segments},
  author    = {Takimoto, Yusuke and Takehara, Hikari and Sato, Hiroyuki and
               Zhu, Zihao and Zheng, Bo},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision
               and Pattern Recognition (CVPR)},
  pages     = {20601--20611},
  year      = {2024},
  doi       = {10.1109/CVPR52733.2024.01947}
}
```

## License

MIT (see [LICENSE](LICENSE)). Third-party dependencies keep their own licenses:
[VulkanWrapper](https://github.com/unclearness/VulkanWrapper) (MIT, a fork of
[takiyu/VulkanWrapper](https://github.com/takiyu/VulkanWrapper) with Vulkan SDK
1.4.x patches), glad (generated loader), GLFW (zlib), tinyobjloader (MIT), stb
(public domain / MIT), spdlog (MIT), GoogleTest (BSD-3), tinyexr (BSD).

## Layout

```
include/dressi/   public API (Appendix A style)
src/core/         graph nodes, F:: ops, BuildBackward, CPU evaluator, IBL math
src/codegen/      substage -> GLSL fragment/compute shader generation
src/pack/         trivial & greedy (RSP) packers, reactive pruning
src/vk/           headless Vulkan context, executor, CPU<->GPU transfer
src/python/       nanobind module (dressi._C) + dressi.torch eager API
examples/         image_fitting / texture_optimization / silhouette_optimization
                  / pbr_shading / pbr_material_optimization / pbr_envmap_optimization
android/          one-APK Android app (all six examples; Kotlin + JNI)
tests/            GoogleTest suites (ctest labels: cpu / gpu) + tests/python
```
