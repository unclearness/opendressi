# opendressi

A C++/Vulkan re-implementation of the **Dressi-AD** layer from the paper
*"Dressi: A Hardware-Agnostic Differentiable Renderer with Reactive Shader
Packing and Soft Rasterization"* (Takimoto et al., Eurographics 2022,
[arXiv:2204.01386](https://arxiv.org/abs/2204.01386)).

Dressi-AD is a define-and-run reverse-mode automatic differentiation
framework specialized for differentiable rendering: computational graphs of
2D images are compiled into GLSL fragment shaders, packed into a minimal
number of Vulkan render passes (Reactive Shader Packing), and executed on
any Vulkan-capable GPU. The public C++ API follows Appendix A of the paper.

Vertex-position gradients are provided by two techniques: **HardSoftRas**
from the Dressi paper and the differentiable **screen-space anti-aliasing**
from *"Dr.Hair: Reconstructing Scalp-Connected Hair Strands without
Pre-training via Differentiable Rendering of Line Segments"* (Takimoto et
al., CVPR 2024, [arXiv:2403.17496](https://arxiv.org/abs/2403.17496)); see
[References](#references) for how to cite.

## Milestone 3: vertex-position gradients (HardSoftRas + screen-space AA)

- `F::RasterizeSoft(...)`: HardSoftRas (Dressi Alg. 2, `--peels=K` depth
  peeling) — GPU-enlarged triangles supply a face-ID buffer with the
  paper's Eq. 3 depth shift; the pixel-to-edge distance is built from
  ordinary image-space ops over `F::LookupFaces` / `F::FaceFetch`'d corner
  positions, so the backward is generated entirely by the AD. Gradients
  cross the indexed reads with the paper's inverse-UV philosophy
  (stochastic jittered edge samples, Alg. 1).
- `F::AntiAlias(...)` + `F::RasterizeFaceId(...)`: the Dr.Hair screen-space
  AA (Eq. 3-5) — blends across triangle-ID boundaries by the
  pixel-to-silhouette-edge distance; differentiable w.r.t. the image and
  the vertex positions (exact or stochastic backward via `--samples`).
  Works on general triangle meshes.
- `examples/silhouette_optimization`: deforms an icosphere into the bunny
  (or self-reconstructs a glTF mesh via `--mesh=`, e.g. the paper's
  Avocado) from multi-view silhouettes. The whole iteration is
  GPU-resident (the paper's transfer model): in-graph projections, soft
  geometry, uniform-Laplacian / normal-consistency regularizers and Adam
  with GPU-resident state (`DressiAD::addUpdate`); per-iteration CPU
  traffic is zero. Mask IoU ~0.98 for both techniques.

## Performance

Measured on the paper's Table 4 mesh — the glTF sample **Avocado**
(406 vertices / 682 faces) — with silhouette self-reconstruction: one
full iteration = rasterization + backward to the 3D vertex positions +
regularizers + Adam, one view. Hardware: RTX PRO 6000 (Blackwell);
the paper used an RTX 2080 (roughly 10x less raw throughput).

### vs the paper (Table 4, ms per iteration)

| resolution | paper (RTX 2080) | ours HardSoftRas K=1 | ours K=3 | ours AA |
|---|---|---|---|---|
| 256²  | 0.304 | 0.20 | 0.39 | 0.49 |
| 512²  | 0.442 | 0.28 | 0.65 | 0.52 |
| 1024² | 1.034 | 0.58 | 1.38 | 0.54 |
| 2048² | 3.301 | 1.54 | 4.49 | 0.60 |
| 4096² | —     | ~5.5 | —    | 0.81 |

The scaling shape matches the paper (fixed per-pass overhead dominates at
low resolutions; pixel work takes over above 1024²). K=1 beats the
paper's absolute numbers at every resolution; accounting for the GPU
generation an effective ~3-5x remains, which maps onto the known
unimplemented items (`{1,1}` values as real uniforms, zero-copy optimizer
aliasing).

### vs nvdiffrast (same GPU, same task; `scripts/nvdiffrast_bench.py`)

The current nvdiffrast (0.4.0) has removed the OpenGL rasterizer
(`RasterizeGLContext` delegates to CUDA), so the paper-era GL backend is
measured on a pinned v0.2.6 (late 2021 — the version the paper compared
against; the CUDA rasterizer only appeared in v0.3.0, 2022-03).

| resolution | nvdiffrast GL v0.2.6 | nvdiffrast CUDA v0.4.0 | ours HSR K=1 | ours AA |
|---|---|---|---|---|
| 256²  | 1.78 | 1.11 | **0.20** | 0.49 |
| 512²  | 1.76 | 1.18 | **0.28** | 0.52 |
| 1024² | 1.82 | 1.21 | **0.58** | 0.54 |
| 2048² | 1.70 | 1.28 | 1.54 | **0.60** |
| 4096² | 3.47 | 3.28 | ~5.5 | **0.81** |

### Discussion

- **Why Dressi wins below 2048²:** every nvdiffrast version/backend
  measures flat (~1.1-1.8 ms) here — at this mesh size a PyTorch
  iteration is bound by host-side kernel launches, not GPU work. Dressi
  bakes the whole forward + backward + optimizer into pre-recorded Vulkan
  command buffers with zero per-iteration host involvement, so it runs at
  the cost of the actual GPU work (0.2 ms).
- **Why the paper's "gap grows with resolution" does not reproduce
  as-is:** on the RTX 2080, nvdiffrast's 2048² iteration was
  GPU-work-bound (5.4-8.0 ms in Table 4). On a Blackwell-class GPU the
  same pixel work costs well under 1 ms and hides beneath the host-launch
  floor until 4096². The advantage source is the same as in the paper —
  it just saturates earlier on modern hardware.
- **The AA technique is nearly resolution-independent** (0.49 → 0.81 ms
  for a 256x pixel increase): its 8-neighbor forward early-outs on equal
  triangle IDs, making the cost silhouette-perimeter-bound rather than
  area-bound, and its stochastic vertex backward is O(faces). It beats
  both nvdiffrast backends at every resolution, including 4x at 4096².
- **Honest limitation:** in the pixel-bound regime (HardSoftRas at
  4096²) nvdiffrast's hand-tuned CUDA kernels are leaner per pixel than
  our generated full-screen RGBA32F passes (3.3-3.5 vs ~5.5 ms); reducing
  the loss/AA chains' image traffic is the known lever there.
- Quality on this benchmark: self-reconstruction reaches silhouette IoU
  0.986 (HardSoftRas, K=3) / 0.977 (AA); the stochastic backwards trade
  a little per-iteration accuracy for the O(faces) cost and recover it
  with iterations and `--samples`/`--peels`, exactly as in the paper.

## Milestone 2: deferred rasterization + texture optimization

- `F::Rasterize(clip_pos, attrib, faces, screen_size)`: depth-tested indexed
  rasterization of a vertex attribute into a screen-space image (G-buffer
  channel; deferred shading per the paper). Non-differentiable with respect
  to geometry (HardSoftRas triangle enlargement is a later milestone).
- `F::Texture(tex, uv, inv_uv)`: nearest sampling, differentiable with
  respect to the texture via the paper's inverse-UV lookup (the mesh
  rasterized in UV space with screen positions as the attribute), with an
  occlusion guard and on-the-fly dilation for chart-boundary texels.
- Sobol-jittered sampling (the paper's quasi-random countermeasure against
  texels that never receive updates): each iteration offsets the sampling
  UVs by a 2D Sobol point and renders the target in-graph with the GT
  texture under the same jitter, so fresh texels are constrained every step.
- `examples/texture_optimization`: multi-view unlit rendering of
  `data/bunny`. Starting from a black texture, the summed per-view MSE
  densely recovers the atlas (~640k texels, zero-jitter rendered loss
  ~5e-6). Saves target views and first/last renders under `texopt_out/`.
- spdlog-based performance logging: `SPDLOG_LEVEL=debug` prints per-phase
  execStep timings (backward / packing / codegen / Vulkan build / upload /
  GPU execute) and rebuild summaries.

## Milestone 1 scope

- Graph core: `Variable` / `Function` handles over a shared node graph,
  type/size inference with `{1,1}` uniform broadcasting
- `namespace F` operators with forward GLSL snippets and backward
  generators: arithmetic, the PBR math set (trig/exp/log/pow/min/max/
  clamp/mix/step/smoothstep), vector ops (dot/cross/normalize/reflect/...),
  matrix ops, spatial reductions (`Sum`/`Mean` via log-step 2x2 reduction)
- `BuildBackward`: reverse-mode gradient graph construction with
  gradient-shape reduction for broadcast inputs
- CPU reference evaluator used as the test oracle (forward parity and
  central finite-difference gradient checks)
- GLSL code generation and a headless Vulkan executor built on
  [VulkanWrapper](https://github.com/unclearness/VulkanWrapper)
  (fork of [takiyu/VulkanWrapper](https://github.com/takiyu/VulkanWrapper)
  with Vulkan SDK 1.4.x compatibility patches)
- Two packing modes: **Naive** (1 function = 1 pass, correctness baseline)
  and **RSP** (greedy substage/stage packing with subpass input attachments)
- Reactive cache: dirty tracking, FAST/FULL rebuild thresholds, pruning of
  clean cached branches, automatic re-expansion when inputs change
- `DressiAD` driver: `setLossVar` / `setOptimizer` / `sendImg` / `recvImg` /
  `execStep` (InitStatus ladder per the paper)

Not yet implemented (later milestones): depth-peeling execution (K > 1),
texture backward w.r.t. UV, compute-shader substages, int-image execution,
matrix images on GPU.

## Building

Requirements: Windows, Visual Studio 2022, CMake >= 3.28,
Vulkan SDK (auto-detected under `C:\VulkanSDK`), a Vulkan 1.1+ GPU.

```sh
git clone --recursive <this repo>
cmake --preset msvc
cmake --build --preset release
ctest --preset release            # all tests
ctest --preset release -LE gpu    # CPU-only tests (no GPU required)
./build/examples/Release/image_fitting.exe
```

## Example

```cpp
Variable param(VEC3, {64, 64});
Variable target(VEC3, {64, 64});
Variable rendered = param / (param + 1.f);       // graph of F:: ops
Variable loss = F::Mean((target - rendered) * (target - rendered));
param.setRequiresGradRecursively();

DressiAD ad;
ad.setLossVar(loss);
ad.setOptimizer([](Variables xs, Variables gxs) {  // SGD
    Variables updated;
    for (size_t i = 0; i < xs.size(); i++) {
        updated.push_back(xs[i] - gxs[i] * 0.01f);
    }
    return updated;
});
ad.sendImg(param, init_img);
ad.sendImg(target, target_img);
for (int i = 0; i < 300; i++) {
    ad.execStep();
}
CpuImage optimized = ad.recvImg(param);
```

## References

This repository re-implements techniques from the following papers. **If
you use this code in academic work, please cite them** — the Dressi paper
for the renderer as a whole (Dressi-AD, Reactive Shader Packing,
HardSoftRas), and additionally the Dr.Hair paper when you use the
screen-space anti-aliasing vertex gradients (`F::AntiAlias`).

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

MIT (see [LICENSE](LICENSE)). Third-party dependencies keep their own
licenses: [VulkanWrapper](https://github.com/unclearness/VulkanWrapper)
(MIT), glad (generated loader), GLFW (zlib), tinyobjloader (MIT), stb
(public domain / MIT), spdlog (MIT), GoogleTest (BSD-3).

## Layout

```
include/dressi/   public API (Appendix A style)
src/core/         graph nodes, F:: ops, BuildBackward, CPU evaluator
src/codegen/      substage -> GLSL fragment shader generation
src/pack/         trivial & greedy (RSP) packers, reactive pruning
src/vk/           headless Vulkan context, executor, CPU<->GPU transfer
tests/            GoogleTest suites (ctest labels: cpu / gpu)
examples/         image_fitting / texture_optimization /
                  silhouette_optimization demos
```
