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
| 256²  | 0.304 | 0.15 | 0.24 | 0.49 |
| 512²  | 0.442 | 0.18 | 0.35 | 0.49 |
| 1024² | 1.034 | 0.30 | 0.77 | 0.51 |
| 2048² | 3.301 | 0.83 | 2.42 | 0.57 |
| 4096² | —     | 2.87 | —    | 0.78 |

The scaling shape matches the paper (fixed per-pass overhead dominates at
low resolutions; pixel work takes over above 1024²). K=1 beats the
paper's absolute numbers at every resolution. The paper's remaining
machinery is implemented: `{1,1}` leaf values ride in real uniforms
(uif_vars), the optimizer writes updated values directly into the input
images (zero-copy aliasing via input+color attachments in eGeneral with
a subpass self-dependency), long same-pixel chains fuse into single
render passes of chained subpasses, whole-image sums reduce in two
levels, COMP substages execute as compute dispatches, and consumers
rematerialize cheap forward chains instead of reading them through
fp32 attachments (the packer's clone pass plus dead-output
elimination — the main lever in the pixel-bound regime). Two of these
are deliberately scoped by measurement on this GPU: GPU-generated
`{1,1}` scalars stay `texelFetch(0,0)` (a mid-frame UBO refresh costs
more than the broadcast fetch), and hot band-pass gathers stay FRAG
(graphics↔compute switches plus lost elementwise fusion cost more than
the render passes they remove).

### vs nvdiffrast and DRTK (same GPU, same task; `scripts/nvdiffrast_bench.py`, `scripts/drtk_bench.py`)

The current nvdiffrast (0.4.0) has removed the OpenGL rasterizer
(`RasterizeGLContext` delegates to CUDA), so the paper-era GL backend is
measured on a pinned v0.2.6 (late 2021 — the version the paper compared
against; the CUDA rasterizer only appeared in v0.3.0, 2022-03).
[DRTK](https://github.com/facebookresearch/DRTK) (Meta's PyTorch CUDA
differentiable-rasterization kit) is measured with its canonical
silhouette pipeline: `rasterize` + `render` + `edge_grad_estimator`
for the boundary gradients.

| resolution | nvdiffrast GL v0.2.6 | nvdiffrast CUDA v0.4.0 | DRTK 0.1.0 | ours HSR K=1 | ours AA |
|---|---|---|---|---|---|
| 256²  | 1.78 | 1.11 | 2.12 | **0.15** | 0.49 |
| 512²  | 1.76 | 1.18 | 6.15 | **0.18** | 0.49 |
| 1024² | 1.82 | 1.21 | 19.6 | **0.30** | 0.51 |
| 2048² | 1.70 | 1.28 | 67-108 | 0.83 | **0.57** |
| 4096² | 3.47 | 3.28 | ~220 | 2.87 | **0.78** |

Both techniques now beat every baseline at every resolution.

### Discussion

- **Why Dressi wins below 2048²:** every nvdiffrast version/backend
  measures flat (~1.1-1.8 ms) here — at this mesh size a PyTorch
  iteration is bound by host-side kernel launches, not GPU work. Dressi
  bakes the whole forward + backward + optimizer into pre-recorded Vulkan
  command buffers with zero per-iteration host involvement, so it runs at
  the cost of the actual GPU work (0.15 ms).
- **Why the paper's "gap grows with resolution" does not reproduce
  as-is:** on the RTX 2080, nvdiffrast's 2048² iteration was
  GPU-work-bound (5.4-8.0 ms in Table 4). On a Blackwell-class GPU the
  same pixel work costs well under 1 ms and hides beneath the host-launch
  floor until 4096². The advantage source is the same as in the paper —
  it just saturates earlier on modern hardware.
- **The AA technique is nearly resolution-independent** (0.49 → 0.78 ms
  for a 256x pixel increase): its 8-neighbor forward early-outs on equal
  triangle IDs, making the cost silhouette-perimeter-bound rather than
  area-bound, and its stochastic vertex backward is O(faces). It beats
  both nvdiffrast backends and DRTK at every resolution, including 4x
  vs nvdiffrast (280x vs DRTK) at 4096².
- **DRTK is pixel-bound from 512² up** — its brute-force CUDA rasterizer
  (no hardware rasterization, no coarse binning) scales ~4x per
  resolution step (static-geometry step: 1.2 / 1.5 / 4.6 / 17 / 67 ms).
  The ≥2048² optimization numbers are inflated further and vary
  run-to-run because per-triangle cost grows with covered pixels: once
  the silhouette-only gradients deform the mesh (interior vertices are
  unconstrained), transient large/degenerate triangles multiply the
  rasterization cost. DRTK targets differentiable *shading* pipelines;
  a bare silhouette task exercises exactly its weakest component.
- **The former pixel-bound limitation is closed by rematerialization:**
  the HardSoftRas backward used to materialize ~59 fp32 intermediates
  per stage as attachments and lost to nvdiffrast's hand-tuned kernels
  at 4096² (4.37 vs 3.3 ms). With the packer's clone pass (consumer
  subpasses recompute cheap forward chains — including re-issuing
  `FaceFetch` — instead of loading them) the crossing count drops ~40%
  and 4096² lands at 2.87 ms, ahead of all nvdiffrast backends.
- Quality on this benchmark: self-reconstruction reaches silhouette IoU
  0.986 (HardSoftRas, K=3) / 0.977 (AA); the stochastic backwards trade
  a little per-iteration accuracy for the O(faces) cost and recover it
  with iterations and `--samples`/`--peels`, exactly as in the paper.

## Milestone 4: PyTorch bindings (`dressi.torch`)

An nvdiffrast-compatible PyTorch API backed by the Dressi Vulkan engine —
runs on any Vulkan GPU, no CUDA required, permissive licensing:

```python
import dressi.torch as dr

ctx     = dr.RasterizeVulkanContext()
rast, _ = dr.rasterize(ctx, pos, tri, resolution=[H, W])  # (u,v,z/w,tri_id+1)
col, _  = dr.interpolate(attr, rast, tri)                 # exact attr VJP
col     = dr.texture(tex, uv)                             # inverse-UV tex grad
col     = dr.antialias(col, rast, pos, tri)               # color + pos grads
rs      = dr.rasterize_soft(ctx, pos, tri, [H, W],        # HardSoftRas ext.
                            radius_px=3.0, peels=3)       # (dist,id,z,cov)
```

Dressi is define-and-run while PyTorch is eager; the layer reconciles them
with a **shape-signature-keyed engine cache**: the first call with a new
(resolution, mesh size, params) signature compiles a `DressiAD` graph
(packing + GLSL + Vulkan objects), subsequent calls only upload changed
leaf values, execute the cached command buffer, and download results.
External gradients enter through a surrogate loss `output * grad_seed`
(BuildBackward seeds 1 everywhere, so leaf gradients equal the exact VJP
for the uploaded `grad_output`).

Same Avocado silhouette benchmark (ms/iter; fixed view, Adam on positions):

| resolution | dressi.torch (eager) | dressi native fused (Python) | nvdiffrast CUDA | DRTK |
|---|---|---|---|---|
| 256²  | 2.3 | **0.21** | 1.11 | 2.12 |
| 512²  | 3.4 | **0.22** | 1.18 | 6.15 |
| 1024² | 8.5 | **0.22** | 1.21 | 19.6 |

The eager layer batches aggressively: N-item calls unroll into ONE engine
(one execStep per forward and one per backward for a whole minibatch),
uploads/downloads ride single staging submits (`sendImgs`/`recvImgs`,
same-shape batches land in one stacked buffer), forward runs a
gradient-free engine (backward stages never execute in forward), and CPU
copies of device tensors are cached across op boundaries. The engine-side
win that helps every path (including the C++ examples): the per-vertex
gather backwards (`AntiAliasBwdVtx`, `GatherDistGrad`) now run WIDE —
{V, max_deg} partials (one incident face per thread) reduced by a
per-column sum — instead of one thread per vertex scanning all incident
faces; measured 2.9 → 0.44 ms (AA) and 14.4 → sub-ms (HardSoftRas) on an
8-view batch.

Multi-view silhouette example (8 views, 128², 300 iters, CUDA tensors):
aa 39 → **10.7 ms/iter**, hardsoftras 97 → **23.8** (nvdiffrast aa: 6.2).
The remaining eager gap is ~6 ms of host glue (CUDA↔CPU staging copies +
submit fences) that only GPU interop can remove.

- The eager drop-in path pays per-op CPU round trips (v1 transfers via CPU
  tensors); it beats DRTK everywhere.
- The **native fused path** (`dressi._C`, see
  `scripts/dressi_native_bench.py`) builds the whole training step —
  transform, raster, AA, MSE, in-graph Adam — as one Dressi graph and
  matches the C++ example from Python (zero per-iteration PCIe traffic).
- Transfer layer lessons (9x eager speedup): readback staging must be
  HostCached (memcpy from write-combined memory runs at ~100 MB/s),
  staging buffers and one-shot command buffers are persistent per context,
  and channel-identical images skip the re-stride pass.

v1 limitations: CPU-tensor transfer only (CUDA tensors are auto-moved with
a warning); no `rast_db`/barycentric derivatives; `rasterize`/`interpolate`
produce no position gradients (use `antialias`/`rasterize_soft`, matching
the silhouette workflow); nearest texture filtering, no mipmaps, no grad
w.r.t. UV; no near-plane clipping; batches loop per item. Planned next:
zero-copy GPU interop — export Vulkan memory (`VK_KHR_external_memory`) and
import it into CUDA as torch tensors (the nvdiffrast-GL pattern; the
reverse direction is not possible because torch's caching allocator does
not export its allocations).

Build & test (dependencies are managed with [uv](https://docs.astral.sh/uv/);
`uv sync` creates `.venv` with CUDA torch on Windows/Linux, and
`uv sync --extra nvdiffrast` adds the comparison backend):

```sh
uv sync
cmake --preset msvc -DDRESSI_BUILD_PYTHON=ON -DPython_EXECUTABLE=.venv/Scripts/python.exe
cmake --build --preset release --target dressi_py
uv run pytest tests/python
```

Python ports of the C++ examples live in `examples/python/` — everything
outside the rasterizer API is plain torch (cameras, regularizers, Adam),
and because the API is nvdiffrast drop-in the same code runs on either
backend for a direct comparison:

```sh
uv run python examples/python/silhouette_optimization.py                       # dressi (Vulkan)
uv run python examples/python/silhouette_optimization.py --backend nvdiffrast  # CUDA
uv run python examples/python/silhouette_optimization.py --technique hardsoftras
uv run python examples/python/texture_optimization.py
```

Silhouette (icosphere → bunny, 128², 8 views, 300 iters): IoU 0.97 (aa) /
0.96 (hardsoftras), final mesh saved as OBJ. Texture (bunny atlas from
gray, 256², 6 views, jittered): rendered-view PSNR > 50 dB.

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
