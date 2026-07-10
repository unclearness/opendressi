# CLAUDE.md

opendressi: a C++/Vulkan re-implementation of the Dressi differentiable
renderer (arXiv:2204.01386, EG 2022). The public API follows the paper's
Appendix A (`Variable` / `Function` / `namespace F` / `DressiAD`); when in
doubt about API shape, that appendix is the reference. Gradient formulas
follow PyTorch's derivatives when unclear.

## Build & test

```sh
cmake --preset msvc                  # VS 2022, x64; sets VULKAN_SDK env
cmake --build --preset release
ctest --preset release               # all 92 tests
ctest --preset release -LE gpu       # CPU-only (no Vulkan device needed)
./build/tests/Release/dressi_tests_cpu.exe --gtest_filter=Backward.*
./build/examples/Release/texture_optimization.exe data/bunny
./build/examples/Release/silhouette_optimization.exe data/bunny --technique=hardsoftras
./build/examples/Release/silhouette_optimization.exe data/bunny --technique=aa
```

- Vulkan SDK is auto-located under `C:/VulkanSDK/*` (`cmake/FindVulkanSdk.cmake`);
  the `VULKAN_SDK` env var is NOT set system-wide on this machine.
- The SDK's `glslang-config.cmake` is broken (bad internal path); glslang
  libs are linked directly by path in the top-level CMakeLists with per-config
  d-suffix selection. Only the Release config is exercised regularly.
- `third_party/VulkanWrapper` is a git submodule pointing at the fork
  `unclearness/VulkanWrapper`, branch `dressi2-sdk-compat` (SDK 1.4.x compat
  patches). Never `add_subdirectory` it — its bundled third_party breaks under
  CMake 4.x; the `dressi_vkw` target compiles `vkw.cpp` against SDK libs.
- `SPDLOG_LEVEL=debug` prints per-phase execStep timings plus a per-stage
  GPU timestamp breakdown every 100 frames; `DRESSI_VK_DEBUG=1` enables
  the Vulkan validation layers.

## Architecture (src/)

Pipeline per `DressiAD::execStep()` (the paper's InitStatus ladder):
build backward graph → wire optimizer → traverse → pack substages → pack
stages → build Vulkan objects → submit. Each step reruns only when its
inputs changed (reactive cache).

- `core/` — graph nodes (`node.h`: handle/body over `shared_ptr`, weak
  back-edges; creation id doubles as topological order), type/size inference
  with `{1,1}` uniform broadcasting, `f_ops.cpp` (every op = GLSL snippet +
  backward generator + CPU kernel), `build_backward.cpp`, `cpu_eval.cpp`
  (CPU interpreter = the test oracle), `cpu_raster.cpp` (CPU reference
  rasterizer + the signed-distance / AA-pair kernels shared by GLSL
  mirrors and cpu_funcs).
- `pack/` — SubStage/Stage IR, `trivial_packer` (1 func = 1 pass, the
  correctness baseline; `DressiAD::setPackingMode`), `greedy_packer` (RSP:
  back-to-front greedy fusion under Vulkan limits, then
  `RematerializeSubStages` clones cheap producer chains into consumers
  and drops dead outputs), `reactive.cpp` (prunes clean cached branches).
- `codegen/` — substage → GLSL fragment shader (`{y}`/`{x0}`/`{s0}` snippet
  markers, substage-local `v0..vN` names for golden tests and shader-cache
  hits); special-cased ops by marker string (`__reduce_2x2_sum__`,
  `__gather_inv_uv__`, `__rasterize__`, `__upsample_nearest_2x__`,
  `__rasterize_soft__ r=<px>`, `__rasterize_face_id__`,
  `__gather_dist_grad__`, `__antialias__`, `__antialias_bwd_img__`,
  `__antialias_bwd_vtx__`, `__sum_all__`, `__sum_partial__`).
- `vk/` — headless context, executor (`ParseStagesAsVulkanObjects`
  equivalent), CPU↔GPU transfer with VEC3→RGBA32F padding.

## Conventions & invariants

- Public headers in `include/dressi/` mirror Appendix A names; internal
  code includes `"core/..."` etc. relative to `src/`.
- `operator<` on `Variable` builds comparison graphs (appendix semantics:
  `0.f < x`); map ordering uses a `std::less` specialization by node id.
- Component-wise ops rely on default inference (`infer == nullptr`); codegen
  uses that to decide GLSL scalar→vector promotion (`pow()` etc. do not
  broadcast scalars).
- Every differentiable op registers a `cpu_func`; tests verify GPU forward
  against the CPU evaluator and backward against central finite differences.
  New F:: ops need all three (snippet, bwd, cpu) plus a `test_backward.cpp`
  gradient check.
- GPU-resident state (`plan.imgs`, `plan.vtx_bufs`, `plan.textures`,
  `plan.uif_bufs`) MUST persist across `BuildGpuPlan` rebuilds — dropping
  it silently breaks reactive-cache runs (see commit 36d7a6e for the
  failure mode).
- Raster vertex data: leaves are uploaded into vertex/index buffers by
  `sendImg` (`CpuImage` stores face indices as floats; converted to uint32
  at upload); COMPUTED clip positions are also legal raster inputs — the
  executor copies the producer's image into a device-local vertex buffer
  each frame (padding-free types only: FLOAT/VEC2/VEC4; faces must stay
  CPU leaves). A leaf used both as geometry and as a texelFetch image gets
  both resources; one `sendImg` updates both.
- Vertex-position gradients (`F::RasterizeSoft` = HardSoftRas,
  `F::AntiAlias` = Dr.Hair screen-space AA) flow through in-graph
  projections back to a 3D position leaf. Their screen->vertex backward
  ops are per-vertex gathers bounded by adjacency-texture bboxes (no
  atomics/COMP needed). Static topology rides in adjacency textures
  (`VertexFacesTex` / `VertexNeighborsTex` / `FaceNeighborsTex` in the
  example utils).
- Transfer model (the paper's): upload leaves once, download results at
  the end; nothing crosses PCIe per iteration. Optimizer outputs write
  DIRECTLY into their input leaf's image (zero-copy aliasing: the update
  substage binds the shared image as input+color attachment in eGeneral
  with a subpass self-dependency — legal because a fullscreen pass has
  one fragment per pixel). Pairs whose writer substage does not consume
  the input same-pixel fall back to an end-of-frame CopyImage.
  GPU-resident optimizer STATE uses
  `DressiAD::addUpdate(state_leaf, new_state)` (in-graph Adam:
  see `examples/silhouette_optimization`). Regularizers enter the
  optimizer's update graph via forward-only mesh ops (`F::SoftClip`,
  `F::VertexNeighborMean`, `F::NormalConsistencyFaceTerm/VertexGrad`).
- The loss may be any float image (per the paper): `BuildBackward` seeds
  gradient 1 at every pixel/component (= derivative of the implicit sum),
  so no forward reduction chain to `{1,1}` is needed. Keep losses as
  images for performance; reduce with `F::Mean`/`F::Sum` only when the
  scalar value itself is required (or sum the recvImg'd image on the CPU).

## Known deviations from the paper (documented, intentional)

- Appendix typos fixed: `Sin` backward, `BuildBackward` null check.
- Uniforms (`uif_vars`): `{1,1}` LEAF values ride in vec4 UBOs;
  GPU-generated `{1,1}` scalars stay `texelFetch(0,0)` slt reads — the
  mid-frame UBO refresh copies measured slower than the broadcast fetch
  they replace (documented in log/202607102000).
- COMP substages execute as compute dispatches (storage-image outputs,
  same-pixel inputs become samplers fetched at the invocation coord;
  FRAG funcs may fuse INTO a COMP substage, never the reverse). By
  measurement only `GatherDistGrad` defaults to COMP: for the hot
  band-pass gathers, graphics<->compute switches plus lost elementwise
  fusion cost more than the render passes they remove.
- Rematerialization (RSP only): after substage packing, consumers clone
  cheap same-pixel producer chains (whitelisted markers: `__face_fetch__`,
  `__screen_coord__`) instead of reading them through attachments, then
  undemanded outputs and dead substages are dropped. Invariant: a cloned
  value must never also be an slt/tex input of the same substage (GLSL
  name collision — guarded in `RematerializeSubStages`).
- Int/matrix images on GPU are not implemented yet.
- HardSoftRas depth peeling: `--peels=K` blends Eq.6 layers (K=3 matches
  the exact-backward reference); triangle enlargement is centroid scaling
  (`F::SoftClip` on the GPU; no geometry shader, no obtuse bbox split).
- The AA technique picks the boundary edge by owner preference
  {tri(neighbor), tri(self)} instead of an exact closest-depth test, and
  its forward keeps the scheme's inherent residual discontinuity when a
  pixel-center coverage flips (gradients, not forward continuity, are the
  deliverable).
- Not yet from the paper: texture backward w.r.t. UV.

## Work logs

Write a work log for each session/topic to `log/YYYYMMDDHHMM_TITLE.md`
(timestamp = when the log is written, TITLE = short snake_case English).
Record: goal, what was done (with commit hashes), key decisions and why,
measurements, and open issues. Commit the log together with the work.

## Testing gotchas

- `recvImg` only works for variables that own a GPU image; under RSP
  packing intermediates get fused away — use `DressiAD::markOutput(var)`
  or Naive mode in tests that read intermediates.
- A variable is only executed if it is reachable from the loss/optimizer
  roots (`TraverseFuncs`); an unused G-buffer channel silently has no image.
- GPU tests carry the ctest label `gpu`; keep new CPU-only tests in
  `dressi_tests_cpu` so headless CI can run `-LE gpu`.
