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
ctest --preset release               # all 74 tests
ctest --preset release -LE gpu       # CPU-only (no Vulkan device needed)
./build/tests/Release/dressi_tests_cpu.exe --gtest_filter=Backward.*
./build/examples/Release/texture_optimization.exe data/bunny
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
- `SPDLOG_LEVEL=debug` prints per-phase execStep timings.

## Architecture (src/)

Pipeline per `DressiAD::execStep()` (the paper's InitStatus ladder):
build backward graph → wire optimizer → traverse → pack substages → pack
stages → build Vulkan objects → submit. Each step reruns only when its
inputs changed (reactive cache).

- `core/` — graph nodes (`node.h`: handle/body over `shared_ptr`, weak
  back-edges; creation id doubles as topological order), type/size inference
  with `{1,1}` uniform broadcasting, `f_ops.cpp` (every op = GLSL snippet +
  backward generator + CPU kernel), `build_backward.cpp`, `cpu_eval.cpp`
  (CPU interpreter = the test oracle).
- `pack/` — SubStage/Stage IR, `trivial_packer` (1 func = 1 pass, the
  correctness baseline; `DressiAD::setPackingMode`), `greedy_packer` (RSP:
  back-to-front greedy fusion under Vulkan limits), `reactive.cpp` (prunes
  clean cached branches).
- `codegen/` — substage → GLSL fragment shader (`{y}`/`{x0}`/`{s0}` snippet
  markers, substage-local `v0..vN` names for golden tests and shader-cache
  hits); special-cased ops by marker string (`__reduce_2x2_sum__`,
  `__gather_inv_uv__`, `__rasterize__`, `__upsample_nearest_2x__`).
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
- GPU-resident state (`plan.imgs`, `plan.vtx_bufs`, `plan.textures`) MUST
  persist across `BuildGpuPlan` rebuilds — dropping it silently breaks
  reactive-cache runs (see commit 36d7a6e for the failure mode).
- Geometry (`F::Rasterize` inputs) must be leaf variables; the executor
  uploads them into vertex/index buffers (`CpuImage` stores face indices as
  floats; converted to uint32 at upload).
- Optimizer outputs are copied back into their input images at end of frame
  (no aliasing; a render pass must not read an image it writes).
- Loss must be scalar FLOAT `{1,1}` (`F::Mean` guarantees this).

## Known deviations from the paper (documented, intentional)

- Appendix typos fixed: `Sin` backward, `BuildBackward` null check.
- `{1,1}` GPU-generated values are read via `texelFetch(0,0)` instead of
  UBOs; uniforms-as-UBO is not implemented yet.
- COMP shader substages, int/matrix images on GPU, and zero-copy optimizer
  aliasing are not implemented yet.
- Not yet from the paper: HardSoftRas (geometry gradients), depth peeling
  execution, texture backward w.r.t. UV.

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
