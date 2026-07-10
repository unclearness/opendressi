# dressi2

A C++/Vulkan re-implementation of the **Dressi-AD** layer from the paper
*"Dressi: A Hardware-Agnostic Differentiable Renderer with Reactive Shader
Packing and Soft Rasterization"* (Takimoto et al., Eurographics 2022,
[arXiv:2204.01386](https://arxiv.org/abs/2204.01386)).

Dressi-AD is a define-and-run reverse-mode automatic differentiation
framework specialized for differentiable rendering: computational graphs of
2D images are compiled into GLSL fragment shaders, packed into a minimal
number of Vulkan render passes (Reactive Shader Packing), and executed on
any Vulkan-capable GPU. The public C++ API follows Appendix A of the paper.

## Milestone 2: deferred rasterization + texture optimization

- `F::Rasterize(clip_pos, attrib, faces, screen_size)`: depth-tested indexed
  rasterization of a vertex attribute into a screen-space image (G-buffer
  channel; deferred shading per the paper). Non-differentiable with respect
  to geometry (HardSoftRas triangle enlargement is a later milestone).
- `F::Texture(tex, uv, inv_uv)`: nearest sampling, differentiable with
  respect to the texture via the paper's inverse-UV lookup (the mesh
  rasterized in UV space with screen positions as the attribute), with an
  occlusion guard and on-the-fly dilation for chart-boundary texels.
- `examples/texture_optimization`: multi-view unlit rendering of
  `data/bunny`. GT views are rendered with the GT atlas; starting from a
  black texture, the summed per-view MSE recovers the atlas on every texel
  observed by any view (rendered loss converges to zero).

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

Not yet implemented (later milestones): rasterization (`F::Rasterize` /
HardSoftRas), texture sampling with inverse-UV backward, compute-shader
substages, int-image execution, matrix images on GPU.

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

## Layout

```
include/dressi/   public API (Appendix A style)
src/core/         graph nodes, F:: ops, BuildBackward, CPU evaluator
src/codegen/      substage -> GLSL fragment shader generation
src/pack/         trivial & greedy (RSP) packers, reactive pruning
src/vk/           headless Vulkan context, executor, CPU<->GPU transfer
tests/            GoogleTest suites (ctest labels: cpu / gpu)
examples/         image_fitting end-to-end optimization demo
```
