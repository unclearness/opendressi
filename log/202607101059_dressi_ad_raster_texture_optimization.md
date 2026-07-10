# Dressi-AD core, deferred rasterization, and dense texture optimization

Date: 2026-07-10
Commits: 3c3e6f1 .. bc9a731 (published to
https://github.com/unclearness/opendressi, branch `main`)

## Goal

Re-implement the Dressi paper (arXiv:2204.01386) in C++/Vulkan. This
session: Milestone 1 (Dressi-AD layer), Milestone 2 (deferred
rasterization + unlit shading), and dense texture recovery on the bunny
dataset with a live viewer and performance logging.

## What was done

- `3c3e6f1` **Milestone 1: Dressi-AD layer.** Appendix-A-faithful graph API
  (`Variable`/`Function`/`F::`/`DressiAD`), reverse-mode `BuildBackward`
  verified against central finite differences via a CPU interpreter oracle,
  GLSL codegen, headless Vulkan executor on the VulkanWrapper fork
  (`unclearness/VulkanWrapper@dressi2-sdk-compat`, SDK 1.4.x compat patches),
  Naive + greedy RSP packing modes, reactive cache (dirty tracking,
  FAST/FULL rebuild, pruning).
- `804ce7e` **Milestone 2: deferred rasterization + differentiable
  texturing.** `F::Rasterize(clip, attrib, faces, size)` renders G-buffer
  channels (depth-tested indexed draw, own render pass; geometry
  non-differentiable). `F::Texture(tex, uv, inv_uv)` differentiable w.r.t.
  the texture through the paper's inverse-UV table (mesh rasterized in UV
  space carrying screen positions); gather guards occlusion by re-verifying
  the forward sample. tinyobjloader/stb asset utilities.
- `41d28b1` **Sobol jitter + spdlog phase timing + markOutput.**
- `5f2abac` **OBJ UV V-flip fix** (vt bottom-left vs image top-left; targets
  had subtly misplaced patterns) and **glad+GLFW live viewer**
  (self-contained Vulkan presentation fed by `recvImg`).
- `cf482c4` **Tiled fixed-view comparison windows** (GT targets vs current
  renders, same 3x2 layout).
- `36d7a6e` **UV jitter replaced by sub-pixel camera jitter; geometry
  buffers persist across rebuilds** (see decisions below).
- `275c947`/`bc9a731` project renamed to opendressi, pushed, MIT license.

## Key decisions

- **Vulkan wrapper**: forked takiyu/VulkanWrapper instead of patching
  vendored code in place; 6-line patch for glslang 15 / `vk::detail::` moves.
- **Optimizer copy-back instead of image aliasing**: writing the update into
  the input's image within one render pass is a feedback loop; an explicit
  end-of-frame `CopyImage` is spec-clean. Zero-copy aliasing deferred.
- **Jitter semantics**: offsetting the sampling UV slides texture content
  across the surface (chart-level wobble in the live view, reported by the
  user). Correct approach: TAA-style sub-pixel projection jitter -- a surface
  point always samples its own UV, pixel centers sweep their footprints, all
  visible texels get constrained over iterations. Targets are rendered
  in-graph with the GT texture under the same jitter so every iteration is
  self-consistent.
- **Gather robustness**: the inverse-UV entry is quantized/affine, so the
  gradient gather searches a 7x7 screen neighborhood and dilates
  chart-boundary texels (9x9 entry search). Without dilation the loss
  plateaued at ~2.5e-3.

## Bugs found

- **Geometry buffers dropped on rebuild** (`36d7a6e`): `BuildGpuPlan`
  carried `imgs` but recreated `vtx_bufs`/`textures` empty. Static-geometry
  runs masked it (raster stages were pruned at the same rebuild); the
  camera-jitter workload rendered nothing (degenerate zero faces) and
  produced a fake loss=0. GPU-resident state now persists across rebuilds.

## Measurements (RTX PRO 6000, bunny 16k verts / 30k faces / 1024^2 atlas,
6 views at 256^2, 4000 iterations)

- Initial build: ~230 ms (vulkan-build ~160 ms = shader compile + pipelines)
- Fast rebuild: ~8 ms; steady state: ~1.0 ms/iter (reactive cache prunes
  55 -> 26 stages with per-iteration camera jitter)
- Live viewer: ~3.7 ms/iter, excluded from optimization timing
- Texture recovery: 600,505 texels updated, 99.4% within 0.1 of GT;
  zero-jitter final loss 8.4e-6

## Open issues / next steps

- HardSoftRas milestone: triangle enlargement (`Enlarge`/`SignedDist`),
  depth peeling execution, geometry gradients.
- Texture backward w.r.t. UV; linear sampling.
- COMP substages, int/matrix images on GPU, UBO uniforms, zero-copy
  optimizer aliasing.
- ~0.6% of updated texels still short of GT after 4000 iters (slowly
  converging rare samples); more iterations or per-texel step normalization
  (Adam) would close it.
