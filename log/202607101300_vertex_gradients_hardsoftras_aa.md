# Vertex-position gradients: HardSoftRas + screen-space AA, silhouette optimization

Date: 2026-07-10
Commits: 2bc965d..d88c8dd (+ this log)

## Goal

Make rasterization differentiable w.r.t. triangle-mesh vertex positions with
two independent techniques, each verified by deforming an icosphere into the
bunny from multi-view silhouettes:

1. **HardSoftRas** (Dressi paper arXiv:2204.01386, Alg.2): HW rasterization
   of enlarged triangles; signed screen-space distance to the hard triangle
   drives `sigmoid(dist / (r/7))`.
2. **Screen-space AA** (Dr.Hair, arXiv:2403.17496 Eq.3-5): hard rasterization
   plus a differentiable 8-neighbor blend across triangle-ID boundaries via
   the pixel-to-silhouette-edge distance `r = min(|e_s|/L, 1)`. Works on
   general triangle meshes, not just hair billboards.

## What was done

- `2bc965d` **F::Sigmoid** with backward and CPU kernel.
- `3c9442a` **CPU reference rasterizer** (`src/core/cpu_raster.*`):
  depth-tested perspective-correct rasterization matching the executor
  conventions, registered as `Rasterize`'s `cpu_func` (prerequisite for
  finite-difference `CheckGrad` on raster graphs); shared signed-distance
  helpers.
- `eca1c73` **RASTER stages can bind texelFetch textures**: packers route
  `InputAccess::TexelFetch` raster inputs to `slt_vars`, the executor gives
  raster pipelines a CombinedImageSampler descriptor set; devices created
  with `GetPhysicalFeatures2` (Geometry capability for `gl_PrimitiveID`).
- `ff8e313` **Example utilities**: icosphere generator, OBJ writer, vertex
  adjacency + uniform-Laplacian gradient, HardSoftRas soft-geometry builder
  (screen-space centroid scaling `k = 1 + r/d_min`, clamped), clip->position
  chain rule, flat Adam.
- `ce8e88b` **F::RasterizeSoft forward**: soft mesh = per-face unwelded,
  CPU-enlarged triangles carrying the face index as a vertex attribute; FS
  fetches the hard triangle from clip/faces textures, outputs VEC4
  (signed_dist_px, face_id, hard_z, coverage) and applies the Eq.3 depth
  shift via gl_FragDepth (hard region [0,0.5], soft rim [0.5,1]). Radius
  rides on the `__rasterize_soft__ r=<px>` marker.
- `c1b2506` **HardSoftRas backward** (`GatherDistGrad`,
  `__gather_dist_grad__`): per output texel (= hard vertex) scans all
  covered pixels, recomputes the argmin edge (envelope theorem) and
  accumulates `gy.x * d dist/d clip` with the screen->clip Jacobian
  (z gradient identically 0). CPU CheckGrad vs central differences passes,
  incl. a w != 1 vertex.
- `9f49e5d` **HardSoftRas GPU tests** + codegen fix: when RSP fusion makes
  one variable both a same-pixel input attachment and a texelFetch input of
  a special op, the generic slt load redefined the local name; subpassLoad
  now wins.
- `7478360` **Screen-space AA**: `F::RasterizeFaceId`
  (`float(gl_PrimitiveID)+1`, background 0) and `F::AntiAlias(img, tri_id,
  vtx_clip, faces)` with backwards w.r.t. the image (exact linear
  transpose) and the clip leaf (per-vertex gather, analytic `d r/d clip`).
  Shared `EvalAaPair` CPU kernel mirrors the GLSL helper branch-for-branch.
- `d88c8dd` **examples/silhouette_optimization**: sphere -> bunny, both
  techniques behind `--technique=`, CPU Adam + relative Laplacian, live
  tiled viewers, PNG/OBJ outputs.

## Key decisions and why

- **CPU-side vertex updates (v1)**: clip positions stay per-view leaves;
  the optimizer lambda is an identity update that captures `gxs` and
  `markOutput`s them (safe: during execStep the OPTIMIZER phase precedes
  TRAVERSE, so the markOutput guard is a no-op and the same step packs the
  grads as roots). The loop recvImg's the clip grads, chain-rules through
  the fixed MVPs and runs Adam on the shared 3D positions, re-sending
  leaves like texture_optimization's jitter re-send. In-graph vertex
  updates (vertex-buffer copy-back, in-graph MVP) stay future work.
- **Screen->vertex gradients as gather ops, not scatter**: FRAG codegen is
  gather-only, so both backwards are `{V,1}`-output special ops that loop
  over all screen pixels per vertex (O(V*W*H), early-outs on background /
  zero gy / foreign faces). No atomics or COMP shaders needed; V=642 at
  128^2 x 8 views is well within budget.
- **Hard/soft data via textures**: `vtx_clip` {V,1} VEC4 and float-valued
  `faces` {F,1} VEC3 are uploaded as images (int images are unsupported on
  GPU; float indices exact < 2^24). For AA the SAME leaf feeds the vertex
  buffer and the AA texture (`sendImg` fills both), so the gradient lands
  on the leaf the loop reads.
- **face_id as a vertex attribute in RasterizeSoft** (constant per face =>
  exact interpolation) rather than gl_PrimitiveID: the soft mesh is
  unwelded anyway; RasterizeFaceId does use gl_PrimitiveID (welded mesh).
- **AA owner preference {tri(n), tri(s)}** approximates closest-depth edge
  selection without a depth buffer: at an interior occlusion boundary only
  the front triangle's edges separate the two pixel centers (verified by
  the OcclusionBoundaryFrontEdgeOwnsGradient test: back vertices get
  exactly zero).
- **Example, hardsoftras path**: `pred = max(StopGradient(hard_mask),
  coverage * sigmoid(dist/sigma))` — the paper's Eq.5 edge/non-edge split.
  Pure `sigmoid(dist)` leaves interior seams at K=1 (every pixel near an
  internal edge reads dist ~ 0 -> 0.5); the max keeps interiors hard while
  the sigmoid band supplies silhouette gradients. Loss floor dropped
  0.175 -> 0.051 and IoU 0.940 -> 0.971.
- **Example, aa path**: the target is the AA'd bunny mask so pred can match
  it exactly at convergence (no persistent boundary residual).
- **Relative Laplacian weight**: `g += lambda * rms(g_data)/rms(g_lap) *
  g_lap` — smoothing fades as the fit converges, no absolute scale tuning.

## Bugs found

- RSP fusion could make one variable both an `inp_var` and an `slt_var` of
  the same substage (same-pixel consumer + texelFetch special op); codegen
  emitted two loads with the same local name (`v0` redefinition). Fixed in
  `9f49e5d` by skipping the generic slt load when the variable is already
  subpassLoaded.
- AA descent with a too-large step collapses the triangle to zero coverage;
  with no ID boundary left the gradient is zero forever (noted in the test:
  keep steps below ~1 px).

## Measurements (RTX PRO 6000, bunny 16k v / 30k f targets, icosphere level 3 = 642 v / 1280 f, 8 views, 128x128, 300 iters, Adam lr 0.01, laplacian 0.5)

| technique   | loss                  | mean IoU | opt ms/iter |
|-------------|-----------------------|----------|-------------|
| hardsoftras | 2.142 -> 0.051 (42x)  | 0.971    | ~27         |
| aa          | 2.060 -> 0.012 (174x) | 0.981    | ~221        |

- The AA per-iteration cost is dominated by `__antialias_bwd_vtx__`
  (O(V*W*H*8) with two tri-id fetches per pair before the early-out).
- Single-triangle GPU sanity fits: hardsoftras >10x loss drop in 200 GD
  steps; aa >200x in 50 steps.
- Test suite: 92 tests (68 cpu + 24 gpu), all passing.

## Open issues / next steps

- Depth peeling (K > 1) and the Eq.6 multi-layer silhouette product;
  obtuse-triangle bbox-split enlargement (currently centroid scaling,
  clamped at 8x, slightly over-covers thin triangles).
- In-graph vertex optimization: vertex-buffer copy-back for optimizer
  outputs + in-graph MVP transform, removing the CPU round-trip.
- Gather-op culling (per-face screen bbox or tiling) if V or the screen
  grows; the AA backward is the current bottleneck.
- AA closest-depth tie-break for silhouette-over-silhouette crossings
  (measure-zero; owner preference used instead).
- HardSoftRas color/shading path (Eq.5 blending of shaded G-buffers) — only
  the silhouette channel is exercised here.
