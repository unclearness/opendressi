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

## Follow-up: flipped faces / normal consistency (`e6ac1ac`)

With only the positional Laplacian, silhouette-unconstrained regions kept
flipped/creased faces (user report confirmed by counting adjacent face
pairs with `n_f . n_g < 0`). Added `NormalConsistencyGrad` — the analytic
gradient of `sum over adjacent face pairs of (1 - n_f . n_g)` through the
normalized cross products — as `--normal=` (default 0.5, same relative
data-gradient-RMS scaling as `--laplacian`), plus the flipped-pair count in
the final report. Results (300 iters, defaults otherwise):

| technique   | --normal | loss reduction | IoU    | flipped pairs /1920 |
|-------------|----------|----------------|--------|---------------------|
| hardsoftras | 0        | 41.6x          | 0.9713 | 62                  |
| hardsoftras | 0.5      | 37.4x          | 0.9706 | 8                   |
| hardsoftras | 1.0      | 16.3x          | 0.9425 | 2 (over-smoothed)   |
| aa          | 0        | 174x           | 0.9813 | 105                 |
| aa          | 0.5      | 163x           | 0.9810 | 11                  |
| aa          | 1.0      | 143x           | 0.9805 | 10                  |

0.5 keeps the fit while removing ~90% of the flips; 1.0 over-smooths
hardsoftras. Remaining flips cluster where no view's silhouette constrains
the surface (more/better-placed views or a remeshing step would address
them).

Second sweep (`a5115fb`): the hardsoftras deformation was under-fit at
0.5/0.5 (user report) — its sigmoid-band gradients are weaker than the AA
boundary gradients, so the same relative regularization over-smooths.
Weight sweep (lap x normal): 0.5/0.5 IoU 0.9706 flips 8; 0.25/0.25 0.9801
/22; 0.1/0.25 0.9825/27; **0.15/0.4 0.9824/16 (new hardsoftras default)**;
0/0.25 0.9847/121. aa keeps 0.5/0.5. Same commit fixes TileImages reading
c=0..2 from 1-channel sources (neighboring pixels bled into G/B: green
speckles on black in the live viewer, colored silhouette fringes in the
saved tiles).

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

## Follow-up: image-valued losses (`e2bcfca`)

The M1-era `BuildBackward` required a scalar FLOAT `{1,1}` loss, which is
NOT the paper's design (author feedback): seeding gradient 1 at every
pixel differentiates the implicit sum, so the loss can stay an image and
the log-step reduction chain + its upsample backward never run. Changes:
loss check relaxed to any float image; `SumToShape` upcasts uniform
gradients through the new `F::Broadcast` ({1,1} -> image, backward = Sum,
inline-const seeds become literals); the silhouette example keeps per-view
`{S,S}` loss images scaled by 1/(W*H) and CPU-sums the recv'd image only
at logging points. Final loss/IoU reproduce bit-exactly (gradient
equivalence). Saves ~16 passes per view per iteration (~0.5 ms at 256^2,
1 view); the per-vertex gradient gather (O(V*W*H), only V fragments of
parallelism) remains the dominant cost vs the paper's pixel-parallel
backward -- per-vertex screen-bbox culling is the next lever.

Performance vs the paper (fwd+bwd, 1 view): paper RTX 2080 geometry
optimization 256^2 = 0.304-0.364 ms (406-25.7k verts, K<=5); ours on an
RTX PRO 6000 (~10x faster raw) = ~8.4 ms at 256^2 / ~2.5 ms at 128^2
(HSR, 642 verts, K=1), AA ~97 ms at 256^2. M2 texture optimization
(~0.5 ms/iter, 256^2 x 6 views) is in the paper's ballpark after GPU
correction; the gap is specific to the gather backwards.

## Follow-up: per-vertex bbox culling of the gather backwards (`5fc260c`)

The gather ops' pixel loops are now bounded by each vertex's incident-face
screen bbox, found by an O(F) face scan per vertex thread (cheap next to
the former O(W*H) pixel scan). `__gather_dist_grad__` reconstructs the
exact soft-triangle bbox in-shader (identical centroid-scaling formula and
clamps as `BuildSoftGeometry`; the radius now rides on its marker too);
`__antialias_bwd_vtx__` pads the hard bbox by the 8-neighborhood reach.
Both are strict supersets of the contributing pixels: default runs
reproduce bit-exactly and the GPU/CPU parity tests pass unchanged.

fwd+bwd GPU per iteration (1 view): hardsoftras 256^2 8.4 -> 4.0 ms,
aa 256^2 97 -> 8.1 ms (12x). Default example (8 views, 128^2): aa 220 ->
43 ms/iter; hardsoftras stays ~28 ms/iter, now dominated by the CPU-side
loop (per-view BuildSoftGeometry + leaf re-sends + grad readbacks).
Remaining gap to the paper's 0.30 ms (RTX 2080, 256^2): still only V
fragment threads of parallelism plus the O(V*F) face scan -- next levers
are a precomputed vertex->face adjacency texture, tiled partial sums, or
an atomic pixel-parallel scatter, and moving the vertex update in-graph.
