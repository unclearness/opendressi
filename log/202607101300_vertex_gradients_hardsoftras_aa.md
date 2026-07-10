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

## Follow-up: GPU-resident optimization loop (`abb7505`, `b2f5587`, example)

Adopted the paper's transfer model end to end (author direction: CPU<->GPU
traffic only at initial upload and final download; small scalars and debug
reads excepted):

- Vertex->incident-face adjacency textures (`abb7505`) replace the gather
  backwards' O(V*F) in-shader face scan (latency-neutral at demo scale --
  only V fragment threads -- but kills the quadratic term for real meshes).
- Executor accepts COMPUTED raster geometry: the producer's image is
  copied into a device-local vertex buffer with a transfer->vertexInput
  barrier each frame; optimizer-updated geometry leaves refresh at frame
  start. Greedy packer keeps GPU-generated vtx inputs as producer outputs.
- `DressiAD::addUpdate` gives optimizer state the same end-of-frame
  copy-back as parameters -> in-graph Adam (m/v/t leaves live on the GPU).
- In-graph mesh ops: `F::SoftClip` (HardSoftRas enlargement),
  `F::VertexNeighborMean` (uniform Laplacian), and the analytic
  `F::NormalConsistencyFaceTerm`/`VertexGrad` pair (face-adjacency +
  vertex->face textures); relative data-RMS weighting is a handful of
  {1,1} reduction ops inside the optimizer graph.
- `BuildBackward` sums gradient contributions as a tree of <=4-ary
  SumPixelWise nodes: 8 views x (GetX/GetY/GetZ) = 24 contributions at the
  position leaf exceeded Vulkan's guaranteed input-attachment minimum as a
  single n-ary node.
- The silhouette example's loop body is now `execStep()` alone; loss
  logging (a small image summed on the CPU at print intervals) and the
  live viewers are the only readbacks, both optional.

Results unchanged within float noise (300 iters, defaults): hardsoftras
68.9x / IoU 0.9809 / 14 flipped at 23.5 ms/iter (was 28.7 with the CPU
loop); aa 158x / IoU 0.9810 / 13 flipped at 42.5 ms/iter. New GPU test
`SoftRasGpu.FullyInGraphSilhouetteFit` locks the machinery (in-graph
projection + SoftClip + momentum SGD via addUpdate, zero per-iteration
transfers). Next parallelism lever for the gathers stays tiled partial
sums / atomic scatter.

## Follow-up: paper-faithful HardSoftRas re-architecture (`14d29ee`)

Author feedback: the implementation likely deviated from the paper. A full
re-read of the PDF (main body + Appendix A) confirmed two structural
deviations that explained the remaining ~100x performance gap:

1. **Backward mechanism.** The paper computes the pixel-edge distance from
   per-face corner variables (`LookupFaces` -> `{F,1}`) indexed per pixel
   by a rasterized ID buffer, with ordinary image-space ops ("All our DR
   algorithms are written in AD, and there are no special backward
   declarations"; `F::Rasterize` itself has NO backward). Gradients cross
   the indexed read the same way texture gradients cross `texture()`: the
   inverse-UV philosophy (Alg.1) -- sparse jittered samples, NOT an exact
   per-vertex sum ("current APIs of Vulkan do not fully support atomic
   float add operations"). My exact O(V*pixels) gather (V fragment threads,
   latency-bound) was solving a harder problem than the paper.
2. **Raster-headed substage fusion.** `IsSubStageVkLimitsSatisfied` allows
   FRAG functions behind a top RASTER function in ONE substage ("Dressi-AD
   can pack shader codes throughout HardSoftRas"); my packer isolates
   every RASTER function (still unfixed -- stage 2).

Stage-1 changes: `F::ScreenCoord` / `F::LookupFaces` (exact adjacency
backward) / `F::FaceFetch` (stochastic edge-band sampling backward,
Wang-hash jitter keyed by the in-graph iteration counter; bit-identical
GLSL/C++). The example's SignedDist() is now an elementwise-op chain --
the whole backward is AD-generated. `BuildBackward` materializes
broadcast gy to full shape before invoking backwards.

| metric (RTX PRO 6000) | exact gather | paper-faithful |
|---|---|---|
| 256^2 x 1 view, ms/iter | 4.0 | **0.22-0.34** (paper on RTX 2080: 0.304-0.364) |
| 128^2 x 8 views, ms/iter | 23.5 | **~0.9** |
| IoU @300 iters | 0.981 | 0.94 |
| IoU @2000 iters, 4/8/16 samples | - | 0.946 / 0.961 / 0.968 |

The quality/speed trade is now iteration- and sample-count-driven exactly
as in the paper (their geometry experiments run 2500 iterations). The
exact-gather ops (`RasterizeSoft` backward, `AntiAliasBwdVtx`) remain
available for comparison. Open: stage-2 raster-headed fusion, applying
the FaceFetch pattern to the AA technique, K>1 depth peeling.

## Follow-up: raster-headed fusion, stochastic AA backward, depth peeling
(`294dbe0`, `510376f`, this commit)

- **Raster-headed fusion** (Appendix A packing rule): a RASTER function
  may top a FRAG substage; GenerateRasterShaders now delegates to the
  common FRAG codegen (rasterization markers are in-body special
  branches), raster render passes support multiple color attachments.
  Bit-identical results; the win grows with fused shading chains.
- **AA stochastic backward**: AntiAlias(seed, n_samples) -- n=0 exact
  bbox scan (kept for FD tests), n>0 jittered edge samples evaluating the
  same EvalAaPair math. 256^2/1view 8.1 -> 0.53 ms; defaults 42.5 -> ~4 ms
  (AA forward/image-bwd remain the pixel-parallel cost). IoU 0.964 @2000
  iters/32 samples vs 0.981 exact; both modes via --samples.
- **Depth peeling (K > 1)**: RasterizeSoft peeling overload discards
  fragments at or in front of the previous layer's Eq.3-shifted depth
  (threshold rebuilt from output channels with elementwise ops); example
  blends K peels with Eq.6. IoU 0.968/0.975/0.980 at K=1/2/3 (2000 iters,
  16 samples) -- K=3 matches the exact-backward reference; 0.40 ms
  (256^2/1view) / 2.25 ms (defaults) per iteration.

Summary vs paper (256^2, 1 view, full iteration): paper RTX 2080
0.304-0.364 ms; ours RTX PRO 6000 0.22 ms (K=1) / 0.40 ms (K=3).
Same architecture, same knobs (r, K, iterations, samples).

## Resolution scaling vs paper Table 4 (1 view, full iteration, ms; ours =
## sphere 642v/1280f on RTX PRO 6000, paper = Avocado 406v/682f on RTX 2080)

| resolution | paper | ours K=1 | ours K=3 | ours AA (stoch) |
|---|---|---|---|---|
| 256^2  | 0.304 | 0.21 | 0.41 | 0.52 |
| 512^2  | 0.442 | 0.27 | 0.68 | 0.53 |
| 1024^2 | 1.034 | 0.54 | 1.56 | 0.54 |
| 2048^2 | 3.301 | 1.54 | 4.67 | 0.60 |

Same sublinear shape as the paper (fixed per-pass overhead dominates at
low resolutions, pixel work takes over above 1024^2). K=1 beats the
paper's absolute numbers at every resolution; accounting for the ~10x
raw-GPU gap an effective ~3-5x remains (pass-count overhead: regularizer
RMS reduction chains, {1,1} texelFetch uniforms, per-pass fixed cost) --
the remaining known deviations (uif_vars uniforms, zero-copy optimizer
aliasing, COMP reductions) map exactly onto it. The AA technique is
nearly resolution-independent thanks to the same-ID early-out in its
8-neighbor forward.

## Follow-up: pass-overhead squeeze + --single-view (this commit)

F::SumAll collapses the regularizer RMS reduction chains (~30 tiny
passes -> 3); the hardsoftras hard mask is derived from the frontal
peel's channels (one raster pass saved per view); F::PixelFetch /
F::TileFetch select per-iteration parameters (MVP columns, target tiles)
from static atlases with an in-graph index. --single-view implements the
paper's random-viewpoint methodology (sine-hash camera pick per
iteration; a deterministic cycle resonates with Adam's ~10-step momentum
window and loses ~0.03 IoU): 0.23-0.33 ms/iter at defaults (vs ~0.85
all-views), but sparse per-view gradients + Adam second-moment
statistics plateau around IoU 0.92 (16000 iters, K=3, 16 samples) vs
0.98 for all-views -- kept as an opt-in speed mode. The remaining
levers toward the paper's per-pass cost are uif_vars (real uniforms)
and zero-copy optimizer aliasing.
