# COMP substages + rematerialization (HSR 4096² lever)

Continuation of log/202607102000: the remaining backlog items, in the
agreed order — (1) COMP shader substages, (2) rematerialization,
(3) HSR 4096² pixel cost. Same benchmark: Avocado 406v/682f silhouette
self-reconstruction, 1 view, median execStep, RTX PRO 6000.

## 1. COMP substages — implemented; hot ops stay FRAG by measurement

Full pipeline support for `ShaderType::COMP`:
- Packer: a COMP function opens a COMP substage; FRAG functions may fuse
  INTO it (snippets are shader-agnostic), never the reverse — the
  substage type is set by its latest function, which keeps the aliased
  Adam-update substage FRAG. `Sampled` inputs are rejected in COMP
  (implicit-LOD `texture()` is illegal in compute). COMP substages are
  their own stage (a dispatch, no render pass).
- Codegen: `GenerateCompShader` shares the FRAG body generator;
  local_size 64×1, `dressi_coord` from `gl_GlobalInvocationID` with a
  bounds check, inp inputs become samplers fetched at the invocation
  coordinate, outputs are `writeonly image2D` (r32f/rg32f/rgba32f)
  written by `imageStore`.
- Executor: compute pipeline + descriptor sets (samplers, uif UBOs,
  storage images), batched eUndefined→eGeneral / eGeneral→eShaderRead
  barriers around the dispatch (transfer read included for downstream
  vertex-buffer/UBO copies). Var images gained eStorage usage.

Measurement (256², all 7 band-pass gathers marked COMP): 97.5 → 115 us.
Two causes: marking `LookupFacesBwd` COMP pulled it out of the fused
elementwise gradient substage (+8 us), and each graphics↔compute switch
on the single queue costs ~1.5-2 us (SoftClip 2.8→4.6, LookupFaces
2.4→4.1). Decision: default assignment reverted to FRAG for the hot
ops; `GatherDistGrad` (exact-backward op, exercised by the SoftRas
tests, absent from the stochastic hot path) stays COMP so ctest covers
the compute path end to end. Validation-clean.

## 2. Rematerialization — the big one

`RematerializeSubStages` (greedy_packer, RSP only), after substage
packing:
1. For every substage input attachment, a DFS tries to clone the
   producing chain into the substage: clone-safe = FRAG, same img size,
   plain snippet or whitelisted coordinate-pure marker
   (`__face_fetch__`, `__screen_coord__`); TexelFetch inputs join
   slt_vars (budget: max_sampled_images), `{1,1}` values follow the
   uif/slt leaf rule, already-available values (gen/inp/planned) are
   reused. Budget: added funcs ≤ max(32, current size) per substage.
2. Dead-output elimination in reverse execution order: an output
   survives only if a later substage or a root still demands it;
   empty substages are dropped.

Guards (the second one found by the author running the example while
this was uncommitted): a cloned value must never also be an slt/tex
input of the same substage, and a clone plan is rejected if any of its
TexelFetch inputs is generated (or being cloned) in the same shader —
both would emit duplicate GLSL declarations (`v2` redefinition).

Effect at 256²: the backward mega-stage went o=59/i=68/fn=270 →
o=35/i=39/fn=523 (recompute instead of materialize; the two
SumPixelWise follow-up substages fused away entirely), 27.4 → 21.6 us,
total 94.5 → 84.2 us. The win grows with resolution — this is
attachment-bandwidth relief.

## 3. HSR 4096² — closed by (2)

| resolution | HSR K=1 before → after | K=3 before → after | AA |
|---|---|---|---|
| 256²  | 0.16 → **0.15** | 0.27 → **0.24** | 0.49 |
| 512²  | 0.21 → **0.18** | 0.43 → **0.35** | 0.49 |
| 1024² | 0.41 → **0.30** | 1.03 → **0.77** | 0.51 |
| 2048² | 1.22 → **0.83** | 3.47 → **2.42** | 0.57 |
| 4096² | 4.37 → **2.87** | — | 0.78 |

HSR at 4096² now beats nvdiffrast (2.87 vs 3.28 CUDA / 3.47 GL) — both
techniques are faster than every measured baseline at every resolution.
Cumulative over the two perf sessions: 256² 0.22→0.15, 4096² ~5.5→2.87.

Quality: bunny defaults HSR IoU 0.9587 / AA 0.9404 (pre-session
0.9573/0.9361 — run-to-run FMA noise); Avocado K=3 0.9730@50it;
texture_optimization unaffected (4000 iters, final loss 8.4e-6, 99.4%
texels). 98/98 tests pass; validation layers clean.

## Notes / open items

- `DRESSI_DUMP_STAGES=1` dumps per-substage packing (funcs, inp/out var
  ids) on rebuild — used to find the remat candidates.
- Remat is depth-limited by "all inputs available or clonable"; chains
  rooted in non-whitelisted markers stop there. Extending the whitelist
  (e.g. `__antialias__`) could shave the AA chain similarly if ever
  needed.
- Possible future COMP win: batching CONSECUTIVE dispatches without
  intervening render passes (avoids the per-switch cost) or a dedicated
  compute queue; not pursued since FRAG fusion currently wins.
