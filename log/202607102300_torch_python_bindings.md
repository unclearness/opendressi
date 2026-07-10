# PyTorch bindings (dressi.torch): nvdiffrast-compatible API on the Vulkan engine

## Goal

Provide an nvdiffrast-drop-in PyTorch API (`import dressi.torch as dr`)
backed by the Dressi engine: `rasterize` (non-differentiable) /
`interpolate` / `texture` / `antialias`, plus `rasterize_soft`
(HardSoftRas) as an extension — vendor-neutral (any Vulkan GPU),
permissively licensed, with performance approaching the C++ examples
(user requirement mid-session: 「C++と同等程度の速度が出ることが目標」).
v1 decisions (user): drop-in nvdiffrast signatures, CPU-tensor transfer,
HardSoftRas as an extension function. Python package name: `dressi`
(user request; the project keeps the name opendressi).

## The define-and-run ⇄ eager reconciliation

- Every torch op is a `torch.autograd.Function` over a **shape-signature-
  keyed engine cache** (`RasterizeVulkanContext._engines`): cache miss
  builds a `DressiAD` graph (pack + GLSL + Vulkan objects), cache hit is
  sendImg → execStep → recvImg on the cached command buffer. Same shape
  keying trick nvdiffrast uses for its GL buffers.
- **External gradient injection**: `BuildBackward` seeds gradient 1 at
  every pixel (build_backward.cpp:87), so the surrogate loss
  `output * seed_leaf` (seed non-requires-grad) makes each leaf gradient
  the exact VJP for `grad_output == seed`. backward() = upload seed →
  execStep → recvImg grads.
- Reactive pruning disabled per engine (`setRebuildCounts(0,0)`): the
  fwd/bwd dirty patterns alternate ({inputs} vs {seed}) and would thrash
  SUBSTAGE repacks. Consequence: a training step costs 2 full graph
  executions (fwd with stale seed + bwd with the real seed); forward-only
  calls use a separate grad-free engine.

## What was done

### C++ core extensions
- `DressiAD(VkContextPtr)` + `DressiAD::createContext()` — one shared
  Vulkan device across all cached engines (was: one instance+device per
  DressiAD).
- `setGradOutputsEnabled` / `inputGrads` — keeps `input_grad_vars` as
  graph roots (the markOutput mechanism) so raw gradients are
  recvImg-readable **without an optimizer**. The previously considered
  identity-optimizer trick is broken: `upd_inp_map[leaf]=leaf` records a
  full-overlap `CopyImage(img, img)` (Vulkan UB). Tests:
  `tests/test_grad_export.cpp` (VJP vs analytic, parity with the
  optimizer-capture path, repeated seed updates, shared context).
- `FaceFetch` exact backward (`n_samples == 0`): per-face screen-bbox scan
  accumulating gy where `idx(p)==f+1` (GLSL variant of the
  `__face_fetch_bwd__` marker + CPU kernel). Needed because the stochastic
  edge-band sampler is badly biased for `interpolate`'s interior-support
  VJP. FD-checked (`Backward.FaceFetchExactBackward`) + GPU parity
  (`GradExport.FaceFetchExactBwdGpuParity`).
- Mesh/topology builders promoted from `examples/common` into the library:
  `include/dressi/mesh_utils.h` (`VertexFacesTex`, `VertexNeighborsTex`,
  `FaceNeighborsTex`, `BuildSoftGeometry`, `UvAsClip`, `ScreenAttr`,
  adjacency builders); examples keep thin using-aliases.
- `CpuImageView` + `sendImg(Variable, CpuImageView)` — borrowed-buffer
  upload without the intermediate `CpuImage` vector copy.

### Transfer-layer performance (the big one)
The eager benchmark started at **35.7 ms/iter @512²**; profiling showed
`recvImg` of the 4 MB rast image took **23 ms**: vkw's memory-type search
picks the first HostVisible|HostCoherent type, which on NVIDIA is
write-combined (uncached) — CPU memcpy *reads* from it run at ~100 MB/s.
Fixes (src/vk/transfer.cpp, context.{h,cpp}):
- Readback staging now requests HostVisible|HostCoherent|**HostCached**
  (fallback to plain coherent when unavailable).
- Persistent grow-on-demand staging buffers per context (send/recv) —
  no per-transfer vkAllocateMemory.
- `RunOneShot` reuses a cached command pool/buffer + fence (was: created
  per transfer).
- Channel-identical images (n_logical == n_phys) skip the re-stride pass
  on both directions; `_stack_batch` avoids the torch.stack copy for N=1.
Result: 35.7 → **4.0 ms/iter @512²** (9x).

### Python layer (`python/`)
- nanobind (`dressi._C`): DressiAD (incl. set_optimizer/add_update),
  Variable (+ graph-building operators), F subset, mesh utils; images
  cross as (H, W, C) float32 ndarrays (numpy out via moved-vector
  capsule, zero-copy in via CpuImageView). GIL released around
  exec/send/recv. `nb::set_leak_warnings(false)` (engine caches
  legitimately live until exit).
- `dressi.torch`: `context.py` (shared device + engine/mesh caches),
  `engine.py` (leaves, surrogate-loss finalize incl. multi-output for
  peels, dirty-skip uploads), `meshdata.py` (faces in BOTH encodings:
  IVEC3 for rasterizers, float VEC3 for gathers; `vtx_faces_tex`; unweld
  index), `provenance.py` (WeakValueDictionary side table: rast → {pos,
  tri, resolution, ctx}, uv → {vtx_uv, rast}; nvdiffrast signatures drop
  data the Dressi backwards need), `convert.py` (GL→Vulkan depth remap
  `z_vk = 0.5(z_gl+w)` and its transpose for pos grads), `ops.py` (the
  autograd.Functions).
- Op graphs:
  - `rasterize`: unwelded corners with constant (u,v) barycentric
    attributes + interpolated (z_gl, w) via `F::Rasterize` (perspective-
    correct) + `F::RasterizeFaceId` → rast=(u,v,z/w,tri_id+1). x/y match
    nvdiffrast at the tensor level (verified); tri_id+1 convention
    matches natively.
  - `interpolate`: per-corner `LookupFaces` → exact `FaceFetch` gather by
    the rast tid channel → barycentric combination. Exact VJP w.r.t. attr.
  - `texture`: inverse-UV table rasterized in-graph from provenance
    (UvAsClip + ScreenAttr); train engines only — forward-only sampling
    needs no provenance.
  - `antialias`: `F::AntiAlias` with per-engine iteration seed; grads for
    color and pos (n_samples=0 exact mode for tests).
  - `rasterize_soft`: `F::SoftClip` → `F::RasterizeSoft` chain with the
    example's in-graph Eq.3 shifted-depth reconstruction between peels;
    returns [N,K,H,W,4] (dist,face_id,depth,cov); sigmoid/Eq.6 composed
    in torch.
- Upload dirty-skip tokens: `(data_ptr, _version, shape)` **plus a strong
  reference** to the source buffer — without the pin, a freed tensor's
  address can be reused by the next allocation with `_version` 0 and the
  token falsely hits (this actually happened: FD tensors alternated into
  the same address and texture gradients came back zero).

### Tests
- C++: 105 (72 cpu / 33 gpu) all green, including the new grad-export and
  FaceFetch-exact suites.
- Python (`tests/python`, 20 tests, also a gpu-labeled ctest entry):
  analytic barycentrics/perspective/depth for rasterize; interpolate
  forward vs manual bary + FD grads (linear → exact); texture nearest
  forward + FD grads on the 1:1 mapping; antialias boundary-only forward,
  color FD, pos-grad cosine vs FD, silhouette optimization convergence
  (offset recovered to ±0.03); rasterize_soft channels, two-layer peeling,
  pos-grad cosine, silhouette convergence; texture-fit E2E; engine-cache
  reuse and upload-skip behavior.

## Measurements (Avocado 406v/682f, fixed view, Adam; ms/iter)

| resolution | eager start | eager final | native fused (Python) | C++ example AA | nvdiffrast CUDA | DRTK |
|---|---|---|---|---|---|---|
| 256²  | -    | 3.3 | 0.55 | 0.49 | 1.11 | 2.12 |
| 512²  | 35.7 | 4.0 | 0.57 | 0.49 | 1.18 | 6.15 |
| 1024² | -    | 9.0 | 0.55 | 0.51 | 1.21 | 19.6 |

- `scripts/dressi_torch_bench.py` (eager) / `scripts/dressi_native_bench.py`
  (fused: whole step incl. in-graph Adam as ONE Dressi graph driven from
  Python — **meets the C++-parity goal**; the remaining ~0.06 ms is the
  Python loop + exec_step call overhead).
- Eager profile @512² after fixes: rasterize 1.0 (read 0.5), AA fwd 1.7,
  AA bwd 1.6 — bounded by per-op CPU round trips and the fwd recompute in
  backward, both inherent to the drop-in eager model with CPU tensors.

## Key decisions & lessons

- nanobind over pybind11: native DLPack/ndarray, no libtorch dependency
  (no per-torch-version rebuilds).
- `setRequiresGradRecursively` propagates DOWNSTREAM (to users) — must be
  called AFTER graph construction; calling setRequiresGrad on a leaf
  before building silently yields an empty backward.
- Identity optimizer for grad readback = self-CopyImage UB; the
  grad-export API is the clean path.
- Data-ptr-based caching of torch tensors requires pinning the source
  buffer (address reuse otherwise forges cache hits).
- Write-combined staging memory is fine to write, catastrophic to read.

## Open issues / next

- **Zero-copy GPU interop (user request)**: direction must be
  Vulkan-allocates → CUDA-imports (VK_KHR_external_memory +
  `cudaImportExternalMemory`; OPAQUE_WIN32 handles on Windows), exposing
  engine images/buffers to torch as CUDA tensors — the reverse (importing
  torch's CUDA allocations into Vulkan) is not possible because torch's
  caching allocator does not export. Needs: exportable-memory allocation
  path in the executor, device matching (LUID), and semaphore or
  wait-idle synchronization; torch-side via `__cuda_array_interface__`
  (cuda-python/cupy) or a small CUDA shim. Non-NVIDIA GPUs keep the CPU
  path as fallback.
- Eager-path headroom: batch per-run uploads into one submit; a
  dirty-pattern-keyed plan cache in the engine would let the backward
  exec skip the clean forward branch (currently recomputed).
- Batches loop per item; range mode unsupported; texture is nearest-only,
  no grad w.r.t. UV (engine limitation, documented).
- Packaging (scikit-build-core wheel) not yet added; the tested dev loop
  is `-DDRESSI_BUILD_PYTHON=ON` + `PYTHONPATH=python`.
- `.venv` (Python 3.13) created at the repo root for the binding tests;
  not committed.

## Addendum 2: eager-path performance round (user goal: beat nvdiffrast)

Target workload: silhouette example, 8 views, 128², CUDA tensors.
Start 39.2 ms/iter (dressi) vs 13.1 (nvdiffrast, per-view loop).

Implemented (all engine-cache-transparent to the API):
- **Batched engines**: an N-item call builds ONE engine with N unrolled
  per-item subgraphs (leaves "<base>#<i>", static topology shared;
  Engine.finalize handles multi-output surrogate losses via a <=4-ary
  SumPixelWise tree). 24 execSteps/iter -> 3. interpolate keeps a single
  shared `attr` leaf when attr is [V,C] (per-item gradient contributions
  accumulate in-graph).
- **Batched transfers**: `DressiAD::sendImgs`/`recvImgs` — one staging
  buffer, one submit, offset BufferImageCopy regions (own record;
  vkw::SetImageLayout for transitions). Engine queues uploads and flushes
  at run(); all outputs/gradients download in one submit. Upload flush
  measured 0.09 ms for 27 images.
- **Forward/backward engine split**: forward runs the gradient-free
  engine; backward gets the train engine (its exec recomputes the forward
  internally). Keeps the AA vertex-gradient stage (2.9 ms) out of forward
  entirely.
- **CPU-copy cache** (convert.py): id+_version-keyed cache of device
  tensors' CPU copies; ops register their outputs, so rast produced on
  CUDA doesn't re-cross for interpolate/antialias.
- Per-item AA jitter seeds (a shared seed correlates the stochastic
  backward across views and measurably hurts convergence).

Two real bugs found by the batching work:
- MeshData cache keyed by tri's (data_ptr, _version, shape) must PIN the
  source tensor — a freed tri's address reused by a different topology
  produced order-dependent "image size mismatch" failures in the suite.
- An in-graph unweld (SoftClip radius=0, kept as a relaxed >=0 check) was
  tried to eliminate the CPU corner-gather + {3F,4} uploads; it PASSED
  all unit tests but degraded AA silhouette IoU 0.97 -> 0.94
  reproducibly, and was reverted (cause not yet isolated — the fp
  screen-space round trip only perturbs positions ~1e-6; suspicious).

Results: silhouette aa 39 -> **15.7 ms/iter** (IoU 0.971 preserved),
hardsoftras 97 -> 61.7, Avocado 1-view eager 3.3/4.0/9.0 ->
2.8/3.6/9.0. nvdiffrast batched: 6.2 ms — still 2.5x ahead. Measured
remaining gap: (1) AntiAliasBwdVtx GPU 2.9 ms — {V,1} layout = 642
threads with the 8-view batch serialized inside each thread
(latency-bound; fix = batch-aware {V,N} gather stage, a C++ op/codegen
change); (2) ~6 ms host glue (CUDA<->CPU staging copies, ~6 submit
fences, numpy conversions) — the decisive fix is Vulkan-export -> CUDA
-import interop. `_C.set_log_level("debug")` now exposes the per-stage
GPU timestamp breakdown from Python (how the BwdVtx wall was found).

## Addendum 3: wide per-vertex gathers + stacked downloads (2026-07-11)

The GPU timestamp breakdown identified the actual walls after Addendum 2:
`AntiAliasBwdVtx` 2.83 ms and `GatherDistGrad` **14.4 ms** GPU for the
8-view batch — both are {V,1}-target gathers, i.e. only V (=642)
fragment/compute threads with the whole batch AND all incident faces
serialized inside each thread.

- **Wide layout** (default for both): output {V, max_deg} — thread (v, d)
  handles ONE incident face — reduced by a new `ColSum` op
  (`__col_sum__`, {W,H}→{W,1}). For GatherDistGrad the wide thread scans
  only ITS face's soft bbox with an exact `face_id == f` pixel guard
  (pixels belong to exactly one face → the split cannot double count),
  which also shrinks the scan area vs the union bbox. Measured:
  AA 2826 → 435 us; hsr example 58.8 → 23.8 ms/iter; the native fused
  Avocado bench 0.57 → **0.218 ms/iter** (the same stage dominated it).
  CPU kernels mirror the row attribution for exact GPU/CPU parity.
- **Stacked batched downloads**: `recvImgsStacked` strips same-shape
  images into ONE contiguous {w, h*n} buffer (one submit, no per-item
  allocations, no torch.stack copy); rasterize prep vectorized (one
  batched unweld gather + concat).
- Silhouette example (8 views, 128², CUDA): aa 15.7 → **10.7 ms/iter**
  (from 39 at the start of the perf work; IoU 0.971-0.973 preserved),
  hardsoftras 97 → 23.8. Avocado eager 2.3/3.4/8.5 @256/512/1024².
  nvdiffrast aa: 6.2 — remaining gap is ~6 ms host glue (CUDA<->CPU
  staging + submit fences); the decisive next step stays Vulkan-export ->
  CUDA-import interop.
- Rejected this round: in-graph unweld via SoftClip(radius=0) — passes
  all unit tests but reproducibly degrades AA convergence (IoU
  0.97 -> 0.94, cause not isolated); rasterize keeps the CPU-side unweld.

## Addendum: Python examples + uv packaging (same session)

- `pyproject.toml` (uv-managed, `package = false`): CUDA torch from the
  cu130 index on win32/linux (pinned `!=2.13.0` — that one wheel
  persistently fails uv's hash check on this network; 2.12.1+cu130
  installs fine), pillow/numpy, `--extra nvdiffrast` pulls nvdiffrast
  **v0.3.3 by git tag** (newer versions compile their CUDA extension at
  pip-build time and fail in uv's isolated build env; 0.3.3 JIT-compiles
  at import). `.python-version` = 3.13. Run everything via `uv run`
  (puts `.venv/Scripts` on PATH — torch's JIT needs `ninja` there).
- nvdiffrast-on-Windows shims live in `examples/python/common.py`
  `resolve_backend`: lenient MSVC banner decoding (broken 'oem' codec on
  Japanese Windows) and inserting the torch-extensions build dirs into
  `sys.path` (recent torch no longer does; nvdiffrast 0.3.3's
  `importlib.import_module` relies on it). nvdiffrast's `texture` needs a
  batched `[N,H,W,C]` texture (ours accepts both).
- `examples/python/{silhouette,texture}_optimization.py` +` common.py`:
  ports of the C++ examples using ONLY the dr.* API + torch (user
  directive). Backend switch `--backend dressi|nvdiffrast` runs the same
  code on both (drop-in compat demo); `--technique hardsoftras` is
  dressi-only. Device defaults to CUDA when available (dressi v1 then
  pays CUDA→CPU staging per call).
- Convergence: C++ regularizer weights do NOT transfer 1:1 (user called
  it): with C++'s aa 0.5/0.5 the torch port plateaus at IoU 0.93;
  retuned aa defaults 0.2/0.2 → IoU 0.972 (300 iters), hsr keeps
  0.15/0.4 → 0.962. Regularization mirrors the C++ scheme (gradient-level
  composition scaled by data-grad RMS; normal-consistency gradient via
  `torch.autograd.grad` of the energy). Initial sphere radius 0.55 (C++
  value), saved images flipped vertically for viewing (row 0 = GL
  bottom; nvdiffrast-sample convention), final mesh saved as OBJ (user
  request). Texture example: sub-pixel Halton camera jitter (the C++
  Sobol countermeasure) rendering the target under the same jitter;
  PSNR 52 dB @500 iters (dressi), 47 dB @60 iters (nvdiffrast smoke).
