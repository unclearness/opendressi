# Benchmarks

Detailed performance data for dressi. See the [README](../README.md) for the
headline summary.

## Methodology

Every figure is the **median steady-state `execStep` wall time** with a 20-iter
warmup excluded — reproduce it exactly; do NOT report a raw mean-over-all-iters.

- **Exclude a warmup.** The first `execStep` runs the full build (pack + GLSL
  compile + Vulkan objects, ~250 ms) and the reactive cache does a fast rebuild
  at iter 2 and a full rebuild at iter 8. Skip ≥20 iters before timing.
- **Median, not mean.** At ≤512² the wall is dominated by host submit/fence
  latency (GPU work is ~90 µs while the wall is ~0.3 ms), so the mean is pulled
  up by OS-scheduling outliers. Median is the stable metric.
- **CUDA baselines synchronize in blocks.** An eager CUDA pipeline must
  synchronize before/after each timed sample; the Python examples use
  10-iteration blocks after the warmup and report the median block-average.

Every example writes a `bench.json` (GPU device, host os/cpu/ram — phone model
on Android — parameters, median steady-state ms/iter, warmup, and quality) into
its `--out-dir`; `scripts/bench_summary.py <dirs>` renders the comparison table.

## Cross-device — 8 Vulkan devices

The identical backward graph runs across three discrete NVIDIA GPUs, an Intel
iGPU, two phones, a tablet, and a CPU software rasterizer — to the same quality.

Devices:

- **NVIDIA RTX PRO 6000 Blackwell Max-Q** — Ryzen Threadripper 9970X, Win 11 Pro
- **NVIDIA RTX 6000 Ada** — Xeon w7-3465X, Ubuntu 24.04
- **NVIDIA GeForce RTX 3090** — Core i9-10850K, Win 11 Pro
- **Intel Iris Xe Graphics** — Core i7-1370P, Win 11 (integrated GPU)
- **Adreno 740** — Snapdragon 8 Gen 2, Xiaomi Pad 6S, Android 16 (tablet)
- **Mali-G1-Ultra MC12** — MediaTek Dimensity 9500, Xiaomi 17T Pro, Android 16 (phone)
- **Mali-G715-Immortalis MC11** — MediaTek Dimensity 9200+, Xiaomi 13T Pro, Android 15 (phone)
- **llvmpipe (LLVM 20.1.2)** — Mesa lavapipe, CPU software Vulkan

Median ms/iter (default parameters), `packing f→s→st` = unpacked backward-graph
ops → substages → Vulkan render passes:

| example | device | median ms/iter | warmup ms (first) | packing f→s→st | quality |
| --- | --- | ---: | ---: | ---: | --- |
| **image_fitting**<br>64px 150it | RTX PRO 6000 Blackwell | 0.056 | 269 (266) | 36→8→8 | max err 0.0005 |
| | RTX 6000 Ada | **0.048** | 288 (284) | 36→8→8 | max err 0.0005 |
| | RTX 3090 | 0.116 | 592 (578) | 36→8→8 | max err 0.0005 |
| | Iris Xe (iGPU) | 0.320 | 136 (127) | 36→8→8 | max err 0.0005 |
| | Adreno 740 (tablet) | 0.225 | 193 (136) | 36→8→8 | max err 0.0005 |
| | Mali-G1-Ultra (phone) | 1.007 | 171 (123) | 36→8→8 | max err 0.0005 |
| | Mali-G715 (phone) | 2.038 | 345 (256) | 36→8→8 | max err 0.0005 |
| | llvmpipe (CPU) | 0.729 | 160 (138) | 36→8→8 | max err 0.0005 |
| **texture_optimization**<br>256px 6v 1500it | RTX PRO 6000 Blackwell | 0.958 | 243 (214) | 232→25→25 | 98.7% |
| | RTX 6000 Ada | **0.889** | 362 (347) | 232→25→25 | 98.7% |
| | RTX 3090 | 2.248 | 635 (580) | 232→25→25 | 98.7% |
| | Iris Xe (iGPU) | 15.695 | 338 (293) | 232→27→25 | 98.7% |
| | Adreno 740 (tablet) | 48.361 | 706 (516) | 232→26→25 | 98.7% |
| | Mali-G1-Ultra (phone) | 23.565 | 1103 (531) | 232→26→25 | 98.7% |
| | Mali-G715 (phone) | 24.255 | 1689 (772) | 232→26→25 | 98.7% |
| | llvmpipe (CPU) | 129.7 | 614 (561) | 232→25→25 | 98.7% |
| **silhouette (aa)**<br>128px 8v 150it | RTX PRO 6000 Blackwell | **1.213** | 644 (496) | 802→38→38 | IoU 0.9529 |
| | RTX 6000 Ada | 1.261 | 595 (506) | 802→38→38 | IoU 0.9529 |
| | RTX 3090 | 2.813 | 2256 (1749) | 802→38→38 | IoU 0.9529 |
| | Iris Xe (iGPU) | 3.339 | 393 (284) | 802→61→53 | IoU 0.9544 |
| | Adreno 740 (tablet) | 21.572 | 1612 (798) | 802→53→47 | IoU 0.9447 |
| | Mali-G1-Ultra (phone) | 11.519 | 8131 (2012) | 802→53→47 | IoU 0.9550 |
| | Mali-G715 (phone) | 21.244 | 2728 (1228) | 802→53→47 | IoU 0.9559 |
| | llvmpipe (CPU) | 28.20 | 1352 (666) | 802→38→38 | IoU 0.9473 |
| **silhouette (hardsoftras)**<br>128px 8v 150it | RTX PRO 6000 Blackwell | **2.163** | 1179 (776) | 3962→123→58 | IoU 0.9514 |
| | RTX 6000 Ada | 3.009 | 951 (695) | 3962→123→58 | IoU 0.9514 |
| | RTX 3090 | 6.824 | 2900 (2167) | 3962→123→58 | IoU 0.9514 |
| | Iris Xe (iGPU) | 9.956 | 907 (584) | 3962→202→68 | IoU 0.9491 |
| | Adreno 740 (tablet) | 23.661 | 3502 (1587) | 3962→185→64 | IoU 0.9406 |
| | Mali-G1-Ultra (phone) | 32.455 | 5520 (1794) | 3962→174→64 | IoU 0.9542 |
| | Mali-G715 (phone) | 36.626 | 6576 (2818) | 3962→174→64 | IoU 0.9512 |
| | llvmpipe (CPU) | 46.91 | 1384 (806) | 3962→123→58 | IoU 0.9506 |
| **pbr_material_optimization**<br>albedo 256px tex512 6v 1200it | RTX PRO 6000 Blackwell | **2.727** | 1362 (964) | 3593→74→42 | PSNR 20.65 dB |
| | RTX 6000 Ada | 3.510 | 1220 (927) | 3593→74→42 | PSNR 20.65 dB |
| | RTX 3090 | 6.042 | 3539 (2661) | 3593→74→42 | PSNR 20.65 dB |
| | Iris Xe (iGPU) | 17.923 | 1019 (788) | 3593→146→49 | PSNR 20.65 dB |
| | Adreno 740 (tablet) | 55.171 | 3786 (1868) | 3593→135→49 | PSNR 20.65 dB |
| | Mali-G1-Ultra (phone) | 40.891 | 7667 (3336) | 3593→125→49 | PSNR 20.65 dB |
| | Mali-G715 (phone) | 46.994 | 7875 (3632) | 3593→125→49 | PSNR 20.65 dB |
| | llvmpipe (CPU) | 117.7 | 2098 (1353) | 3593→74→42 | PSNR 20.48 dB |
| **pbr_envmap_optimization**<br>192px env64 4v 1500it | RTX PRO 6000 Blackwell | 8.532 | 1980 (1642) | 2436→43→36 | PSNR 18.26 dB |
| | RTX 6000 Ada | **8.193** | 904 (777) | 2436→43→36 | PSNR 18.26 dB |
| | RTX 3090 | 14.012 | 6494 (4582) | 2436→43→36 | PSNR 18.26 dB |
| | Iris Xe (iGPU) | 105.30 | 833 (792) | 2436→54→38 | PSNR 18.26 dB |
| | Adreno 740 (tablet) | 304.1 | 3124 (2041) | 2436→53→38 | PSNR 18.26 dB |
| | Mali-G1-Ultra (phone) | 181.1 | 8252 (3800) | 2436→50→37 | PSNR 18.23 dB |
| | Mali-G715 (phone) | 200.2 | 10312 (5076) | 2436→50→37 | PSNR 18.26 dB |
| | llvmpipe (CPU) | 669.0 | 2043 (1757) | 2436→43→36 | PSNR 18.25 dB |
| **pbr_shading**<br>512px 60 frames¹ | RTX PRO 6000 Blackwell | 1.477 (677 FPS) | 305 (289) | 286→5→5 | 677 FPS |
| | RTX 6000 Ada | **1.040** (961 FPS) | 538 (526) | 286→5→5 | 961 FPS |
| | RTX 3090 | 2.948 (339 FPS) | 789 (764) | 286→5→5 | 339 FPS |
| | Iris Xe (iGPU) | 8.660 (116 FPS) | 702 (654) | 286→11→5 | 116 FPS |
| | Adreno 740 (tablet) | 17.717 (56 FPS) | 825 (638) | 286→11→5 | 56 FPS |
| | Mali-G1-Ultra (phone) | 53.746 (19 FPS) | 779 (671) | 286→10→5 | 19 FPS |
| | Mali-G715 (phone) | 24.044 (42 FPS) | 1206 (973) | 286→10→5 | 42 FPS |
| | llvmpipe (CPU) | 11.97 (84 FPS) | 894 (719) | 286→5→5 | 84 FPS |

¹ `pbr_shading` is an interactive viewer; on Android it runs a deterministic
200-frame orbit (vs desktop headless 60) so it self-terminates and emits
`bench.json`. The metric is exec+readback per frame, so the frame budget does
not change the median.

Observations:

- **Quality is device-independent** — the metric matches across all eight
  devices to within stochastic noise (texture 98.7% everywhere, pbr_envmap
  18.23–18.26 dB, silhouette IoU 0.941–0.956), validating the engine.
- **Packing is device-dependent** — the func count is identical, but each
  device fuses to a different substage count under its own Vulkan limits: the
  NVIDIA cards fuse hardest (pbr_material →74), the Intel iGPU least (→146), the
  phones in between (both Mali →125 / Adreno →135). The render-pass (stage) count
  that costs draw calls stays low everywhere.
- **Mobile spans the same graph** — the three mobile devices (Xiaomi 17T Pro and
  13T Pro phones, Xiaomi Pad 6S tablet) are ~1-2 orders slower on the heavy
  optimizers but converge to the same quality. The Dimensity 9500 (Mali-G1-Ultra,
  17T Pro) beats the Snapdragon 8 Gen 2 (Adreno 740, Pad 6S) on the heavy
  optimizers yet is slower on light/forward paths and pays a much larger
  first-build warmup — its shader compiler, not steady-state throughput, is the
  bottleneck. The older Dimensity 9200+ (Mali-G715, 13T Pro) trails both on the
  heavy optimizers but leads the Dimensity 9500 on light/forward paths
  (pbr_shading 42 vs 19 FPS) — the same compile-bound signature, one generation
  back.

## vs the paper (Table 4)

One-view **Avocado** (406 v / 682 f) self-reconstruction, median `execStep`.
Hardware: RTX PRO 6000 (Blackwell); the paper used an RTX 2080 (~10× less raw
throughput).

| resolution | paper (RTX 2080) | ours HardSoftRas K=1 | ours K=3 | ours AA |
|---|---|---|---|---|
| 256²  | 0.304 | 0.34 | 0.77 | 0.24 |
| 512²  | 0.442 | 0.38 | 0.88 | 0.24 |
| 1024² | 1.034 | 0.50 | 1.30 | 0.25 |
| 2048² | 3.301 | 1.02 | 2.95 | 0.31 |
| 4096² | —     | 3.08 | 9.64 | 0.51 |

The scaling shape matches the paper: fixed per-pass overhead dominates at low
resolution; pixel work takes over above 1024². At ≤512² the wall is bound by
host submit/fence latency, not GPU work.

## vs nvdiffrast and DRTK

Same GPU, same task (`scripts/nvdiffrast_bench.py`, `scripts/drtk_bench.py`).
nvdiffrast 0.4.0 dropped the OpenGL rasterizer, so the paper-era GL backend is
measured on a pinned v0.2.6; [DRTK](https://github.com/facebookresearch/DRTK)
runs its canonical `rasterize` + `render` + `edge_grad_estimator` silhouette
pipeline.

| resolution | nvdiffrast GL v0.2.6 | nvdiffrast CUDA v0.4.0 | DRTK 0.1.0 | ours HSR K=1 | ours AA |
|---|---|---|---|---|---|
| 256²  | 1.78 | 1.11 | 2.12 | **0.15** | 0.49 |
| 512²  | 1.76 | 1.18 | 6.15 | **0.18** | 0.49 |
| 1024² | 1.82 | 1.21 | 19.6 | **0.30** | 0.51 |
| 2048² | 1.70 | 1.28 | 67-108 | 0.83 | **0.57** |
| 4096² | 3.47 | 3.28 | ~220 | 2.87 | **0.78** |

Both techniques beat every baseline at every resolution.

- **Why dressi wins below 2048²:** every nvdiffrast backend measures flat
  (~1.1-1.8 ms) here — at this mesh size a PyTorch iteration is bound by
  host-side kernel launches, not GPU work. dressi bakes the whole
  forward+backward+optimizer into pre-recorded Vulkan command buffers with zero
  per-iteration host involvement, so it runs at the cost of the GPU work itself.
- **Why the paper's "gap grows with resolution" doesn't reproduce as-is:** on
  the RTX 2080 nvdiffrast's 2048² step was GPU-bound (5-8 ms); on Blackwell the
  same pixel work costs well under 1 ms and hides beneath the host-launch floor
  until 4096². Same advantage source, saturating earlier.
- **AA is nearly resolution-independent** (0.49 → 0.78 ms for a 256× pixel
  increase): its 8-neighbor forward early-outs on equal triangle IDs
  (perimeter-bound), and its stochastic vertex backward is O(faces).
- **DRTK is pixel-bound from 512² up** — a brute-force CUDA rasterizer with no
  hardware rasterization; silhouette-only gradients deform the mesh into
  transient large/degenerate triangles that multiply covered-pixel cost.
- **Rematerialization closed the former 4096² gap:** the HardSoftRas backward
  used to materialize ~59 fp32 intermediates per stage as attachments; the
  packer's clone pass recomputes cheap forward chains in consumer subpasses
  instead, dropping the crossing count ~40% (4.37 → 2.87 ms at 4096²).

Quality on this benchmark: self-reconstruction reaches silhouette IoU 0.986
(HardSoftRas, K=3) / 0.977 (AA); the stochastic backwards trade a little
per-iteration accuracy for the O(faces) cost and recover it with iterations and
`--samples` / `--peels`, exactly as in the paper.

## PyTorch (`dressi.torch`) performance

Same Avocado silhouette benchmark (median steady-state ms/iter; fixed view,
Adam on positions). The **native fused** column drives the *same* dressi graph
as the C++ `silhouette_optimization` example and lands within noise of it (C++
0.26 ms @512²) — Python is not magically faster; both run the compiled graph
with zero per-iteration PCIe traffic:

| resolution | dressi.torch (eager) | native fused (Python ≈ C++) | nvdiffrast CUDA | DRTK |
|---|---|---|---|---|
| 256²  | 2.3 | **0.21** | 1.11 | 2.12 |
| 512²  | 3.4 | **0.22** | 1.18 | 6.15 |
| 1024² | 8.5 | **0.22** | 1.21 | 19.6 |

Multi-view silhouette example (8 views, 128², 300 iters, CUDA tensors): aa
39 → **8.0 ms/iter**, hardsoftras 97 → **21.6 ms** (nvdiffrast aa on the same
example code: **3.1 ms**). Profiling corrected an earlier assumption: the eager
"host glue" is not PCIe-bound (CUDA↔CPU copies are only ~0.3 ms) — it is per-op
Python/submit overhead (transfer+`execStep` submit/fences, staging memcpy, torch
autograd plumbing), which the native-fused path (one graph, one submit) avoids
entirely. The eager engine batches N-item calls into ONE engine (one execStep
per forward/backward for a whole minibatch), and the per-vertex gather backwards
run WIDE ({V, max_deg} partials + column-sum) — the lever that also sped up the
C++ paths (AA 2.9 → 0.44 ms on an 8-view batch).

## Reproduce

```sh
# desktop batch: runs the examples, writes bench.json into per-example dirs
python scripts/run_benches.py
python scripts/bench_summary.py bench_run bench_run_lvp   # lvp = llvmpipe

# llvmpipe (CPU software Vulkan): select the lavapipe ICD
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json python scripts/run_benches.py

# Android: run each example in the app (RELEASE build), then pull + aggregate
scripts/pull_android_benches.sh  -o bench_results/<device>     # Git Bash / WSL
scripts/pull_android_benches.ps1 -OutRoot bench_results/<device>  # PowerShell
python scripts/bench_summary.py bench_results/<device>
```

`bench_summary.py` needs a real Python (`.venv\Scripts\python` or `uv run
python` on Windows — the bare `python` may hit the Store alias).
