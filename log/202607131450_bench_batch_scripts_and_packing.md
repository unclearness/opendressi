# Batch bench runners, Win11 OS string, packing-reduction records

## Goal

Make it easy to run every example's benchmark in one go and aggregate the
results, and record two things the summary was missing: the correct OS name on
Windows 11 and the stage/substage packing reduction (device-dependent).

## What was done

### Batch bench runners (execution and aggregation stay separate)

- `scripts/run_benches.ps1` (Windows), `scripts/run_benches.sh` (Linux/macOS):
  run each example exe with headless flags and a per-example `--out-dir`, one
  `bench.json` per run under an output root (default `bench_results`). They do
  NOT summarize — aggregate afterwards with `scripts/bench_summary.py`.
  Headless flags per example: silhouette `--no-view=1`, texture/pbr_material/
  pbr_envmap `--view-interval=0`, pbr_shading `--frames=60`, image_fitting none.
- `scripts/pull_android_benches.sh`: Android has no headless adb entry point
  (app is MAIN/LAUNCHER only), so this COLLECTS the `files/out/<example>/
  bench.json` the on-device app already produced, via `adb run-as`.
- `scripts/run_benches.py`: cross-platform Python entry point that just
  dispatches to the right script (`--target auto|windows|linux|android`).
  On Windows it prefers Git Bash for the bash targets (system32 `bash` is the
  WSL launcher, which can't open Windows drive paths) and passes forward-slash
  paths.

### Windows 11 OS string (bench.cpp)

`RegGetValueA(... ProductName)` still returns "Windows 10 ..." on Windows 11
(Microsoft never updated the string). Fixed `OsString()` to promote to
"Windows 11" when `CurrentBuildNumber >= 22000` (the real boundary). Verified:
build 26200 now reports "Windows 11 Home (build 26200)".

### Packing-reduction records (device-dependent)

- Added `DressiAD::getFuncCount()` (unpacked backward-graph op count = the
  trivial 1-func-per-pass baseline) alongside the existing
  getSubStageCount()/getStageCount().
- `BenchRecord::addPacking(funcs, substages, stages)` writes the three keys;
  all six examples now record the STEADY-STATE build's counts (after reactive
  pruning, e.g. image_fitting settles 13->8 stages, so 8 is recorded).
- `bench_summary.py` renders a `packing f->s->st` column (ASCII arrows to avoid
  UnicodeEncodeError on redirect under a cp932 console).
- These packed counts are device-dependent: greedy fusion is bounded by the
  physical device's Vulkan limits, so the same graph packs differently per GPU.

## Measurements (Intel Iris Xe, i7-1370P, Win11)

| example | params | median ms/iter | packing f->s->st | quality |
| --- | --- | ---: | ---: | --- |
| image_fitting | 64px 150it | 0.320 | 36->8->8 | max err 0.0005 |
| texture_optimization | 256px 6v 1500it | 15.695 | 232->27->25 | 98.7% |
| silhouette aa | 128px 8v 150it | 3.339 | 802->61->53 | IoU 0.9544 |
| silhouette hardsoftras | 128px 8v 150it | 9.956 | 3962->202->68 | IoU 0.9491 |
| pbr_material | albedo 256px 6v 1200it | 17.923 | 3593->146->49 | PSNR 20.65 dB |
| pbr_envmap | 192px env64 4v 1500it | 105.303 | 2436->54->38 | PSNR 18.26 dB |
| pbr_shading | 512px 60 frames | 8.660 | 286->11->5 | 116 FPS |

## Open issues

- `gh` CLI is not installed on this machine and there is no open PR yet, so the
  benchmark table could not be posted as a PR comment programmatically.
