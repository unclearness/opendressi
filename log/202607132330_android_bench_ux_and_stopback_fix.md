# Android: release bench build, cross-device records, Stop/Back + residual-frame fixes

Branch: `cross-device-bench` (PR unclearness/dressi#1)

## Goal

Make the Android app measurement-ready by default, add two more
cross-device benchmark records, and fix two on-device UI bugs surfaced
while benchmarking (Stop/Back unresponsive; previous frame lingering
after Back → new example).

## What was done

Commits (this session):

- `3ea606d` **Android release benchmarking build + reliable per-example
  bench.json**
  - `build.gradle.kts`: the `release` variant is now a benchmarking build.
    The debug native `.so` is unoptimized (measured: only `-g`, no
    `-O`/`-DNDEBUG` in `.cxx/Debug/.../compile_commands.json`), so its
    execStep timings are meaningless. Release pins the native lib to CMake
    Release via `arguments += "-DCMAKE_BUILD_TYPE=Release"` (verified -O3
    ×410 files + `-DNDEBUG`), signs with the debug key so `adb install`
    works, and stays `isDebuggable = true` so `run-as` (bench pull) can read
    `files/out/*/bench.json`. (AGP would otherwise pick CMake "Debug" for a
    debuggable variant — hence the explicit pin.)
  - `MainActivity.kt`: silhouette runs as two technique buttons
    (hardsoftras/aa), each into its own out-dir
    (`out/silhouette_optimization_<technique>`) so both records coexist;
    `defaultArgs(name, technique)` + `startExample(name, technique?)`.
  - `MainActivity.kt`: `pbr_shading` gets `--frames=200` on Android. It is
    an interactive viewer that never self-terminates (time-based orbit), so
    it only wrote bench.json on a >20-frame Stop and its out-dir was empty
    on device. A fixed frame budget makes it deterministic, self-terminating
    and always emit bench.json (per-frame metric, budget-independent).
  - `scripts/pull_android_benches.ps1`: Windows/PowerShell port of the pull
    script; `.sh` and `.ps1` now target the per-technique silhouette dirs.
  - `README-android.md`: document the release benchmarking build and that
    the pull scripts run on the host PC (Git Bash / PowerShell), not phone.

- `55b025a` **Fix Stop/Back untouchable under edge-to-edge (Android 15+)**
  Root cause of "STOP/BACK don't respond, both devices, all examples". On
  Android 15+ (app targets SDK 36) edge-to-edge is enforced; the bottom
  Stop/Back row rendered UNDER the navigation bar. Measured on 24018RPACG
  (1280x2772): buttons at `[.,2616]..[.,2772]`, nav-bar inset
  `[0,2619][1280,2772]` — the nav bar's gesture host stole every tap. The
  buttons were `enabled=true` (state machine was fine — verified onFinished
  fires and Back enables on natural finish) but physically unhittable. Fix:
  pad the root by system-bar insets (`getInsets(systemBars())` API30+,
  deprecated `systemWindowInset*` fallback). Verified: button row moves up
  to end exactly at the nav-bar edge (`..2619`).

- `62a1b71` **Clear the SurfaceView between runs (no residual frame)**
  Start clears log/stream bar but not the SurfaceView, so after Back → pick
  another example the previous last frame lingered until the new run's first
  frame (seconds during first build). Clear every buffer to black in
  `SurfaceState::setSurface` when a surface is attached (posted 3× for
  double/triple buffering).

## Cross-device records + PR comment

Added two devices to `bench_results/` and expanded the PR's main benchmark
comment (issue comment 4954097153) from 5 → 7 devices:

- **RTX 3090** (`oldmachine`, i9-10850K / Win11 Pro)
- **Mali-G1-Ultra MC12** (`17tpro`, MediaTek MT6993 = **Dimensity 9500** /
  Xiaomi 2602EPTC0R / Android 16) — 2nd mobile, previously-unverified D9500.

Highlights (median ms/iter): the engine now spans ~4 orders on one graph.
On mobile the D9500 Mali beats the SD-8-Gen-2 Adreno on heavy optimizers
(texture 23.6 vs 48.4, pbr_envmap 181 vs 304) but is slower on light/forward
paths (pbr_shading 19 vs 56 FPS) and pays a much larger first-build warmup
(silhouette-aa 8.1 s vs 1.6 s) — its shader compiler is the bottleneck, not
steady-state throughput. Quality stays device-independent (texture 98.7%,
envmap 18.23-18.26 dB, silhouette IoU 0.941-0.955).

Also re-measured the Blackwell workstation with the browser closed: no
speedup — all deltas within ±few % measurement noise (image_fitting/
pbr_shading swing most, being the most latency-bound). Confirms these
sub-ms/low-ms walls are host submit/fence-latency bound, not CPU-background
bound; PR table left on the original Blackwell numbers.

## Key decisions

- Release build made *debuggable* on purpose: benchmarking needs both
  `adb install` (signing) and `run-as` (debuggable). Native optimization is
  preserved by pinning CMAKE_BUILD_TYPE regardless of the debuggable flag.
- `pbr_shading` deterministic frame budget over "fix the Stop path": a
  wall-clock orbit is non-reproducible and unfit for a cross-device bench;
  self-termination also sidesteps the manual-Stop dependency.
- Edge-to-edge fixed with inset padding (no androidx dependency: the app has
  none) rather than opting out, which the framework may ignore on SDK 35+.

## Measurements / verification

- Native opt flags confirmed via `compile_commands.json` (debug: `-g` only;
  release: `-O3` + `-DNDEBUG`).
- Button geometry confirmed via `uiautomator dump` + `dumpsys window
  displays` insets before/after the inset fix.
- Natural-finish state machine confirmed by polling `uiautomator` through a
  full texture_optimization run (Stop enabled during run → Back enabled at
  iter 1499).

## Open issues

- Could not auto-test *taps*: `adb shell input tap` is blocked by MIUI
  (INJECT_EVENTS SecurityException), so the final tap-responds check is left
  to the user on-device. Structural verification (button bounds now clear of
  the nav bar; surface recompiled with the black-clear) is done.
- `bench_results/` raw JSON is not committed here (only the PR comment table
  was updated); confirm with the user whether the new records should also be
  checked in.
