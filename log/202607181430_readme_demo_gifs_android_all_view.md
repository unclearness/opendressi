# README demo GIFs + Android "all" tiled view

## Goal

- Put animated demos of every example at the top of the README, hosted
  outside the repository (GitHub wiki `demo/` + a user-attachments video)
  so the repo carries no binaries.
- Make the Android app record-friendly for shape_texture_optimization:
  target / prediction / recovered atlas visible on one screen instead of
  toggling between streams.

## What was done

### examples: `--snapshot=N` progress dumps (all 7 examples)

Every example gained an opt-in `--snapshot=N` (default 0 = off) that saves
progress PNGs (`snap_%04d.png` etc.) into its out dir every N iterations
plus the final one. Source data for the GIFs is produced with
`--snapshot=1 --view-interval=0` (headless, every iteration). Details:

- texture_optimization / pbr_material also dump the recovered atlas
  (`snap_tex_*`; the 1024^2 bunny atlas is DownsampleHalf'd to 256^2 to
  keep 1500 files small), shape_texture dumps both phases
  (`snap1_*`, `snap2_*`, `snap2_tex_*`), pbr_envmap dumps render + tonemapped
  env (`snap_env_*`), pbr_shading dumps orbit frames.
- image_fitting got a live viewer ("rendered | target" tile, per-iter,
  `--view-interval`): it previously had NO viewer at all, so the desktop
  demo showed nothing and the Android app showed a black surface.
  Bench note: with the viewer on, the measured execStep median rises
  0.093 -> 0.211 ms on the RTX PRO 6000 box (display-path sync side
  effects; the timed section itself is unchanged). `--view-interval=0`
  restores 0.063 ms, matching the old viewer-less behavior — use it for
  clean benching (same role as silhouette's `--no-view`). Quality
  unchanged (max_err 0.0005).
- shape_texture defaults tightened for demos: shape 200->150 iters
  (IoU 0.9682 vs ~0.97), texture 300->100 iters (final loss 0.081 vs
  0.078) — negligible quality cost, ~2.5x shorter phone runs. Phase 1
  additionally holds the initial (undeformed sphere) frame for 3 s when a
  viewer is open (100 ms chunks polling host.cancelled(), so Stop works)
  so recordings show the starting state; headless runs are unaffected.
- Steady-state medians of the other examples re-checked while generating
  the GIF data: quality figures match the README table (texture 98.7%,
  silhouette IoU 0.951/0.953, albedo PSNR 20.65 dB, env PSNR 18.26 dB).

### scripts/make_readme_gifs.sh

Rebuilds all 8 gallery GIFs from the snapshot sequences with ffmpeg
(palettegen/paletteuse two-pass). Key decisions:

- **Log-spaced frame selection** for the optimization GIFs: progress is
  roughly logarithmic, so uniform stride finished visually in 1-2 frames.
  Geometric index spacing (dense early, sparse late; deduped) gives every
  frame similar visual progress. 0.8 s initial + 1.2 s final hold.
- Atlas-producing examples render as vertical stacks (render on top of
  the recovered atlas / env map); shape_texture concatenates phase 1
  (padded) and phase 2 on a 256x384 canvas.
- Output goes to `out/gifs/` (git-ignored). Upload to the wiki repo's
  `demo/` folder; the README references
  `https://raw.githubusercontent.com/wiki/unclearness/dressi/demo/*.gif`.

### README

- Gallery table (4x2, HTML) of the 8 wiki-hosted GIFs right below the
  opening paragraph, each cell linking to its example dir.
- Inline video player of the phone run (shape_texture on the Xiaomi 17T
  Pro, camera capture): GitHub only renders a player for
  `github.com/user-attachments` uploads (10 MB limit — the original
  84.5 MB capture was re-encoded to 9.85 MB, 1080p crf 25, no audio, via
  ffmpeg; kept locally as `out/gifs/shape_texture_android_web.mp4`).
  A full-quality link points at the wiki copy. Note: user-attachments
  URLs 404 anonymously until the containing content is published.

### Android app

- `SurfaceState` gained a synthetic **"all"** stream: once >= 2 real
  streams exist, the notified title list gets an extra "all" entry (its
  index == real-stream count maps to `kAllStream`); selecting it tiles
  every stream's cached latest frame vertically (widest-stream width,
  aspect-preserving nearest, 2 px separators, reused scratch canvas) and
  every blit refreshes the composite, so it is live during optimization
  and correct across phase transitions / surface re-attach. Kotlin needed
  no stream-logic changes (buttons are built from the notified titles).
- Default selection is now the "all" view (both the native default at
  second-stream registration and Kotlin's `titles.size - 1`).
- Stream buttons made compact (minWidth 0, isAllCaps false, 10 dp side
  padding) so shape_texture's four buttons fit without scrolling.
- Release APK built and installed on the 17T Pro; visual check done by
  the user (recording published as the README video).

## Open issues

- README cross-device table: image_fitting rows predate the live viewer;
  future re-measurements must use `--view-interval=0` (documented above)
  or the numbers will absorb display-sync overhead.
- pbr_shading.gif is 4.7 MB (largest asset); could be trimmed to 60
  frames if wiki size ever matters.
- The "all" composite recomposes on every blit; fine on the tested
  devices, but a dirty-flag could halve the work if a slow phone cares.
