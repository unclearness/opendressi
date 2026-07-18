#!/usr/bin/env bash
# Build the README gallery GIFs from the --snapshot=1 PNG sequences.
#
# Source data (run from the repo root; every iteration saved):
#   build/examples/Release/image_fitting.exe --snapshot=1
#   build/examples/Release/silhouette_optimization.exe data/bunny \
#       --technique=hardsoftras --snapshot=1 --view-interval=0 --no-view=1
#   build/examples/Release/silhouette_optimization.exe data/bunny \
#       --technique=aa --snapshot=1 --view-interval=0 --no-view=1
#   build/examples/Release/texture_optimization.exe data/bunny \
#       --snapshot=1 --view-interval=0
#   build/examples/Release/shape_texture_optimization.exe data/bunny \
#       --snapshot=1 --view-interval=0 --no-view
#   build/examples/Release/pbr_material_optimization.exe \
#       --snapshot=1 --view-interval=0
#   build/examples/Release/pbr_envmap_optimization.exe \
#       --snapshot=1 --view-interval=0
#   build/examples/Release/pbr_shading.exe --frames=120 --snapshot=1 \
#       --view-interval=0
#
# GIFs land in out/gifs/ (git-ignored); upload them to the GitHub wiki
# (demo/ folder in the wiki repo) so the README can reference them without
# bloating this repository.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=out/gifs
mkdir -p "$OUT"

FPS=15
# 1.2 s hold on the converged last frame.
HOLD="tpad=stop_mode=clone:stop_duration=1.2"
# Log-spaced GIFs additionally hold the initial state for 0.8 s.
HOLD2="tpad=start_mode=clone:start_duration=0.8:stop_mode=clone:stop_duration=1.2"
PAL="split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"

gif() { # gif <src_pattern> <stride> <scale_filter> <dst>
    ffmpeg -y -loglevel error -i "$1" -filter_complex \
        "[0:v]select='not(mod(n,$2))',setpts=N/$FPS/TB,$3,$HOLD,$PAL" \
        -r $FPS "$OUT/$4"
    echo "$OUT/$4"
}

# Optimization progress is roughly logarithmic: with a uniform stride the
# visible change is over in the first couple of GIF frames. Log (geometric)
# index spacing gives every frame a similar amount of visual progress —
# dense at the start (every iteration), sparse once converged.
logsel() { # logsel <n_iters> <n_frames> -> deduped ascending indices
    awk -v n="$1" -v m="$2" 'BEGIN {
        prev = -1;
        for (k = 0; k < m; k++) {
            idx = int(exp(log(n) * k / (m - 1)) + 0.5) - 1;
            if (idx > n - 1) idx = n - 1;
            if (idx != prev) { print idx; prev = idx; }
        }
    }'
}

stage_seq() { # stage_seq <dir> <printf_pattern> <indices...>
    local dir=$1 pat=$2 i=0 idx
    shift 2
    for idx in "$@"; do
        # shellcheck disable=SC2059
        cp "$(printf "$pat" "$idx")" "$dir/$(printf '%04d' $i).png"
        i=$((i + 1))
    done
}

loggif() { # loggif <src_pattern> <n_iters> <n_frames> <scale_filter> <dst>
    local stage
    stage=$(mktemp -d)
    # shellcheck disable=SC2046
    stage_seq "$stage" "$1" $(logsel "$2" "$3")
    ffmpeg -y -loglevel error -framerate $FPS -i "$stage/%04d.png" \
        -filter_complex "[0:v]$4,$HOLD2,$PAL" -r $FPS "$OUT/$5"
    echo "$OUT/$5  ($(ls "$stage" | wc -l) log-spaced frames)"
    rm -rf "$stage"
}

NEAREST256="scale=256:256:flags=neighbor"

# Same, but vstacking two per-iteration sequences (render on top of the
# recovered texture / env map).
loggif2() { # loggif2 <patA> <patB> <n_iters> <n_frames> <scaleA> <scaleB> <dst>
    local stage_a stage_b indices
    stage_a=$(mktemp -d)
    stage_b=$(mktemp -d)
    indices=$(logsel "$3" "$4")
    # shellcheck disable=SC2046,SC2086
    stage_seq "$stage_a" "$1" $indices
    # shellcheck disable=SC2046,SC2086
    stage_seq "$stage_b" "$2" $indices
    ffmpeg -y -loglevel error -framerate $FPS -i "$stage_a/%04d.png" \
        -framerate $FPS -i "$stage_b/%04d.png" -filter_complex \
        "[0:v]$5[t];[1:v]$6[b];[t][b]vstack,$HOLD2,$PAL" \
        -r $FPS "$OUT/$7"
    echo "$OUT/$7  ($(ls "$stage_a" | wc -l) log-spaced frames)"
    rm -rf "$stage_a" "$stage_b"
}

loggif imgfit_out/snap_%04d.png  150 60 "$NEAREST256" image_fitting.gif
loggif2 texopt_out/snap_%04d.png texopt_out/snap_tex_%04d.png 1500 72 \
    "$NEAREST256" "$NEAREST256" texture_optimization.gif
loggif2 pbrmat_out/snap_%04d.png pbrmat_out/snap_tex_%04d.png 1200 72 \
    "$NEAREST256" "$NEAREST256" pbr_material_optimization.gif
gif silopt_out_hardsoftras/snap_%04d.png 2 "$NEAREST256" silhouette_hardsoftras.gif
gif silopt_out_aa/snap_%04d.png          2 "$NEAREST256" silhouette_aa.gif

# shape_texture: phase 1 (silhouette; padded to the phase-2 canvas) then
# phase 2 (render on top of the 2:1 recovered atlas)
ffmpeg -y -loglevel error -i shapetex_out/snap1_%04d.png \
    -i shapetex_out/snap2_%04d.png -i shapetex_out/snap2_tex_%04d.png \
    -filter_complex \
    "[0:v]select='not(mod(n,3))',setpts=N/$FPS/TB,$NEAREST256,pad=256:384:0:0:black[p1];\
[1:v]select='not(mod(n,2))',setpts=N/$FPS/TB,$NEAREST256[p2r];\
[2:v]select='not(mod(n,2))',setpts=N/$FPS/TB,scale=256:128:flags=lanczos[p2t];\
[p2r][p2t]vstack[p2];\
[p1][p2]concat=n=2:v=1,$HOLD,$PAL" \
    -r $FPS "$OUT/shape_texture_optimization.gif"
echo "$OUT/shape_texture_optimization.gif"

# pbr_envmap: render view (top) + recovered env map (bottom, 2:1)
loggif2 pbrenv_out/snap_%04d.png pbrenv_out/snap_env_%04d.png 1500 72 \
    "$NEAREST256" "scale=256:128:flags=neighbor" pbr_envmap_optimization.gif

# pbr_shading: one full 120-frame orbit, no hold (seamless loop)
ffmpeg -y -loglevel error -i pbr_out/snap_%04d.png -filter_complex \
    "[0:v]setpts=N/20/TB,scale=256:256:flags=lanczos,$PAL" \
    -r 20 "$OUT/pbr_shading.gif"
echo "$OUT/pbr_shading.gif"

ls -l "$OUT"
