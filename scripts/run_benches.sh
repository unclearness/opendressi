#!/usr/bin/env bash
# Run every example's benchmark into one output root, one bench.json per run.
#
# A thin wrapper that just *executes* the example exes with headless flags and
# per-example --out-dir. It deliberately does NOT summarize: aggregate the
# results separately with
#
#     python3 scripts/bench_summary.py <out-root>
#
# so running and reporting stay decoupled (re-summarize old runs, mix in
# Android bench.json files pulled with adb, etc. without re-running). Each
# bench.json is a MEDIAN steady-state execStep record (warmup excluded) written
# by the example itself via examples/common/bench.h — see the "Benchmarking"
# section of CLAUDE.md for the methodology these encode.
#
# Usage:
#   scripts/run_benches.sh [-o out_root] [-b build_dir] [-d data_dir]
#                          [-n label,label] [-l]
#   -o  output root (default: bench_results)
#   -b  dir holding the example exes (default: build/examples/Release)
#   -d  dataset root for the mesh examples (default: data)
#   -n  comma-separated label substrings; only matching benchmarks run
#   -l  list the benchmark commands and exit
#
# Example:
#   scripts/run_benches.sh
#   python3 scripts/bench_summary.py bench_results
set -u

out_root='bench_results'
build_dir='build/examples/Release'
data_dir='data'
only=''
list_only=0

while getopts 'o:b:d:n:lh' opt; do
    case "$opt" in
        o) out_root=$OPTARG ;;
        b) build_dir=$OPTARG ;;
        d) data_dir=$OPTARG ;;
        n) only=$OPTARG ;;
        l) list_only=1 ;;
        h|*) sed -n '2,30p' "$0"; exit 0 ;;
    esac
done

# Resolve the repo root (this script lives in scripts/) and cd there.
repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo" || exit 1

# Benchmark table: "label|exe|space-separated headless args". Each run appends
# its own --out-dir=<out_root>/<label>. Edit here to add/change coverage.
benches=(
    "image_fitting|image_fitting|"
    "texture_optimization|texture_optimization|$data_dir/bunny --view-interval=0"
    "silhouette_hardsoftras|silhouette_optimization|$data_dir/bunny --technique=hardsoftras --no-view=1"
    "silhouette_aa|silhouette_optimization|$data_dir/bunny --technique=aa --no-view=1"
    "pbr_material_optimization|pbr_material_optimization|--view-interval=0"
    "pbr_envmap_optimization|pbr_envmap_optimization|--view-interval=0"
    "pbr_shading|pbr_shading|--frames=60"
)

# Optional label filter (-n foo,bar keeps labels containing foo or bar).
match_only() {
    [ -z "$only" ] && return 0
    local label=$1 pat
    IFS=',' read -ra pats <<< "$only"
    for pat in "${pats[@]}"; do
        [ -n "$pat" ] && case "$label" in *"$pat"*) return 0 ;; esac
    done
    return 1
}

if [ "$list_only" -eq 1 ]; then
    for row in "${benches[@]}"; do
        IFS='|' read -r label exe args <<< "$row"
        match_only "$label" || continue
        printf '%-28s %s/%s %s --out-dir=%s/%s\n' \
            "$label" "$build_dir" "$exe" "$args" "$out_root" "$label"
    done
    exit 0
fi

mkdir -p "$out_root"

fail=0
declare -a summary
for row in "${benches[@]}"; do
    IFS='|' read -r label exe args <<< "$row"
    match_only "$label" || continue

    bin="$build_dir/$exe"
    [ -x "$bin" ] || bin="$build_dir/$exe.exe"   # tolerate a Windows layout
    if [ ! -x "$bin" ]; then
        echo "WARNING: $label: missing $build_dir/$exe (build first) - skipped" >&2
        summary+=("$label	MISSING")
        continue
    fi
    out="$out_root/$label"

    # shellcheck disable=SC2086  # args is an intentional word-split arg string
    echo "==> $label: $bin $args --out-dir=$out"
    # shellcheck disable=SC2086
    "$bin" $args "--out-dir=$out"
    code=$?

    if [ "$code" -eq 0 ]; then
        summary+=("$label	OK")
    else
        summary+=("$label	EXIT $code")
        echo "WARNING: $label exited with code $code" >&2
        fail=1
    fi
done

echo
echo '=== bench run summary ==='
printf '%s\n' "${summary[@]}" | column -t -s "$(printf '\t')" 2>/dev/null \
    || printf '%s\n' "${summary[@]}"
echo "Aggregate with:  python3 scripts/bench_summary.py $out_root"

exit "$fail"
