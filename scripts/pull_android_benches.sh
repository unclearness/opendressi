#!/usr/bin/env bash
# Collect the Android app's per-example bench.json into a local output root.
#
# On Android the examples run inside the app (org.dressi.examples) driven by
# its UI — there is NO headless adb entry point, so this script does NOT start
# runs. Run each example on-device from the app first (they write
# files/out/<example>/bench.json), then pull the records here for aggregation:
#
#     scripts/pull_android_benches.sh -o bench_results/android_pixel8
#     python3 scripts/bench_summary.py bench_results/android_pixel8
#
# Files are read out of the app's private storage with `run-as`, so a debuggable
# build is required (the assembleDebug APK is). adb is identical on every OS, so
# this runs under Git Bash on Windows as well as Linux/macOS.
#
# Usage:
#   scripts/pull_android_benches.sh [-o out_root] [-s serial] [-p package]
#                                   [-n label,label] [-l]
#   -o  local output root (default: bench_results/android)
#   -s  adb device serial (default: the only connected device)
#   -p  app package (default: org.dressi.examples)
#   -n  comma-separated example-name substrings; only matching ones are pulled
#   -l  list the on-device example names and exit
set -u

out_root='bench_results/android'
serial=''
pkg='org.dressi.examples'
only=''
list_only=0

while getopts 'o:s:p:n:lh' opt; do
    case "$opt" in
        o) out_root=$OPTARG ;;
        s) serial=$OPTARG ;;
        p) pkg=$OPTARG ;;
        n) only=$OPTARG ;;
        l) list_only=1 ;;
        h|*) sed -n '2,28p' "$0"; exit 0 ;;
    esac
done

# Example registry names == the on-device out/<name> dirs (see
# examples/common/example_registry.cpp and MainActivity.defaultArgs).
examples=(
    image_fitting
    texture_optimization
    silhouette_optimization
    pbr_shading
    pbr_material_optimization
    pbr_envmap_optimization
)

match_only() {
    [ -z "$only" ] && return 0
    local name=$1 pat
    IFS=',' read -ra pats <<< "$only"
    for pat in "${pats[@]}"; do
        [ -n "$pat" ] && case "$name" in *"$pat"*) return 0 ;; esac
    done
    return 1
}

if [ "$list_only" -eq 1 ]; then
    printf '%s\n' "${examples[@]}"
    exit 0
fi

command -v adb >/dev/null 2>&1 || { echo "error: adb not found on PATH" >&2; exit 1; }
adb=(adb)
[ -n "$serial" ] && adb=(adb -s "$serial")

# Fail fast if no device is reachable.
if ! "${adb[@]}" get-state >/dev/null 2>&1; then
    echo "error: no adb device (check 'adb devices' / USB debugging / -s serial)" >&2
    exit 1
fi

mkdir -p "$out_root"

fail=0
declare -a summary
for name in "${examples[@]}"; do
    match_only "$name" || continue

    remote="files/out/$name/bench.json"
    local_dir="$out_root/$name"
    mkdir -p "$local_dir"

    # run-as streams the file out of the app's private storage; capture stdout,
    # keep stderr for the failure message. Empty output == not produced yet.
    if json="$("${adb[@]}" shell run-as "$pkg" cat "$remote" 2>/dev/null)" \
            && [ -n "$json" ]; then
        printf '%s' "$json" > "$local_dir/bench.json"
        summary+=("$name	pulled")
    else
        echo "WARNING: $name: no $remote (run it in the app first?)" >&2
        summary+=("$name	MISSING")
        fail=1
    fi
done

echo
echo '=== android pull summary ==='
printf '%s\n' "${summary[@]}" | column -t -s "$(printf '\t')" 2>/dev/null \
    || printf '%s\n' "${summary[@]}"
echo "Aggregate with:  python3 scripts/bench_summary.py $out_root"

exit "$fail"
