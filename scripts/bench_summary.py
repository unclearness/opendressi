#!/usr/bin/env python3
"""Summarize cross-device benchmark records into Markdown tables.

Every example writes a one-object ``bench.json`` into its ``--out-dir``
at the end of a run: GPU device name, host info (os / cpu / ram), the
parameters, the MEDIAN steady-state ms per iteration (warmup excluded),
the one-time build/warmup overhead in ms (pack + GLSL compile + Vulkan
build + reactive rebuilds; ``first_build_ms`` is iter 0 alone), and the
example's quality metric. Collect those files from every
machine/phone and render one table per example, one row per run, sorted
by median time.

Usage:
    python scripts/bench_summary.py [path ...]

Each path may be a bench*.json file or a directory (searched
recursively; build trees and .git are skipped). With no arguments the
current directory is searched — e.g. after:

    # desktop runs write ./texopt_out/bench.json etc. in place
    ./build/examples/Release/texture_optimization.exe data/bunny
    ./build/examples/Release/silhouette_optimization.exe data/bunny --no-view=1

    # pull one file per example per Android device (any filename works)
    adb shell run-as org.dressi.examples cat files/out/texture_optimization/bench.json > bench_results/pixel_texopt.json

    python scripts/bench_summary.py texopt_out silopt_out_hardsoftras bench_results
"""

import json
import sys
from pathlib import Path

SKIP_DIRS = {".git", ".cxx", ".gradle", "_deps", "node_modules"}


def find_records(paths):
    files = []
    for p in map(Path, paths):
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            for f in p.rglob("bench*.json"):
                parts = set(f.parts)
                if parts & SKIP_DIRS or any(
                        part.startswith("build") for part in f.parts):
                    continue
                files.append(f)
    records = []
    for f in sorted(set(files)):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
            rec["_file"] = str(f)
            records.append(rec)
        except (json.JSONDecodeError, OSError) as e:
            print(f"warning: skipping {f}: {e}", file=sys.stderr)
    return records


def params_of(rec):
    ex = rec.get("example")
    if ex == "silhouette_optimization":
        p = (f"{rec.get('technique', '?')} {rec.get('screen', '?')}px "
             f"{rec.get('views', '?')}v {rec.get('iters', '?')}it "
             f"s{rec.get('samples', '?')} K{rec.get('peels', '?')}")
        if rec.get("single_view"):
            p += " single-view"
        return p
    if ex == "texture_optimization":
        return (f"{rec.get('screen', '?')}px {rec.get('views', '?')}v "
                f"{rec.get('iters', '?')}it")
    if ex == "image_fitting":
        return f"{rec.get('screen', '?')}px {rec.get('iters', '?')}it"
    if ex == "pbr_material_optimization":
        return (f"{rec.get('optimize', '?')} {rec.get('screen', '?')}px "
                f"tex{rec.get('tex', '?')} {rec.get('views', '?')}v "
                f"{rec.get('iters', '?')}it")
    if ex == "pbr_envmap_optimization":
        return (f"{rec.get('screen', '?')}px env{rec.get('envres', '?')} "
                f"{rec.get('views', '?')}v {rec.get('iters', '?')}it")
    if ex == "pbr_shading":
        return f"{rec.get('screen', '?')}px {rec.get('frames', '?')} frames"
    return ""


def warmup_of(rec):
    # One-time build/warmup overhead (ms), with the single largest build
    # (iter 0) in parentheses. "-" for older records without the field.
    if "warmup_ms" not in rec:
        return "-"
    s = f"{rec['warmup_ms']:.0f}"
    if "first_build_ms" in rec:
        s += f" ({rec['first_build_ms']:.0f})"
    return s


def packing_of(rec):
    # Packing reduction "funcs->substages->stages": the unpacked backward-graph
    # op count collapsed by substage packing then by stage packing into Vulkan
    # render passes. Device-dependent (greedy fusion is bounded by the GPU's
    # Vulkan limits). "-" for older records without the fields.
    if "stages" not in rec:
        return "-"
    funcs = rec.get("funcs")
    substages = rec.get("substages")
    stages = rec["stages"]
    if funcs is not None and substages is not None:
        return f"{funcs}->{substages}->{stages}"
    return str(stages)


def quality_of(rec):
    if "mean_iou" in rec:
        return f"IoU {rec['mean_iou']:.4f}"
    if "accurate_pct" in rec:
        return f"{rec['accurate_pct']:.1f}% accurate"
    if "psnr_db" in rec:
        return f"PSNR {rec['psnr_db']:.2f} dB"
    if "fps" in rec:
        return f"{rec['fps']:.0f} FPS"
    if "max_err" in rec:
        return f"max err {rec['max_err']:.4f}"
    return ""


def host_of(rec):
    parts = []
    if rec.get("cpu"):
        cpu = rec["cpu"]
        if rec.get("cpu_cores"):
            cpu += f" ({rec['cpu_cores']}c)"
        parts.append(cpu)
    if rec.get("ram_gb"):
        parts.append(f"{rec['ram_gb']:.0f} GB")
    if rec.get("os"):
        parts.append(rec["os"])
    return ", ".join(parts) or rec.get("platform", "")


def main():
    records = find_records(sys.argv[1:] or ["."])
    if not records:
        print("no bench*.json records found", file=sys.stderr)
        return 1
    for example in sorted({r.get("example", "?") for r in records}):
        rows = [r for r in records if r.get("example") == example]
        rows.sort(key=lambda r: r.get("median_ms_per_iter", float("inf")))
        print(f"## {example}\n")
        print("| device | host | parameters | median ms/iter "
              "| warmup ms (first) | packing f->s->st | quality |")
        print("| --- | --- | --- | ---: | ---: | ---: | --- |")
        for r in rows:
            print(f"| {r.get('device', '?')} | {host_of(r)} "
                  f"| {params_of(r)} "
                  f"| {r.get('median_ms_per_iter', float('nan')):.3f} "
                  f"| {warmup_of(r)} "
                  f"| {packing_of(r)} "
                  f"| {quality_of(r)} |")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
