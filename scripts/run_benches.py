#!/usr/bin/env python3
"""Cross-platform entry point for the benchmark run wrappers.

Thin dispatcher: it just invokes the right platform script with translated
flags — nothing more. Execution and aggregation stay separate, so this only
*runs* the benchmarks; summarize afterwards with:

    python scripts/bench_summary.py <out-root>

Targets:
  windows   -> powershell scripts/run_benches.ps1   (desktop exes)
  linux     -> bash       scripts/run_benches.sh     (desktop exes)
  android   -> bash       scripts/pull_android_benches.sh
               (collects bench.json the on-device app already produced; there
               is no headless adb run — start each example from the app first)

`--target auto` (default) picks windows or linux from the host OS; android must
be requested explicitly because it talks to a connected device, not this host.

Examples:
  python scripts/run_benches.py                       # run all, this OS
  python scripts/run_benches.py --only silhouette     # subset
  python scripts/run_benches.py --list                # show commands, run none
  python scripts/run_benches.py --target android -o bench_results/pixel8
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def _find_bash():
    """Locate a POSIX bash. On Windows prefer Git Bash — the system32 `bash`
    is the WSL launcher, which cannot open Windows drive paths."""
    if sys.platform == "win32":
        candidates = [
            Path(os.environ.get("ProgramFiles", "")) / "Git/bin/bash.exe",
            Path(os.environ.get("ProgramFiles(x86)", "")) / "Git/bin/bash.exe",
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Programs/Git/bin/bash.exe",
        ]
        for c in candidates:
            if c.is_file():
                return str(c)
    return shutil.which("bash")


def build_command(args):
    """Return the argv list for the selected platform script."""
    if args.target == "android":
        script = SCRIPTS / "pull_android_benches.sh"
        cmd = _bash(script)
        cmd += ["-o", args.out_root]
        if args.only:
            cmd += ["-n", args.only]
        if args.serial:
            cmd += ["-s", args.serial]
        if args.package:
            cmd += ["-p", args.package]
        if args.list:
            cmd += ["-l"]
        return cmd

    if args.target == "windows":
        script = SCRIPTS / "run_benches.ps1"
        pwsh = shutil.which("pwsh") or shutil.which("powershell")
        if not pwsh:
            sys.exit("error: neither pwsh nor powershell found on PATH")
        cmd = [pwsh, "-NoProfile", "-ExecutionPolicy", "Bypass",
               "-File", str(script), "-OutRoot", args.out_root,
               "-BuildDir", args.build_dir, "-DataDir", args.data_dir]
        if args.only:
            cmd += ["-Only", args.only]  # PowerShell splits "a,b" into an array
        if args.list:
            cmd += ["-List"]
        return cmd

    # linux (and macOS)
    script = SCRIPTS / "run_benches.sh"
    cmd = _bash(script)
    cmd += ["-o", args.out_root, "-b", args.build_dir, "-d", args.data_dir]
    if args.only:
        cmd += ["-n", args.only]
    if args.list:
        cmd += ["-l"]
    return cmd


def _bash(script):
    bash = _find_bash()
    if not bash:
        sys.exit("error: bash not found (needed for this target; on Windows "
                 "install Git Bash)")
    # Forward-slash form so Git Bash opens the Windows-drive path correctly.
    return [bash, script.as_posix()]


def main():
    p = argparse.ArgumentParser(
        description="Run the dressi example benchmarks via the platform "
                    "wrapper (aggregate separately with bench_summary.py).")
    p.add_argument("--target", choices=["auto", "windows", "linux", "android"],
                   default="auto", help="platform wrapper to invoke "
                   "(default: auto = this host's desktop wrapper)")
    p.add_argument("-o", "--out-root", default="bench_results",
                   help="output root receiving one bench.json subdir per run")
    p.add_argument("-b", "--build-dir", default="build/examples/Release",
                   help="dir holding the example exes (desktop targets)")
    p.add_argument("-d", "--data-dir", default="data",
                   help="dataset root for the mesh examples (desktop targets)")
    p.add_argument("-n", "--only", default="",
                   help="comma-separated label substrings; only matches run")
    p.add_argument("--serial", default="", help="adb device serial (android)")
    p.add_argument("--package", default="", help="app package (android)")
    p.add_argument("-l", "--list", action="store_true",
                   help="list the commands/examples and exit")
    args = p.parse_args()

    if args.target == "auto":
        args.target = "windows" if sys.platform == "win32" else "linux"

    cmd = build_command(args)
    print("+ " + " ".join(cmd), file=sys.stderr)
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
