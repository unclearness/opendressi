<#
.SYNOPSIS
    Run every example's benchmark into one output root, one bench.json per run.

.DESCRIPTION
    A thin wrapper that just *executes* the example exes with headless flags
    and per-example --out-dir. It deliberately does NOT summarize: aggregate
    the results separately with

        python scripts/bench_summary.py <OutRoot>

    so running and reporting stay decoupled (re-summarize old runs, mix in
    Android bench.json files pulled with adb, etc. without re-running).

    Each bench.json is a MEDIAN steady-state execStep record (warmup excluded)
    written by the example itself via examples/common/bench.h — see the
    "Benchmarking" section of CLAUDE.md for the methodology these encode.

.PARAMETER OutRoot
    Directory that receives one subdir per benchmark (default: bench_results).

.PARAMETER BuildDir
    Where the example exes live (default: build/examples/Release).

.PARAMETER DataDir
    Dataset root passed to the mesh examples (default: data).

.PARAMETER Only
    One or more label substrings; only matching benchmarks run
    (e.g. -Only silhouette,texture). Default: run all.

.PARAMETER List
    Print the benchmark labels/commands and exit without running anything.

.EXAMPLE
    ./scripts/run_benches.ps1
    python scripts/bench_summary.py bench_results

.EXAMPLE
    ./scripts/run_benches.ps1 -Only silhouette -OutRoot bench_results/rerun
#>
[CmdletBinding()]
param(
    [string]  $OutRoot  = 'bench_results',
    [string]  $BuildDir = 'build/examples/Release',
    [string]  $DataDir  = 'data',
    [string[]]$Only,
    [switch]  $List
)

$ErrorActionPreference = 'Stop'
# Resolve paths relative to the repo root (this script lives in scripts/).
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# Benchmark table: Label -> exe basename + headless args. Each run adds its own
# --out-dir=<OutRoot>/<Label>. Edit here to add/change coverage.
$benches = @(
    @{ Label = 'image_fitting'
       Exe   = 'image_fitting'
       Args  = @() }

    @{ Label = 'texture_optimization'
       Exe   = 'texture_optimization'
       Args  = @("$DataDir/bunny", '--view-interval=0') }

    @{ Label = 'silhouette_hardsoftras'
       Exe   = 'silhouette_optimization'
       Args  = @("$DataDir/bunny", '--technique=hardsoftras', '--no-view=1') }

    @{ Label = 'silhouette_aa'
       Exe   = 'silhouette_optimization'
       Args  = @("$DataDir/bunny", '--technique=aa', '--no-view=1') }

    @{ Label = 'pbr_material_optimization'
       Exe   = 'pbr_material_optimization'
       Args  = @('--view-interval=0') }

    @{ Label = 'pbr_envmap_optimization'
       Exe   = 'pbr_envmap_optimization'
       Args  = @('--view-interval=0') }

    @{ Label = 'pbr_shading'
       Exe   = 'pbr_shading'
       Args  = @('--frames=60') }
)

if ($Only) {
    $benches = $benches | Where-Object {
        $label = $_.Label
        $Only | Where-Object { $label -like "*$_*" }
    }
    if (-not $benches) { throw "no benchmark labels match: $($Only -join ', ')" }
}

if ($List) {
    foreach ($b in $benches) {
        $exe = Join-Path $BuildDir "$($b.Exe).exe"
        "{0,-28} {1} {2} --out-dir={3}" -f $b.Label, $exe, ($b.Args -join ' '),
            (Join-Path $OutRoot $b.Label)
    }
    return
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

$results = @()
foreach ($b in $benches) {
    $exe = Join-Path $BuildDir "$($b.Exe).exe"
    if (-not (Test-Path $exe)) {
        Write-Warning "$($b.Label): missing $exe (build with 'cmake --build --preset release') - skipped"
        $results += [pscustomobject]@{ Label = $b.Label; Status = 'MISSING' }
        continue
    }
    $out = Join-Path $OutRoot $b.Label
    $runArgs = @($b.Args) + "--out-dir=$out"

    Write-Host "==> $($b.Label): $exe $($runArgs -join ' ')" -ForegroundColor Cyan
    & $exe @runArgs
    $code = $LASTEXITCODE

    $status = if ($code -eq 0) { 'OK' } else { "EXIT $code" }
    $results += [pscustomobject]@{ Label = $b.Label; Status = $status }
    if ($code -ne 0) { Write-Warning "$($b.Label) exited with code $code" }
}

Write-Host ''
Write-Host '=== bench run summary ===' -ForegroundColor Green
$results | Format-Table -AutoSize | Out-String | Write-Host
Write-Host "Aggregate with:  python scripts/bench_summary.py $OutRoot"

if ($results | Where-Object { $_.Status -notin @('OK', 'MISSING') }) { exit 1 }
