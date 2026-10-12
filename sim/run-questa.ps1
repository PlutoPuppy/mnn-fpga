param([string]$QuestaRoot = 'C:/intelFPGA_lite/23.1std/questa_fse/win64')
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path "$PSScriptRoot/..").Path
$simulationRoot = Join-Path $repoRoot 'build/questa'
New-Item -ItemType Directory -Force -Path $simulationRoot | Out-Null
Push-Location $simulationRoot
try {
    & "$QuestaRoot/vlib.exe" work
    if ($LASTEXITCODE -ne 0) { throw 'vlib failed' }
    & "$QuestaRoot/vlog.exe" -sv "$repoRoot/rtl/systolic_pe.sv" "$repoRoot/rtl/systolic_array.sv" "$repoRoot/rtl/systolic_mmio.sv" "$PSScriptRoot/mmio_tb.sv"
    if ($LASTEXITCODE -ne 0) { throw 'vlog failed' }
    & "$QuestaRoot/vsim.exe" -c work.mmio_tb -do 'onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'
    if ($LASTEXITCODE -ne 0) { throw 'RTL simulation failed (see transcript)' }
} finally { Pop-Location }
