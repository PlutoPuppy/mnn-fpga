param(
    [string]$QuartusRoot = 'C:/intelFPGA_lite/23.1std/quartus',
    [string]$BaseProject = "$PSScriptRoot/../../../tinyvit.qpf",
    [switch]$GenerateOnly
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path "$PSScriptRoot/..").Path
$baseQpf = (Resolve-Path -LiteralPath $BaseProject).Path
$baseQsf = [IO.Path]::ChangeExtension($baseQpf, '.qsf')
$assignments = Get-Content -LiteralPath $baseQsf
if (-not ($assignments -match '^set_global_assignment -name DEVICE 5CSEMA5F31C6$')) {
    throw 'This bring-up target requires the DE1-SoC 5CSEMA5F31C6 base project.'
}
$buildRoot = Join-Path $repoRoot 'build/quartus-jtag'
New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
$ipPath = ($PSScriptRoot -replace '\\', '/') + '/*,$'
function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    # Windows PowerShell treats redirected native stderr (including Qsys INFO)
    # as error records. The executable exit code determines success.
    $savedPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Executable @Arguments
    } finally { $ErrorActionPreference = $savedPreference }
    if ($LASTEXITCODE -ne 0) { throw "$Executable failed with exit code $LASTEXITCODE" }
}
Push-Location $buildRoot
try {
    Invoke-Checked "$QuartusRoot/sopc_builder/bin/qsys-script.exe" @(
        "--script=$PSScriptRoot/create_jtag_system.tcl", "--search-path=$ipPath")
    Invoke-Checked "$QuartusRoot/sopc_builder/bin/qsys-generate.exe" @(
        'gemm_jtag.qsys', '--synthesis=VERILOG', "--search-path=$ipPath")
    # Reuse the supplied project device and the exact pins of our exposed ports.
    # Other legacy peripherals and demonstration modules are excluded.
    $pins = @($assignments | Where-Object {
        $_ -match '^set_location_assignment .* -to (CLOCK_50|KEY\[[0-3]\]|LEDR\[[0-9]\])$'
    })
    if ($pins.Count -ne 15) { throw 'Expected clock, four key and ten LED pin assignments.' }
    $qsf = @(
        'set_global_assignment -name FAMILY "Cyclone V"',
        'set_global_assignment -name DEVICE 5CSEMA5F31C6',
        'set_global_assignment -name TOP_LEVEL_ENTITY tinyvit_jtag',
        'set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files',
        'set_global_assignment -name SYSTEMVERILOG_FILE ../../platform/tinyvit_jtag.sv',
        'set_global_assignment -name SDC_FILE ../../platform/tinyvit_jtag.sdc',
        'set_global_assignment -name QIP_FILE gemm_jtag/synthesis/gemm_jtag.qip',
        'set_global_assignment -name NUM_PARALLEL_PROCESSORS 1'
    ) + $pins + @(
        'set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to CLOCK_50',
        'set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to KEY[*]',
        'set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LEDR[*]'
    )
    Set-Content -LiteralPath tinyvit_jtag.qsf -Value $qsf -Encoding ascii
    Set-Content -LiteralPath tinyvit_jtag.qpf -Value @(
        'QUARTUS_VERSION = "23.1"', 'PROJECT_REVISION = "tinyvit_jtag"') -Encoding ascii
    Set-Content -LiteralPath base-project.txt -Value $baseQpf -Encoding utf8
    if (-not $GenerateOnly) {
        Invoke-Checked "$QuartusRoot/bin64/quartus_sh.exe" @('--flow', 'compile', 'tinyvit_jtag')
    }
} finally { Pop-Location }
