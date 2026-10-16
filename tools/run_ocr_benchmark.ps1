param(
    [switch]$Adaptive,
    [switch]$Variants,
    [switch]$Strict,
    [string]$Output = '/data/tiktok/.cache/regression/ocr-after.json',
    [string]$Container = 'tiktok-translator'
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker CLI no está instalado.' }
foreach ($File in @('tools/benchmark_ocr.py','tools/regression_metrics.py','tests/regression/expected.json')) {
    $SourceFile = Join-Path $ProjectRoot $File
    $TargetName = Split-Path -Leaf $File
    & docker cp $SourceFile "${Container}:/tmp/$TargetName"
    if ($LASTEXITCODE -ne 0) { throw "No se pudo preparar $TargetName en el contenedor." }
}
$BenchmarkArgs = @('exec','-e','PYTHONPATH=/app:/tmp',$Container,'python','/tmp/benchmark_ocr.py',
    '--dataset','/tmp/expected.json','--media-root','/data/tiktok','--output',$Output)
if ($Adaptive) { $BenchmarkArgs += '--adaptive' }
if ($Variants) { $BenchmarkArgs += '--variants' }
if ($Strict) { $BenchmarkArgs += '--strict' }
& docker @BenchmarkArgs
exit $LASTEXITCODE
