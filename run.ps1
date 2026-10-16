$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'launcher\start.ps1') @args
exit $LASTEXITCODE
