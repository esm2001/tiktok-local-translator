$ErrorActionPreference = 'Stop'
$projectDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$venvPython = Join-Path $projectDir '.venv\Scripts\python.exe'
Set-Location -LiteralPath $projectDir
if (Test-Path -LiteralPath $venvPython) {
    & $venvPython (Join-Path $PSScriptRoot 'main.py') @args
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        Write-Host 'No se ha encontrado Python. Instala Python 3.11 o posterior y vuelve a abrir el launcher.'
        exit 1
    }
    & $pythonCommand.Source (Join-Path $PSScriptRoot 'main.py') @args
}
exit $LASTEXITCODE
