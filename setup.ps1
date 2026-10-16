[CmdletBinding()]
param(
    [switch]$PullModels,
    [switch]$InstallBrowser,
    [switch]$BuildBackend,
    [switch]$Development,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot
$venvPython = Join-Path $projectDir '.venv\Scripts\python.exe'
$setupExit = 1

function Invoke-PythonChecked {
    param([string]$Executable, [string[]]$PythonArguments)
    & $Executable @PythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python termino con codigo $LASTEXITCODE."
    }
}

Push-Location -LiteralPath $projectDir
try {
    if ($CheckOnly -and ($PullModels -or $InstallBrowser -or $BuildBackend -or $Development)) {
        throw '-CheckOnly no admite opciones de instalacion o descarga.'
    }
    if ($env:OS -ne 'Windows_NT') {
        throw 'Este setup utiliza Docker Desktop y el launcher de Windows.'
    }
    $pythonExecutable = $venvPython
    if (-not (Test-Path -LiteralPath $pythonExecutable)) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pythonCommand) {
            throw 'Instala Python 3.11 o posterior, anadelo al PATH y abre otra consola.'
        }
        $pythonExecutable = $pythonCommand.Source
    }
    Invoke-PythonChecked -Executable $pythonExecutable -PythonArguments @(
        '-c', 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)'
    )
    if (-not $CheckOnly) {
        if (-not (Test-Path -LiteralPath $venvPython)) {
            Write-Host 'Creando .venv para el navegador local...'
            Invoke-PythonChecked -Executable $pythonExecutable -PythonArguments @('-m', 'venv', '.venv')
        }
        $pythonExecutable = $venvPython
        & $pythonExecutable -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('playwright') else 1)"
        $playwrightMissing = $LASTEXITCODE -ne 0
        $requirements = Join-Path $projectDir 'requirements-launcher.txt'
        $requirementsHash = (Get-FileHash -LiteralPath $requirements -Algorithm SHA256).Hash.ToLowerInvariant()
        $stateFile = Join-Path $projectDir '.cache\launcher-host-dependencies.json'
        $previousHash = $null
        if (Test-Path -LiteralPath $stateFile) {
            try { $previousHash = (Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json).requirements }
            catch { $previousHash = $null }
        }
        if ($playwrightMissing -or ($previousHash -and $previousHash -ne $requirementsHash)) {
            Write-Host 'Instalando dependencias Python del navegador desde PyPI. Esto descarga paquetes; no descarga modelos ni Chromium.'
            Invoke-PythonChecked -Executable $pythonExecutable -PythonArguments @('-m', 'pip', 'install', '-r', 'requirements-launcher.txt')
        }
        if ($Development) {
            Write-Host 'Instalando dependencias de desarrollo solicitadas desde requirements-dev.txt...'
            Invoke-PythonChecked -Executable $pythonExecutable -PythonArguments @('-m', 'pip', 'install', '-r', 'requirements-dev.txt')
        }
    }
    $runtimeArguments = @('-u', 'tools/setup_runtime.py')
    if ($PullModels) { $runtimeArguments += '--pull-models' }
    if ($InstallBrowser) { $runtimeArguments += '--install-browser' }
    if ($BuildBackend) { $runtimeArguments += '--build-backend' }
    if ($CheckOnly) { $runtimeArguments += '--check-only' }
    & $pythonExecutable @runtimeArguments
    $setupExit = $LASTEXITCODE
} catch {
    Write-Host "Setup incompleto: $($_.Exception.Message)" -ForegroundColor Red
} finally {
    Pop-Location
}
exit $setupExit
