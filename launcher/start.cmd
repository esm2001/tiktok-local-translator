@echo off
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "launcher\main.py" %*
) else (
    python "launcher\main.py" %*
)
set "launcher_exit=%errorlevel%"
exit /b %launcher_exit%
