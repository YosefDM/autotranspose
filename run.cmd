@echo off
rem Launcher that works regardless of the PowerShell execution policy, which
rem blocks run.ps1 on a default Windows install.
rem
rem   run devices
rem   run detect
rem   run run
setlocal
pushd "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo No virtual environment found. Creating one...
    python -m venv "%~dp0.venv"
    if errorlevel 1 (
        echo Could not create the virtual environment. Is Python on your PATH?
        popd
        exit /b 1
    )
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 (
        echo Dependency install failed.
        popd
        exit /b 1
    )
)

"%PY%" -m autotranspose %*
set "CODE=%ERRORLEVEL%"
popd
exit /b %CODE%
