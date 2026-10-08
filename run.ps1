# Convenience launcher: activates the venv and forwards arguments.
#   .\run.ps1 devices
#   .\run.ps1 detect
#   .\run.ps1 run --capture "CABLE Output" --output "Speakers"
$ErrorActionPreference = "Stop"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "No venv found. Creating one..." -ForegroundColor Yellow
    python -m venv (Join-Path $PSScriptRoot ".venv")
    & $py -m pip install --upgrade pip
    & $py -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
}
& $py -m autotranspose @args
