param(
    [string]$Device = "auto",
    [int]$Epochs = 3
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Environment not found. Run .\scriptsootstrap.ps1 first."
}
Set-Location $Root
& $Python -m scripts.smoke_test --device $Device --epochs $Epochs
