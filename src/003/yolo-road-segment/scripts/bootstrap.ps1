param(
    [string]$PythonVersion = "3.11",
    [string]$TorchBackend = "auto",
    [switch]$Recreate,
    [switch]$SkipAssets,
    [switch]$ForceAssets
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv was not found. Install it first: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
}

Write-Host "[1/5] Installing Python $PythonVersion..."
uv python install $PythonVersion

if ($Recreate -and (Test-Path ".venv")) {
    Write-Host "Removing existing .venv..."
    Remove-Item -Recurse -Force ".venv"
}

if (-not (Test-Path ".venv")) {
    Write-Host "[2/5] Creating virtual environment..."
    uv venv --python $PythonVersion
} else {
    Write-Host "[2/5] Reusing existing .venv."
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Virtual-environment Python was not created: $Python"
}

Write-Host "[3/5] Installing PyTorch backend '$TorchBackend'..."
uv pip install --python $Python torch torchvision --torch-backend=$TorchBackend

Write-Host "[4/5] Installing project dependencies..."
uv pip install --python $Python -e .

if (-not $SkipAssets) {
    Write-Host "[5/5] Downloading model and COCO8-Seg assets..."
    $AssetArgs = @("-m", "scripts.prepare_assets")
    if ($ForceAssets) { $AssetArgs += "--force" }
    & $Python @AssetArgs
} else {
    Write-Host "[5/5] Asset download skipped."
}

uv pip freeze --python $Python | Set-Content -Encoding utf8 "environment.lock.txt"

Write-Host ""
Write-Host "Environment ready."
& $Python -c "import torch, ultralytics; print('PyTorch:', torch.__version__); print('Ultralytics:', ultralytics.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
Write-Host ""
Write-Host "Next: .\scripts\smoke_test.ps1"
