$ErrorActionPreference = 'Stop'

$bundledPython = Join-Path $HOME '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$systemPython = Get-Command python -ErrorAction SilentlyContinue
$python = if (Test-Path -LiteralPath $bundledPython) { $bundledPython } elseif ($systemPython) { $systemPython.Source } else { $null }

if (-not $python) {
    throw 'Python 3.10+ was not found.'
}

$deps = Join-Path $PSScriptRoot '.deps'
New-Item -ItemType Directory -Path $deps -Force | Out-Null

& $python -m pip install --target $deps torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
& $python -m pip install --target $deps h5py==3.15.1 huggingface_hub==0.35.3 matplotlib numpy

Write-Host "Dependencies installed at $deps"
