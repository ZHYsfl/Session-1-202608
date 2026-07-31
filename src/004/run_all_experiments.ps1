param(
    [switch]$Quick
)

$ErrorActionPreference = 'Stop'

$bundledPython = Join-Path $HOME '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$systemPython = Get-Command python -ErrorAction SilentlyContinue

if (Test-Path -LiteralPath $bundledPython) {
    $python = $bundledPython
}
elseif ($systemPython) {
    $python = $systemPython.Source
}
else {
    throw 'Python 3 was not found. See REPRODUCTION.md for setup instructions.'
}

$arguments = @(Join-Path $PSScriptRoot 'run_all_experiments.py')
if ($Quick) {
    $arguments += '--quick'
}

& $python @arguments
exit $LASTEXITCODE
