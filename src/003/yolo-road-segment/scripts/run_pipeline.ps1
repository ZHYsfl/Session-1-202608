param(
    [ValidateSet("check", "train", "val", "predict", "all")]
    [string]$Stage = "all",
    [string]$RunName = "road_yolo26n_seg",
    [string]$Device = "auto",
    [string]$Source = "datasets/road/images/test",
    [switch]$CleanRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Environment not found. Run .\scriptsootstrap.ps1 first."
}
Set-Location $Root

$TrainDir = Join-Path $Root "outputs	rain\$RunName"
$BestModel = Join-Path $TrainDir "weightsest.pt"

function Invoke-Check {
    & $Python -m scripts.check_dataset --data configs/road_seg.yaml
}

function Invoke-Train {
    if ($CleanRun -and (Test-Path $TrainDir)) {
        Remove-Item -Recurse -Force $TrainDir
    }
    & $Python -m src.train --name $RunName --device $Device --exist-ok
}

function Invoke-Val {
    if (-not (Test-Path $BestModel)) {
        throw "Best model not found: $BestModel"
    }
    & $Python -m src.validate --model $BestModel --device $Device --name "${RunName}_val" --exist-ok
}

function Invoke-Predict {
    if (-not (Test-Path $BestModel)) {
        throw "Best model not found: $BestModel"
    }
    & $Python -m src.predict --model $BestModel --source $Source --device $Device --name "${RunName}_predict" --exist-ok
}

switch ($Stage) {
    "check"   { Invoke-Check }
    "train"   { Invoke-Check; Invoke-Train }
    "val"     { Invoke-Val }
    "predict" { Invoke-Predict }
    "all"     { Invoke-Check; Invoke-Train; Invoke-Val; Invoke-Predict }
}
