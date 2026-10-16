param(
    [ValidateSet('cpu', 'cu124')][string]$Backend = 'cpu',
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$setupRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$setupEnv = Join-Path $setupRoot ('.venv-v2-' + $Backend)
$setupPython = Join-Path $setupEnv 'Scripts\python.exe'
if (Test-Path -LiteralPath $setupEnv) {
    throw "Environment already exists: $setupEnv. It will not be overwritten."
}
Push-Location -LiteralPath $setupRoot
try {
    & $Python -m venv $setupEnv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python virtual environment.' }
    & $setupPython -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { throw 'Could not prepare pip.' }
    $setupIndex = 'https://download.pytorch.org/whl/' + $Backend
    & $setupPython -m pip install 'torch==2.6.0' 'torchvision==0.21.0' --index-url $setupIndex
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch install failed; check Python/platform/network/driver compatibility.' }
    & $setupPython -m pip install 'ultralytics==8.4.175'
    if ($LASTEXITCODE -ne 0) { throw 'Ultralytics install failed.' }
    & $setupPython -m pip install --editable .
    if ($LASTEXITCODE -ne 0) { throw 'Project install failed.' }
    & $setupPython -m research_pipeline.v2 doctor --device auto
    if ($LASTEXITCODE -ne 0) { throw 'Runtime verification failed.' }
    # Record only after successful verification. A later cli.ps1 honors this
    # selection before any historical .venv and remains relocatable with source.
    $setupMarker = Join-Path $setupRoot '.train-config-runtime.json'
    $setupMarkerTemp = Join-Path $setupRoot '.train-config-runtime.json.tmp'
    @{ schema_version = 1; interpreter = ".venv-v2-$Backend\Scripts\python.exe"; backend = $Backend } |
        ConvertTo-Json | Set-Content -LiteralPath $setupMarkerTemp -Encoding UTF8
    Move-Item -LiteralPath $setupMarkerTemp -Destination $setupMarker -Force
    Write-Host "Ready: $setupPython -m research_pipeline.v2 optimize --data DATASET --device auto"
    Write-Host "Launcher selection saved: $setupMarker"
    Write-Host 'Successful installation is not an optimum or unseen-GPU certificate.'
} finally {
    Pop-Location
}
