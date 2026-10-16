param([string]$Python)

$ErrorActionPreference = 'Stop'
$cliRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$cliPython = $Python
$cliMarker = Join-Path $cliRoot '.train-config-runtime.json'
if (-not $cliPython -and (Test-Path -LiteralPath $cliMarker -PathType Leaf)) {
    $cliSaved = Get-Content -LiteralPath $cliMarker -Raw | ConvertFrom-Json
    if ($cliSaved.schema_version -ne 1 -or -not $cliSaved.interpreter) {
        throw "Invalid runtime selection file: $cliMarker. Use -Python PATH_TO_PYTHON explicitly."
    }
    $cliPython = [System.IO.Path]::GetFullPath((Join-Path $cliRoot $cliSaved.interpreter))
    $cliRootPrefix = $cliRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    if (-not $cliPython.StartsWith($cliRootPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or -not (Test-Path -LiteralPath $cliPython -PathType Leaf)) {
        throw "Selected environment no longer exists inside this project. Use -Python PATH_TO_PYTHON explicitly; saved selection: $cliMarker"
    }
}
if (-not $cliPython) {
    foreach ($cliCandidate in @('.venv\Scripts\python.exe', '.venv-v2-cu124\Scripts\python.exe', '.venv-v2-cpu\Scripts\python.exe', '.venv-cpu\Scripts\python.exe')) {
        $cliPath = Join-Path $cliRoot $cliCandidate
        if (Test-Path -LiteralPath $cliPath -PathType Leaf) {
            $cliPython = $cliPath
            break
        }
    }
}
if ($Python) {
    $cliExplicitPath = if ([System.IO.Path]::IsPathRooted($Python)) { $Python } else { Join-Path $cliRoot $Python }
    if (Test-Path -LiteralPath $cliExplicitPath -PathType Leaf) {
        $cliPython = [System.IO.Path]::GetFullPath($cliExplicitPath)
    } else {
        $cliCommand = Get-Command -Name $Python -CommandType Application -ErrorAction SilentlyContinue
        if (-not $cliCommand) { throw "Python interpreter not found: $Python" }
        $cliPython = $cliCommand.Source
    }
}
if (-not $cliPython) {
    throw 'No project environment found. Run setup-v2.ps1 first, or provide .\cli.ps1 -Python PATH_TO_PYTHON.'
}
Push-Location -LiteralPath $cliRoot
try {
    Write-Host "Python: $cliPython"
    & $cliPython -m research_pipeline.v2
    $cliExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $cliExitCode
