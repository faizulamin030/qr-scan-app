$ErrorActionPreference = "Stop"

$venvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$appFile = Join-Path $PSScriptRoot "qr_payment.py"

if (Test-Path $venvPython) {
    & $venvPython $appFile
} else {
    py $appFile
}
