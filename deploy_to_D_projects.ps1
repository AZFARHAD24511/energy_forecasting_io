$ErrorActionPreference = "Stop"
$Target = "D:\projects\energy_forecasting_io"
$Zip = Join-Path $PSScriptRoot "energy_forecasting_io.zip"

if (-not (Test-Path "D:\projects")) {
    throw "D:\projects does not exist. Nothing was changed."
}
if (Test-Path $Target) {
    throw "Target already exists: $Target . Nothing was overwritten. Rename/remove it manually only if you intend to."
}
if (-not (Test-Path $Zip)) {
    throw "ZIP not found next to this script: $Zip"
}

Expand-Archive -LiteralPath $Zip -DestinationPath "D:\projects" -Force:$false
Write-Host "Project created at $Target" -ForegroundColor Green
Write-Host "No existing project folders were changed."
Write-Host "Next: cd $Target ; .\setup_project.ps1"
