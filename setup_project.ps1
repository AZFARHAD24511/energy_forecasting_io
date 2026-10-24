$ErrorActionPreference = "Stop"
$python = "D:\projects\.venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Python executable not found: $python"
}

Write-Host "`n--- Python environment ---"
& $python -c "import sys; print(sys.executable); print(sys.version)"
if ($LASTEXITCODE -ne 0) { throw "Python check failed." }

Write-Host "`n--- Checking dependencies (no installation) ---"
& $python -c "import importlib.util; modules = {'numpy':'numpy', 'pandas':'pandas', 'scikit-learn':'sklearn', 'matplotlib':'matplotlib', 'pytest':'pytest', 'xgboost':'xgboost'}; [print('{}: {}'.format(name, 'installed' if importlib.util.find_spec(module) else 'MISSING')) for name, module in modules.items()]"
if ($LASTEXITCODE -ne 0) { throw "Dependency check failed." }

Write-Host "`nDependency check completed."
