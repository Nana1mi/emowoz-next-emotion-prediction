param(
    [ValidateSet('all', 'minority', 'semantic')][string]$Route = 'all',
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [string]$OutputDir = 'runs\controlled_v2'
)

$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskPython = (Get-Command python -ErrorAction Stop).Source
if (-not (Test-Path -LiteralPath $taskPython)) { throw "kg_project Python not found: $taskPython" }
Set-Location -LiteralPath $taskRoot
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'

# Offline synthetic verification only. No dataset download, encoding or official training.
& $taskPython -u (Join-Path $taskRoot 'verify_controlled.py') --device $Device
if ($LASTEXITCODE -ne 0) { throw 'Controlled pipeline verification failed. Training was not started.' }

# Runtime logs are kept outside the output folder so provenance checks see a new empty output.
$taskLogDirectory = Join-Path $taskRoot 'runs\controlled_logs'
New-Item -ItemType Directory -Path $taskLogDirectory -Force | Out-Null
$taskLogPath = Join-Path $taskLogDirectory ('controlled_' + $Route + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
$taskArgs = @('--out', $OutputDir, '--reuse-from', (Join-Path $taskRoot 'runs\full'),
              '--route', $Route, '--device', $Device, '--seeds', '42', '43', '44')
Write-Host "Python: $taskPython"
Write-Host "Output: $OutputDir"
Write-Host "Route: $Route; three seeds; fixed configs; validation-only selection."
& $taskPython -u (Join-Path $taskRoot 'run_controlled_experiments.py') @taskArgs 2>&1 | Tee-Object -FilePath $taskLogPath
if ($LASTEXITCODE -ne 0) { throw "Controlled experiments stopped. See $taskLogPath; baseline files were not overwritten." }
Write-Host "Finished. Read controlled_ablation_report.md and validation_selection.json in $OutputDir"

# Examples:
# .\run_controlled.ps1                    # all predefined configurations
# .\run_controlled.ps1 -Route minority   # minority route first
# .\run_controlled.ps1 -Route semantic   # extend the same verified run with semantic route
# .\run_controlled.ps1 -OutputDir 'runs\controlled_v3' # use a NEW folder after code/config changes
# Demo after training (does not run automatically):
# python .\demo_world_model.py --ckpt .\runs\controlled_v2
