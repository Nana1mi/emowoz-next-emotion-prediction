param(
    [ValidateSet('check','screen','confirm','all')][string]$Stage = 'all',
    [string]$OutputDir = 'runs\role_interaction_v1',
    [string]$ReferenceDir = 'runs\encoder_minilm_v1'
)

$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskPython = (Get-Command python -ErrorAction Stop).Source
if (-not (Test-Path -LiteralPath $taskPython)) { throw "kg_project Python not found: $taskPython" }
Set-Location -LiteralPath $taskRoot
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
$env:TOKENIZERS_PARALLELISM = 'false'

Write-Host 'Running CPU-only synthetic checks (no real experiment or GPU workload)...'
& $taskPython -u (Join-Path $taskRoot 'verify_role_interaction.py')
if ($LASTEXITCODE -ne 0) { throw 'CPU synthetic verification failed; formal experiments not started.' }
$taskLogDirectory = Join-Path $taskRoot 'runs\role_interaction_logs'
New-Item -ItemType Directory -Path $taskLogDirectory -Force | Out-Null
$taskLogPath = Join-Path $taskLogDirectory ('role_interaction_' + $Stage + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
$taskArgs = @('--stage', $Stage, '--out', $OutputDir, '--reference', $ReferenceDir)
Write-Host "Stage: $Stage; output: $OutputDir; read-only reference: $ReferenceDir"
Write-Host 'Three matched-capacity variants x three seeds; same history, 256 tokens, single encoder pass.'
$taskPreviousErrorPreference = $ErrorActionPreference
$taskExitCode = -1
try {
    $ErrorActionPreference = 'Continue'
    & $taskPython -u (Join-Path $taskRoot 'run_role_interaction.py') @taskArgs 2>&1 | Tee-Object -FilePath $taskLogPath
    $taskExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $taskPreviousErrorPreference
}
if ($taskExitCode -ne 0) { throw "Role run stopped. See $taskLogPath. Rerun IDENTICAL command to resume; previous results unchanged." }
Write-Host "Finished $Stage. Log: $taskLogPath"
if ($Stage -eq 'all' -or $Stage -eq 'confirm') { Write-Host "Read role_interaction_report.txt and summary.json in $OutputDir" }
