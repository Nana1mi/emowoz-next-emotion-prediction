param(
    [ValidateSet('check','teacher','screen','confirm','all')][string]$Stage = 'all',
    [string]$OutputDir = 'runs\emotion_kd_v1',
    [string]$ReferenceDir = 'runs\encoder_minilm_v1',
    [ValidateSet('cpu','cuda')][string]$TeacherDevice = 'cpu'
)

$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskPython = (Get-Command python -ErrorAction Stop).Source
if (-not (Test-Path -LiteralPath $taskPython)) { throw "kg_project Python not found: $taskPython" }
Set-Location -LiteralPath $taskRoot
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
$env:TOKENIZERS_PARALLELISM = 'false'

# Tiny random CPU models only: no real training, downloads or extra GPU stress.
& $taskPython -u (Join-Path $taskRoot 'verify_emotion_distillation.py')
if ($LASTEXITCODE -ne 0) { throw 'Synthetic distillation verification failed; formal experiments not started.' }

$taskLogDirectory = Join-Path $taskRoot 'runs\emotion_kd_logs'
New-Item -ItemType Directory -Path $taskLogDirectory -Force | Out-Null
$taskLogPath = Join-Path $taskLogDirectory ('emotion_kd_' + $Stage + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
$taskArgs = @('--stage', $Stage, '--out', $OutputDir, '--reference', $ReferenceDir, '--teacher-device', $TeacherDevice)
Write-Host "Stage: $Stage; output: $OutputDir"
Write-Host "Read-only CE reference: $ReferenceDir; OOF teacher: $TeacherDevice"
Write-Host 'Student matches previous full-finetuning settings. Teacher never runs alongside the student on GPU.'
$taskPreviousErrorPreference = $ErrorActionPreference
$taskExitCode = -1
try {
    $ErrorActionPreference = 'Continue'
    & $taskPython -u (Join-Path $taskRoot 'run_emotion_distillation.py') @taskArgs 2>&1 | Tee-Object -FilePath $taskLogPath
    $taskExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $taskPreviousErrorPreference
}
if ($taskExitCode -ne 0) { throw "Distillation stopped. See $taskLogPath. Rerun IDENTICAL command to resume; reference results unchanged." }
Write-Host "Finished $Stage. Log: $taskLogPath"
if ($Stage -eq 'all' -or $Stage -eq 'confirm') { Write-Host "Read emotion_distillation_report.txt and summary.json in $OutputDir" }
