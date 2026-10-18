param(
    [ValidateSet('check','benchmark','screen','confirm','all')][string]$Stage = 'screen',
    [ValidateSet('minilm','roberta','all')][string]$Suite = 'minilm',
    [string]$OutputDir = '',
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda',
    [ValidateSet('auto','fp32','bf16','fp16')][string]$Precision = 'auto',
    [int]$MicroBatch = 16,
    [int]$Accumulation = 4,
    [int]$EvalBatch = 32,
    [int]$Epochs = 8,
    [int]$Patience = 3,
    [double]$EncoderLR = 0.00002,
    [double]$HeadLR = 0.001,
    [switch]$AllowDownload,
    [switch]$GradientCheckpointing
)

$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskPython = (Get-Command python -ErrorAction Stop).Source
if (-not (Test-Path -LiteralPath $taskPython)) { throw "kg_project Python not found: $taskPython" }
Set-Location -LiteralPath $taskRoot
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
if ([string]::IsNullOrWhiteSpace($OutputDir)) { $OutputDir = 'runs\encoder_' + $Suite + '_v1' }

# Only tiny random synthetic Transformers are trained by this verifier, never real data.
& $taskPython -u (Join-Path $taskRoot 'verify_encoder_baselines.py') --device $Device
if ($LASTEXITCODE -ne 0) { throw 'Synthetic encoder verification failed; official training not started.' }

$taskLogDirectory = Join-Path $taskRoot 'runs\encoder_logs'
New-Item -ItemType Directory -Path $taskLogDirectory -Force | Out-Null
$taskLogPath = Join-Path $taskLogDirectory ('encoder_' + $Suite + '_' + $Stage + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
$taskArgs = @('--stage', $Stage, '--suite', $Suite, '--out', $OutputDir, '--device', $Device,
              '--precision', $Precision, '--micro-batch', "$MicroBatch", '--accumulation', "$Accumulation",
              '--eval-batch', "$EvalBatch", '--epochs', "$Epochs", '--patience', "$Patience",
              '--encoder-lr', "$EncoderLR", '--head-lr', "$HeadLR")
if ($AllowDownload) { $taskArgs += '--allow-download' }
if ($GradientCheckpointing) { $taskArgs += '--gradient-checkpointing' }
Write-Host "Python: $taskPython"
Write-Host "Stage: $Stage; suite: $Suite; output: $OutputDir"
Write-Host 'History-only prediction. Screen uses validation only. Old results stay read-only.'
# Windows PowerShell may treat ordinary native stderr warnings as terminating
# errors under Stop. Capture them in the log, then use Python's exit code.
$taskPreviousErrorPreference = $ErrorActionPreference
$taskExitCode = -1
try {
    $ErrorActionPreference = 'Continue'
    & $taskPython -u (Join-Path $taskRoot 'run_encoder_baselines.py') @taskArgs 2>&1 | Tee-Object -FilePath $taskLogPath
    $taskExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $taskPreviousErrorPreference
}
if ($taskExitCode -ne 0) { throw "Encoder run stopped. See $taskLogPath. Rerun with identical settings to resume." }
Write-Host "Finished $Stage. Log: $taskLogPath"
if ($Stage -eq 'screen') { Write-Host 'Next: rerun the same command with -Stage confirm. Do not change training settings/output.' }
