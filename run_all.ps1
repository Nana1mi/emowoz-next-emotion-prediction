param(
    [switch]$Quick,
    [int]$EncodeBatch = 128
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonExe = (Get-Command python -ErrorAction Stop).Source
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "kg_project Python not found: $pythonExe"
}
Set-Location -LiteralPath $projectRoot
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
$env:HF_HUB_DOWNLOAD_TIMEOUT = '120'
$env:HF_HUB_ETAG_TIMEOUT = '30'
# Keep the model download cache in the project. Existing caches are not deleted.
$env:HF_HOME = Join-Path $projectRoot 'data\hf_cache'

& $pythonExe -u (Join-Path $projectRoot 'run_experiments.py') smoke
if ($LASTEXITCODE -ne 0) { throw 'Offline self-check failed. Do not start training.' }

if ($Quick) {
    $outputDir = Join-Path $projectRoot 'runs\quick'
    $runArgs = @('all', '--out', $outputDir, '--max-dialogues', '2000', '--seeds', '42', '--epochs', '10')
} else {
    $outputDir = Join-Path $projectRoot 'runs\full'
    $runArgs = @('all', '--out', $outputDir, '--seeds', '42', '43', '44', '--epochs', '15')
}
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$logFile = Join-Path $outputDir ('run_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
$runArgs += @('--encode-batch', [string]$EncodeBatch)
Write-Host "Using: $pythonExe"
Write-Host "Output: $outputDir"
& $pythonExe -u (Join-Path $projectRoot 'run_experiments.py') @runArgs 2>&1 | Tee-Object -FilePath $logFile
if ($LASTEXITCODE -ne 0) { throw "Experiment stopped. See: $logFile" }
Write-Host "Completed. Read metrics.csv and 第二次报告草稿.md in $outputDir"
