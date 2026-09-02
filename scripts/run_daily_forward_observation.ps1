$ErrorActionPreference = "Stop"

$repository = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repository ".venv\Scripts\python.exe"
$logDirectory = Join-Path $repository "reports\forward_observation"
$logPath = Join-Path $logDirectory "scheduled_run.log"

New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
Set-Location $repository

$runDate = Get-Date -Format "yyyy-MM-dd"
$ErrorActionPreference = "Continue"
& $python -B -m src.daily_forward_pipeline --end-date $runDate *>&1 |
    Tee-Object -FilePath $logPath -Append
$pipelineExitCode = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($pipelineExitCode -ne 0) {
    throw "Forward observation pipeline failed with exit code $pipelineExitCode"
}
