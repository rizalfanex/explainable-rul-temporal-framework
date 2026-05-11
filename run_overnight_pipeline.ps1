$ErrorActionPreference = "Stop"

cd E:\predictive-maintenance-rul-ieee
conda activate main

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "outputs\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = "$logDir\overnight_pipeline_$timestamp.log"

Start-Transcript -Path $logFile -Force

Write-Host "============================================================"
Write-Host "OVERNIGHT PIPELINE STARTED"
Write-Host "Project: E:\predictive-maintenance-rul-ieee"
Write-Host "Env    : main"
Write-Host "Time   : $(Get-Date)"
Write-Host "============================================================"

Write-Host "`n[0] Environment check"
python src\00_check_environment.py

Write-Host "`n[9] Ablation study"
python src\09_ablation_rul_features.py

Write-Host "`n[10] Statistical validation"
python src\10_statistical_validation.py

Write-Host "`n[11] Q1 research assets summary"
python src\11_generate_q1_research_assets.py

Write-Host "`n[Final outputs]"
Get-ChildItem outputs\tables
Get-ChildItem outputs\figures
Get-ChildItem paper\notes

Write-Host "============================================================"
Write-Host "OVERNIGHT PIPELINE FINISHED"
Write-Host "Time: $(Get-Date)"
Write-Host "Log : $logFile"
Write-Host "============================================================"

Stop-Transcript
