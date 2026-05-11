$ErrorActionPreference = "Stop"

cd E:\predictive-maintenance-rul-ieee
conda activate main

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "outputs\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = "$logDir\final_q1_pipeline_$timestamp.log"

Start-Transcript -Path $logFile -Force

Write-Host "============================================================"
Write-Host "FINAL Q1 PIPELINE STARTED"
Write-Host "Project: E:\predictive-maintenance-rul-ieee"
Write-Host "Env    : main"
Write-Host "Time   : $(Get-Date)"
Write-Host "============================================================"

Write-Host "`n[0] Environment check"
python src\00_check_environment.py

Write-Host "`n[12] Deep sequence baselines"
python src\12_deep_sequence_baselines.py

Write-Host "`n[13] Hybrid feature-sequence model"
python src\13_hybrid_feature_sequence_model.py

Write-Host "`n[14] Final IEEE assets"
python src\14_generate_final_ieee_assets.py

Write-Host "`n[15] Manuscript draft notes"
python src\15_generate_manuscript_draft.py

Write-Host "`n[Final tables]"
Get-ChildItem outputs\tables

Write-Host "`n[Final figures]"
Get-ChildItem outputs\figures

Write-Host "`n[Paper notes]"
Get-ChildItem paper\notes

Write-Host "============================================================"
Write-Host "FINAL Q1 PIPELINE FINISHED"
Write-Host "Time: $(Get-Date)"
Write-Host "Log : $logFile"
Write-Host "============================================================"

Stop-Transcript
