@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo First run START_HERE_WINDOWS.bat
    pause
    exit /b 1
)
echo This creates SYNTHETIC COLOR IMAGES, NOT MEDICAL RESULTS.
if exist data\synthetic_smoke\ADI\ADI_000.png (
   echo Synthetic images already exist; skipping regeneration.
) else (
   .venv\Scripts\python.exe -m onco_agent synthetic-data --output data\synthetic_smoke --count 16 --size 48
   if errorlevel 1 goto :failed
)
.venv\Scripts\python.exe -m onco_agent split --root data\synthetic_smoke --output results\smoke_splits.json --image-size 48
if errorlevel 1 goto :failed
.venv\Scripts\python.exe -m onco_agent train --manifest results\smoke_splits.json --output results\smoke_fast --architecture tiny_cnn --epochs 1 --size 48 --batch-size 16 --device cpu
if errorlevel 1 goto :failed
.venv\Scripts\python.exe -m onco_agent cache --model results\smoke_fast\best.pt --manifest results\smoke_splits.json --subset val --output results\smoke_fast_val.npz --batch-size 12 --device cpu
if errorlevel 1 goto :failed
.venv\Scripts\python.exe -m onco_agent cache --model results\smoke_fast\best.pt --manifest results\smoke_splits.json --subset policy --output results\smoke_fast_policy.npz --batch-size 12 --device cpu
if errorlevel 1 goto :failed
.venv\Scripts\python.exe -m onco_agent calibrate --fast results\smoke_fast_val.npz --strong results\smoke_fast_val.npz --output results\smoke_calibration.json
if errorlevel 1 goto :failed
.venv\Scripts\python.exe -m onco_agent fit-policy --fast results\smoke_fast_policy.npz --strong results\smoke_fast_policy.npz --calibration results\smoke_calibration.json --output results\smoke_policy.json
if errorlevel 1 goto :failed
.venv\Scripts\python.exe -m onco_agent freeze --calibration results\smoke_calibration.json --learned results\smoke_policy.json --output results\smoke_frozen.json
if errorlevel 1 goto :failed
.venv\Scripts\python.exe -m onco_agent evaluate --fast results\smoke_fast_val.npz --strong results\smoke_fast_val.npz --config results\smoke_frozen.json --output results\smoke_study --split val
if errorlevel 1 goto :failed
.venv\Scripts\python.exe -m onco_agent plot --metrics results\smoke_study\val_metrics.json --output results\smoke_study\policy_comparison.png
if errorlevel 1 goto :failed
echo.
echo Software smoke run completed. ALL RESULTS ARE SYNTHETIC AND NOT RESEARCH EVIDENCE.
echo Next follow docs\WINDOWS_COMMANDS.md to use verified real data.
pause
exit /b 0
:failed
echo Smoke test failed. Share the error output for a fix.
pause
exit /b 1
