@echo off
setlocal
cd /d "%~dp0"
echo ===========================================================
echo  ONCO-AGENT - Windows setup + automated tests
 echo  RESEARCH SOFTWARE ONLY - NOT FOR CLINICAL USE
 echo ===========================================================
where py >nul 2>&1
if errorlevel 1 (
    echo Python launcher 'py' not found. Install Python 3.11 for Windows.
    pause
    exit /b 1
)
py -3.11 -m venv .venv
if errorlevel 1 goto :failed
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :failed
.venv\Scripts\python.exe -m pip install -e ".[dev,dashboard]"
if errorlevel 1 goto :failed
.venv\Scripts\python.exe -m pytest -q
if errorlevel 1 goto :failed
.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
echo.
echo Setup finished. Now open README.md and docs\WINDOWS_COMMANDS.md.
echo For a synthetic execution test, run WINDOWS_SMOKE_TEST.bat.
pause
exit /b 0
:failed
echo Setup failed. Please share the full terminal output for troubleshooting.
pause
exit /b 1
