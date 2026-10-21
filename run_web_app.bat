@echo off
title Quantum Classifier Web Explorer
echo ========================================================
echo Launching Quantum Classifier Web Explorer...
echo ========================================================
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run_app.py
) else (
    python run_app.py
)
pause
