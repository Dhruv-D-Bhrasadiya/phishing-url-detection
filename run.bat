@echo off
:: ============================================================
::  run.bat  —  CyberShield Master Launcher
:: ============================================================
title CyberShield - Launcher
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

:MENU
cls
echo ============================================================
echo   CYBERSHIELD - PHISHING DETECTION & RESEARCH PLATFORM
echo ============================================================
echo.
echo   Select an option to run:
echo.
echo   [1] Launch Interactive Web UI  (Recommended - Port 5000)
echo   [2] Run Quick Demo Smoke-Test  (Synthetic data, ~30s)
echo   [3] Run Full ML Experiments    (Train all models & metrics)
echo   [4] Open Jupyter Notebooks
echo   [5] Run Unit Tests             (pytest tests/)
echo   [6] Exit
echo.
echo ============================================================
set /p opt="Enter choice [1-6]: "

if "%opt%"=="1" goto WEB_UI
if "%opt%"=="2" goto DEMO
if "%opt%"=="3" goto FULL
if "%opt%"=="4" goto NOTEBOOKS
if "%opt%"=="5" goto TESTS
if "%opt%"=="6" goto EXIT
goto MENU

:WEB_UI
cls
echo.
echo Starting Web UI on http://127.0.0.1:5000 ...
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:5000"
.venv\Scripts\python.exe src\app.py
pause
goto MENU

:DEMO
cls
echo.
echo Running Quick Smoke-Test...
.venv\Scripts\python.exe -m src.experiments --quick-check
echo.
pause
goto MENU

:FULL
cls
echo.
echo Running Full Experiment Pipeline on D1 & D2...
.venv\Scripts\python.exe -m src.experiments --d1 data\raw\dataset_phishing_uci.csv --d2 data\raw\phiusiil_phishing_url.csv --no-tune
echo.
pause
goto MENU

:NOTEBOOKS
cls
echo.
echo Opening Jupyter Notebooks in notebooks/ ...
echo (Press Ctrl+C in this terminal when finished)
.venv\Scripts\jupyter.exe notebook notebooks\
pause
goto MENU

:TESTS
cls
echo.
echo Running Unit Tests...
.venv\Scripts\python.exe -m pytest tests/ -v
echo.
pause
goto MENU

:EXIT
exit /b 0
