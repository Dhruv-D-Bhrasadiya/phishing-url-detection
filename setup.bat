@echo off
:: ============================================================
::  setup.bat  —  One-time environment setup
::  Run this ONCE before anything else.
:: ============================================================
echo.
echo ============================================================
echo   Phishing URL Detection  -  Setup
echo ============================================================
echo.

cd /d "%~dp0"

:: Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from https://python.org
    pause & exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 ( echo [ERROR] venv creation failed. & pause & exit /b 1 )
    echo      Done.
) else (
    echo [1/3] Virtual environment already exists - skipping.
)

:: Install requirements
echo [2/3] Installing packages (this may take 2-3 minutes)...
.venv\Scripts\python -m pip install --upgrade pip --quiet
.venv\Scripts\python -m pip install -r requirements.txt --quiet
if errorlevel 1 ( echo [ERROR] pip install failed. & pause & exit /b 1 )
echo      Done.

:: Generate synthetic datasets
echo [3/3] Generating synthetic CSV datasets...
.venv\Scripts\python scripts\generate_datasets.py
if errorlevel 1 ( echo [ERROR] Dataset generation failed. & pause & exit /b 1 )
echo      Done.

echo.
echo ============================================================
echo   Setup complete! Now run:
echo     run_ui.bat         (launch interactive Web UI)
echo     run_demo.bat       (quick smoke-test)
echo     run_full.bat       (full experiments)
echo     run_notebooks.bat  (open Jupyter)
echo ============================================================
echo.
pause
