@echo off
:: ============================================================
::  run_ui.bat  —  Launch CyberShield Phishing URL Web UI
:: ============================================================
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

echo.
echo ============================================================
echo   CYBERSHIELD - PHISHING DETECTION & RESEARCH UI
echo ============================================================
echo.
echo   Starting web server on http://127.0.0.1:5000 ...
echo   Opening browser in 3 seconds...
echo.

:: Launch browser in background after short delay
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:5000"

:: Run Flask server with project virtualenv
.venv\Scripts\python.exe src\app.py
pause
