@echo off
title Coronary RWS Analyser - Backend
echo ========================================
echo  Coronary RWS Analyser - Backend Server
echo ========================================
echo.

cd /d "%~dp0python-backend"

echo Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

echo.
echo Starting FastAPI server on http://127.0.0.1:8000
echo Press Ctrl+C to stop
echo.

python -m uvicorn app.main:app --reload --port 8000 --host 127.0.0.1

pause
