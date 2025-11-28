@echo off
title Coronary RWS Analyser - Installation
echo ========================================
echo  Coronary RWS Analyser - Setup
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Installing Node.js dependencies...
call npm install
if errorlevel 1 (
    echo ERROR: npm install failed!
    pause
    exit /b 1
)

echo.
echo [2/3] Installing Python dependencies...
cd python-backend
pip install -r requirements.txt
if errorlevel 1 (
    echo WARNING: Some Python packages may have failed
)

echo.
echo [3/3] Installing opencv-contrib-python (for CSRT tracker)...
pip uninstall -y opencv-python opencv-python-headless 2>nul
pip install opencv-contrib-python==4.10.0.84

cd ..

echo.
echo ========================================
echo  Installation Complete!
echo ========================================
echo.
echo To start the application:
echo   1. Run START_BACKEND_WINDOWS.bat
echo   2. Run START_FRONTEND_WINDOWS.bat
echo   3. Open http://localhost:1420
echo.
pause
