@echo off
title PhishGuard
cd /d "%~dp0"

:: Check if Flask is already running on port 5000
netstat -ano | findstr ":5000" | findstr "LISTENING" >nul 2>&1
if %errorlevel% == 0 (
    echo PhishGuard already running, opening browser...
    start http://phishguard.local:5000
    exit /b
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Suppress TensorFlow noise
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0

:: Start Flask in background
start /B python app.py > flask_out.log 2>&1

:: Wait for Flask to boot
timeout /t 4 /nobreak >nul

:: Open browser with friendly URL
start http://phishguard.local:5000
