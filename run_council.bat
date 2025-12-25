@echo off
title AI Trading Council
echo.
echo ============================================================
echo                  AI TRADING COUNCIL
echo ============================================================
echo.
echo  Make sure Ollama is running with your AI models!
echo.
echo  Starting server at http://localhost:8000
echo.
echo ============================================================
echo.

cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Install required packages
echo Installing required packages...
pip install fastapi uvicorn httpx websockets aiofiles --quiet

REM Start the server
echo.
echo Starting AI Trading Council server...
echo.
start "" http://localhost:8000
python ai_council_server.py

pause
