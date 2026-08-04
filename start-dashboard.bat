@echo off
cd /d "%~dp0"
start "AI-PR-Summary Server" cmd /k py server.py
timeout /t 3 /nobreak >nul
start "" http://127.0.0.1:5000
