@echo off
cd /d "%~dp0"
start "AI-PR-Summary Server" cmd /k py server.py
timeout /t 2 /nobreak >nul
start "" "dashboard\index.html"
