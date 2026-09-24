@echo off
REM  Runs the RAGLab HTTP service (Windows, no Docker) - CMD wrapper for
REM  run_server.ps1. The local equivalent of "docker compose up".
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_server.ps1" %*
if errorlevel 1 (echo. & echo run_server.bat FAILED - see the message above. & pause & exit /b 1)
