@echo off
REM  The RAGLab console over REST (Windows, no Docker) - CMD wrapper for
REM  run_front.ps1. The service must be running (run_server.bat).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_front.ps1" %*
if errorlevel 2 (echo. & echo The service is unreachable - start it first: run_server.bat & pause & exit /b 2)
if errorlevel 1 (exit /b 1)
