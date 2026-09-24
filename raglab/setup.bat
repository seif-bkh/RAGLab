@echo off
REM  RAGLab one-time setup (Windows, no Docker) - CMD wrapper for setup.ps1.
REM  See raglab\WINDOWS.md.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
if errorlevel 1 (echo. & echo setup.bat FAILED - see the message above. & pause & exit /b 1)
