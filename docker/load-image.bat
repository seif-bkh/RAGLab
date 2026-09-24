@echo off
REM  Verify + docker load a RAGLab image export (.zip / .tar.gz / .tar) on
REM  Windows - CMD wrapper for load-image.ps1. Pass the file as the argument.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0load-image.ps1" %*
if errorlevel 1 (echo. & echo load-image.bat FAILED - see the message above. & pause & exit /b 1)
