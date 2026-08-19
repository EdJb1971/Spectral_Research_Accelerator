@echo off
title SpectralEarth Platform Launcher
echo ==========================================================
echo    SpectralEarth Scientific Platform Local Launcher
echo ==========================================================
echo.
echo [*] Launching setup and servers via PowerShell...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_platform.ps1"
pause
