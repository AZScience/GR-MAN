@echo off
title PhysioNet Downloader Launcher
echo ======================================================
echo    KHOI DONG UNG DUNG TAI DU LIEU PHYSIONET
echo ======================================================
cd /d "%~dp0"
python app_gui.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Co loi khi chay ung dung GUI. Dang thu chuyen sang ban CLI...
    python download_cli.py
    pause
)
