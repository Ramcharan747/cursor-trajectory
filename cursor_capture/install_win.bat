@echo off
:: ═══════════════════════════════════════════════════════
::  CursorCapture — One-Click Installer for Windows
::  Double-click this file to install.
:: ═══════════════════════════════════════════════════════

title CursorCapture Installer

echo.
echo ╔══════════════════════════════════════════════════╗
echo ║     CursorCapture — Windows Installer           ║
echo ╠══════════════════════════════════════════════════╣
echo ║                                                 ║
echo ║  This installs a tiny background app that       ║
echo ║  records cursor movement for research.          ║
echo ║                                                 ║
echo ║  - Runs silently in the background              ║
echo ║  - Auto-starts on every login                   ║
echo ║  - Uses less than 5MB RAM                       ║
echo ║  - Data saved to your home folder               ║
echo ║  - No screenshots, no keystrokes — just         ║
echo ║    cursor position and time                     ║
echo ║                                                 ║
echo ╚══════════════════════════════════════════════════╝
echo.

:: Check if binary exists in the same folder
set "SCRIPT_DIR=%~dp0"
set "BINARY=%SCRIPT_DIR%cursor_capture.exe"

if not exist "%BINARY%" (
    echo   X Error: Cannot find cursor_capture.exe in the same folder.
    echo     Make sure this script is next to cursor_capture.exe
    echo.
    pause
    exit /b 1
)

:: Set install directory
set "INSTALL_DIR=%LOCALAPPDATA%\CursorCapture"
set "DATA_DIR=%USERPROFILE%\cursor_capture_data"

echo   [1/4] Creating install directory...
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
copy /Y "%BINARY%" "%INSTALL_DIR%\cursor_capture.exe" >nul
echo         Done: %INSTALL_DIR%\

echo.
echo   [2/4] Creating data directory...
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
echo         Done: %DATA_DIR%\

echo.
echo   [3/4] Setting up auto-start on login...

:: Method 1: Add to Startup folder (simplest, most reliable)
set "STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"

:: Create a VBScript wrapper that launches silently (no console window)
(
    echo Set WshShell = CreateObject^("WScript.Shell"^)
    echo WshShell.Run """%INSTALL_DIR%\cursor_capture.exe"" run", 0, False
) > "%STARTUP_DIR%\CursorCapture.vbs"

echo         Done: Added to Windows Startup

echo.
echo   [4/4] Starting CursorCapture...

:: Launch silently using VBScript (no console window)
set "TEMP_LAUNCHER=%TEMP%\start_cursor_capture.vbs"
(
    echo Set WshShell = CreateObject^("WScript.Shell"^)
    echo WshShell.Run """%INSTALL_DIR%\cursor_capture.exe"" run", 0, False
) > "%TEMP_LAUNCHER%"

cscript //nologo "%TEMP_LAUNCHER%"
del "%TEMP_LAUNCHER%" >nul 2>nul

:: Wait and verify
timeout /t 3 /nobreak >nul

:: Check if process is running
tasklist /FI "IMAGENAME eq cursor_capture.exe" 2>nul | find /I "cursor_capture.exe" >nul
if %errorlevel% equ 0 (
    echo         Done: CursorCapture is running!
) else (
    echo         Warning: Process may not have started. Try restarting your PC.
)

echo.
echo ╔══════════════════════════════════════════════════╗
echo ║                                                 ║
echo ║  Installation complete!                         ║
echo ║                                                 ║
echo ║  CursorCapture is running in the background     ║
echo ║  and will auto-start on every login.            ║
echo ║                                                 ║
echo ║  You can close this window now.                 ║
echo ║                                                 ║
echo ║  Data is saved to:                              ║
echo ║    %DATA_DIR%\                                  ║
echo ║                                                 ║
echo ║  To uninstall, delete:                          ║
echo ║    %INSTALL_DIR%\                               ║
echo ║    %STARTUP_DIR%\CursorCapture.vbs              ║
echo ║                                                 ║
echo ╚══════════════════════════════════════════════════╝
echo.
pause
