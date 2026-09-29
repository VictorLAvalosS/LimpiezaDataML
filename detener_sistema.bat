@echo off
chcp 65001 >nul
title 🛑 DETENER - Data Cleaner PRO + ML (Puerto 8501)
color 0C
cls
cd /d "%~dp0"

set "APP_DIR=%~dp0"
set "PID_FILE=%APP_DIR%data\.server_pid.txt"
set "SERVER_PORT=8501"
if not exist "%APP_DIR%data" mkdir "%APP_DIR%data" 2>nul

echo.
echo  ╔══════════════════════════════════════════════════════════════════╗
echo  ║       🛑 DETENIENDO EL SERVIDOR Streamlit (Puerto %SERVER_PORT%)         ║
echo  ╚══════════════════════════════════════════════════════════════════╝
echo.

REM === 1) Parar por PID guardado ===
if exist "%PID_FILE%" (
    echo  1/3 Buscando por PID archivo guardado...
    setlocal enabledelayedexpansion
    set /p SPID=<"%PID_FILE%" 2>nul
    if defined SPID (
        echo      → PID detectado: !SPID!
        tasklist /FI "PID eq !SPID!" 2>nul | find /I "!SPID!" >nul 2>nul
        if !errorlevel! equ 0 (
            taskkill /PID !SPID! /T /F >nul 2>nul
            echo      ✅ Proceso !SPID! terminado.
        ) else (
            echo      ℹ️  PID !SPID! no está activo.
        )
    )
    endlocal
    del "%PID_FILE%" 2>nul
) else (
    echo  1/3 (no había PID archivo guardado — saltando)
)

REM === 2) Matar por PUERTO 8501 (100% fiable) ===
echo.
echo  2/3 Buscando y matando procesos que ocupan el puerto %SERVER_PORT%...
setlocal enabledelayedexpansion
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr /R /C:":%SERVER_PORT%.*LISTENING"') do (
    echo      🎯 Encontrado PID %%a ocupando :%SERVER_PORT% — taskkill.
    taskkill /PID %%a /T /F >nul 2>nul
)
endlocal

REM === 3) Matar ventana con título "Streamlit" ===
echo.
echo  3/3 Buscando ventanas CMD con título "Streamlit*"...
for /f "tokens=2" %%a in ('tasklist /v /FI "WINDOWTITLE eq Streamlit*" /FO list 2^>nul ^| findstr /I "PID:"') do (
    echo      🎯 Encontrada ventana PID %%a — cerrando.
    taskkill /PID %%a /T /F >nul 2>nul
)

REM === Limpiar PID final ===
del "%PID_FILE%" 2>nul

echo.
echo  ──────────────────────────────────────────────────────────────────
echo   ✅  ACCIÓN FINALIZADA.  El servidor Streamlit (Puerto %SERVER_PORT%)
echo      debería estar DETENIDO ahora mismo.
echo.
echo      💡 Para volver a INICIAR: ejecuta  menu_sistema.bat  →  opción 1
echo  ──────────────────────────────────────────────────────────────────
echo.
pause
exit /b 0
