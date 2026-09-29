@echo off
chcp 65001 >nul
title EJECUTAR - Data Cleaner PRO + ML (VENV DIRECTO)
color 0A
cls
cd /d "%~dp0"

REM =======================================================
REM 📌 CÓMO EJECUTAR ESTE ARCHIVO EN PowerShell:
REM     ❌ NO escribas:   run.bat
REM     ✅ SÍ escribe:   .\run.bat
REM     🆗 O MEJOR USA:   .\menu_sistema.bat  (menú interactivo con control total)
REM =======================================================

REM Este script NO activa el venv (evita ExecutionPolicy de PowerShell)
REM usa DIRECTAMENTE python.exe DENTRO de la carpeta venv (100% fiable).

set "VENV_PY=%~dp0venv\Scripts\python.exe"
set "APP_FILE=%~dp0app.py"
set "PID_FILE=%~dp0data\.server_pid.txt"
set "PORT_FILE=%~dp0data\.server_port.txt"
set "SERVER_PORT=8501"
if not exist "%~dp0data" mkdir "%~dp0data" 2>nul

echo =======================================================
echo  🧹 Data Cleaner PRO + Machine Learning Suite 🧠
echo  (Ejecución via VENV · sin conflictos)
echo =======================================================
echo.
echo  💡 ATAJO RÁPIDO: Pulsa tecla Windows + R y escribe:
echo       %~dp0menu_sistema.bat     (menú de control)
echo     O bien doble-clic sobre  menu_sistema.bat
echo.

REM === VERIFICACION ===
set "PY_EXE=%VENV_PY%"
if not exist "%VENV_PY%" (
    echo ⚠️  VENV no detectado → intentando PYTHON GLOBAL del sistema...
    where python >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] No hay Python (ni venv ni global).
        echo   Solucion: Ejecuta PRIMERO install.bat para crear el venv.
        pause
        exit /b 1
    )
    set "PY_EXE=python"
)

echo [1/3] Python detectado:
"%PY_EXE%" --version
echo.

echo [2/3] Chequeo librerías core (instalación silenciosa si faltan)...
"%PY_EXE%" -m pip install -r "%~dp0requirements.txt" --quiet --no-warn-script-location 2>nul
echo      Listo.
echo.

netstat -ano | find "LISTENING" | find ":%SERVER_PORT%" >nul 2>nul
if not errorlevel 1 (
    echo.
    echo ⚠️  Puerto %SERVER_PORT% OCUPADO. Puedes:
    echo    · [Detener ANTES] → ejecuta  detener_sistema.bat
    echo    · O usa → menu_sistema.bat Opción 2
    echo    (se intentará detener automáticamente en 3 segundos...)
    timeout /t 3 /nobreak >nul
    call :MATAR_POR_PUERTO
    timeout /t 2 /nobreak >nul
)

echo [3/3] Iniciando Streamlit...
echo %SERVER_PORT% >"%PORT_FILE%"
echo.
echo   🌐 URL LOCAL:        http://localhost:%SERVER_PORT%
echo   🛑 PARA PARAR:        (A) Ctrl + C en ESTA ventana
echo                         (B) Cierra esta ventana
echo                         (C) Ejecuta:  .\detener_sistema.bat
echo   🎮 CONTROL TOTAL:    .\menu_sistema.bat  (menú 6 opciones)
echo =======================================================
echo.

REM Lanzar y guardar PID al detectar puerto
start /B "" cmd /c "timeout /t 8 /nobreak >nul & for /f "tokens=5" %%%%a in ('netstat -ano 2^^^>nul ^| findstr /R /C "::%SERVER_PORT%.*LISTENING"') do @echo %%%%a > "%PID_FILE%""

"%PY_EXE%" -m streamlit run "%APP_FILE%" --server.headless false --server.address localhost --server.port %SERVER_PORT% --server.runOnSave true

if errorlevel 1 (
    echo.
    echo [ERROR] Streamlit no se pudo iniciar.
    echo   Solucion 1: Ejecuta  menu_sistema.bat  → Opción 4 (actualiza libs)
    echo   Solucion 2: Puerto 8501 ocupado? Ejecuta:  .\detener_sistema.bat
    echo   Solucion 3: Verifica tener internet (primer run instala paquetes).
)

del "%PID_FILE%" 2>nul
echo.
pause
exit /b 0

:MATAR_POR_PUERTO
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr /R /C:":%SERVER_PORT%.*LISTENING"') do (
    echo      → Matando proceso PID %%a (ocupa :%SERVER_PORT%).
    taskkill /PID %%a /T /F >nul 2>nul
)
exit /b 0

