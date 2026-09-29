@echo off
chcp 65001 >nul
title ╔═══ MENÚ DE CONTROL TOTAL ═══╗  Data Cleaner PRO + ML Suite
color 0A
mode con cols=100 lines=35
cls
setlocal enabledelayedexpansion

cd /d "%~dp0"
set "APP_DIR=%~dp0"
set "PID_FILE=%APP_DIR%data\.server_pid.txt"
set "PORT_FILE=%APP_DIR%data\.server_port.txt"
set "LOG_FILE=%APP_DIR%data\.server_log.txt"
set "SERVER_PORT=8501"
set "VENV_PY=%APP_DIR%venv\Scripts\python.exe"
set "APP_FILE=%APP_DIR%app.py"
if not exist "%APP_DIR%data" mkdir "%APP_DIR%data" 2>nul

:MENU
cls
echo.
echo  ╔══════════════════════════════════════════════════════════════════════════╗
echo  ║         🧹 DATA CLEANER PRO + MACHINE LEARNING SUITE  🧠                 ║
echo  ║                    MENÚ DE CONTROL TOTAL - Windows                       ║
echo  ╠══════════════════════════════════════════════════════════════════════════╣
echo  ║                                                                          ║
echo  ║   Estado del servidor:                                                    ║
if exist "%PID_FILE%" (
    set /p SPID=<"%PID_FILE%" 2>nul
    tasklist /FI "PID eq !SPID!" 2>nul | find /I "!SPID!" >nul 2>nul
    if !errorlevel! equ 0 (
        echo  ║      🟢 ✅ EN EJECUCIÓN  (PID = !SPID!   ·   Puerto %SERVER_PORT%)       ║
        echo  ║      🌐 URL:   http://localhost:%SERVER_PORT%                            ║
    ) else (
        echo  ║      🔴 ⛔ DETENIDO   (archivo PID antiguo encontrado, pero sin proceso) ║
        del "%PID_FILE%" 2>nul
    )
) else (
    echo  ║      🔴 ⛔ DETENIDO   (servidor no iniciado)                              ║
)
echo  ║                                                                          ║
echo  ╠══════════════════════════════════════════════════════════════════════════╣
echo  ║                                                                          ║
echo  ║   SELECCIONA UNA OPCIÓN (pulsa NÚMERO + ENTER):                           ║
echo  ║                                                                          ║
echo  ║      [ 1 ]  🚀 INICIAR  la APP Streamlit (Abre navegador web)            ║
echo  ║      [ 2 ]  🛑 DETENER  el servidor (sin cerrar esta ventana)            ║
echo  ║      [ 3 ]  🌐 ABRIR  el navegador en http://localhost:%SERVER_PORT%     ║
echo  ║      [ 4 ]  📦 INSTALAR / ACTUALIZAR  las librerías requirements.txt     ║
echo  ║      [ 5 ]  👀 VER ÚLTIMOS 20 LOGS  del servidor (si hay)                ║
echo  ║      [ 6 ]  ❌ SALIR  de este menú                                        ║
echo  ║                                                                          ║
echo  ╚══════════════════════════════════════════════════════════════════════════╝
echo.
set /p "OPT=  Tu opción [1-6]: "

if "%OPT%"=="1" goto INICIAR
if "%OPT%"=="2" goto DETENER
if "%OPT%"=="3" goto ABRIR
if "%OPT%"=="4" goto INSTALAR
if "%OPT%"=="5" goto VERLOGS
if "%OPT%"=="6" goto SALIR
echo.
echo  ❌ Opción no válida, introduce 1, 2, 3, 4, 5 o 6.
pause >nul
goto MENU

:INICIAR
cls
echo.
echo  [1/5] Comprobando Python VENV...
if not exist "%VENV_PY%" (
    echo      ⚠️  VENV no detectado. Intentando con python GLOBAL del sistema...
    where python >nul 2>nul
    if !errorlevel! neq 0 (
        echo      ❌ ERROR: No hay Python (ni venv ni global). Ejecuta install.bat primero.
        pause & goto MENU
    )
    set "PY_EXE=python"
) else (
    set "PY_EXE=%VENV_PY%"
)
echo      OK.

echo  [2/5] Comprobando puerto %SERVER_PORT%...
netstat -ano | find "LISTENING" | find ":%SERVER_PORT%" >nul 2>nul
if !errorlevel! equ 0 (
    echo      ⚠️  Puerto %SERVER_PORT% OCUPADO — vamos a DETENER el proceso anterior...
    call :MATAR_POR_PUERTO %SERVER_PORT%
    timeout /t 2 /nobreak >nul
) else (
    echo      Libre ✔
)

echo  [3/5] Instalación rápida de librerías si faltan (modo silencioso)...
"%PY_EXE%" -m pip install -r "%APP_DIR%requirements.txt" --quiet --no-warn-script-location 2>nul
echo      OK.

echo  [4/5] Guardando variables de control (PID / Port)...
echo %SERVER_PORT% >"%PORT_FILE%"
echo      Puerto = %SERVER_PORT%

echo  [5/5] 🚀 LANZANDO Streamlit en NUEVA VENTANA (no bloqueas este menú)...
echo.
echo      🌐 En ~5 segundos abre:  http://localhost:%SERVER_PORT%
echo      🛑 Para parar: Vuelve a este menú → pulsa 2, o cierra la ventana nueva.
echo.

start "Streamlit Data Cleaner (Cierra esta ventana para DETENER)" /MIN cmd /k "chcp 65001 >nul & cd /d ""%APP_DIR%"" & title 🟢 Streamlit corriendo — Ctrl+C o cierra para DETENER & echo 🧹 Iniciando Data Cleaner PRO + ML... & echo. & echo 🛑 PARA PARAR: Ctrl+C O cierra esta ventana. & echo. & ""%PY_EXE%"" -m streamlit run ""%APP_FILE%"" --server.headless false --server.address localhost --server.port %SERVER_PORT% --server.runOnSave true > ""%LOG_FILE%"" 2>&1 & echo. & echo ✅ Proceso finalizado. & pause"

timeout /t 2 /nobreak >nul
call :GRABAR_PID
echo.
echo      ✅ Servidor lanzado en SEGUNDO PLANO. Este menú sigue disponible.
timeout /t 3 /nobreak >nul
goto ABRIR_OPCIONAL

:ABRIR_OPCIONAL
cls
echo.
echo  🌐 ¿Quieres abrir YA el navegador en http://localhost:%SERVER_PORT%  ?
echo      Pulsa S = Sí   ·   N = No (volver al menú)
set /p "ABR=N (S/N): "
if /I "%ABR%"=="S" goto ABRIR
goto MENU

:DETENER
cls
echo.
echo  🛑 DETENIENDO el servidor Streamlit...
echo.
if exist "%PID_FILE%" (
    set /p SPID=<"%PID_FILE%" 2>nul
    echo  🎯 1) Intentando parar por PID guardado (PID=!SPID!)...
    tasklist /FI "PID eq !SPID!" 2>nul | find /I "!SPID!" >nul 2>nul
    if !errorlevel! equ 0 (
        taskkill /PID !SPID! /T /F >nul 2>nul
        echo      ✅ Proceso !SPID! terminado.
    ) else (
        echo      ℹ️  PID no activo, saltando.
    )
    del "%PID_FILE%" 2>nul
) else (
    echo  ℹ️  No había archivo PID guardado.
)

echo  🎯 2) Búsqueda SEGURA por PUERTO %SERVER_PORT% (fallback)...
call :MATAR_POR_PUERTO %SERVER_PORT%

echo  🎯 3) Búsqueda por NOMBRE (streamlit / python que ejecuta streamlit)...
for /f "tokens=2" %%a in ('tasklist /v /FI "WINDOWTITLE eq Streamlit*" /FO list 2^>nul ^| findstr /I "PID:"') do (
    taskkill /PID %%a /T /F >nul 2>nul
    echo      ✅ Proceso %%a (ventana Streamlit) terminado.
)

del "%PID_FILE%" 2>nul
echo.
echo  ✅ Operación completada: Servidor DETENIDO.
echo  ℹ️  Si seguía abierta la ventana del servidor, se cerró automáticamente.
pause
goto MENU

:ABRIR
cls
echo.
echo  🌐 Abriendo http://localhost:%SERVER_PORT% en tu navegador predeterminado...
start "" "http://localhost:%SERVER_PORT%"
timeout /t 1 /nobreak >nul
goto MENU

:INSTALAR
cls
echo.
echo  📦 INSTALANDO / ACTUALIZANDO librerías desde requirements.txt...
echo.
if exist "%VENV_PY%" (set "PY_EXE=%VENV_PY%") else (set "PY_EXE=python")
"%PY_EXE%" -m pip install --upgrade pip --quiet --no-warn-script-location 2>nul
"%PY_EXE%" -m pip install -r "%APP_DIR%requirements.txt" --upgrade --no-warn-script-location
echo.
echo  ✅ Librerías actualizadas.
pause
goto MENU

:VERLOGS
cls
echo.
echo  👀 ÚLTIMAS 20 LÍNEAS de los logs del servidor (archivo %LOG_FILE%):
echo  ════════════════════════════════════════════════════════════════════════════
if exist "%LOG_FILE%" (
    powershell -NoProfile -Command "Get-Content '%LOG_FILE%' -Tail 20"
) else (
    echo  ℹ️  (todavía no hay logs — inicia primero el servidor con opción 1)
)
echo  ════════════════════════════════════════════════════════════════════════════
echo.
pause
goto MENU

:GRABAR_PID
REM Intentamos durante 8s detectar el PID que abre el puerto 8501
for /L %%i in (1,1,16) do (
    timeout /t 1 /nobreak >nul
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:":%SERVER_PORT%.*LISTENING"') do (
        echo %%a >"%PID_FILE%"
        exit /b 0
    )
)
exit /b 0

:MATAR_POR_PUERTO
REM Param 1 = puerto
set "PUERTO_KILL=%1"
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr /R /C:":!PUERTO_KILL!.*LISTENING"') do (
    echo      🎯 Matando proceso PID %%a (ocupa puerto !PUERTO_KILL!)...
    taskkill /PID %%a /T /F >nul 2>nul
)
exit /b 0

:SALIR
cls
echo.
echo  👋 Saliendo del menú de control.
echo     Nota: si el servidor estaba en ejecución SIGUE CORRIENDO (para detenerlo,
echo           vuelve a ejecutar  menu_sistema.bat  → opción 2).
echo.
timeout /t 2 /nobreak >nul
endlocal
exit /b 0
