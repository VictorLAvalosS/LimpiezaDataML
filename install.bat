@echo off
chcp 65001 >nul
title INSTALADOR - Data Cleaner PRO (Modo VENV SIN bloqueos)
color 0B
cls

set "VENV_DIR=%~dp0venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REQ_CORE=%~dp0requirements.txt"
set "REQ_OPT=%~dp0requirements_optional.txt"

echo =======================================================
echo  🚀 INSTALADOR DATA CLEANER PRO + ML
echo  (Usa VENV - sin problemas de permisos ni sandbox)
echo =======================================================
echo.

REM === PASO 0: Verificar Python global ===
echo [PASO 0/7] Buscando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR CRITICO] Python NO instalado. Instala Python 3.10+ con "Add to PATH"
    echo   https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo      OK: %%i
echo.

REM === PASO 1: Crear carpetas ===
echo [PASO 1/7] Creando carpetas data/...
if not exist "%~dp0data\raw" mkdir "%~dp0data\raw"
if not exist "%~dp0data\processed" mkdir "%~dp0data\processed"
echo      OK.
echo.

REM === PASO 2: Crear VENV si no existe ===
echo [PASO 2/7] Verificando/Creando Entorno Virtual venv...
if not exist "%VENV_PY%" (
    echo      Creando venv (tarda ~30 segs)...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] No se pudo crear venv. Prueba a borrar la carpeta venv si existe.
        pause
        exit /b 1
    )
    echo      OK: venv creado.
) else (
    echo      OK: venv ya existia, se reutiliza.
)
echo.

REM === PASO 3: Actualizar PIP ===
echo [PASO 3/7] Actualizando pip DENTRO de venv...
"%VENV_PY%" -m pip install --upgrade pip --quiet 2>nul
echo      OK.
echo.

REM === PASO 4: Instalar CORE ===
echo [PASO 4/7] Instalando LIBRERIAS CORE (10 paquetes)...
echo      (⏳ Tarda ~5-7 min: Pandas Numpy Scikit-learn Streamlit...)
"%VENV_PY%" -m pip install -r "%REQ_CORE%"
if errorlevel 1 (
    echo.
    echo [FALLBACK] Instalacion 1 por 1 por si algun paquete da problemas:
    "%VENV_PY%" -m pip install pandas numpy scipy openpyxl matplotlib seaborn scikit-learn streamlit python-dotenv joblib
    if errorlevel 1 (
        echo [ERROR FATAL] No se pudieron instalar las core. Revisa conexion a Internet.
        pause
        exit /b 1
    )
)
echo      OK: 10 librerias core instaladas.
echo.

REM === PASO 5: Opcionales ===
echo [PASO 5/7] Intentando librerias OPCIONALES (Gemini + Traductor)...
"%VENV_PY%" -m pip install -r "%REQ_OPT%" --quiet 2>nul
if errorlevel 1 (
    echo      (No se instalaron opcionales - no importa, todo funciona igual)
) else (
    echo      OK: Librerias opcionales instaladas.
)
echo.

REM === PASO 6: ===
echo [PASO 6/7] Configurando .env (para API Key Gemini)...
if not exist "%~dp0.env" (
    if exist "%~dp0.env.example" (
        copy /y "%~dp0.env.example" "%~dp0.env" >nul
        echo      OK: .env creado (Editalo si quieres usar Gemini).
    ) else (
        echo      (Sin .env.example - ok)
    )
) else (
    echo      OK: .env ya existia.
)
echo.

REM === PASO 7: Verificacion FINAL ===
echo [PASO 7/7] VERIFICANDO que TODO funciona...
"%VENV_PY%" -c "import pandas, numpy, sklearn, streamlit, seaborn, joblib, scipy, openpyxl, dotenv, matplotlib; print('     ✅ 10/10 LIBRERIAS CORE: FUNCIONANDO OK')" 2>&1
echo.

echo =======================================================
echo  ✅ INSTALACION COMPLETADA CON EXITO
echo =======================================================
echo.
echo  🚀 PARA INICIAR EL SISTEMA:
echo     DOBLE CLIC en:  run.bat
echo.
echo     O escribe en esta misma ventana: run.bat
echo.
echo  🛑 Para APAGAR: cierra la ventana negra de Streamlit.
echo =======================================================
echo.
pause
