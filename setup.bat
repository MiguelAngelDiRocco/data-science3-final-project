@echo off
REM ============================================
REM Script de Setup para Windows
REM Proyecto: NLP Sentiment Analysis
REM ============================================

echo.
echo ========================================
echo   SETUP DEL PROYECTO
echo   NLP Sentiment Analysis
echo ========================================
echo.

REM Verificar Python
echo [1/6] Verificando Python...
python --version
if errorlevel 1 (
    echo ERROR: Python no esta instalado o no esta en PATH
    echo Por favor, instala Python 3.10+ desde python.org
    pause
    exit /b 1
)
echo OK: Python encontrado
echo.

REM Crear ambiente virtual
echo [2/6] Creando ambiente virtual...
if exist venv (
    echo Ambiente virtual ya existe
) else (
    python -m venv venv
    echo OK: Ambiente virtual creado
)
echo.

REM Activar ambiente virtual
echo [3/6] Activando ambiente virtual...
call venv\Scripts\activate.bat
echo OK: Ambiente virtual activado
echo.

REM Actualizar pip
echo [4/6] Actualizando pip...
python -m pip install --upgrade pip
echo OK: pip actualizado
echo.

REM Instalar dependencias
echo [5/6] Instalando dependencias...
pip install -r requirements.txt
echo OK: Dependencias instaladas
echo.

REM Descargar recursos NLTK
echo [6/6] Descargando recursos de NLTK...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')"
echo OK: Recursos NLTK descargados
echo.

echo ========================================
echo   SETUP COMPLETADO EXITOSAMENTE
echo ========================================
echo.
echo Proximos pasos:
echo 1. Coloca el archivo Reviews.csv en: data\raw\
echo 2. Abre VSCode en esta carpeta
echo 3. Ejecuta el notebook: notebooks\00_Setup_and_DataLoad.ipynb
echo.
echo Para activar el ambiente virtual en el futuro:
echo    venv\Scripts\activate
echo.
pause
