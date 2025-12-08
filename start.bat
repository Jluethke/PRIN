@echo off
REM =============================================================================
REM NeuroPRIN — Official Build Start Script (Jonathan’s Founder Build v1.2)
REM =============================================================================

setlocal ENABLEEXTENSIONS

REM --------------------
REM Required Python version (locked)
set REQUIRED_MAJOR=3
set REQUIRED_MINOR=11

REM --------------------
REM Check Python version (full semantic parsing)
for /f "tokens=2 delims= " %%i in ('python --version') do set PYTHON_VER=%%i
for /f "tokens=1,2,3 delims=." %%a in ("%PYTHON_VER%") do (
    set MAJOR=%%a
    set MINOR=%%b
    set PATCH=%%c
)

echo [INFO] Detected Python version: %MAJOR%.%MINOR%.%PATCH%

if not "%MAJOR%"=="%REQUIRED_MAJOR%" (
    goto version_error
)
if not "%MINOR%"=="%REQUIRED_MINOR%" (
    goto version_error
)

REM --------------------
REM Create venv if missing
if not exist "venv\Scripts\python.exe" (
    echo [1/6] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/6] Virtual environment already exists.
)

REM --------------------
REM Activate venv
echo [2/6] Activating virtual environment...
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo ERROR: Could not activate virtual environment.
    pause
    exit /b 1
)

REM --------------------
REM Upgrade pip
echo [3/6] Upgrading pip...
python -m pip install --upgrade pip

REM --------------------
REM Install pinned requirements
if exist "requirements.txt" (
    echo [4/6] Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo WARNING: requirements.txt not found. Skipping.
)

pip install "G:\othreflashdrive\projects2\PRIN-master\PRIN-master\PandasTA-v0.3.14b source code.tar.gz"


REM --------------------
REM Install NeuroPRIN local package in editable mode
echo [5/6] Installing NeuroPRIN package (editable mode)...
pip install -e .

REM --------------------
REM Launch the GUI
echo [6/6] Launching NeuroPRIN GUI...
python examples\gui.py

echo.
echo [DONE] NeuroPRIN GUI closed. Review logs if issues occurred.
pause
exit /b 0

REM --------------------
:version_error
echo.
echo [ERROR] Python %REQUIRED_MAJOR%.%REQUIRED_MINOR% is required. Found: Python %MAJOR%.%MINOR%.%PATCH%
echo Please install Python %REQUIRED_MAJOR%.%REQUIRED_MINOR%.x from https://www.python.org/downloads/
pause
exit /b 1
