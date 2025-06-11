@echo off
REM =============================================================================
REM start.bat — Create & activate Python venv, install deps, register pkg, launch GUI
REM Usage: Double-click or run from CMD/PowerShell in project root.
REM =============================================================================

REM 1. Create venv if missing
if not exist "venv\Scripts\python.exe" (
    echo [1/6] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/6] Virtual environment already exists.
)

REM 2. Activate venv
echo [2/6] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate via activate.bat; trying PowerShell...
    powershell -ExecutionPolicy Bypass -Command "Set-ExecutionPolicy Bypass -Scope Process; .\venv\Scripts\Activate.ps1"
    if errorlevel 1 (
        echo ERROR: Could not activate virtual environment.
        pause
        exit /b 1
    )
)

REM 3. Upgrade pip
echo [3/6] Upgrading pip...
python -m pip install --upgrade pip

REM 4. Install requirements.txt
if exist "requirements.txt" (
    echo [4/6] Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo WARNING: requirements.txt not found. Skipping.
)

REM 5. Install the local neuroprin package in editable mode
echo [5/6] Installing neuroprin package (editable)...
pip install -e .

REM 6. Launch the GUI
echo [6/6] Launching NeuroPRIN GUI...
python examples\gui.py

echo.
echo Done! If the GUI didn’t stay open, check for errors above.
pause
