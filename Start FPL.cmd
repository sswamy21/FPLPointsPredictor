@echo off
setlocal
cd /d "%~dp0"
set "FPL_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%~dp0..\..\work\fpl-venv\Scripts\python.exe" set "FPL_PYTHON=%~dp0..\..\work\fpl-venv\Scripts\python.exe"
if not exist "%FPL_PYTHON%" (
    python -m venv .venv
    if errorlevel 1 goto failed
)
"%FPL_PYTHON%" -c "import flask, joblib, numpy, pandas, requests, sklearn" >nul 2>nul
if errorlevel 1 (
    "%FPL_PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 goto failed
)
"%FPL_PYTHON%" -m fpl serve
if not errorlevel 1 exit /b 0
:failed
echo.
echo The FPL dashboard could not start. See the messages above.
pause
exit /b 1
