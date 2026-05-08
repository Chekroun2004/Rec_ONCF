@echo off
:: Daily ONCF retrain job — called by Windows Task Scheduler
:: Logs stdout+stderr to logs\retrain_YYYY-MM-DD.log

setlocal

set PROJECT_ROOT=%~dp0..
set LOGDIR=%PROJECT_ROOT%\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

:: Build log filename with today's date (YYYY-MM-DD)
for /f "tokens=1-3 delims=-" %%a in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do (
    set TODAY=%%a-%%b-%%c
)
set LOGFILE=%LOGDIR%\retrain_%TODAY%.log

echo [%DATE% %TIME%] Starting ONCF retrain >> "%LOGFILE%"

"%PROJECT_ROOT%\.venv\Scripts\python.exe" "%PROJECT_ROOT%\scripts\07_retrain.py" >> "%LOGFILE%" 2>&1
set EXIT_CODE=%ERRORLEVEL%

echo [%DATE% %TIME%] Retrain finished (exit code %EXIT_CODE%) >> "%LOGFILE%"

:: Keep only the last 30 log files (rotate)
for /f "skip=30 delims=" %%f in ('dir /b /o-d "%LOGDIR%\retrain_*.log" 2^>nul') do del "%LOGDIR%\%%f" >nul 2>&1

exit /b %EXIT_CODE%
