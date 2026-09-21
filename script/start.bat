@echo off
REM Start backend (port 8765), then frontend; stop backend when frontend exits.
REM Kept ASCII-only: cmd parses UTF-8 Chinese as GBK and breaks lines.
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
    echo [start] .venv not found, running uv sync ...
    uv sync
    if errorlevel 1 exit /b 1
)

echo [start] starting backend on :8765 ...
start "veluxia-backend" /min .venv\Scripts\python.exe -m src.backend.server --port 8765

echo [start] starting frontend ...
REM Run as a module so absolute src.* imports resolve from the project root.
.venv\Scripts\python.exe -m src.main
set FRONT_EXIT=%errorlevel%

echo [start] frontend exited, stopping backend ...
taskkill /fi "WINDOWTITLE eq veluxia-backend*" /t /f >nul 2>&1

exit /b %FRONT_EXIT%
