@echo off
REM Dev launcher: the frontend starts the backend (8765) and guard (8756) itself.
REM Kept ASCII-only: cmd parses UTF-8 Chinese as GBK and breaks lines.
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
    echo [dev] .venv not found, running uv sync ...
    uv sync
    if errorlevel 1 exit /b 1
)

echo [dev] starting frontend ...
REM Run as a module so absolute src.* imports resolve from the project root.
.venv\Scripts\python.exe -m src.main
