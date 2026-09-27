@echo off
REM Start the frontend; it launches the backend (8765) and guard (8756) itself
REM and stops them when it exits. Starting the backend here too made the frontend
REM see the port as taken and kill it.
REM Kept ASCII-only: cmd parses UTF-8 Chinese as GBK and breaks lines.
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
    echo [start] .venv not found, running uv sync ...
    uv sync
    if errorlevel 1 exit /b 1
)

echo [start] starting frontend ...
REM Run as a module so absolute src.* imports resolve from the project root.
.venv\Scripts\python.exe -m src.main
