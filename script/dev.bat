@echo off
REM Veluxia 一键开发启动：后端(8765) + 前端
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
    echo [dev] 未找到 .venv，执行 uv sync ...
    uv sync
    if errorlevel 1 exit /b 1
)

echo [dev] 启动后端 :8765 ...
start "veluxia-backend" .venv\Scripts\python.exe -m src.backend.server --port 8765

echo [dev] 启动前端 ...
.venv\Scripts\python.exe src\main.py
