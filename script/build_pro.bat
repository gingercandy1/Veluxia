@echo off
rem Refresh UI translations: extract tr() strings into .ts, then compile .qm.
rem After running, translate new entries in Qt Linguist (pyside6-linguist.exe), then run again.
cd /d "%~dp0.."

set BIN=.venv\Scripts
set TS=resource\translations

%BIN%\pyside6-lupdate.exe -extensions py src\main.py src\app -no-obsolete -locations relative ^
    -ts %TS%\app_zh_CN.ts %TS%\app_en_US.ts %TS%\app_ja_JP.ts %TS%\app_ko_KR.ts %TS%\app_ru_RU.ts %TS%\app_es_ES.ts
if errorlevel 1 exit /b 1

for %%f in (%TS%\*.ts) do (
    %BIN%\pyside6-lrelease.exe %%f
    if errorlevel 1 exit /b 1
)
