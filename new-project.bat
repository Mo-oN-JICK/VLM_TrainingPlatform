@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"

if not exist "%PY%" goto :noenv

set "DIR=%~1"
if "%DIR%"=="" set /p DIR=만들 폴더 경로 (예: C:\Work\my_task): 
if "%DIR%"=="" exit /b 1
set "NAME=%~2"
if "%NAME%"=="" set /p NAME=과제 이름: 

"%PY%" -m vlm_trainer.cli.main new "%DIR%" --name "%NAME%" --edit
if errorlevel 1 pause
exit /b 0

:noenv
echo [오류] .venv 를 찾을 수 없습니다: %PY%
pause
exit /b 1
