@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"

if not exist "%PY%" goto :noenv

set "SPEC=%~1"
if "%SPEC%"=="" set "SPEC=solutions\vlm_parts\projects\01_parts\project.yaml"
if not exist "%SPEC%" goto :nospec

echo.
echo   프로젝트: %SPEC%
echo   이 창이 서버입니다. 닫으면 편집기가 꺼집니다.
echo.
"%PY%" -m vlm_trainer.cli.main edit "%SPEC%" --open
if errorlevel 1 pause
exit /b 0

:noenv
echo [오류] .venv 를 찾을 수 없습니다: %PY%
echo        uv venv --python 3.12 .venv
echo        .venv\Scripts\python.exe -m pip install -e .
pause
exit /b 1

:nospec
echo [오류] 프로젝트 파일이 없습니다: %SPEC%
echo        새로 만들려면 "새 프로젝트.bat" 을 실행하세요.
pause
exit /b 1
