@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
set "SPEC=solutions\vlm_parts\projects\01_parts\project.yaml"

if not exist "%PY%" goto :noenv

echo === 1) 테스트 ===
"%PY%" -m pytest tests -q
echo.
echo === 2) 네 게이트 ===
"%PY%" -m vlm_trainer.cli.main compile %SPEC%
"%PY%" -m vlm_trainer.cli.main dryrun  %SPEC%
"%PY%" -m vlm_trainer.cli.main budget  %SPEC%
echo.
pause
exit /b 0

:noenv
echo [오류] .venv 를 찾을 수 없습니다: %PY%
pause
exit /b 1
