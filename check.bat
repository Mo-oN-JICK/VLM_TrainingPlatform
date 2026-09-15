@echo off
setlocal
cd /d "%~dp0"
rem This file is ASCII on purpose. Korean text is printed by Python, which
rem writes Unicode to the console no matter what the code page is (949 or 65001).
rem Do NOT add `chcp` here: it shifts cmd's parse position in the middle of the file.
rem Keep CRLF line endings; .gitattributes pins them.
rem Output redirected to a file would otherwise use the cp949 locale and die on
rem characters it cannot encode. This only affects files/pipes; the console is
rem written through the wide API and renders the same either way.
set "PYTHONUTF8=1"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" goto :noenv
set "SPEC=solutions\vlm_parts\projects\01_parts\project.yaml"

echo === 1) tests ===
"%PY%" -m pytest tests -q
echo.
echo === 2) gates ===
"%PY%" -m vlm_trainer.cli.main compile "%SPEC%"
"%PY%" -m vlm_trainer.cli.main dryrun  "%SPEC%"
"%PY%" -m vlm_trainer.cli.main budget  "%SPEC%"
echo.
pause
exit /b 0

:noenv
echo [ERROR] .venv not found: %PY%
echo         Run setup.bat first.
pause
exit /b 1
