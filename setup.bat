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

set "BOOT="
if exist "%PY%" set "BOOT=%PY%"
if not defined BOOT (
  py -3.12 -c "pass" >nul 2>&1 && set "BOOT=py -3.12"
)
if not defined BOOT (
  py -3 -c "pass" >nul 2>&1 && set "BOOT=py -3"
)
if not defined BOOT (
  python -c "pass" >nul 2>&1 && set "BOOT=python"
)
if not defined BOOT goto :nopy

%BOOT% "%~dp0tools\setup.py"
echo.
pause
exit /b 0

:nopy
echo [ERROR] Python not found.
echo         Install Python 3.12 from python.org, then run setup.bat again.
pause
exit /b 1
