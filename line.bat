@echo off
setlocal
cd /d "%~dp0"
rem Double-click launcher for the 5-node screen. ASCII on purpose (see editor.bat).
rem Keep CRLF line endings; .gitattributes pins them.
rem   no argument      : asks for a folder. New folder -> empty project, existing -> opens it.
rem   line.yaml or dir : drag it onto this file, or pass it as the first argument.
rem Everything (prepare, pre-run, train, verify) is run by the Run button in the window.
set "PYTHONUTF8=1"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" goto :noenv

set "DIR=%~1"
if not "%DIR%"=="" goto :have
echo.
echo Project folder (blank = bundled example):
set /p DIR=folder path (ex: C:\Work\my_line):
if "%DIR%"=="" set "DIR=solutions\vlm_open\projects\01_open"

:have
if /i "%DIR:~-5%"==".yaml" goto :open
if exist "%DIR%\line.yaml" goto :openDir
set /p NAME=new project name:
"%PY%" -m vlm_trainer.cli.main new-line "%DIR%" --name "%NAME%" --edit
goto :done

:openDir
set "DIR=%DIR%\line.yaml"
:open
"%PY%" -m vlm_trainer.cli.main edit "%DIR%"

:done
if errorlevel 1 pause
exit /b %errorlevel%

:noenv
echo [ERROR] .venv not found: %PY%
echo         Run setup.bat first.
pause
exit /b 1
