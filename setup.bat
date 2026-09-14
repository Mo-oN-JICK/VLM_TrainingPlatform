@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "VENV=%~dp0.venv"
set "PY=%VENV%\Scripts\python.exe"

echo ============================================
echo   VLM Trainer 설치
echo ============================================
echo.

if exist "%PY%" (
  echo [1/4] .venv 가 이미 있습니다. 건너뜁니다.
  goto :deps
)
echo [1/4] .venv 를 만듭니다 (Python 3.12)...
where uv >nul 2>&1
if errorlevel 1 (
  py -3.12 -m venv "%VENV%"
  if errorlevel 1 goto :nopy
) else (
  uv venv --python 3.12 "%VENV%"
)
"%PY%" -m ensurepip --upgrade >nul 2>&1

:deps
echo.
echo [2/4] 패키지를 설치합니다 (pyyaml, numpy, pillow, vlmt 명령)...
"%PY%" -m pip install -q --upgrade pip
"%PY%" -m pip install -q -e .
if errorlevel 1 goto :failinstall

echo.
echo [3/4] 예시 데이터를 만듭니다...
"%PY%" tools\make_parts_dataset.py --n 48
if exist "solutions\dummy_ecg\data\dummy\index.jsonl" (
  echo   dummy_ecg 데이터는 이미 있습니다.
) else (
  "%PY%" tools\make_dummy_dataset.py --n 24
)

echo.
echo [4/4] 확인합니다...
"%PY%" -m vlm_trainer.cli.main compile solutions\vlm_parts\projects\01_parts\project.yaml
echo.

"%PY%" -c "import torch" 2>nul
if errorlevel 1 goto :notorch
echo   torch 확인됨. 학습까지 가능합니다.
goto :done

:notorch
echo --------------------------------------------
echo   torch 가 없습니다. 학습을 뺀 나머지는 전부 동작합니다.
echo   (편집기 / compile / dryrun / budget / run / materialize)
echo.
echo   학습까지 하려면 자신의 CUDA 에 맞는 torch 를 설치하세요:
echo     .venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130
echo   CUDA 가 없다면:
echo     .venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
echo --------------------------------------------

:done
echo.
echo 끝났습니다. editor.bat 을 실행하면 편집기가 열립니다.
pause
exit /b 0

:nopy
echo.
echo   [오류] Python 3.12 를 찾을 수 없습니다.
echo          python.org 에서 설치하거나 uv 를 설치한 뒤 다시 실행하세요.
pause
exit /b 1

:failinstall
echo   [오류] 패키지 설치에 실패했습니다.
pause
exit /b 1
