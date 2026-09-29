"""설치 절차 본체. `setup.bat` 이 이 파일을 부른다.

한국어 안내가 배치 파일이 아니라 여기 있는 이유: cmd 의 `echo` 는 콘솔 코드페이지를
그대로 타기 때문에, 배치에 한국어를 적으면 그 PC 의 코드페이지가 949 냐 65001 이냐에 따라
갈린다. 어느 한쪽에 맞춰 저장하면 다른 쪽에서 깨진다. 파이썬은 콘솔에 유니코드를 직접
쓰므로 코드페이지가 무엇이든 같은 글자가 나온다. 그래서 배치는 ASCII 만 두고,
사람이 읽을 말은 전부 이 파일이 맡는다.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
PY = VENV / "Scripts" / "python.exe"
SPEC = ROOT / "solutions" / "vlm_parts" / "projects" / "01_parts" / "project.yaml"

CUDA_URL = "https://download.pytorch.org/whl/cu130"
# 학습과 실물 백본 추론에 필요한 것. (사람이 읽을 이름, 임포트 이름).
# torch 만 보면 안 된다 — transformers 가 없으면 굽기까지 가고 학습에서 죽는다.
TRAIN_DEPS = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),          # Qwen2-VL 프로세서가 비디오 처리기를 끌고 온다
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("huggingface_hub", "huggingface_hub"),
)
CPU_URL = "https://download.pytorch.org/whl/cpu"


def say(*parts: str) -> None:
    print(*parts, flush=True)


def run(cmd: list[str], quiet: bool = False) -> int:
    """없는 실행 파일은 예외가 아니라 0 이 아닌 값으로 돌려준다 — 호출부가 분기만 보게."""
    try:
        return subprocess.call(cmd, stdout=subprocess.DEVNULL if quiet else None)
    except (FileNotFoundError, OSError):
        return 127


def make_venv() -> bool:
    if PY.exists():
        say("[1/4] .venv 가 이미 있습니다. 건너뜁니다.")
        return True
    say("[1/4] .venv 를 만듭니다 (Python 3.12)...")
    if run(["uv", "venv", "--python", "3.12", str(VENV)]) == 0 and PY.exists():
        return True
    for base in (["py", "-3.12"], ["py", "-3"], [sys.executable]):
        if run(base + ["-m", "venv", str(VENV)]) == 0 and PY.exists():
            return True
    say("")
    say("  [오류] Python 3.12 를 찾을 수 없습니다.")
    say("         python.org 에서 설치하거나 uv 를 설치한 뒤 다시 실행하세요.")
    return False


def install() -> bool:
    say("")
    say("[2/4] 패키지를 설치합니다 (pyyaml, numpy, pillow, 편집기, vlmt 명령)...")
    run([str(PY), "-m", "ensurepip", "--upgrade"], quiet=True)
    run([str(PY), "-m", "pip", "install", "-q", "--upgrade", "pip"])
    # `[app]` 은 편집기(PySide6)다. 없으면 editor.bat 이 안내만 하고 멈춘다.
    # 헤드리스 서버라면 `pip install -e .` 로 이것만 빼면 된다 — 나머지는 전부 돈다.
    if run([str(PY), "-m", "pip", "install", "-q", "-e", str(ROOT) + "[app]"]) != 0:
        say("  [오류] 패키지 설치에 실패했습니다.")
        return False
    return True


def datasets() -> None:
    say("")
    say("[3/4] 예시 데이터를 만듭니다...")
    run([str(PY), str(ROOT / "tools" / "make_parts_dataset.py"), "--n", "48"])
    if (ROOT / "solutions/dummy_ecg/data/dummy/index.jsonl").exists():
        say("  dummy_ecg 데이터는 이미 있습니다.")
    else:
        run([str(PY), str(ROOT / "tools" / "make_dummy_dataset.py"), "--n", "24"])


def verify() -> None:
    say("")
    say("[4/4] 확인합니다...")
    run([str(PY), "-m", "vlm_trainer.cli.main", "compile", str(SPEC)])
    say("")
    missing = [name for name, mod in TRAIN_DEPS if run([str(PY), "-c", f"import {mod}"],
                                                       quiet=True) != 0]
    if not missing:
        say("  학습에 필요한 것이 전부 있습니다.")
        return

    # **torch 만 안내하면 4090 에서 굽기까지 가고 학습에서 죽는다.** 실물 백본은
    # transformers 를 쓰고, Qwen2-VL 프로세서는 torchvision 을 함께 끌고 온다.
    # 빠진 것을 전부 세어서 한 번에 알려 준다.
    say("-" * 44)
    say(f"  빠진 것: {', '.join(missing)}")
    say("  이것 없이도 편집기 / compile / dryrun / budget / run / materialize 는 돕니다.")
    say("  **학습과 실물 백본 추론은 안 됩니다.**")
    say("")
    say("  CUDA GPU 가 있다면 (드라이버에 맞는 cu 버전을 고르세요):")
    say(rf"    .venv\Scripts\python.exe -m pip install torch torchvision --index-url {CUDA_URL}")
    say(r"    .venv\Scripts\python.exe -m pip install transformers accelerate safetensors huggingface_hub")
    say("  CUDA 가 없다면 첫 줄만 이렇게:")
    say(rf"    .venv\Scripts\python.exe -m pip install torch torchvision --index-url {CPU_URL}")
    say("-" * 44)


def main() -> int:
    os.chdir(ROOT)
    say("=" * 44)
    say("  VLM Trainer 설치")
    say("=" * 44)
    say("")
    if not make_venv():
        return 1
    if not install():
        return 1
    datasets()
    verify()
    say("")
    say("끝났습니다. editor.bat 을 실행하면 편집기가 열립니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
