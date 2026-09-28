"""`.bat` 이 부르는 명령이 실제로 파서를 통과하는가.

`editor.bat` 이 `vlmt edit <spec> --open` 을 부르고 있었다. `--open` 은 웹 편집기 시절의
옵션이고 지금 파서는 그것을 거부한다. 그래서 **편집기가 한 번도 열리지 않았다** —
`setup.bat` 이 끝나며 "editor.bat 을 실행하면 편집기가 열립니다" 라고 말하는데.

첫 실행에서 깨지는 것이 제일 나쁘다. 파이썬 쪽은 테스트가 촘촘한데 `.bat` 은 아무도
안 봤다. 여기서 본다.
"""

from __future__ import annotations

import io
import os
import re
import shlex

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BATS = ["setup.bat", "editor.bat", "new-project.bat", "check.bat"]


def _lines(name: str):
    """줄 끝이 무엇이든 줄 단위로 본다. **줄 끝 자체는 아래의 CRLF 검사가 맡는다** —
    여기서 CRLF 만 인정하면 LF 파일이 한 줄로 뭉쳐서, 엉뚱한 검사가 엉뚱한 이유로
    터진다. 검사는 제 이유로 터져야 읽는 사람이 원인을 안다."""
    return io.open(os.path.join(ROOT, name), encoding="utf-8").read().splitlines()


def _invocations(name: str):
    """`vlm_trainer.cli.main` 을 부르는 줄에서 인자만 뽑는다.
    `%SPEC%` 같은 변수는 자리표시자로 바꾼다 — 파서가 보는 것은 **모양**이다."""
    out = []
    for line in _lines(name):
        if "vlm_trainer.cli.main" not in line:
            continue
        args = line.split("vlm_trainer.cli.main", 1)[1]
        args = re.sub(r'"?%~?[A-Za-z0-9_]+%?"?', "PLACEHOLDER", args)
        out.append(shlex.split(args.replace("\\", "/")))
    return out


@pytest.mark.parametrize("name", BATS)
def test_every_command_a_bat_runs_is_one_the_parser_accepts(name):
    """이것이 없어서 `--open` 이 살아남았다."""
    from vlm_trainer.cli.main import build_parser

    for args in _invocations(name):
        assert args, f"{name}: 빈 명령"
        try:
            build_parser().parse_args(args)
        except SystemExit as exc:
            pytest.fail(f"{name}: `vlmt {' '.join(args)}` 를 파서가 거부한다 (exit {exc.code})")


@pytest.mark.parametrize("name", BATS)
def test_every_spec_path_a_bat_names_exists(name):
    """기본 스펙이 없는 파일을 가리키면 두 번째로 나쁜 첫 실행이 된다."""
    missing = []
    for line in _lines(name):
        for m in re.finditer(r"solutions[\\/][^\s\"]+\.yaml", line):
            p = os.path.join(ROOT, m.group(0).replace("\\", os.sep))
            if not os.path.exists(p):
                missing.append(m.group(0))
    assert not missing, f"{name} 이 가리키는데 없는 스펙: {missing}"


@pytest.mark.parametrize("name", BATS)
def test_the_bat_files_stay_ascii_and_crlf(name):
    """한글을 넣으면 콘솔 코드페이지(949/65001)에 따라 깨진다 — 4090 PC 에서 겪었다.
    한국어는 파이썬이 출력한다. 줄 끝이 LF 로 바뀌면 cmd 가 `set "PY=..."` 를
    잘못 읽는다(`.gitattributes` 가 `*.bat -text` 로 고정한다)."""
    raw = io.open(os.path.join(ROOT, name), "rb").read()
    try:
        raw.decode("ascii")
    except UnicodeDecodeError as e:
        pytest.fail(f"{name}: ASCII 가 아닌 바이트 — {e}")
    assert b"\r\n" in raw, f"{name}: CRLF 가 아니다"
    assert not re.search(rb"(?<!\r)\n", raw), f"{name}: LF 만 있는 줄이 섞였다"


@pytest.mark.parametrize("name", BATS)
def test_no_bat_changes_the_console_code_page(name):
    """`chcp` 는 파일 중간에서 cmd 의 읽기 위치를 옮긴다. 실제로 이것 때문에
    `"PY"` 가 명령으로 해석돼 첫 실행이 실패했다."""
    # 주석(`rem`)은 세지 않는다 — 이 파일들은 "chcp 를 넣지 마라" 고 적어 두고 있다.
    live = [ln for ln in _lines(name)
            if ln.strip() and not ln.strip().lower().startswith("rem")]
    guilty = [ln for ln in live if "chcp" in ln.lower()]
    assert not guilty, f"{name}: chcp 가 실행된다 — {guilty}"


def test_setup_points_at_a_bat_that_works():
    """`setup.bat` 이 끝나며 다음에 무엇을 누르라고 말한다. 그것이 깨져 있으면
    설치를 막 끝낸 사람이 첫 화면을 못 본다."""
    tools = io.open(os.path.join(ROOT, "tools", "setup.py"), encoding="utf-8").read()
    named = set(re.findall(r"([a-z-]+\.bat)", tools))
    assert named, "setup.py 가 다음에 무엇을 누를지 말하지 않는다"
    for b in named:
        assert b in BATS, f"setup.py 가 없는 {b} 를 안내한다"
