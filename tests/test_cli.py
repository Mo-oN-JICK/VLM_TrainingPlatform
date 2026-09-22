"""CLI가 사람에게 말하는 방식.

게이트 에러는 이미 형식이 정해져 있다. 여기서 보는 것은 **게이트가 아닌 실패**다 —
노드가 계약을 어기고 일반 예외를 던졌을 때 CLI가 트레이스백을 뱉지 않아야 한다.
"""

from __future__ import annotations

import os

import pytest

from vlm_trainer.cli.main import main

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPEC = os.path.join(ROOT, "solutions", "dummy_ecg", "projects", "01_dummy", "project.yaml")

pytestmark = pytest.mark.skipif(not os.path.exists(SPEC), reason="예시 Solution이 없다")


def test_a_bad_set_value_is_not_a_traceback(capsys):
    """편집기는 이 경우를 잡는데 CLI만 날것으로 터지고 있었다."""
    code = main(["compile", SPEC, "--set", "n_plot.size=64"])
    assert code == 2

    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "TypeError" in err, "무엇이 터졌는지는 여전히 보여야 한다"
    assert "--set" in err and "n_plot.size=" in err, "고치는 법을 말해야 한다"


def test_a_gate_error_keeps_its_own_format(capsys):
    """게이트 에러는 위 안내를 덧붙이지 않는다 — 이미 네 가지를 말하고 있다."""
    code = main(["compile", SPEC, "--set", "n_img.color_space=BGR"])
    assert code == 2

    err = capsys.readouterr().err
    assert "레시피가 덮어쓸 수 없다" in err and "허용:" in err
    assert "VLMT_TRACE" not in err


def test_a_good_value_still_compiles(capsys):
    assert main(["compile", SPEC, "--set", "n_plot.size=[640,320]"]) == 0
    assert "compile OK" in capsys.readouterr().out


def test_the_core_runs_without_the_editor(monkeypatch):
    """**PySide6 없이도 전부 돌아야 한다.**

    `CLAUDE.md` 는 코어가 표준 라이브러리 + yaml 만 쓰기를 요구한다. 편집기는 선택
    사항(`pip install -e .[app]`)이고, 학습만 돌리는 PC 나 헤드리스 서버에는 있을
    이유가 없다. 어딘가에서 최상위 임포트가 하나만 생겨도 이 경계가 무너진다.
    """
    import builtins
    import importlib

    real = builtins.__import__

    def no_qt(name, *a, **k):
        if name == "PySide6" or name.startswith("PySide6."):
            raise ImportError("PySide6 없음 (테스트가 막았다)")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_qt)
    for mod in ("vlm_trainer.cli.main", "vlm_trainer.ui.render",
                "vlm_trainer.ui.layout", "vlm_trainer.ui.api",
                "vlm_trainer.engine.runner", "vlm_trainer.engine.materialize"):
        importlib.reload(importlib.import_module(mod))

    from vlm_trainer.ui import app as app_mod

    assert not app_mod.available()
    with pytest.raises(SystemExit) as e:
        app_mod.launch("nope.yaml")
    assert "pip install" in str(e.value)
