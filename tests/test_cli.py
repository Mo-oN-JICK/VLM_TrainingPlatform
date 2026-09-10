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
