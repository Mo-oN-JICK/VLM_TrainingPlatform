"""실행 따라가기 — 지금 도는 노드가 어디인지, 거기서 얼마나 쓰고 있는지.

세 가지를 지킨다.

1. 진행 파일이 **지금 도는 노드**를 가리킨다 (화면이 그쪽으로 옮겨 가는 근거다)
2. 끝난 노드에 **건수와 걸린 시간**이 붙는다
3. 도는 노드에 **이번에 들어가서 흐른 시간**과 누계가 함께 붙는다

**시간 간격만으로 기록을 거르면 이것이 정확히 반대로 동작한다.** 느린 노드는 앞 기록
직후에 시작하므로 간격 조건에 늘 탈락하고, 그 노드가 만든 공백 덕에 그 뒤의 빠른
노드가 기록된다. 실측으로 겪었다 — `n_infer` 가 샘플마다 1.5초를 먹는데 진행 파일은
한 번도 그것을 가리키지 않고 4ms 걸리는 `n_answers` 만 가리켰다. 보고 싶은 노드가
유일하게 안 보였다.
"""

from __future__ import annotations

import time

import pytest

from vlm_trainer.engine import runner as R
from vlm_trainer.ui import layout as L


def _report(**kw):
    rep = R.RunReport()
    for k, v in kw.items():
        setattr(rep, k, v)
    return rep


# ── 어느 노드를 가리키는가 ──────────────────────────────────────────────
def test_a_slow_node_is_always_worth_writing():
    """느린 노드가 도는 동안 화면이 멈춰 보이지 않아야 한다. 이 노드가 **간격 조건과
    무관하게** 기록되어야 하는 이유다."""
    rep = _report(node_ms={"n_infer": 15_000.0},
                  node_state={"n_infer": {"success": 10}})       # 평균 1.5초
    assert R._worth_writing(rep, "n_infer", every=0.4)


def test_a_fast_node_is_left_to_the_timer():
    """4ms 걸리는 노드를 매번 기록하면 기록 횟수만 늘고 볼 것은 없다."""
    rep = _report(node_ms={"n_answers": 40.0},
                  node_state={"n_answers": {"success": 10}})     # 평균 4ms
    assert not R._worth_writing(rep, "n_answers", every=0.4)


def test_a_node_never_seen_counts_as_slow():
    """모르는 것을 빠르다고 가정하면 첫 샘플에서 무거운 노드가 그대로 안 보인다.
    시작할 때 화면이 한 번은 채워져야 한다."""
    assert R._worth_writing(_report(), "처음보는노드", every=0.4)


def test_the_gap_a_slow_node_makes_is_not_credited_to_the_next_node():
    """이것이 실제로 있었던 버그다. `n_infer`(1.5초)가 끝나면 `n_answers`(4ms)가
    시작할 때 간격 조건이 충족되어 있고, 그래서 **빠른 노드만** 기록됐다."""
    rep = _report(node_ms={"n_infer": 15_000.0, "n_answers": 40.0},
                  node_state={"n_infer": {"success": 10}, "n_answers": {"success": 10}})
    assert R._worth_writing(rep, "n_infer", every=0.4)
    assert not R._worth_writing(rep, "n_answers", every=0.4), (
        "느린 노드가 만든 공백이 그다음 빠른 노드의 기록 근거가 되고 있다")


def test_the_snapshot_carries_the_running_node_only_while_running():
    """끝난 실행이 여전히 어딘가를 가리키면 화면이 다 끝난 그래프를 따라간다."""
    rep = _report(active="n_infer", active_since=time.perf_counter() - 1.2,
                  node_ms={"n_infer": 1000.0})
    live = R.snapshot(rep, run_id="r", total=10, phase="running")
    assert live["active"] == "n_infer" and live["active_ms"] >= 1000

    done = R.snapshot(rep, run_id="r", total=10, phase="done")
    assert done["active"] == "" and done["active_ms"] == 0


def test_the_elapsed_time_survives_the_round_trip():
    """앱은 진행 파일만 읽는다. 여기서 시작 시각이 사라지면 도는 노드의 숫자가 0에
    붙어 있고, 보는 사람은 멈춘 줄로 안다."""
    rep = _report(active="n_infer", active_since=time.perf_counter() - 2.0)
    back = R.report_from_snapshot(R.snapshot(rep, run_id="r", total=10, phase="running"))
    assert back.active == "n_infer"
    assert 1.5 <= (time.perf_counter() - back.active_since) <= 2.5


# ── 카드에 무엇이 적히는가 ──────────────────────────────────────────────
def test_a_finished_node_shows_its_count_and_how_long_it_took():
    rep = _report(node_state={"n_img": {"success": 10}}, node_ms={"n_img": 119.0})
    state, extra = L.state_of(rep, "n_img")
    assert state == "success"
    assert "10건" in extra and "119ms" in extra


def test_a_running_node_shows_time_spent_in_this_pass_and_the_total():
    """앞 숫자만 있으면 이 노드가 전체에서 얼마나 무거운지 모르고, 뒤 숫자만 있으면
    지금 한 건이 유난히 오래 걸리는 중인지 알 수 없다."""
    rep = _report(active="n_infer", active_since=time.perf_counter() - 0.9,
                  node_state={"n_infer": {"success": 8}}, node_ms={"n_infer": 12_300.0})
    state, extra = L.state_of(rep, "n_infer")
    assert state == "running"
    assert "900ms" in extra or "0.9s" in extra, f"이번 경과가 없다: {extra!r}"
    assert "누계" in extra and "12.3s" in extra, f"누계가 없다: {extra!r}"


def test_a_node_that_just_started_still_reads_as_the_current_place():
    """카운트가 0이어도 화면에는 여기가 현재 위치라고 나와야 한다 — 첫 샘플의 첫 통과."""
    rep = _report(active="n_cols", active_since=time.perf_counter())
    assert L.state_of(rep, "n_cols")[0] == "running"


def test_a_failed_node_keeps_its_time_too():
    """5ms 만에 터진 것과 40초를 쓰고 터진 것은 원인이 다르다."""
    rep = _report(node_state={"n_infer": {"failed": 2}}, node_ms={"n_infer": 40_000.0})
    state, extra = L.state_of(rep, "n_infer")
    assert state == "failed" and "40" in extra


def test_a_folded_procedure_reports_the_box_not_its_insides():
    """캔버스는 접힌 Procedure 를 상자 하나로 그린다. 안쪽 노드 id 로 답하면
    그 상자는 실행 내내 아무 색도 바뀌지 않는다."""
    rep = _report(active="p_prep/n_resize", active_since=time.perf_counter() - 0.5,
                  node_state={"p_prep/n_wrap": {"success": 6}},
                  node_ms={"p_prep/n_wrap": 5.0})
    state, _ = L.state_of_many(rep, ["p_prep/n_resize", "p_prep/n_wrap"])
    assert state == "running", "안쪽 노드가 도는데 상자가 running 이 아니다"


def test_pending_stays_pending_without_a_report():
    assert L.state_of(None, "n_img") == ("pending", "")
