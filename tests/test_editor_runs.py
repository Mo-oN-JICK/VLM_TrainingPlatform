"""실행·진행·경과 시간 — `test_editor.py` 에서 갈라 나왔다.

1234줄 한 파일이라 무엇이 어디 있는지 찾기 어려웠다. 테스트 이름과 내용은 그대로다.
"""

from __future__ import annotations

import os
import shutil

import pytest

from vlm_trainer.spec import recipe as recipe_mod
from vlm_trainer.ui import tokens as T
from vlm_trainer.ui.api import Editor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SOLUTION = os.path.join(ROOT, "solutions", "dummy_ecg")
SPEC_DIR = os.path.join(SOLUTION, "projects", "01_dummy")
PROJECT = os.path.join(SPEC_DIR, "project.yaml")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "solutions", "dummy_ecg", "data", "dummy", "index.jsonl")),
    reason="더미 데이터가 없다. python tools/make_dummy_dataset.py 를 먼저 실행하라.",
)


@pytest.fixture
def ed(tmp_path):
    """진짜 Solution을 복사해 연다 — 원본은 편집하지 않는다.

    Procedure는 Solution 아래에 있으므로 프로젝트 디렉터리만 복사하면 찾지 못한다.
    데이터는 컴파일에 필요 없으므로 뺀다.
    """
    work = tmp_path / "dummy_ecg"
    # edit_history.jsonl은 append-only라 앞선 세션의 기록이 딸려오면 안 된다
    shutil.copytree(
        SOLUTION,
        work,
        ignore=shutil.ignore_patterns("data", "runs", "*.lock.yaml", "edit_history*"),
    )
    return Editor.open(str(work / "projects" / "01_dummy" / "project.yaml"))


@pytest.fixture
def ed_with_data(tmp_path):
    """데이터까지 복사한다. 실제로 그래프를 돌리는 테스트 하나만 쓴다."""
    work = tmp_path / "dummy_ecg"
    shutil.copytree(
        SOLUTION, work, ignore=shutil.ignore_patterns("runs", "*.lock.yaml", "edit_history*")
    )
    return Editor.open(str(work / "projects" / "01_dummy" / "project.yaml"))


# ── 호환성 표 ───────────────────────────────────────────────────────────



def _running(ed, tmp_path, *, processed: int, total: int, ago: float):
    """진행 중인 작업 하나를 흉내낸다. 시계를 믿지 않도록 값을 직접 놓는다."""
    import json as _json
    import time as _t

    from vlm_trainer.engine import runner as runner_mod

    class _Alive:
        def poll(self):
            return None

    ed.proc = _Alive()
    ed.launched_at = _t.time() - ago
    ed.finished_at = 0.0
    ed.progress_path = str(tmp_path / "progress.json")

    first = ed.compiled.order[0]
    rep = runner_mod.RunReport(order=list(ed.compiled.order), started=_t.time() - ago)
    rep.processed = processed
    rep.active, rep.active_since = first, _t.perf_counter() - 4.0
    rep.node_ms[first] = 8200.0
    with open(ed.progress_path, "w", encoding="utf-8") as fh:
        _json.dump(runner_mod.snapshot(rep, run_id="r", total=total, phase="running"), fh)
    return first




def test_run_uses_the_same_cli_command(ed):
    """편집기에 전용 실행 경로는 없다. 사람이 터미널에 그대로 쳐도 같아야 한다."""
    ed.extra_modules = ("fixture_nodes",)
    ed.recipe_select(2)
    cmd = ed.run_command(limit=4, debug_output=True)

    assert cmd[1:4] == ["-m", "vlm_trainer.cli.main", "run"]
    assert cmd[4] == ed.path
    assert "--trigger" in cmd and cmd[cmd.index("--trigger") + 1] == "ui"
    assert "--debug-output" in cmd
    assert cmd[cmd.index("--nodes") + 1] == "fixture_nodes"
    # 화면에 보이는 값 그대로 돈다 — 오버레이는 CLI의 --set 으로 넘어간다
    assert "n_stats.z_thresh=4.5" in cmd


def test_run_is_refused_while_the_spec_on_disk_differs(ed):
    ed.set_param("n_stats", "z_thresh", 4.5)
    res = ed.run_start()
    assert not res["ok"] and "저장하지 않은 변경" in res["reason"]
    assert "화면과 다른 그래프가 돈다" in res["detail"]

    assert ed.save()["ok"]
    assert ed.run_command()  # 저장한 뒤에는 막을 이유가 없다


def test_run_is_refused_while_the_graph_is_incomplete(ed):
    ed.add_node("ts.stats@1.0.0")
    res = ed.run_start()
    assert not res["ok"] and "필수 입력" in res["detail"]


def test_progress_snapshot_round_trips(ed):
    from vlm_trainer.engine import runner as runner_mod

    rep = runner_mod.RunReport(order=list(ed.compiled.order))
    rep.count("n_stats", runner_mod.SUCCESS)
    rep.node_ms["n_stats"] = 12.5
    rep.processed = 3
    rep.quarantine.append(runner_mod.Quarantined("s1", "n_ev", "이유", "힌트"))

    data = runner_mod.snapshot(rep, run_id="r1", total=4, phase="running")
    back = runner_mod.report_from_snapshot(data)

    assert back.states_of("n_stats") == {"success": 1}
    assert back.node_ms["n_stats"] == 12.5
    assert back.quarantine[0].node_id == "n_ev"
    from vlm_trainer.ui.layout import state_of

    state, extra = state_of(back, "n_stats")
    assert state == "success" and "1건" in extra


def test_run_state_reads_the_snapshot_a_run_leaves(ed, tmp_path):
    import json as _json

    from vlm_trainer.engine import runner as runner_mod

    rep = runner_mod.RunReport(order=list(ed.compiled.order))
    for nid in ed.compiled.order:
        rep.count(nid, runner_mod.SUCCESS)
    rep.processed = 2

    ed.progress_path = str(tmp_path / "progress.json")
    with open(ed.progress_path, "w", encoding="utf-8") as fh:
        _json.dump(runner_mod.snapshot(rep, run_id="r1", total=2, phase="done"), fh)

    st = ed.run_state()
    assert st["running"] is False and st["phase"] == "done"
    assert st["processed"] == 2
    assert st["states"]["n_stats"]["state"] == "success"


def test_the_editor_actually_runs_the_graph(ed_with_data, tmp_path, monkeypatch):
    """끝에서 끝까지 — 편집기가 띄운 프로세스가 진행 파일을 남기고 상태가 칠해진다.

    저장소 밖에서 돌린다. 하위 프로세스가 CWD에 기대 임포트하면 여기서 걸린다.
    """
    import time as _time

    ed = ed_with_data
    monkeypatch.chdir(tmp_path)
    assert ed.run_start(limit=2)["ok"]

    for _ in range(1200):  # 최대 120초. 전체 실행 중에는 앞선 테스트의 프로세스와 겹친다
        st = ed.run_state()
        if not st["running"]:
            break
        _time.sleep(0.1)
    else:
        raise AssertionError("끝나지 않았다")

    assert st["phase"] == "done", st.get("console", "")
    assert st["exit"] == 0, st.get("console", "")
    assert st["states"]["n_answer"]["state"] in ("success", "cached")


def test_a_stopped_run_is_not_reported_as_a_failure(ed_with_data, tmp_path, monkeypatch):
    """사람이 멈춘 것과 죽은 것은 다르다. 빨간 글씨로 같이 묶으면 신호가 죽는다."""
    import time as _time

    ed = ed_with_data
    monkeypatch.chdir(tmp_path)
    assert ed.run_start(limit=24)["ok"]
    assert ed.run_stop()["ok"]

    for _ in range(300):
        st = ed.run_state()
        if not st["running"]:
            break
        _time.sleep(0.1)

    assert st["stopped"] is True
    assert "console" not in st, "중지는 실패가 아니므로 콘솔 꼬리를 들이밀지 않는다"


# ── 세션을 넘는 History ─────────────────────────────────────────────────


def test_previews_are_served_only_from_this_runs_folder(ed, tmp_path, monkeypatch):
    """경로를 그대로 실어 보내면 서버가 아무 파일이나 내주는 문이 된다."""
    monkeypatch.chdir(tmp_path)
    ed.run_id = "ui_test"
    os.makedirs(ed.preview_dir, exist_ok=True)
    with open(os.path.join(ed.preview_dir, "n_img.png"), "wb") as fh:
        fh.write(b"PNG-ish")
    with open(tmp_path / "secret.txt", "w", encoding="utf-8") as fh:
        fh.write("남의 파일")

    assert ed.preview_file("n_img.png") == b"PNG-ish"
    for probe in ("../../secret.txt", r"..\..\secret.txt", "n_img.txt", "", "secret.txt"):
        assert ed.preview_file(probe) is None, probe


def test_the_run_writes_preview_images_when_the_toggle_is_on(ed_with_data, tmp_path, monkeypatch):
    """Debug Output이 그림을 보여주지 못하면 크롭이 어긋났는지 알 수 없다."""
    import time as _time

    ed = ed_with_data
    monkeypatch.chdir(tmp_path)
    assert ed.run_start(limit=1, debug_output=True)["ok"]

    for _ in range(600):
        st = ed.run_state()
        if not st["running"]:
            break
        _time.sleep(0.1)

    assert st["phase"] == "done", st.get("console", "")
    imgs = {k: v["image"] for k, v in st["previews"].items() if v.get("image")}
    assert imgs, "이미지 미리보기가 하나도 없다"
    assert ed.preview_file(os.path.basename(imgs["n_img"])), "서버가 그 파일을 못 찾는다"


def test_training_before_materializing_says_so(ed):
    res = ed.train_start()
    assert not res["ok"] and "Materialize" in res["reason"]
    assert "vlmt materialize" in res["detail"], "터미널로도 할 수 있다는 것을 말한다"


def test_materialize_refuses_an_incomplete_graph(ed):
    ed.add_node("ts.stats@1.0.0")
    assert not ed.materialize_start()["ok"]


def test_only_one_thing_runs_at_a_time(ed_with_data, tmp_path, monkeypatch):
    ed = ed_with_data
    monkeypatch.chdir(tmp_path)
    assert ed.materialize_start()["ok"]
    second = ed.materialize_start()
    assert not second["ok"] and "돌고 있다" in second["reason"]
    ed.run_stop()


def test_the_editor_materializes_and_then_trains(ed_with_data, tmp_path, monkeypatch):
    """끝에서 끝까지 — 두 버튼이 두 CLI 명령을 띄우고 학습이 완주한다."""
    import time as _time

    ed = ed_with_data
    monkeypatch.chdir(tmp_path)

    def wait():
        for _ in range(1200):  # 최대 120초
            st = ed.run_state()
            if not st["running"]:
                return st
            _time.sleep(0.1)
        raise AssertionError("끝나지 않았다")

    assert ed.materialize_start()["ok"]
    st = wait()
    assert st["exit"] == 0, st.get("console", "")
    assert os.path.isdir(os.path.join(tmp_path, "runs", ed.run_id, "materialized"))

    assert ed.train_start()["ok"]
    st = wait()
    assert st["exit"] == 0, st.get("console", "")
    assert st["kind"] == "train"
    assert st["train"]["step"] > 0 and st["train"]["stage"]
    assert os.path.exists(
        os.path.join(tmp_path, "runs", ed.run_id, "train", "inference_contract.json")
    )


# ── 물질화 경계와 실행 프로파일 ─────────────────────────────────────────


def test_a_graph_that_does_not_train_is_not_asked_about_a_boundary():
    """루프가 없으면 '루프 안에서 돈다'는 위험 자체가 없다."""
    from vlm_trainer.core.compiler import compile_project

    demo = os.path.join(ROOT, "tests", "data", "solution", "projects", "01_demo", "project.yaml")
    if not os.path.exists(demo):
        pytest.skip("데모 스펙이 없다")
    cg = compile_project(demo)  # 경계가 있든 없든 학습 노드가 없으면 통과한다
    assert cg.order


def test_snapshot_carries_the_node_being_run(ed):
    """누계(node_ms)만으로는 "어디까지 왔나"에 답이 안 된다.

    느린 노드 앞에서 멈춘 것인지 그 노드가 도는 중인지 구별되지 않기 때문이다.
    """
    import time as _t

    from vlm_trainer.engine import runner as runner_mod
    from vlm_trainer.ui.layout import state_of

    rep = runner_mod.RunReport(order=list(ed.compiled.order))
    rep.active, rep.active_since = "n_stats", _t.perf_counter() - 3.0

    data = runner_mod.snapshot(rep, run_id="r1", total=4, phase="running")
    assert data["active"] == "n_stats"
    assert data["active_ms"] >= 2900

    back = runner_mod.report_from_snapshot(data)
    state, extra = state_of(back, "n_stats")
    # 카운트가 아직 0이어도(첫 샘플의 첫 통과) 현재 위치로 보여야 한다
    assert state == "running" and extra.endswith("s")


def test_a_finished_run_highlights_nothing(ed):
    """다 끝난 그래프에 노드 하나가 계속 빛나고 있으면 그것이 마지막으로 돈 노드인지
    지금 도는 노드인지 화면만 보고는 알 수 없다."""
    from vlm_trainer.engine import runner as runner_mod

    rep = runner_mod.RunReport(order=list(ed.compiled.order))
    rep.active, rep.active_since = "n_stats", 0.0

    for phase in ("done", "aborted"):
        data = runner_mod.snapshot(rep, run_id="r1", total=4, phase=phase)
        assert data["active"] == "", phase
        assert data["active_ms"] == 0.0, phase


def test_elapsed_time_shows_on_finished_nodes(ed):
    """실패한 노드도 5ms 만에 터진 것과 40초를 쓰고 터진 것은 원인이 다르다."""
    from vlm_trainer.engine import runner as runner_mod
    from vlm_trainer.ui.layout import state_of

    rep = runner_mod.RunReport(order=list(ed.compiled.order))
    rep.count("n_stats", runner_mod.SUCCESS)
    rep.node_ms["n_stats"] = 1234.0
    rep.count("n_ev", runner_mod.FAILED)
    rep.node_ms["n_ev"] = 42_000.0

    assert state_of(rep, "n_stats") == ("success", "1건 · 1.2s")
    assert state_of(rep, "n_ev") == ("failed", "1건 · 42.0s")


def test_run_state_answers_with_the_ids_the_canvas_draws(ed, tmp_path):
    """접힌 Procedure는 안쪽 노드 id로 그려져 있지 않다.

    compiled.order를 그대로 내보내면 상자가 실행 내내 아무 색도 바뀌지 않는다.
    """
    import json as _json

    from vlm_trainer.engine import runner as runner_mod
    from vlm_trainer.ui.layout import fold

    shown, _ = fold(ed.compiled, ed.expanded)

    rep = runner_mod.RunReport(order=list(ed.compiled.order))
    rep.active, rep.active_since = ed.compiled.order[0], 0.0

    ed.progress_path = str(tmp_path / "progress.json")
    with open(ed.progress_path, "w", encoding="utf-8") as fh:
        _json.dump(runner_mod.snapshot(rep, run_id="r1", total=2, phase="running"), fh)

    st = ed.run_state()
    assert set(st["states"]) == set(shown), "캔버스에 없는 id로 답하면 칠할 곳이 없다"
    assert st["active"] in shown
    assert st["states"][st["active"]]["state"] == "running"


# ── Procedure 노출 파라미터 편집 ─────────────────────────────────────────


def test_elapsed_time_counts_from_when_the_job_started(ed, tmp_path):
    """노드별 누계를 다 더해도 이 값이 나오지 않는다.

    캐시 적중, 샘플 적재, shard 쓰기처럼 어느 노드에도 속하지 않는 시간이 있다.
    """
    _running(ed, tmp_path, processed=20, total=110, ago=30.0)
    st = ed.run_state()
    assert 29_000 <= st["elapsed_ms"] <= 31_000


def test_elapsed_time_stops_when_the_job_does(ed, tmp_path):
    """다 끝난 작업의 숫자가 계속 올라가면 그것은 경과 시간이 아니라 시계다."""
    import time as _t

    class _Dead:
        returncode = 0

        def poll(self):
            return 0

    _running(ed, tmp_path, processed=110, total=110, ago=12.0)
    ed.proc = _Dead()
    first = ed.run_state()["elapsed_ms"]
    _t.sleep(0.05)
    again = ed.run_state()["elapsed_ms"]
    assert first == again, "끝난 뒤에도 경과 시간이 흐른다"


def test_remaining_time_is_withheld_until_the_rate_means_something(ed, tmp_path):
    """처음 한두 건으로 "남은 40분"을 띄우면 그 수를 믿고 자리를 뜨게 된다."""
    _running(ed, tmp_path, processed=3, total=110, ago=30.0)
    assert ed.run_state()["eta_ms"] == 0.0

    _running(ed, tmp_path, processed=20, total=110, ago=30.0)
    eta = ed.run_state()["eta_ms"]
    # 20건에 30초 -> 남은 90건은 135초
    assert 130_000 <= eta <= 140_000


def test_a_running_node_shows_this_pass_and_the_running_total(ed, tmp_path):
    """앞만 있으면 이 노드가 전체에서 얼마나 무거운지 모르고,
    뒤만 있으면 지금 한 건이 유난히 오래 걸리는 중인지 알 수 없다."""
    first = _running(ed, tmp_path, processed=20, total=110, ago=30.0)
    st = ed.run_state()
    extra = st["states"][st["active"]]["extra"]
    assert extra.startswith("4."), extra          # 이번에 들어가서 4초
    assert "누계 8.2s" in extra, extra            # 지금까지 8.2초
