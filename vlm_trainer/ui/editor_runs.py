"""실행 — 하위 프로세스를 띄우고 그 진행 파일을 읽는다

`Editor` 가 한 클래스에 아홉 관심사를 담고 있어 믹스인으로 갈랐다.
**메서드 이름과 동작은 하나도 바뀌지 않는다** — `Editor` 가 이것을 상속한다.

여기 있는 메서드는 `self._try` · `self._recompile` · `self.graph` 처럼 Editor 본체가
들고 있는 것을 쓴다. 그것이 믹스인이 독립 클래스가 아닌 이유다.
"""

from __future__ import annotations

from .editor_common import _short



import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from ..engine import runner as runner_mod
from ..engine import samples as samples_mod
from ..spec import recipe as recipe_mod
from ..spec.decompile import decompile, dump_yaml

class RunsMixin:
    """실행 — 하위 프로세스를 띄우고 그 진행 파일을 읽는다"""

    # ── 실행 ────────────────────────────────────────────────────────────
    #
    # 편집기는 실행 경로를 따로 갖지 않는다. `vlmt run`을 그대로 하위 프로세스로 띄우고
    # 그 프로세스가 남기는 진행 스냅샷을 읽을 뿐이다. UI에서만 되는 실행은 존재하지 않는다.

    def run_command(self, limit: int = 8, debug_output: bool = False) -> List[str]:
        """띄울 명령. 사람이 터미널에 그대로 쳐도 같은 결과가 나와야 한다."""
        cmd = [
            sys.executable, "-m", "vlm_trainer.cli.main",
            "run", self.path,
            "--run-id", self.run_id or "ui",
            "--limit", str(int(limit)),
            "--trigger", "ui",
            "--skip-budget",
        ]
        if debug_output:
            cmd.append("--debug-output")
        for mod in self.extra_modules:
            cmd += ["--nodes", mod]
        # 화면에 보이는 값 그대로 돈다. 레시피 오버레이는 CLI의 --set 으로 넘긴다.
        for path, value in sorted(self.overlay.items()):
            cmd += ["--set", f"{path}={json.dumps(value, ensure_ascii=False)}"]
        return cmd

    def _spawn(self, cmd: List[str], phase: str) -> Dict[str, Any]:
        """하위 프로세스 하나를 띄운다. **버튼 하나가 CLI 명령 하나다** —
        편집기가 여러 명령을 엮어 돌리기 시작하면 그것이 UI 전용 실행 경로다."""
        if self.proc is not None and self.proc.poll() is None:
            return {"ok": False, "reason": f"이미 {self.phase or '작업'}이 돌고 있다"}

        run_dir = os.path.join(os.getcwd(), "runs", self.run_id)
        os.makedirs(run_dir, exist_ok=True)
        self.progress_path = os.path.join(run_dir, "progress.json")
        self.train_progress_path = os.path.join(run_dir, "train_progress.json")
        self.console_path = os.path.join(run_dir, f"{phase}.log")
        self.phase = phase
        self.stopped = False
        # 하위 프로세스를 띄운 시각. 진행 파일의 started 보다 이쪽이 이르고,
        # materialize 처럼 진행 파일을 쓰지 않는 단계에서도 경과 시간이 나온다.
        self.launched_at = time.time()
        self.finished_at = 0.0

        env = dict(os.environ)
        pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env["PYTHONPATH"] = os.pathsep.join([p for p in (pkg_root, env.get("PYTHONPATH", "")) if p])
        try:
            fh = open(self.console_path, "w", encoding="utf-8")
            self.proc = subprocess.Popen(
                cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=os.getcwd(), env=env
            )
        except OSError as exc:
            return {"ok": False, "reason": f"실행을 띄우지 못했다: {exc}"}
        return {"ok": True, "run_id": self.run_id, "phase": phase, "command": " ".join(cmd)}

    def _base_command(self, sub: str) -> List[str]:
        cmd = [sys.executable, "-m", "vlm_trainer.cli.main", sub, self.path,
               "--run-id", self.run_id or "ui"]
        for mod in self.extra_modules:
            cmd += ["--nodes", mod]
        for path, value in sorted(self.overlay.items()):
            cmd += ["--set", f"{path}={json.dumps(value, ensure_ascii=False)}"]
        return cmd

    def _ready_to_launch(self) -> Dict[str, Any]:
        if not self.valid:
            return {"ok": False, "reason": _short(self.error), "detail": self.error}
        if self.dirty:
            return {
                "ok": False,
                "reason": "저장하지 않은 변경이 있다 — 실행은 디스크의 스펙을 돈다",
                "detail": (
                    "편집기의 그래프와 project.yaml이 다르다. 지금 실행하면 화면과 다른 그래프가 돈다.\n"
                    "  이 검사가 없었다면: 방금 고친 값이 아니라 예전 값으로 돈 결과를 보고 판단하게 된다.\n"
                    "  Save를 먼저 누른다."
                ),
            }
        return {"ok": True}

    def materialize_start(self, limit: int = 0) -> Dict[str, Any]:
        """`vlmt materialize` — 학습이 읽을 shard를 만든다."""
        ready = self._ready_to_launch()
        if not ready["ok"]:
            return ready
        self.run_id = self.run_id or ("ui_" + time.strftime("%Y%m%dT%H%M%S"))
        cmd = self._base_command("materialize")
        if limit:
            cmd += ["--limit", str(int(limit))]
        return self._spawn(cmd, "materialize")

    def train_start(self) -> Dict[str, Any]:
        """`vlmt train` — 물질화된 shard로 학습한다.

        물질화를 대신 돌려 주지 않는다. 두 단계는 사람이 터미널에서 치는 두 명령이고,
        편집기가 그것을 엮으면 CLI에 없는 경로가 하나 생긴다.
        """
        ready = self._ready_to_launch()
        if not ready["ok"]:
            return ready
        if not self.run_id:
            return {
                "ok": False,
                "reason": "먼저 Materialize를 눌러야 한다",
                "detail": (
                    "학습은 물질화된 shard를 읽는다. 아직 이 세션에서 만든 것이 없다.\n"
                    "  Materialize를 먼저 누르거나, 터미널에서 vlmt materialize를 돌려라."
                ),
            }
        mat = os.path.join(os.getcwd(), "runs", self.run_id, "materialized")
        if not os.path.isdir(mat):
            return {
                "ok": False,
                "reason": "물질화된 shard가 없다",
                "detail": f"{mat}가 없다. Materialize를 먼저 누른다.",
            }
        return self._spawn(self._base_command("train"), "train")

    def run_start(self, limit: int = 8, debug_output: bool = False) -> Dict[str, Any]:
        ready = self._ready_to_launch()
        if not ready["ok"]:
            return ready
        self.run_id = "ui_" + time.strftime("%Y%m%dT%H%M%S")
        return self._spawn(self.run_command(limit, debug_output), "run")

    def run_stop(self) -> Dict[str, Any]:
        if self.proc is None or self.proc.poll() is not None:
            return {"ok": False, "reason": "돌고 있는 것이 없다"}
        self.stopped = True  # 사람이 멈춘 것과 죽은 것은 다르다
        self.proc.terminate()
        return {"ok": True}

    def run_state(self) -> Dict[str, Any]:
        """스냅샷을 읽어 카드에 칠할 상태로. 실행이 없으면 비어 있는 답이다."""
        from .render import fold, state_of_many

        alive = self.proc is not None and self.proc.poll() is None
        if not alive and self.proc is not None and not self.finished_at:
            self.finished_at = time.time()

        # 작업에 들어선 뒤 흐른 시간. 끝난 뒤에는 멈춘 값이어야 한다 —
        # 다 끝난 작업의 숫자가 계속 올라가면 그것은 경과 시간이 아니라 시계다.
        elapsed_ms = 0.0
        if self.launched_at:
            end = time.time() if alive else (self.finished_at or self.launched_at)
            elapsed_ms = max(0.0, (end - self.launched_at) * 1000)

        out: Dict[str, Any] = {
            "running": alive,
            "run_id": self.run_id,
            "kind": self.phase,
            "train": self._read_train_progress(),
            "exit": None if (self.proc is None or alive) else self.proc.returncode,
            "stopped": self.stopped and self.proc is not None and not alive,
            "states": {},
            "processed": 0,
            "total": 0,
            "phase": "",
            "aborted": "",
            "quarantine": [],
            "active": "",
            "elapsed_ms": elapsed_ms,
            "eta_ms": 0.0,
        }
        # 물질화·학습 중에도 직전 실행이 남긴 카드 상태는 그대로 둔다.
        # 지우면 "아무것도 안 돌았다"로 읽히는데 그것은 사실이 아니다.
        data = self._read_progress()
        if data is None:
            if not alive and self.proc is not None and not self.stopped:
                out["console"] = self._console_tail()
            return out

        rep = runner_mod.report_from_snapshot(data)
        out.update(
            {
                "processed": data.get("processed", 0),
                "total": data.get("total", 0),
                "phase": data.get("phase", ""),
                "aborted": data.get("aborted", ""),
                "quarantine": data.get("quarantine", []),
                "quarantine_total": data.get("quarantine_total", 0),
                "cache": data.get("cache", {}),
                "previews": self._preview_urls(data.get("previews", {})),
            }
        )
        # 남은 시간은 지금까지의 속도로만 민다. 평균이 흔들리는 초반에는 내지 않는다 —
        # 처음 한두 건으로 "남은 시간 40분"을 띄우면 그 수를 믿고 자리를 뜨게 된다.
        done, total = out["processed"], out["total"]
        if alive and done >= 5 and total > done and elapsed_ms > 0:
            out["eta_ms"] = elapsed_ms / done * (total - done)

        # 캔버스가 쓰는 id로 답한다. 접힌 Procedure는 안쪽 노드 id로 그려져 있지 않으므로
        # compiled.order를 그대로 내보내면 상자가 실행 내내 아무 색도 바뀌지 않는다.
        if self.compiled is not None:
            shown, _ = fold(self.compiled, self.expanded)
            for sid, s in shown.items():
                state, extra = state_of_many(rep, s.state_ids)
                out["states"][sid] = {"state": state, "extra": extra}
                if state == "running":
                    out["active"] = sid
        if not alive and self.proc is not None and self.proc.returncode and not self.stopped:
            out["console"] = self._console_tail()
        return out

    def _preview_urls(self, previews: Dict[str, Any]) -> Dict[str, Any]:
        """디스크 경로를 페이지가 부를 수 있는 주소로. 파일명만 남긴다 —
        경로를 그대로 실어 보내면 서버가 아무 파일이나 내주는 문이 된다."""
        out: Dict[str, Any] = {}
        for nid, p in previews.items():
            row = dict(p)
            path = str(row.pop("image_path", "") or "")
            row["image"] = f"/preview/{os.path.basename(path)}" if path else ""
            out[nid] = row
        return out

    @property
    def preview_dir(self) -> str:
        return os.path.join(os.getcwd(), "runs", self.run_id, "preview") if self.run_id else ""

    def preview_file(self, name: str) -> Optional[bytes]:
        """이번 실행의 미리보기 폴더 안에 있는 파일만 내준다."""
        base = self.preview_dir
        if not base or not name:
            return None
        p = os.path.abspath(os.path.join(base, os.path.basename(name)))
        if os.path.dirname(p) != os.path.abspath(base) or not p.lower().endswith(".png"):
            return None  # 경로를 벗어나려는 시도
        try:
            with open(p, "rb") as fh:
                return fh.read()
        except OSError:
            return None

    def _read_train_progress(self) -> Dict[str, Any]:
        """학습은 샘플이 아니라 step 단위로 움직인다. 진행 파일도 따로 있다."""
        if self.phase != "train" or not self.train_progress_path:
            return {}
        try:
            with open(self.train_progress_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    def _read_progress(self) -> Optional[Dict[str, Any]]:
        if not self.progress_path or not os.path.exists(self.progress_path):
            return None
        try:
            with open(self.progress_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None  # 원자 교체 사이에 읽었을 뿐이다. 다음 폴링에서 잡힌다

    def _console_tail(self, n: int = 12) -> str:
        try:
            with open(self.console_path, "r", encoding="utf-8", errors="replace") as fh:
                return "".join(fh.readlines()[-n:])
        except OSError:
            return ""

    def save(self) -> Dict[str, Any]:
        if self.compiled is None:
            return {"ok": False, "reason": "컴파일되지 않은 그래프는 저장하지 않는다"}
        if not self.valid:
            # 편집 중인 미완성 상태는 저장하지 않는다. 스펙 파일은 언제나 돌아가는 그래프여야 한다.
            return {"ok": False, "reason": _short(self.error), "detail": self.error}
        # **오버레이 이전의 그래프**를 쓴다. 레시피 값이 프로젝트 스펙에 스며들면
        # 레시피가 존재할 이유가 사라진다.
        spec = decompile(self.base, keep_procedures=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(dump_yaml(spec))
        os.replace(tmp, self.path)
        self.dirty = False
        written = [self.path]
        if self.book_dirty and self.book is not None:
            written.append(recipe_mod.save_book(self.book))
            self.book_dirty = False
        return {"ok": True, "path": self.path, "written": written}

