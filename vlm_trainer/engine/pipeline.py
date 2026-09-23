"""파이프라인 — 준비 → 굽기 → 학습 → 추론을 한 번에 돈다.

**여기에 실행 코드는 없다.** 그래프를 보고 "무슨 단계가 필요한가" 를 정하고, 그 결과를
파일로 남길 뿐이다. 실제로 도는 것은 `vlmt materialize` · `vlmt train` · `vlmt run` 이고,
`cli/cmd_execute.py` 가 그 세 명령을 이 계획대로 부른다.

이 분리에는 이유가 있다. 파이프라인이 제 실행 경로를 따로 가지면, 터미널에서 세 명령을
따로 쳤을 때와 `vlmt pipeline` 이 다르게 도는 날이 온다. 그때 어느 쪽이 맞는지 아무도
모른다. 단계를 **고르는** 것까지만 여기서 하고, **도는** 것은 하나뿐이어야 한다.

단계는 그래프가 정한다. 사람이 고르지 않는다 — Trainer 가 없는 그래프에 학습 단계를
띄워 놓고 "돌지 않음" 이라고 적는 것보다, 애초에 없는 편이 읽기 쉽다.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..core.compiler import CompiledGraph
from ..core.registry import resolve as resolve_node

PENDING, RUNNING, DONE, FAILED, SKIPPED = "pending", "running", "done", "failed", "skipped"

SNAPSHOT = "pipeline.json"


@dataclass
class Stage:
    """단계 하나. `key` 는 코드가, `label` 은 사람이 읽는다."""

    key: str
    label: str
    command: str = ""          # 이 단계가 실제로 부르는 CLI 하위 명령
    split: str = ""            # 이 단계가 도는 split. 빈 값이면 전부
    state: str = PENDING
    ms: float = 0.0
    note: str = ""

    @property
    def done(self) -> bool:
        return self.state in (DONE, SKIPPED)


@dataclass
class Plan:
    run_id: str = ""
    stages: List[Stage] = field(default_factory=list)
    started: float = 0.0
    finished: float = 0.0

    def get(self, key: str) -> Optional[Stage]:
        return next((s for s in self.stages if s.key == key), None)

    @property
    def active(self) -> str:
        return next((s.key for s in self.stages if s.state == RUNNING), "")

    @property
    def failed(self) -> bool:
        return any(s.state == FAILED for s in self.stages)


# ── 그래프를 읽고 단계를 고른다 ──────────────────────────────────────────
def trainers(cg: CompiledGraph) -> Set[str]:
    """학습 노드 — 샘플마다 돌지 않는 노드다.

    `per_sample=False` 가 "이 노드는 샘플 하나가 아니라 데이터셋 전체를 본다" 는 뜻이고,
    학습이 정확히 그것이다. 노드 타입 이름(`train.vlm_trainer`)으로 찾지 않는다 —
    그러면 남이 만든 학습 노드가 이 파이프라인에서 안 보인다.
    """
    return {i for i in cg.order if not resolve_node(cg.nodes[i].ref).per_sample}


def after_training(cg: CompiledGraph) -> Set[str]:
    """학습보다 **뒤에** 있는 노드 전부. 추론이 여기 산다.

    컴파일러가 `external_call` 게이트에서 쓰는 것과 같은 계산이다. 같은 질문에 두 곳이
    다르게 답하기 시작하면, 게이트를 통과한 그래프가 파이프라인에서 빠지는 날이 온다.
    """
    out: Set[str] = set()
    stack = list(trainers(cg))
    while stack:
        cur = stack.pop()
        for e in cg.edges:
            if e.src_node == cur and e.dst_node not in out:
                out.add(e.dst_node)
                stack.append(e.dst_node)
    return out


def infer_split(cg: CompiledGraph) -> str:
    """추론 단계가 돌 split. 추론 노드가 스스로 선언한 것을 읽는다.

    노드가 `only_split: val` 이라 적어 두고도 파이프라인이 110건 전부를 돌면,
    100건은 전처리를 다 거친 뒤 추론에서 버려진다. 그 시간이 그대로 낭비다.

    선언이 갈리면 아무것도 고르지 않는다 — 한쪽 말만 듣고 나머지를 조용히 굶기는 것보다
    전부 도는 편이 낫다.
    """
    want = {str(cg.nodes[i].params.get("only_split") or "")
            for i in after_training(cg) if "only_split" in cg.nodes[i].params}
    return want.pop() if len(want) == 1 else ""


def plan(cg: CompiledGraph, run_id: str = "", *, bake_split: str = "") -> Plan:
    """이 그래프가 돌아야 할 단계. 그래프에 없는 단계는 만들지 않는다."""
    p = Plan(run_id=run_id)
    p.stages.append(Stage("prepare", "준비", command="budget",
                          note="컴파일과 예산(G4). GPU 를 잡기 전 마지막 문이다"))

    if cg.materialize.get("boundary"):
        p.stages.append(Stage("bake", "굽기", command="materialize", split=bake_split,
                              note="경계까지 미리 구워 shard 로 남긴다"))
    if trainers(cg):
        p.stages.append(Stage("train", "학습", command="train",
                              note="구운 shard 만 읽는다"))
    if after_training(cg):
        p.stages.append(Stage("infer", "추론", command="run", split=infer_split(cg),
                              note="학습한 모델로 답을 만들고 Output 노드까지 돈다"))
    return p


# ── 남기기 ──────────────────────────────────────────────────────────────
def snapshot_path(run_id: str, root: str = "runs") -> str:
    return os.path.join(root, run_id, SNAPSHOT)


def write(p: Plan, path: str) -> None:
    """원자 교체로 남긴다. 실패해도 실행을 막지 않는다 —
    관찰이 실행을 죽여서는 안 된다(`runner.write_progress` 와 같은 규약)."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"run_id": p.run_id, "started": p.started, "finished": p.finished,
                       "stages": [asdict(s) for s in p.stages]}, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        pass


def read(path: str) -> Optional[Plan]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return Plan(
        run_id=str(data.get("run_id") or ""),
        started=float(data.get("started") or 0.0),
        finished=float(data.get("finished") or 0.0),
        stages=[Stage(**{k: v for k, v in s.items() if k in Stage.__annotations__})
                for s in (data.get("stages") or [])],
    )


# ── 이어 받기 ───────────────────────────────────────────────────────────
@dataclass
class Leftover:
    """굽다 만 것. 있으면 **물어본다** — 조용히 지우지도, 조용히 재사용하지도 않는다."""

    run_id: str = ""
    out_dir: str = ""
    shards: int = 0
    samples: int = 0

    def __bool__(self) -> bool:
        return self.shards > 0


def leftover(run_id: str, root: str = "runs") -> Leftover:
    """커밋된 shard 가 남아 있는지. 재사용 여부를 **결정하지는 않는다.**

    굽다 만 것을 말없이 이어받으면, 스펙을 고친 뒤 다시 돌렸을 때 예전 파라미터로 구운
    샘플과 새 파라미터로 구운 샘플이 한 학습에 섞인다. 말없이 지우면 몇 시간이 날아간다.
    양쪽 다 나쁘므로 사람에게 묻는다.
    """
    out_dir = os.path.join(root, run_id, "materialized")
    manifest = os.path.join(out_dir, "manifest.jsonl")
    if not os.path.exists(manifest):
        return Leftover(run_id=run_id, out_dir=out_dir)
    shards, samples = 0, 0
    try:
        with open(manifest, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                shards += 1
                try:
                    samples += int(json.loads(line).get("n") or 0)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
    except OSError:
        return Leftover(run_id=run_id, out_dir=out_dir)
    return Leftover(run_id=run_id, out_dir=out_dir, shards=shards, samples=samples)


# ── 사람이 읽는 줄 ──────────────────────────────────────────────────────
def render(p: Plan) -> str:
    from ..core.humanize import ms as fmt_ms

    mark = {PENDING: "·", RUNNING: ">", DONE: "v", FAILED: "x", SKIPPED: "-"}
    lines = []
    for s in p.stages:
        took = f"  {fmt_ms(s.ms)}" if s.ms else ""
        split = f"  split {s.split}" if s.split else ""
        note = f"  {s.note}" if s.note and s.state in (FAILED, SKIPPED) else ""
        lines.append(f"  {mark.get(s.state, '?')} {s.label:<4} {s.command:<12}{split}{took}{note}")
    return "\n".join(lines)


def headline(p: Plan) -> str:
    """진행 줄 하나. `준비 v · 굽기 > · 학습 · 추론` 처럼 읽힌다."""
    mark = {PENDING: "", RUNNING: " >", DONE: " v", FAILED: " x", SKIPPED: " -"}
    return " · ".join(f"{s.label}{mark.get(s.state, '')}" for s in p.stages)


def as_dict(p: Plan) -> Dict[str, Any]:
    """편집기가 읽는 형태. `Plan` 을 그대로 건네면 UI 가 dataclass 에 묶인다."""
    return {"run_id": p.run_id, "active": p.active, "failed": p.failed,
            "headline": headline(p), "stages": [asdict(s) for s in p.stages]}
