"""5노드 화면이 그릴 것을 계산한다 — Qt 를 모른다.

카드 한 장이 무엇을 보여 줄지(램프, 글자, 통과·NG 건수)와 NG 표의 행을 여기서 정한다.
위젯은 이 결과를 그리기만 한다. 같은 계산을 테스트가 위젯 없이 확인한다.

램프는 다섯이다: 대기 / 진행 / 정상 / NG 있음 / 오류. **오류는 스테이션이 멈춘 것**이고
**NG 있음은 일부 건을 배출하고 계속 간 것**이다. 둘을 섞으면 건 하나가 빠진 라인과
설비가 선 라인이 같은 색이 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PENDING, RUNNING, OK, NG, ERROR = "pending", "running", "ok", "ng", "error"
LAMP_TEXT = {PENDING: "대기", RUNNING: "진행", OK: "정상", NG: "NG 있음", ERROR: "오류"}


@dataclass
class Card:
    id: str
    number: int
    label: str
    hint: str
    lamp: str = PENDING
    passed: int = 0
    ng: int = 0
    line2: str = ""            # 카드 둘째 줄. ④ 는 단계·step·loss, 나머지는 통과·NG
    members: List[str] = field(default_factory=list)   # 컴파일된 노드 id
    settings: Dict[str, Any] = field(default_factory=dict)

    @property
    def lamp_text(self) -> str:
        return LAMP_TEXT[self.lamp]


@dataclass
class NgRow:
    sample_key: str
    card_number: int
    card_label: str
    node_id: str
    cause: str
    hint: str

    @property
    def one_line(self) -> str:
        return (self.cause or "").strip().splitlines()[0][:120] if (self.cause or "").strip() else ""


def _members(cg: Any, declared: List[str]) -> List[str]:
    """스테이션이 선언한 id 는 묶음 상자의 안쪽에서 `p_x/n_y` 로 펼쳐진다."""
    return [n for n in cg.order if any(n == m or n.startswith(m + "/") for m in declared)]


def _sinks(cg: Any, ids: List[str]) -> List[str]:
    """이 스테이션 안에서 더 아래로 이어지는 배선이 없는 노드 — 건이 스테이션을 **나가는** 자리."""
    inner = set(ids)
    has_out = {e.src_node for e in cg.edges if e.src_node in inner and e.dst_node in inner}
    return [i for i in ids if i not in has_out]


def _counts(cg: Any, rep: Any, ids: List[str]) -> tuple:
    """(통과, NG). 통과는 스테이션을 나가는 노드에 도착한 건수(여럿이면 가장 적은 쪽),
    NG 는 안쪽 노드의 실패 합이다."""
    if rep is None or not ids:
        return 0, 0
    ng = sum(int(rep.states_of(i).get("failed", 0)) for i in ids)
    seen = [rep.states_of(i) for i in _sinks(cg, ids)]
    seen = [st for st in seen if st]
    passed = min((int(st.get("success", 0)) + int(st.get("cached", 0)) for st in seen), default=0)
    return passed, ng


def _node_lamp(rep: Any, ids: List[str], ng: int, passed: int) -> str:
    from .layout import state_of_many

    if rep is None:
        return PENDING
    name, _ = state_of_many(rep, ids)
    if name == "running":
        return RUNNING
    if ng and passed == 0:
        return ERROR        # 건이 하나도 이 스테이션을 못 나갔다 — 설비가 선 것이다
    if name in ("failed", "partial") or ng:
        return NG           # 일부를 배출하고 계속 갔다
    if name in ("success", "cached"):
        return OK
    return PENDING


def cards(desc: Dict[str, Any], cg: Any, rep: Any, run: Optional[Dict[str, Any]] = None) -> List[Card]:
    """`desc` 는 `spec.line.describe_file`, `rep` 는 진행 스냅샷에서 만든 RunReport(없으면 None)."""
    run = run or {}
    out: List[Card] = []
    stages = {s["key"]: s for s in (run.get("pipeline") or {}).get("stages") or []}
    train = run.get("train") or {}
    trainers = {n.id for n in cg.nodes.values() if n.id and _is_trainer(n)}

    for s in desc["stations"]:
        ids = _members(cg, s["members"])
        flow = [i for i in ids if i not in trainers]
        passed, ng = _counts(cg, rep, flow)
        c = Card(id=s["id"], number=s["number"], label=s["label"], hint=s["hint"], passed=passed, ng=ng,
                 members=ids, settings=dict(s["settings"]))
        c.lamp = _node_lamp(rep, flow, ng, passed)
        c.line2 = f"통과 {passed} · NG {ng}"

        if trainers & set(ids):
            st = stages.get("train") or {}
            state = st.get("state", "")
            # 학습 카드의 정상은 **학습이 끝났다**는 뜻이다. 앞의 한 건 묶기가 통과했다고
            # 학습한 것처럼 초록이면, 모델이 없는데 정상으로 읽힌다.
            if state == "running":
                c.lamp = RUNNING
            elif state == "failed":
                c.lamp = ERROR
            elif state == "done":
                c.lamp = c.lamp if c.lamp in (NG, ERROR) else OK
            elif c.lamp == OK:
                c.lamp = PENDING
            if train.get("step"):
                c.line2 = (f"{train.get('stage', '')} · step {train['step']} · "
                           f"loss {float(train.get('loss') or 0):.3f}")
            elif state == "done":
                c.line2 = "학습 끝"
            else:
                c.line2 = f"통과 {passed} · NG {ng}"
        out.append(c)
    return out


def _is_trainer(n: Any) -> bool:
    try:
        from ..core.registry import resolve

        return not resolve(n.ref).per_sample
    except Exception:
        return False


def ng_rows(cards_: List[Card], quarantine: List[Dict[str, Any]]) -> List[NgRow]:
    """배출된 건. 어느 스테이션에서 나왔는지는 그 건이 걸린 노드가 속한 카드다."""
    where = {m: c for c in cards_ for m in c.members}
    rows = []
    for q in quarantine or []:
        c = where.get(q.get("node_id", ""))
        rows.append(NgRow(
            sample_key=str(q.get("sample_key", "")),
            card_number=c.number if c else 0, card_label=c.label if c else "",
            node_id=str(q.get("node_id", "")), cause=str(q.get("cause", "")), hint=str(q.get("hint", ""))))
    return rows
