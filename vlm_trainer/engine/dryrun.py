"""G3 — 실행 전 dry-run.

실제 샘플 몇 건을 전 노드에 통과시켜 shape·dtype·스키마를 실측으로 검증한다.
비용은 보통 수십 초다. 그 대가로 학습 시작 후에야 드러날 오류를 전부 앞으로 당긴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.compiler import CompiledGraph
from ..core.node import NodeKind
from . import samples as samples_mod
from .runner import RunOptions, RunReport, execute
from .values import check_against, describe


@dataclass
class DryRunResult:
    samples: int = 0
    processed: int = 0
    type_mismatches: List[str] = field(default_factory=list)
    determinism_failures: List[str] = field(default_factory=list)
    quarantine: List[str] = field(default_factory=list)
    violation_ratio: float = 0.0
    measured: Dict[str, str] = field(default_factory=dict)
    measured_chars: int = 0  # 샘플당 프롬프트+정답 문자 수(최대)
    measured_text: str = ""  # 그 최대 샘플의 본문. 토크나이저가 있으면 세어서 쓴다
    report: Optional[RunReport] = None
    aborted: str = ""

    @property
    def ok(self) -> bool:
        return not (self.type_mismatches or self.determinism_failures or self.aborted)


def dryrun(
    cg: CompiledGraph,
    space: samples_mod.SampleSpace,
    n: int = 3,
    opts: Optional[RunOptions] = None,
    violation_threshold: float = 0.02,
) -> DryRunResult:
    o = opts or RunOptions()
    o.run_outputs = False
    o.use_cache = False  # 실측이 목적이므로 캐시를 쓰지 않는다
    o.determinism_audit = True

    rows = space.pick(n, seed=o.seed)
    rep = execute(cg, space, rows, o)

    res = DryRunResult(
        samples=len(rows),
        processed=rep.processed,
        determinism_failures=sorted(set(rep.determinism_failures)),
        quarantine=[f"{q.sample_key} @{q.node_id}: {q.cause}" for q in rep.quarantine],
        report=rep,
        aborted=rep.aborted,
    )

    # 선언 타입 vs 실측 (마지막으로 성공한 샘플 기준)
    for nid in cg.order:
        node = cg.nodes[nid]
        if node.kind is NodeKind.OUTPUT:
            continue
        for port, declared in node.output_types.items():
            ref = f"{nid}:{port}"
            if ref not in rep.last_values:
                continue
            v = rep.last_values[ref]
            res.measured[ref] = describe(v)
            res.type_mismatches.extend(check_against(declared, v, ref))

    # 정적 추정(G4)이 낙관적으로 기울지 않도록 실측 문자 수를 남긴다
    for ref, v in rep.last_values.items():
        if isinstance(v, dict) and "prompt" in v and "answer" in v:
            n = len(v["prompt"]) + len(v["answer"])
            if n > res.measured_chars:
                res.measured_chars = n
                res.measured_text = v["prompt"] + v["answer"]

    if rep.quarantine:
        viol = sum(1 for q in rep.quarantine if "위반" in q.cause)
        total = rep.processed + len(rep.quarantine)
        res.violation_ratio = viol / total if total else 0.0
        if res.violation_ratio > violation_threshold:
            res.aborted = res.aborted or (
                f"정답 스키마 위반율 {res.violation_ratio:.1%}가 임계 {violation_threshold:.1%}를 넘는다. "
                "정답 생성 규칙이 잘못된 채로 GPU를 태우는 것이 가장 흔한 낭비다."
            )
    return res


def render(res: DryRunResult) -> str:
    lines = [
        f"dry-run: 샘플 {res.samples}건 중 {res.processed}건 통과",
    ]
    if res.type_mismatches:
        lines.append("\n선언 타입과 실측 불일치:")
        lines += [f"  - {m}" for m in res.type_mismatches]
    if res.determinism_failures:
        lines.append("\n결정성 감사 실패(같은 입력, 다른 출력):")
        lines += [f"  - {n}" for n in res.determinism_failures]
        lines.append("  Processing 노드는 순수 함수여야 한다. 캐시가 오염되면 재현이 깨진다.")
    if res.quarantine:
        lines.append(f"\n격리된 샘플 {len(res.quarantine)}건:")
        lines += [f"  - {q}" for q in res.quarantine[:10]]
    if res.aborted:
        lines.append(f"\n중단: {res.aborted}")
    if res.measured_chars:
        lines.append(f"\n실측: 샘플당 프롬프트+정답 {res.measured_chars}자 (최대)")
        lines.append("  토큰 수는 토크나이저가 있으면 세고, 없으면 chars_per_token으로 추정한다.")
    if res.ok:
        lines.append("\n통과. G1·G2·G3를 지났다. 남은 것은 자원 예산(G4)이다.")
    return "\n".join(lines)
