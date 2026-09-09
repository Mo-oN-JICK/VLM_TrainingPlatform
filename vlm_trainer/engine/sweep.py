"""스윕 실행 — 그래프 하나 위에서 레시피만 바꿔 실험 여러 개를 돌린다.

단일 GPU가 전제이므로 병렬이 아니라 **순차 큐**다. 두 가지가 이 설계의 핵심이다.

1. 전개 시점에 전 레시피의 예산을 먼저 본다. 20개 중 7개가 OOM으로 죽는 것을
   다음 날 아침에 발견하는 대신 큐에 넣기 전에 안다.
2. 전처리 파라미터가 같은 레시피는 물질화를 공유한다. 캐시 키가 같으면 같은 bake를
   재사용하므로, 스윕의 실질 비용은 학습 시간만 남는다.

설계 문서 13 §13.5.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..core.compiler import CompiledGraph, compile_project
from ..spec.canonical import hash_parts
from ..spec.recipe import Recipe, RecipeBook
from ..train.config import TrainerConfig
from . import budget as budget_mod
from . import dryrun as dryrun_mod
from . import materialize as mat_mod
from . import samples as samples_mod
from .runner import RunOptions

OK, OVER_BUDGET, FAILED = "ok", "over_budget", "failed"


def materialize_key(cg: CompiledGraph) -> str:
    """물질화 결과를 결정하는 부분만의 지문.

    경계 상류 노드들의 캐시 키를 모은 것이라, Trainer 설정만 다른 레시피들은
    같은 키를 갖고 같은 bake를 공유한다.
    """
    targets = sorted(mat_mod.boundary_targets(cg))
    return hash_parts("materialize", [cg.nodes[i].cache_key for i in targets]).split(":")[-1][:12]


@dataclass
class Entry:
    recipe: Recipe
    status: str = OK
    reason: str = ""
    bake: str = ""
    train_dir: str = ""
    peak_gb: float = 0.0
    steps: int = 0
    last_loss: float = 0.0
    estimate_gb: float = 0.0


@dataclass
class SweepReport:
    entries: List[Entry] = field(default_factory=list)
    bakes: Dict[str, str] = field(default_factory=dict)  # key -> 디렉터리
    baked_now: List[str] = field(default_factory=list)
    reused_bakes: List[str] = field(default_factory=list)

    @property
    def queued(self) -> List[Entry]:
        return [e for e in self.entries if e.status == OK]

    @property
    def rejected(self) -> List[Entry]:
        return [e for e in self.entries if e.status != OK]


def plan(
    project_path: str,
    recipes: Sequence[Recipe],
    *,
    spec_dir: str = "",
    trainer_config: Optional[str] = None,
    device: str = "",
    measured_tokens: Optional[Dict[str, int]] = None,
) -> Tuple[SweepReport, Dict[int, Tuple[CompiledGraph, TrainerConfig, str]]]:
    """컴파일 + 예산 사전 검사. GPU를 잡기 전에 무엇이 돌 수 있는지 전부 안다."""
    rep = SweepReport()
    compiled: Dict[int, Tuple[CompiledGraph, TrainerConfig, str]] = {}
    spec_dir = spec_dir or os.path.dirname(os.path.abspath(project_path))

    for r in recipes:
        e = Entry(recipe=r)
        try:
            cg = compile_project(project_path, recipe_overrides=r.overrides)
        except Exception as exc:  # 컴파일 실패도 큐에 넣기 전에 안다
            e.status, e.reason = FAILED, str(exc).splitlines()[0]
            rep.entries.append(e)
            continue

        key = materialize_key(cg)
        e.bake = key
        tns = budget_mod.train_nodes(cg)
        if not tns:
            e.status, e.reason = FAILED, "Trainer 노드가 없다"
            rep.entries.append(e)
            continue

        path = trainer_config or cg.nodes[tns[0]].params.get("config_path") or "trainer.yaml"
        cfg = TrainerConfig.load(os.path.normpath(os.path.join(spec_dir, path)))
        if device:
            cfg.budget.device = device

        b = budget_mod.for_graph(cg, cfg, tns[0], measured_text_tokens=(measured_tokens or {}).get(key, 0))
        e.estimate_gb = max((s.total for s in b.stages), default=0.0)
        if not b.ok:
            e.status = OVER_BUDGET
            e.reason = b.errors[0].splitlines()[0]
        else:
            compiled[r.id] = (cg, cfg, key)
        rep.entries.append(e)

    return rep, compiled


def run(
    project_path: str,
    recipes: Sequence[Recipe],
    *,
    run_root: str = "runs/sweeps",
    spec_dir: str = "",
    trainer_config: Optional[str] = None,
    limit: int = 0,
    shard_size: int = 64,
    device: str = "",
    torch_device: str = "auto",
    max_steps: int = 0,
    train: bool = True,
    cache_dir: str = ".cache",
    on_event: Optional[Callable[[str], None]] = None,
) -> SweepReport:
    from ..train import loop as loop_mod

    spec_dir = spec_dir or os.path.dirname(os.path.abspath(project_path))
    say = on_event or (lambda _m: None)
    os.makedirs(run_root, exist_ok=True)

    # 1) 예산 사전 검사 (실측 없이 한 번) — 무엇이 큐에 못 들어가는지 먼저 안다
    rep, compiled = plan(
        project_path, recipes, spec_dir=spec_dir, trainer_config=trainer_config, device=device
    )
    for e in rep.rejected:
        say(f"  제외 {e.recipe.label}: {e.reason}")

    # 2) 물질화는 전처리 지문별로 한 번만
    groups: Dict[str, List[int]] = {}
    for rid, (_cg, _cfg, key) in compiled.items():
        groups.setdefault(key, []).append(rid)

    measured: Dict[str, int] = {}
    for key, rids in groups.items():
        cg, cfg, _ = compiled[rids[0]]
        bake_dir = os.path.abspath(os.path.join(run_root, f"bake_{key}"))
        rep.bakes[key] = os.path.join(bake_dir, "materialized")
        existed = os.path.exists(os.path.join(bake_dir, "materialized", "manifest.jsonl"))

        space = samples_mod.load(cg.sample_space, spec_dir)
        ro = RunOptions(run_id=f"bake_{key}", cache_dir=cache_dir, spec_dir=spec_dir)
        say(f"  물질화 {key} ({'재사용' if existed else '새로 굽기'}) · 레시피 {len(rids)}개가 공유")
        mrep = mat_mod.materialize(
            cg,
            space,
            mat_mod.MaterializeOptions(
                out_dir=os.path.join(bake_dir, "materialized"),
                shard_size=shard_size,
                limit=limit,
                resume=existed,
            ),
            ro,
        )
        if not mrep.ok:
            for rid in rids:
                e = next(x for x in rep.entries if x.recipe.id == rid)
                e.status, e.reason = FAILED, mrep.aborted.splitlines()[0]
                compiled.pop(rid, None)
            continue
        (rep.reused_bakes if existed and not mrep.written else rep.baked_now).append(key)

        # 이 bake를 공유하는 레시피들의 텍스트 길이는 같다 — 한 번만 실측한다
        dr = dryrun_mod.dryrun(cg, space, n=1, opts=RunOptions(run_id=f"bake_{key}", cache_dir=cache_dir, spec_dir=spec_dir))
        if dr.measured_chars:
            measured[key] = int(dr.measured_chars / max(0.5, cfg.sequence.chars_per_token))

    # 3) 실측을 반영해 예산을 다시 보고, 통과한 것만 순차 학습
    for rid in list(compiled):
        cg, cfg, key = compiled[rid]
        e = next(x for x in rep.entries if x.recipe.id == rid)
        b = budget_mod.for_graph(cg, cfg, budget_mod.train_nodes(cg)[0], measured_text_tokens=measured.get(key, 0))
        e.estimate_gb = max((s.total for s in b.stages), default=0.0)
        if not b.ok:
            e.status, e.reason = OVER_BUDGET, b.errors[0].splitlines()[0]
            say(f"  제외 {e.recipe.label}: {e.reason}")
            continue
        if not train:
            continue

        out = os.path.abspath(os.path.join(run_root, e.recipe.label, "train"))
        say(f"  학습 {e.recipe.label} (추정 {e.estimate_gb:.1f} GB)")
        try:
            trep = loop_mod.train(
                cfg, rep.bakes[key], out, device=torch_device, max_steps=max_steps
            )
            e.train_dir = out
            e.steps = sum(s.steps for s in trep.stages)
            e.last_loss = trep.stages[-1].last_loss if trep.stages else 0.0
            e.peak_gb = max((s.peak_vram_gb for s in trep.stages), default=0.0)
        except Exception as exc:
            e.status, e.reason = FAILED, f"{type(exc).__name__}: {exc}".splitlines()[0]
            say(f"    실패: {e.reason}")

    return rep


def render(rep: SweepReport) -> str:
    lines = [
        f"스윕: 레시피 {len(rep.entries)}개 · 큐 {len(rep.queued)}개 · 제외 {len(rep.rejected)}개",
        f"  물질화: 새로 {len(rep.baked_now)}개 · 재사용 {len(rep.reused_bakes)}개 "
        f"(전처리 지문 {len(rep.bakes)}종)",
        "",
        f"  {'레시피':<22}{'bake':<14}{'추정':>8}{'peak':>8}{'step':>7}  결과",
    ]
    for e in rep.entries:
        if e.status == OK:
            result = f"loss {e.last_loss:.3f}" if e.steps else "학습 안 함"
        else:
            result = f"{e.status} — {e.reason[:48]}"
        lines.append(
            f"  {e.recipe.label:<22}{e.bake:<14}{e.estimate_gb:>7.1f}G{e.peak_gb:>7.2f}G{e.steps:>7}  {result}"
        )
    return "\n".join(lines)
