"""물질화 — 전처리와 학습이 한 장의 GPU를 두고 경쟁하지 않게 만드는 분리막.

경계까지 전 샘플을 미리 구워 디스크에 두고, 학습은 그 산출물만 읽는다.
전문가 모델은 이 단계에서만 살고 끝나면 프로세스째 종료되어 VRAM을 완전히 반납한다.

커밋 단위는 shard다. shard 디렉터리와 jsonl은 먼저 쓰이고, **매니페스트에 줄이 붙는 순간**
커밋된 것으로 인정한다. 중간에 죽으면 매니페스트에 없는 shard는 고아로 보고 지운 뒤 다시 만든다.
설계 문서 08 §8.2, §8.7.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from ..core.compiler import CompiledGraph
from ..core.errors import StructureError
from . import samples as samples_mod
from . import worker
from .journal import Journal, PHASE_DONE, RUN_START, SHARD_COMMITTED
from .runner import RunOptions, RunReport, ancestors, execute

MANIFEST = "manifest.jsonl"
SHARD_PREFIX = "shard-"


@dataclass
class MaterializeOptions:
    out_dir: str = ""  # 비면 runs/<run_id>/materialized
    shard_size: int = 64  # shard 하나에 담을 샘플 수
    resume: bool = False
    limit: int = 0
    split: str = ""
    strict_spec_match: bool = True


@dataclass
class MaterializeReport:
    out_dir: str = ""
    boundary: List[str] = field(default_factory=list)
    written: int = 0
    reused: int = 0
    shards: List[str] = field(default_factory=list)
    orphans_removed: List[str] = field(default_factory=list)
    quarantine: int = 0
    aborted: str = ""
    run: Optional[RunReport] = None

    @property
    def ok(self) -> bool:
        return not self.aborted


def _shard_name(i: int) -> str:
    return f"{SHARD_PREFIX}{i:05d}"


def _read_manifest(out_dir: str) -> List[Dict[str, Any]]:
    p = os.path.join(out_dir, MANIFEST)
    if not os.path.exists(p):
        return []
    out: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # 잘린 마지막 줄은 커밋되지 않은 것으로 본다
    return out


def _gc_orphans(out_dir: str, committed: Set[str]) -> List[str]:
    """매니페스트에 없는 shard는 커밋되지 않은 것이다. 지우고 다시 만든다."""
    removed: List[str] = []
    if not os.path.isdir(out_dir):
        return removed
    for name in sorted(os.listdir(out_dir)):
        if not name.startswith(SHARD_PREFIX):
            continue
        stem = name[:-6] if name.endswith(".jsonl") else name
        if stem in committed:
            continue
        path = os.path.join(out_dir, name)
        shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else os.remove(path)
        removed.append(name)
    return removed


def boundary_targets(cg: CompiledGraph) -> Set[str]:
    b = [n for n in (cg.materialize.get("boundary") or []) if n in cg.nodes]
    if not b:
        raise StructureError(
            "materialize.boundary가 비어 있다. 물질화할 경계 노드를 스펙에 지정하라.\n"
            "  경계가 없으면 전처리가 학습 루프 안에서 돌며 GPU를 두고 경쟁한다."
        )
    out: Set[str] = set()
    for n in b:
        out |= ancestors(cg, n)
    return out


def materialize(
    cg: CompiledGraph,
    space: samples_mod.SampleSpace,
    opts: Optional[MaterializeOptions] = None,
    run_opts: Optional[RunOptions] = None,
) -> MaterializeReport:
    o = opts or MaterializeOptions()
    ro = run_opts or RunOptions()
    ro.run_outputs = False  # 경계까지만. Output 노드는 학습 단계의 몫이다

    targets = boundary_targets(cg)
    boundary = [n for n in (cg.materialize.get("boundary") or []) if n in cg.nodes]
    out_dir = os.path.abspath(o.out_dir or os.path.join("runs", ro.run_id, "materialized"))
    os.makedirs(out_dir, exist_ok=True)

    rep = MaterializeReport(out_dir=out_dir, boundary=boundary)
    journal = Journal(os.path.join(os.path.dirname(out_dir), "journal.jsonl"))
    replay = journal.replay()

    if o.resume and replay.spec_hash and o.strict_spec_match and replay.spec_hash != cg.spec_hash:
        rep.aborted = (
            f"재개 거부: 스펙이 바뀌었다.\n"
            f"  저널의 spec_hash {replay.spec_hash}\n"
            f"  현재    spec_hash {cg.spec_hash}\n"
            "  같은 run_id로 다른 실험을 이어 붙이면 재현이 깨진다. 새 run_id로 시작하라."
        )
        return rep

    committed = {m["shard"] for m in _read_manifest(out_dir)}
    done_keys: Set[str] = set()
    if o.resume:
        for m in _read_manifest(out_dir):
            done_keys.update(m.get("keys") or ())
        rep.shards = sorted(committed)
        rep.reused = len(done_keys)
    else:
        committed = set()
        done_keys = set()

    rep.orphans_removed = _gc_orphans(out_dir, committed)
    if not o.resume:
        # 새로 시작하면 매니페스트도 비운다
        open(os.path.join(out_dir, MANIFEST), "w", encoding="utf-8").close()
        journal.append(RUN_START, phase="materialize", spec_hash=cg.spec_hash, out_dir=out_dir)
    elif not replay.spec_hash:
        journal.append(RUN_START, phase="materialize", spec_hash=cg.spec_hash, out_dir=out_dir)

    src = space.split(o.split) if o.split else space
    rows = [r for r in src.rows if str(r[src.key]) not in done_keys]
    if o.limit:
        rows = rows[: max(0, o.limit - len(done_keys))]

    buf: List[Dict[str, Any]] = []
    buf_keys: List[str] = []
    next_index = len(committed)

    def flush() -> None:
        nonlocal next_index, buf, buf_keys
        if not buf:
            return
        name = _shard_name(next_index)
        shard_dir = os.path.join(out_dir, name)
        os.makedirs(os.path.join(shard_dir, "images"), exist_ok=True)
        recs = [_write_images(r, shard_dir) for r in buf]

        tmp = os.path.join(out_dir, name + ".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, os.path.join(out_dir, name + ".jsonl"))  # Windows에서도 원자적

        # 여기까지는 아직 커밋이 아니다. 매니페스트에 줄이 붙는 순간이 커밋이다.
        entry = {"shard": name, "n": len(recs), "keys": list(buf_keys)}
        with open(os.path.join(out_dir, MANIFEST), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        journal.append(SHARD_COMMITTED, **entry)

        rep.shards.append(name)
        rep.written += len(recs)
        next_index += 1
        buf, buf_keys = [], []

    def on_sample(key: str, values: Dict[str, Any]) -> None:
        for node_id in boundary:
            for port in cg.nodes[node_id].output_types:
                v = values.get(f"{node_id}:{port}")
                if isinstance(v, dict) and "prompt" in v and "answer" in v:
                    buf.append(v)
                    buf_keys.append(key)
                    break
        if len(buf) >= o.shard_size:
            flush()

    try:
        run = execute(cg, space, rows, ro, targets=targets, on_sample=on_sample)
        rep.run = run
        rep.quarantine = len(run.quarantine)
        if run.aborted:
            rep.aborted = run.aborted
        flush()
    finally:
        # 전문가 모델이 잡고 있던 GPU를 완전히 반납한다. 학습은 그 뒤에 시작한다.
        worker.shutdown()

    if not rep.aborted:
        journal.append(PHASE_DONE, phase="materialize", shards=len(rep.shards), samples=rep.written + rep.reused)
    return rep


def _write_images(sample: Dict[str, Any], shard_dir: str) -> Dict[str, Any]:
    from PIL import Image as PILImage

    rec: Dict[str, Any] = {
        "id": sample.get("id", ""),
        "prompt": sample.get("prompt", ""),
        "answer": sample.get("answer", ""),
        "meta": sample.get("meta", {}),
        "images": [],
    }
    for i, arr in enumerate(sample.get("images") or []):
        rel = os.path.join("images", f"{rec['id']}_{i}.png")
        PILImage.fromarray(np.asarray(arr, dtype=np.uint8)).save(os.path.join(shard_dir, rel))
        rec["images"].append(rel.replace("\\", "/"))
    return rec


def render(rep: MaterializeReport) -> str:
    lines = [
        f"물질화 경계: {', '.join(rep.boundary)}",
        f"  출력: {rep.out_dir}",
        f"  새로 구움 {rep.written}건 · 재사용 {rep.reused}건 · shard {len(rep.shards)}개",
    ]
    if rep.orphans_removed:
        lines.append(
            f"  커밋되지 않은 shard {len(rep.orphans_removed)}개를 지웠다: {', '.join(rep.orphans_removed)}"
        )
    if rep.quarantine:
        lines.append(f"  격리 {rep.quarantine}건")
    if rep.aborted:
        lines.append("")
        lines.append(rep.aborted)
    else:
        lines.append("  학습은 이 산출물만 읽는다. 전문가 모델 워커는 종료되었다.")
    return "\n".join(lines)
