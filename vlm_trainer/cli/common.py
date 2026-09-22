"""여러 명령이 함께 쓰는 헬퍼.

`main.py` 에 두면 명령 모듈이 `main` 을 되불러 순환이 된다. 양쪽이 보는 것만 여기 둔다.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from ..core import humanize, registry
from ..engine import budget as budget_mod
from ..engine import materialize as materialize_mod
from ..engine import sweep as sweep_mod
from ..engine import dryrun as dryrun_mod
from ..engine import preview as preview_mod
from ..engine import samples as samples_mod
from ..engine.runner import RunOptions
from ..spec import recipe as recipe_mod
from ..train import tokens as tokens_mod
from ..train.config import TrainerConfig
from ..train import shards as shards_mod
from ..train import contract as contract_mod




def _took(ms: float) -> str:
    """`core.humanize.ms` 의 다른 이름. 터미널과 창이 같은 표기를 쓴다."""
    return humanize.ms(ms)


def _load_nodes(modules: List[str]) -> None:
    registry.load_builtin_nodes()
    for m in modules or []:
        importlib.import_module(m)


def _recipe_overrides(a: argparse.Namespace) -> Dict[str, Any]:
    """--recipe N 의 오버라이드에 --set 을 얹는다. 손으로 준 값이 마지막에 이긴다."""
    out: Dict[str, Any] = {}
    rid = getattr(a, "recipe", None)
    if rid:
        book = recipe_mod.load(a.spec)
        out.update(book.overrides_for(int(rid)))
    out.update(_overrides(getattr(a, "set", []) or []))
    return out


def _overrides(pairs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--set 은 경로=값 형식이어야 한다: {p!r}")
        k, v = p.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except json.JSONDecodeError:
            out[k.strip()] = v
    return out


def _space(a: argparse.Namespace, cg) -> samples_mod.SampleSpace:
    return samples_mod.load(cg.sample_space, os.path.dirname(os.path.abspath(a.spec)))


def _progress_path(a: argparse.Namespace, run_id: str) -> str:
    """진행 상황 스냅샷의 자리. CLI 실행도 남긴다 — 다른 터미널이나 편집기가 지켜볼 수 있다."""
    p = getattr(a, "progress", "")
    if p == "off":
        return ""
    return p or os.path.join("runs", run_id, "progress.json")


def _run_id(a: argparse.Namespace) -> str:
    import time

    return getattr(a, "run_id", "") or time.strftime("%Y%m%dT%H%M%S")


def _parse_ids(text: str) -> List[int]:
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out




def _trainer_cfg(a: argparse.Namespace, cg) -> Optional[tuple]:
    """(TrainerConfig, train_node_id). Trainer 노드가 없으면 None."""
    nodes = budget_mod.train_nodes(cg)
    if not nodes:
        return None
    tn = nodes[0]
    path = cg.nodes[tn].params.get("config_path") or "trainer.yaml"
    spec_dir = os.path.dirname(os.path.abspath(a.spec))
    cfg = TrainerConfig.load(os.path.normpath(os.path.join(spec_dir, path)))
    if getattr(a, "device", ""):
        cfg.budget.device = a.device
    for kv in getattr(a, "what_if", None) or []:
        k, _, v = kv.partition("=")
        k, v = k.strip(), v.strip()
        if k == "images":
            cfg.vision.max_images_per_sample = int(v)
        elif k == "tiles":
            cfg.vision.max_tiles = int(v)
            cfg.vision.tiling = int(v) > 1
        elif k == "max_len":
            cfg.sequence.max_len = int(v)
        elif k == "backbone":
            cfg.backbone = v
        elif k == "quantization":
            cfg.quantization.mode = v
        elif k == "per_device":
            for st in cfg.stages:
                st.per_device = int(v)
        else:
            raise SystemExit(
                f"--what-if 에 알 수 없는 키: {k!r} "
                "(images|tiles|max_len|backbone|quantization|per_device)"
            )
    return cfg, tn


def _tokenizer_id(cfg) -> str:
    """백본이 보고한 토크나이저. 없으면 백본 id 자체를 쓴다(hf:<id>가 곧 토크나이저다)."""
    from ..plugins.base import resolve_backbone

    try:
        spec = resolve_backbone(cfg.backbone).spec()
    except Exception:
        return ""
    return spec.tokenizer_id or (cfg.backbone[3:] if cfg.backbone.startswith("hf:") else "")


def _measure_tokens(a: argparse.Namespace, cg, cfg) -> Tuple[int, str]:
    """dry-run이 실제로 만든 텍스트를 센다. 토크나이저가 없으면 비율로 추정한다.

    정적 추정이 실측보다 낙관적이면 G4가 잡는다.
    """
    space = _space(a, cg)
    opts = RunOptions(
        run_id=_run_id(a),
        extra_modules=tuple(a.nodes),
        cache_dir=a.cache_dir,
        spec_dir=os.path.dirname(os.path.abspath(a.spec)),
    )
    res = dryrun_mod.dryrun(cg, space, n=1, opts=opts)
    if not res.measured_chars:
        return 0, ""
    return tokens_mod.measure(res.measured_text, res.measured_chars, cfg.sequence, _tokenizer_id(cfg))


def _train_logger(a: argparse.Namespace, run_id: str):
    """콘솔에 찍고, 같은 사실을 진행 파일에도 남긴다.

    실행(`run`)과 같은 통로다 — 다른 터미널이나 편집기가 학습을 지켜볼 수 있어야 하는데,
    stdout만 있으면 그 사실이 프로세스 안에 갇힌다.
    """
    path = _progress_path(a, run_id)
    if path:
        path = os.path.join(os.path.dirname(path), "train_progress.json")
    started = time.time()
    state = {"run_id": run_id, "phase": "train", "stage": "", "step": 0, "loss": 0.0,
             "at": 0.0, "started": started, "elapsed_ms": 0.0}

    def log(stage: str, step: int, loss: float) -> None:
        print(f"    {stage} step {step:>4} loss {loss:.4f}")
        if not path:
            return
        now = time.time()
        state.update(stage=stage, step=int(step), loss=float(loss), at=now,
                     elapsed_ms=(now - started) * 1000)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(state, fh, ensure_ascii=False)
            os.replace(tmp, path)
        except OSError:
            pass  # 관찰이 학습을 죽여서는 안 된다

    return log


