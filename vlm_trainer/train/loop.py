"""학습 루프 — 다단계, 체크포인트, step 단위 재개.

데이터는 물질화된 shard에서만 읽는다. 원본 데이터도 그래프도 다시 읽지 않는다.
단계는 `init_from`으로 이어지고, 각 단계마다 freeze 정책이 실제 requires_grad로 적용된다.
설계 문서 07, 08 §8.7.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from ..engine.journal import Journal
from ..plugins.base import resolve_backbone
from . import freeze as freeze_mod
from . import shards as shards_mod
from .config import Stage, TrainerConfig

CKPT = "ckpt.pt"


class ShardDataset(Dataset):
    """물질화 산출물만 읽는다. 워커로 넘어가야 하므로 경로만 들고 있는다(Windows spawn)."""

    def __init__(self, out_dir: str, with_images: bool = True) -> None:
        self.out_dir = out_dir
        self.with_images = with_images
        self.index: List[Dict[str, Any]] = [
            {"shard": r["_shard"], "id": r["id"], "prompt": r["prompt"], "answer": r["answer"],
             "images": r.get("images") or []}
            for r in shards_mod.iter_samples(out_dir)
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        rec = dict(self.index[i])
        if self.with_images:
            base = os.path.join(self.out_dir, rec["shard"])
            rec["_images"] = [shards_mod._load(os.path.join(base, r)) for r in rec["images"]]
        return rec


@dataclass
class StageResult:
    name: str
    steps: int = 0
    epochs: int = 0
    first_loss: float = 0.0
    last_loss: float = 0.0
    trainable: Dict[str, int] = field(default_factory=dict)
    lora_modules: int = 0
    seconds: float = 0.0
    peak_vram_gb: float = 0.0
    ckpt: str = ""
    resumed_from: int = 0


@dataclass
class TrainReport:
    device: str = "cpu"
    samples: int = 0
    stages: List[StageResult] = field(default_factory=list)
    out_dir: str = ""
    contract: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.stages) and all(s.steps > 0 for s in self.stages)


def _pick_device(requested: str = "auto") -> torch.device:
    if requested not in ("auto", ""):
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_atomic(path: str, payload: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # Windows에서도 원자적


def _stage_dir(out_dir: str, stage: Stage) -> str:
    d = os.path.join(out_dir, stage.name)
    os.makedirs(d, exist_ok=True)
    return d


def train(
    cfg: TrainerConfig,
    materialized_dir: str,
    out_dir: str,
    *,
    device: str = "auto",
    journal: Optional[Journal] = None,
    resume: bool = False,
    max_steps: int = 0,
    log_every: int = 10,
    on_log: Optional[Any] = None,
) -> TrainReport:
    adapter = resolve_backbone(cfg.backbone)
    dev = _pick_device(device)
    ds = ShardDataset(materialized_dir)
    if not len(ds):
        raise FileNotFoundError(
            f"물질화 산출물이 비어 있다: {materialized_dir}\n"
            "  먼저 `vlmt materialize`로 경계까지 구워야 학습이 읽을 것이 생긴다."
        )

    rep = TrainReport(device=str(dev), samples=len(ds), out_dir=out_dir)
    os.makedirs(out_dir, exist_ok=True)
    model = adapter.build(cfg, cfg.stages[0]).to(dev)
    prev_stage_ckpt: Optional[str] = None

    for stage in cfg.stages:
        sdir = _stage_dir(out_dir, stage)
        ckpt_path = os.path.join(sdir, CKPT)
        res = StageResult(name=stage.name)

        # 단계 연결: init_from이 가리키는 단계의 가중치에서 출발한다
        if stage.init_from and prev_stage_ckpt and os.path.exists(prev_stage_ckpt):
            state = torch.load(prev_stage_ckpt, map_location=dev, weights_only=False)
            model.load_state_dict(state["model"], strict=False)

        fr = freeze_mod.apply(model, stage, adapter)
        model.to(dev)
        res.trainable, res.lora_modules = dict(fr.trainable), len(fr.lora_modules)

        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError(
                f'stage "{stage.name}": 학습할 파라미터가 없다. trainable 선언을 확인하라 — {stage.trainable}'
            )
        opt = torch.optim.AdamW(params, lr=stage.lr, weight_decay=0.0)

        start_step, start_epoch, consumed = 0, 0, 0
        if resume and os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=dev, weights_only=False)
            model.load_state_dict(state["model"], strict=False)
            try:
                opt.load_state_dict(state["optimizer"])
            except ValueError:
                pass  # 파라미터 집합이 바뀐 단계는 옵티마이저 상태를 이어받지 않는다
            start_step = int(state.get("step", 0))
            start_epoch = int(state.get("epoch", 0))
            consumed = int(state.get("consumed", 0))
            res.resumed_from = start_step

        loader = DataLoader(
            ds,
            batch_size=stage.per_device,
            shuffle=False,  # 순열은 seed로 재현한다(아래 sampler 대신 결정적 순서)
            num_workers=0,  # Windows spawn 워커는 프로파일에서 켠다
            collate_fn=lambda b: adapter.collate(b, cfg.sequence.max_len),
        )

        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        model.train()
        t0 = time.perf_counter()
        step = start_step
        seen = consumed

        for epoch in range(start_epoch, stage.epochs):
            for i, batch in enumerate(loader):
                if seen and i * stage.per_device < seen:
                    continue  # 재개: 이미 소비한 샘플은 건너뛴다
                out = model(
                    batch["input_ids"].to(dev),
                    batch["images"].to(dev) if batch["images"].numel() else None,
                    labels=batch["labels"].to(dev),
                )
                loss = out["loss"] / max(1, stage.grad_accum)
                loss.backward()
                if (i + 1) % stage.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1

                value = float(out["loss"].detach())
                res.first_loss = res.first_loss or value
                res.last_loss = value
                if on_log and step and step % log_every == 0:
                    on_log(stage.name, step, value)
                if max_steps and step >= max_steps:
                    break
            seen = 0
            res.epochs = epoch + 1
            if max_steps and step >= max_steps:
                break

        res.steps = step
        res.seconds = time.perf_counter() - t0
        if dev.type == "cuda":
            res.peak_vram_gb = torch.cuda.max_memory_allocated(dev) / (1 << 30)

        _save_atomic(
            ckpt_path,
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "step": step,
                "epoch": res.epochs,
                "consumed": 0,
                "stage": stage.name,
                "backbone": cfg.backbone,
                "config": cfg.digest(),
            },
        )
        res.ckpt = ckpt_path
        prev_stage_ckpt = ckpt_path
        if journal is not None:
            journal.append(
                "ckpt", phase="train", stage=stage.name, step=step, path=ckpt_path,
                loss=round(res.last_loss, 5),
            )
        rep.stages.append(res)

    return rep


def render(rep: TrainReport) -> str:
    lines = [f"학습 [{rep.device}] 샘플 {rep.samples}건 · 산출물 {rep.out_dir}"]
    for s in rep.stages:
        tr = ", ".join(f"{k} {v:,}" for k, v in s.trainable.items() if v)
        lines.append(
            f"  {s.name:<18} step {s.steps:>4} · epoch {s.epochs} · "
            f"loss {s.first_loss:.3f} -> {s.last_loss:.3f} · {s.seconds:.1f}s"
            + (f" · peak {s.peak_vram_gb:.2f} GB" if s.peak_vram_gb else "")
        )
        lines.append(f"    학습 파라미터: {tr or '없음'}" + (f" · LoRA {s.lora_modules}개 모듈" if s.lora_modules else ""))
        if s.resumed_from:
            lines.append(f"    step {s.resumed_from}에서 재개")
    if rep.contract:
        lines.append(f"  추론 계약: {rep.contract.get('contract')}")
        lines.append(f"  추론 그래프: {rep.contract.get('graph')}")
    return "\n".join(lines)
