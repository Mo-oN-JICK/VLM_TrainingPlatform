"""freeze 정책 — 선언한 것이 실제 requires_grad와 일치해야 한다.

모듈 그룹별 3상태: "false"(동결) / "true"(전체 학습) / "lora"(어댑터만).
그룹 이름은 백본 어댑터가 매핑하므로 백본이 바뀌어도 스키마 문법은 그대로다.
설계 문서 07 §7.2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """베이스 가중치는 동결하고 저랭크 어댑터만 학습한다."""

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.a = nn.Linear(base.in_features, r, bias=False)
        self.b = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=5**0.5)
        nn.init.zeros_(self.b.weight)  # 학습 시작 시 항등이 되도록
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.b(self.a(self.drop(x))) * self.scale


@dataclass
class FreezeReport:
    trainable: Dict[str, int] = field(default_factory=dict)  # 그룹 -> 학습 파라미터 수
    lora_modules: List[str] = field(default_factory=list)
    total_trainable: int = 0
    total_params: int = 0

    @property
    def ratio(self) -> float:
        return self.total_trainable / self.total_params if self.total_params else 0.0


def inject_lora(root: nn.Module, targets: Tuple[str, ...], r: int, alpha: int, dropout: float) -> List[str]:
    """이름이 targets로 끝나는 nn.Linear를 LoRALinear로 바꾼다."""
    replaced: List[str] = []
    for name, module in list(root.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                replaced.append(f"{name}.{child_name}" if name else child_name)
    return replaced


def apply(model: nn.Module, stage: Any, adapter: Any) -> FreezeReport:
    """선언한 정책을 실제 requires_grad로 옮기고, 무엇이 학습되는지 보고한다."""
    groups = adapter.module_groups(model)
    rep = FreezeReport()

    for p in model.parameters():
        p.requires_grad_(False)  # 기본은 전부 동결. 선언한 것만 연다

    for group, mode in stage.trainable.items():
        mods = groups.get(group) or []
        if mode == "true":
            for m in mods:
                for p in m.parameters():
                    p.requires_grad_(True)
        elif mode == "lora":
            root = adapter.lora_root(model) if hasattr(adapter, "lora_root") else (mods[0] if mods else model)
            rep.lora_modules += inject_lora(
                root, tuple(stage.lora.targets), stage.lora.r, stage.lora.alpha, stage.lora.dropout
            )
            for name, p in root.named_parameters():
                if ".a." in name or ".b." in name:
                    p.requires_grad_(True)

    # 정규식 override — 상위 몇 개 레이어만 여는 식의 세밀한 제어
    for ov in getattr(stage, "overrides", ()) or ():
        pat, mode = re.compile(ov["pattern"]), ov.get("mode", "true")
        for name, p in model.named_parameters():
            if pat.search(name):
                p.requires_grad_(mode == "true")

    for group, mods in groups.items():
        rep.trainable[group] = sum(
            p.numel() for m in mods for p in m.parameters() if p.requires_grad
        )
    rep.total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rep.total_params = sum(p.numel() for p in model.parameters())
    return rep


def verify(model: nn.Module, stage: Any, adapter: Any) -> List[str]:
    """선언과 실제가 어긋난 그룹을 보고한다. 비어 있으면 일치."""
    groups = adapter.module_groups(model)
    bad: List[str] = []
    for group, mode in stage.trainable.items():
        mods = groups.get(group) or []
        params = [p for m in mods for n, p in m.named_parameters() if ".a." not in n and ".b." not in n]
        if not params:
            continue
        any_grad = any(p.requires_grad for p in params)
        all_grad = all(p.requires_grad for p in params)
        if mode == "false" and any_grad:
            bad.append(f"{group}: false로 선언했는데 학습 대상이 있다")
        if mode == "true" and not all_grad:
            bad.append(f"{group}: true로 선언했는데 동결된 파라미터가 있다")
        if mode == "lora" and any_grad:
            bad.append(f"{group}: lora로 선언했는데 베이스 가중치가 열려 있다")
    return bad
