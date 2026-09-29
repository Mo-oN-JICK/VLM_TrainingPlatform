"""학습 산출물을 다시 올린다.

추론 노드와 Export 가 **같은 코드로** 올려야 한다. 둘이 갈리면 "추론에서는 잘 나왔는데
내보낸 모델은 이상하다" 가 되고, 그때 어느 쪽이 학습한 모델인지 아무도 모른다.

여기서 지키는 것 하나: **절반만 올라가는 일이 없어야 한다.** LoRA 를 끼우면 모듈 트리가
바뀌므로(`q_proj` → `q_proj.base`), 같은 트리를 먼저 만들지 않으면 `strict=False` 가
학습한 가중치를 통째로 버린다. 그래도 모델은 답을 내놓는다 — 학습 전의 답을.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from ..core.errors import VlmtError

CONTRACT = "inference_contract.json"


class CheckpointError(VlmtError):
    """체크포인트를 그대로 되살릴 수 없다. 메시지에 무엇이 어긋났는지 담는다."""

    def __init__(self, cause: str, hint: str = "") -> None:
        super().__init__(cause + ("\n" + hint if hint else ""))
        self.cause, self.hint = cause, hint


@dataclass
class BuildCfg:
    """`adapter.build` 가 읽는 만큼만. 학습 때 쓴 설정을 그대로 따라가게 한다."""

    dtype: str = "bf16"
    attn_impl: str = "sdpa"
    quantization: Any = None


@dataclass
class Quant:
    mode: str = "none"
    double_quant: bool = True
    compute_dtype: str = "bfloat16"


def build_cfg(model_dir: str) -> BuildCfg:
    """`inference_contract.json` 에서 학습 때의 자료형과 양자화를 읽는다.

    추측하지 않는다. bf16 으로 학습한 가중치를 fp16 모델에 얹으면 답이 조용히 나빠지고,
    그것을 모델 탓으로 돌리게 된다. 계약 파일이 바로 이런 자리를 위해 있다.
    """
    try:
        with open(os.path.join(model_dir, CONTRACT), "r", encoding="utf-8") as fh:
            bb = json.load(fh).get("backbone") or {}
    except (OSError, ValueError):
        return BuildCfg(quantization=Quant())
    return BuildCfg(dtype=str(bb.get("dtype") or "bf16"),
                    quantization=Quant(mode=str(bb.get("quantization") or "none")))


def latest(model_dir: str) -> str:
    """마지막 단계의 체크포인트. 단계 이름을 모르므로 가장 최근 것을 고른다."""
    if not os.path.isdir(model_dir):
        raise CheckpointError(
            f"{model_dir} 가 없다",
            "  안 잡혔다면: 없는 모델로 답을 만들려 한다.\n"
            "  추정 낭비: 없음(시작하지 않았다).\n"
            "  먼저 학습을 돌려라.")
    ckpts = [os.path.join(model_dir, d, f)
             for d in sorted(os.listdir(model_dir))
             if os.path.isdir(os.path.join(model_dir, d))
             for f in sorted(os.listdir(os.path.join(model_dir, d)))
             if f.endswith(".pt")]
    if not ckpts:
        raise CheckpointError(
            f"{model_dir} 에 체크포인트가 없다",
            "  안 잡혔다면: 학습되지 않은 모델이 답을 내고, 그 답을 보고 판단하게 된다.\n"
            "  추정 낭비: 없음(시작하지 않았다).\n"
            "  먼저 학습을 돌려라.")
    return max(ckpts, key=os.path.getmtime)


def reinject_lora(adapter: Any, model: Any, spec: Dict[str, Any]) -> int:
    """학습 때 끼운 LoRA 를 같은 자리에 다시 끼운다. 끼운 모듈 수를 돌려준다."""
    if not spec:
        return 0
    from .freeze import inject_lora

    root = adapter.lora_root(model) if hasattr(adapter, "lora_root") else model
    return len(inject_lora(root, tuple(spec.get("targets") or ()), int(spec.get("r") or 8),
                           int(spec.get("alpha") or 16), float(spec.get("dropout") or 0.0)))


def exported_format(model_dir: str) -> str:
    """내보낸 폴더면 그 형식(`huggingface` / `state_dict`), 아니면 빈 문자열.

    학습이 남긴 폴더와 Export 한 폴더는 생김새가 다르다. 학습 쪽은 단계별 하위 폴더에
    `.pt` 가 있고, Export 쪽은 합쳐진 모델이 평평하게 놓인다. 둘을 같은 코드로 열려고
    하면 Export 폴더에서 "체크포인트가 없다" 가 난다 — 방금 내보낸 그 모델을 두고.
    """
    try:
        with open(os.path.join(model_dir, CONTRACT), "r", encoding="utf-8") as fh:
            return str((json.load(fh).get("exported") or {}).get("format") or "")
    except (OSError, ValueError):
        return ""


def _load_exported(model_dir: str, backbone: str, fmt: str) -> Tuple[Any, Any]:
    """Export 폴더를 연다. LoRA 는 이미 합쳐져 있으므로 트리를 손댈 일이 없다."""
    from ..plugins.base import resolve_backbone

    if fmt == "huggingface":
        # **그 폴더 자체**를 백본으로 등록한다. 원래 모델 id 로 해소하면 가중치도
        # 프로세서도 원본을 읽게 되고, 내보낸 것이 아니라 학습 전 모델로 답하게 된다.
        from ..plugins import hf_backbone

        adapter = hf_backbone.register(model_dir)
        return adapter, adapter.build(build_cfg(model_dir), None)

    import torch

    adapter = resolve_backbone(backbone)
    model = adapter.build(build_cfg(model_dir), None)
    path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(path):
        raise CheckpointError(
            f"{path} 가 없다",
            "  안 잡혔다면: 학습 전 모델이 답을 내고, 그 답을 보고 판단하게 된다.\n"
            "  추정 낭비: 추론 시간 전부 + 그 뒤의 판단.\n"
            "  내보낸 폴더가 온전한지 확인하라.")
    res = model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False),
                                strict=False)
    unexpected = list(getattr(res, "unexpected_keys", []))
    if unexpected:
        raise CheckpointError(
            f"내보낸 가중치 {len(unexpected)}개가 모델에 들어갈 자리가 없다",
            f"  첫 몇 개: {', '.join(unexpected[:3])}\n"
            "  안 잡혔다면: 절반만 올라간 모델이 답을 낸다.\n"
            "  내보낼 때와 다른 백본으로 열고 있지 않은지 확인하라.")
    return adapter, model


def load(model_dir: str, backbone: str, *, to_cuda: bool = False) -> Tuple[Any, Any]:
    """`(어댑터, 모델)`. 학습 때와 **같은 모듈 트리**로 만들어 얹는다.

    Export 폴더면 그쪽 길로 간다 — LoRA 가 이미 합쳐져 있어 트리를 다시 만들 것이 없다.
    """
    import torch

    from ..plugins.base import resolve_backbone

    fmt = exported_format(model_dir)
    if fmt:
        adapter, model = _load_exported(model_dir, backbone, fmt)
        if to_cuda and torch.cuda.is_available():
            model = model.cuda()
        return adapter, model

    adapter = resolve_backbone(backbone)
    path = latest(model_dir)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        payload = {"model": payload}

    model = adapter.build(build_cfg(model_dir), None)
    reinject_lora(adapter, model, payload.get("lora") or {})

    res = model.load_state_dict(payload.get("model", payload), strict=False)
    unexpected = list(getattr(res, "unexpected_keys", []))
    if unexpected:
        raise CheckpointError(
            f"체크포인트에 있는 가중치 {len(unexpected)}개가 모델에 들어갈 자리가 없다",
            f"  첫 몇 개: {', '.join(unexpected[:3])}\n"
            "  안 잡혔다면: 학습이 안 된 모델이 답을 내놓고, 그 답을 보고 "
            "'파인튜닝이 소용없다' 고 판단하게 된다.\n"
            "  추정 낭비: 학습 시간 전부 + 그 뒤의 모든 판단.\n"
            "  학습 때와 같은 모듈 구조를 먼저 만들어야 한다 — 보통 LoRA 설정이 "
            "체크포인트와 어긋난 경우다.")
    if to_cuda and torch.cuda.is_available():
        model = model.cuda()
    return adapter, model


# ── LoRA 접기 ───────────────────────────────────────────────────────────
def merge_lora(model: Any) -> List[str]:
    """LoRA 를 베이스 가중치에 **더해 넣고** 원래 레이어로 되돌린다.

    내보낸 모델은 이 도구 없이도 열려야 한다. LoRA 모듈을 그대로 두면 받은 사람이
    `vlm_trainer.train.freeze` 를 임포트해야 가중치를 읽을 수 있고, 그것은 모델이
    아니라 우리 저장소에 묶인 물건이다.

    `W' = W + (B @ A) * scale` — 학습 때 `forward` 가 더하던 것과 같은 값이다.
    """
    import torch
    import torch.nn as nn

    from .freeze import LoRALinear

    merged: List[str] = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, LoRALinear):
                continue
            base: nn.Linear = child.base
            with torch.no_grad():
                delta = (child.b.weight @ child.a.weight) * child.scale
                base.weight.add_(delta.to(base.weight.dtype))
            setattr(module, child_name, base)
            merged.append(f"{name}.{child_name}" if name else child_name)
    return merged


@dataclass
class ExportReport:
    out_dir: str = ""
    backbone: str = ""
    merged: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    contract: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.files) and bool(self.contract)


def export(model_dir: str, backbone: str, out_dir: str) -> ExportReport:
    """합친 모델 한 덩어리 + 계약. **계약 없이는 내보내지 않는다.**

    프롬프트를 어떤 형식으로 넣어야 하는지 모르면 가중치만 있어도 쓸 수 없다.
    자리표시자를 어떻게 적는지, 답을 어떤 태그로 받는지가 전부 계약에 있다.
    """
    src = os.path.join(model_dir, CONTRACT)
    if not os.path.exists(src):
        raise CheckpointError(
            f"{src} 가 없다 — 계약 없이는 내보내지 않는다",
            "  안 잡혔다면: 가중치만 받은 사람이 프롬프트 형식을 몰라 "
            "아무 답도 못 얻고, 모델이 나쁘다고 결론 내린다.\n"
            "  추정 낭비: 학습 시간 전부.\n"
            "  학습을 다시 돌려라 — `vlmt train` 이 계약을 함께 남긴다.")

    adapter, model = load(model_dir, backbone)
    rep = ExportReport(out_dir=os.path.abspath(out_dir), backbone=backbone)
    rep.merged = merge_lora(model)

    os.makedirs(rep.out_dir, exist_ok=True)
    rep.files = list(adapter.save(model, rep.out_dir))

    rep.contract = os.path.join(rep.out_dir, CONTRACT)
    with open(src, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # 받은 사람이 알아야 하는 것 둘을 계약에 덧붙인다. 그래프의 `<image>` 가
    # 실제로 어떤 토큰으로 나가는지는 백본만 안다.
    data.setdefault("backbone", {})["image_placeholder"] = adapter.image_placeholder()
    data["exported"] = {
        "merged_lora_modules": len(rep.merged),
        "files": rep.files,
        # 이 폴더를 무엇으로 열어야 하는가. `state_dict` 는 모델 클래스가 따로 필요하다는
        # 뜻이고(우리 `tiny-vlm` 이 그렇다), `huggingface` 는 그냥 열린다.
        "format": getattr(adapter, "export_format", "state_dict"),
    }
    with open(rep.contract, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return rep


def render(rep: ExportReport) -> str:
    lines = [f"Export -> {rep.out_dir}",
             f"  백본: {rep.backbone}",
             f"  LoRA {len(rep.merged)}개 모듈을 베이스에 합쳤다"
             if rep.merged else "  LoRA 없음 (합칠 것이 없다)"]
    for f in rep.files:
        lines.append(f"  파일: {f}")
    lines.append(f"  계약: {rep.contract}")
    lines.append("  계약에 프롬프트 형식과 이미지 자리표시자가 들어 있다 — "
                 "가중치만으로는 쓸 수 없다.")
    return "\n".join(lines)
