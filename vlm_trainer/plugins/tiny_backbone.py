"""로컬 소형 VLM 백본 — 다운로드 없이 학습 경로 전체를 실제로 태우기 위한 것.

가짜 학습이 아니라 진짜 학습이다. 비전 타워, 프로젝터, LLM, LoRA, 다단계 freeze,
체크포인트, 재개가 전부 이 모델로 검증된다. 실물 2B 백본은 같은 인터페이스를 채우면
그래프도 스펙도 그대로 둔 채 `backbone:` 한 줄로 교체된다.

토크나이저는 UTF-8 바이트 단위다. 어휘가 작아 소형 모델에 맞고, 무엇보다 결정적이다.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BackboneAdapter, BackboneSpec, register_backbone

# 바이트 토크나이저: 0..255 = UTF-8 바이트, 그 위에 특수 토큰
IMG_TOKEN = 256
BOS, EOS, PAD = 257, 258, 259
VOCAB = 260

HIDDEN = 128
LAYERS = 4
HEADS = 4
INTERMEDIATE = 256
VISION_DIM = 96
PATCH = 56  # 448 / 56 = 8 -> 8x8 격자를 4x4로 풀링해 16 토큰
TOKENS_PER_TILE = 16
MAX_CONTEXT = 4096


def encode(text: str, placeholder: str = "<image>") -> List[int]:
    """자리표시자를 이미지 토큰 하나로 바꾸고 나머지는 UTF-8 바이트로."""
    out: List[int] = []
    for i, chunk in enumerate(text.split(placeholder)):
        if i:
            out.append(IMG_TOKEN)
        out.extend(chunk.encode("utf-8"))
    return out


def decode(ids: Sequence[int]) -> str:
    return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


class Block(nn.Module):
    def __init__(self, hidden: int, heads: int, inter: int) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(hidden)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.n2 = nn.LayerNorm(hidden)
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.heads = heads

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        b, t, c = x.shape
        h = self.n1(x)
        q, k, v = (
            p(h).view(b, t, self.heads, c // self.heads).transpose(1, 2)
            for p in (self.q_proj, self.k_proj, self.v_proj)
        )
        a = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        x = x + self.o_proj(a.transpose(1, 2).reshape(b, t, c))
        h = self.n2(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class VisionTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, VISION_DIM, kernel_size=PATCH, stride=PATCH)
        self.block = Block(VISION_DIM, 4, VISION_DIM * 2)
        self.norm = nn.LayerNorm(VISION_DIM)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """[N,3,H,W] -> [N, TOKENS_PER_TILE, VISION_DIM]"""
        x = self.patch(images)  # [N, D, g, g]
        side = int(math.sqrt(TOKENS_PER_TILE))
        x = F.adaptive_avg_pool2d(x, (side, side)).flatten(2).transpose(1, 2)
        return self.norm(self.block(x, causal=False))


class TinyVLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision = VisionTower()
        self.proj = nn.Sequential(nn.Linear(VISION_DIM, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN))
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.pos = nn.Embedding(MAX_CONTEXT, HIDDEN)
        self.llm = nn.ModuleList(Block(HIDDEN, HEADS, INTERMEDIATE) for _ in range(LAYERS))
        self.norm = nn.LayerNorm(HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(
        self, input_ids: torch.Tensor, images: Optional[torch.Tensor], labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        b, t = input_ids.shape
        h = self.embed(input_ids)

        if images is not None and images.numel():
            # [B, n_img, 3, H, W] -> 이미지 토큰 자리에 투영된 비전 특징을 꽂는다
            bi, n_img = images.shape[0], images.shape[1]
            feats = self.proj(self.vision(images.flatten(0, 1)))  # [B*n, tokens, HIDDEN]
            feats = feats.view(bi, n_img * TOKENS_PER_TILE, HIDDEN)
            slots = (input_ids == IMG_TOKEN).nonzero(as_tuple=False)
            for row in range(bi):
                idx = slots[slots[:, 0] == row][:, 1]
                take = min(len(idx), n_img)
                if take:
                    # 자리표시자 하나가 이미지 한 장을 대표한다(토큰 평균)
                    pooled = feats[row].view(n_img, TOKENS_PER_TILE, HIDDEN).mean(dim=1)[:take]
                    h[row, idx[:take]] = pooled.to(h.dtype)

        h = h + self.pos(torch.arange(t, device=input_ids.device))[None]
        for blk in self.llm:
            h = blk(h)
        logits = self.lm_head(self.norm(h))

        out: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return out


def _count(module: nn.Module) -> float:
    return float(sum(p.numel() for p in module.parameters()))


@register_backbone
class TinyVlmAdapter(BackboneAdapter):
    """다운로드 없는 로컬 백본. BackboneAdapter 계약의 참조 구현이기도 하다."""

    _spec: Optional[BackboneSpec] = None

    @classmethod
    def spec(cls) -> BackboneSpec:
        if cls._spec is None:
            m = TinyVLM()
            cls._spec = BackboneSpec(
                id="tiny-vlm",
                params_total=_count(m),
                params_by_group={
                    "vision_tower": _count(m.vision),
                    "projector": _count(m.proj),
                    "llm": _count(m.llm) + _count(m.embed) + _count(m.lm_head),
                },
                n_layers=LAYERS,
                hidden=HIDDEN,
                intermediate=INTERMEDIATE,
                vocab=VOCAB,
                tokens_per_tile=TOKENS_PER_TILE,
                max_context=MAX_CONTEXT,
                tokenizer_id="utf8-bytes",
            )
        return cls._spec

    # ── 학습 ────────────────────────────────────────────────────────────
    @classmethod
    def build(cls, cfg: Any, stage: Any) -> nn.Module:
        return TinyVLM()

    @classmethod
    def module_groups(cls, model: nn.Module) -> Dict[str, List[nn.Module]]:
        return {
            "vision_tower": [model.vision],
            "projector": [model.proj],
            "llm": [model.llm, model.embed, model.lm_head, model.norm, model.pos],
        }

    @classmethod
    def lora_root(cls, model: nn.Module) -> nn.Module:
        return model.llm

    @classmethod
    def encode(cls, text: str, placeholder: str = "<image>") -> List[int]:
        return encode(text, placeholder)

    @classmethod
    def generate(cls, model: nn.Module, prompt: str, images: Any, max_new: int = 64) -> str:
        """탐욕적 디코딩. 빔도 샘플링도 없다 — 답이 형식을 지키는지 보는 것이 목적이고,
        그 판단에 무작위성이 끼면 두 번 돌릴 때마다 다른 결론이 난다."""
        dev = next(model.parameters()).device
        ids = encode(prompt) + [BOS]
        img = torch.zeros(0)
        if images is not None and len(images):
            side = images[0].shape[0]
            img = torch.zeros(1, len(images), 3, side, side, dtype=torch.float32)
            for j, a in enumerate(images):
                arr = np.array(a, dtype=np.uint8, copy=True)
                img[0, j] = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            img = img.to(dev)

        model.eval()
        out: List[int] = []
        with torch.no_grad():
            for _ in range(max_new):
                x = torch.tensor([ids + out], dtype=torch.long, device=dev)[:, -MAX_CONTEXT:]
                nxt = int(model(x, img if img.numel() else None)["logits"][0, -1].argmax())
                if nxt == EOS:
                    break
                out.append(nxt)
        return decode(out)

    @classmethod
    def collate(cls, batch: List[Dict[str, Any]], max_len: int) -> Dict[str, torch.Tensor]:
        """프롬프트+정답을 이어 붙이고, 손실은 정답 토큰에만 건다."""
        ids_list, lab_list, imgs_list = [], [], []
        for rec in batch:
            p = encode(rec["prompt"]) + [BOS]
            a = encode(rec["answer"]) + [EOS]
            ids = (p + a)[:max_len]
            labels = ([-100] * len(p) + a)[:max_len]
            ids_list.append(ids)
            lab_list.append(labels)
            imgs_list.append(rec.get("_images") or [])

        width = max(len(x) for x in ids_list)
        input_ids = torch.full((len(batch), width), PAD, dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        for i, (ids, lab) in enumerate(zip(ids_list, lab_list)):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)

        n_img = max((len(x) for x in imgs_list), default=0)
        images = torch.zeros(0)
        if n_img:
            side = imgs_list[0][0].shape[0]
            images = torch.zeros(len(batch), n_img, 3, side, side, dtype=torch.float32)
            for i, arrs in enumerate(imgs_list):
                for j, a in enumerate(arrs[:n_img]):
                    arr = np.array(a, dtype=np.uint8, copy=True)  # 읽기 전용 배열 경고 방지
                    images[i, j] = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return {"input_ids": input_ids, "labels": labels, "images": images}
