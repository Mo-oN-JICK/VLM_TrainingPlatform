"""HuggingFace 백본 어댑터 — 실물 VLM을 같은 인터페이스로 들여온다.

`backbone: hf:<모델 경로 또는 id>` 한 줄이면 그래프도 스펙도 그대로 둔 채 백본이 바뀐다.

핵심 제약 하나: **`spec()`은 가중치를 로드하지 않는다.** `config.json`만 읽어 파라미터 수와
형상을 답한다. 예산 게이트(G4)가 학습 시작 전에 호출하기 때문이고, 그 시점에 5GB를 올리면
게이트의 의미가 사라진다.

`build`/`collate`는 transformers가 있을 때만 동작한다. 실물 모델로 아직 검증되지 않았다 —
`tiny_backbone.py`가 검증된 참조 구현이고, 이 파일은 같은 계약의 HF 판이다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..core.errors import RegistrationError
from .base import BackboneAdapter, BackboneSpec, register_backbone

PREFIX = "hf:"

# config.json의 키 이름은 모델 계열마다 다르다. 흔한 것부터 찾는다.
_HIDDEN = ("hidden_size", "d_model", "n_embd")
_LAYERS = ("num_hidden_layers", "n_layer", "num_layers")
_INTER = ("intermediate_size", "ffn_dim", "n_inner")
_VOCAB = ("vocab_size",)
_CTX = ("max_position_embeddings", "max_sequence_length", "n_positions")


def _pick(d: Dict[str, Any], keys: Tuple[str, ...], default: int = 0) -> int:
    for k in keys:
        if isinstance(d.get(k), int):
            return int(d[k])
    return default


def find_config(ref: str) -> str:
    """모델 참조를 config.json 경로로 해소한다. 없으면 무엇을 해야 하는지 말한다."""
    ref = ref[len(PREFIX) :] if ref.startswith(PREFIX) else ref

    if os.path.isdir(ref) and os.path.exists(os.path.join(ref, "config.json")):
        return os.path.join(ref, "config.json")
    if os.path.isfile(ref) and ref.endswith(".json"):
        return ref

    # HF 캐시에서 찾는다 (다운로드는 하지 않는다)
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    cache = os.path.join(home, "hub", "models--" + ref.replace("/", "--"), "snapshots")
    if os.path.isdir(cache):
        for snap in sorted(os.listdir(cache)):
            cand = os.path.join(cache, snap, "config.json")
            if os.path.exists(cand):
                return cand

    raise RegistrationError(
        f"백본 {ref!r}의 config.json을 찾을 수 없다.\n"
        f"  찾아본 곳: 로컬 디렉터리, {cache}\n"
        "  먼저 모델을 내려받아라:\n"
        f'    huggingface-cli download {ref}\n'
        "  또는 로컬 경로를 직접 지정하라: backbone: hf:D:/models/<dir>"
    )


def _vision_tokens(cfg: Dict[str, Any]) -> int:
    """타일 하나가 만드는 비전 토큰 수. 모델 계열마다 이름이 다르다."""
    v = cfg.get("vision_config") or cfg.get("vision_tower_config") or {}
    if isinstance(v.get("num_image_tokens"), int):
        return int(v["num_image_tokens"])
    img, patch = v.get("image_size"), v.get("patch_size")
    if isinstance(img, int) and isinstance(patch, int) and patch:
        grid = (img // patch) ** 2
        merge = v.get("spatial_merge_size") or cfg.get("spatial_merge_size") or 1
        return max(1, grid // (int(merge) ** 2))
    return 256  # 보수적 기본값. 실물 모델에서는 config가 답한다


def _params_from_config(cfg: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """config만으로 파라미터 수를 추정한다.

    가중치를 열지 않으므로 근사다. 예산은 보수적으로 잡아야 하므로 올림 쪽으로 센다.
    """
    text = cfg.get("text_config") or cfg
    h = _pick(text, _HIDDEN, 2048)
    n = _pick(text, _LAYERS, 24)
    inter = _pick(text, _INTER, h * 4)
    vocab = _pick(text, _VOCAB, 32000)

    attn = 4 * h * h
    mlp = 3 * h * inter
    llm = n * (attn + mlp) + 2 * vocab * h  # 임베딩 + lm_head

    v = cfg.get("vision_config") or {}
    vh = _pick(v, _HIDDEN, 1024)
    vn = _pick(v, _LAYERS, 24)
    vision = vn * (4 * vh * vh + 3 * vh * vh * 4) if vh else 0.0
    projector = 2.0 * vh * h if vh else 0.0

    return float(llm + vision + projector), {
        "llm": float(llm),
        "vision_tower": float(vision),
        "projector": float(projector),
    }


def spec_from_config(path: str, model_id: str) -> BackboneSpec:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    text = cfg.get("text_config") or cfg
    total, groups = _params_from_config(cfg)
    return BackboneSpec(
        id=model_id,
        params_total=total,
        params_by_group=groups,
        n_layers=_pick(text, _LAYERS, 24),
        hidden=_pick(text, _HIDDEN, 2048),
        intermediate=_pick(text, _INTER, _pick(text, _HIDDEN, 2048) * 4),
        vocab=_pick(text, _VOCAB, 32000),
        tokens_per_tile=_vision_tokens(cfg),
        max_context=_pick(text, _CTX, 4096),
        tokenizer_id=model_id,
        supports_quantization=("none", "int8", "nf4"),
        supports_attn=("sdpa", "eager"),
    )


class HFBackbone(BackboneAdapter):
    """모델 하나에 대응하는 어댑터. `register(ref)`가 클래스를 만들어 등록한다."""

    model_ref: str = ""
    config_path: str = ""

    @classmethod
    def spec(cls) -> BackboneSpec:
        return cls.spec_data

    # ── 학습 (transformers 필요, 실물 모델로 아직 미검증) ───────────────
    @classmethod
    def build(cls, cfg: Any, stage: Any) -> Any:
        try:
            import torch
            from transformers import AutoModelForVision2Seq
        except ImportError as e:
            raise RegistrationError(
                "실물 백본을 쓰려면 transformers가 필요하다:\n"
                "  python -m pip install transformers accelerate safetensors\n"
                f"  ({e})"
            ) from None

        kw: Dict[str, Any] = {"dtype": getattr(torch, cfg.dtype, torch.bfloat16)}
        if cfg.quantization.mode in ("nf4", "int8"):
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                raise RegistrationError(
                    "양자화에는 bitsandbytes가 필요하다: python -m pip install bitsandbytes"
                ) from None
            kw["quantization_config"] = (
                BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=cfg.quantization.double_quant,
                    bnb_4bit_compute_dtype=getattr(torch, cfg.quantization.compute_dtype, torch.bfloat16),
                )
                if cfg.quantization.mode == "nf4"
                else BitsAndBytesConfig(load_in_8bit=True)
            )
        model = AutoModelForVision2Seq.from_pretrained(
            os.path.dirname(cls.config_path), attn_implementation=cfg.attn_impl, **kw
        )
        return model

    @classmethod
    def module_groups(cls, model: Any) -> Dict[str, List[Any]]:
        """모델 계열마다 이름이 다르다. 흔한 이름을 훑어 그룹을 만든다."""
        groups: Dict[str, List[Any]] = {"vision_tower": [], "projector": [], "llm": []}
        for name, mod in model.named_children():
            low = name.lower()
            if "vision" in low or "visual" in low or "image_encoder" in low:
                groups["vision_tower"].append(mod)
            elif "proj" in low or "connector" in low or "merger" in low or "mm_" in low:
                groups["projector"].append(mod)
            else:
                groups["llm"].append(mod)
        return groups

    @classmethod
    def lora_root(cls, model: Any) -> Any:
        groups = cls.module_groups(model)
        return groups["llm"][0] if groups["llm"] else model

    @classmethod
    def processor(cls) -> Any:
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(os.path.dirname(cls.config_path))

    @classmethod
    def collate(cls, batch: List[Dict[str, Any]], max_len: int) -> Dict[str, Any]:
        """프로세서로 이미지와 텍스트를 함께 묶고, 손실은 정답 토큰에만 건다."""
        import torch

        proc = cls.processor()
        texts = [f"{r['prompt']}\n{r['answer']}" for r in batch]
        images = [r.get("_images") or [] for r in batch]
        enc = proc(
            text=texts,
            images=images if any(images) else None,
            return_tensors="pt",
            padding=True,
            truncation=False,  # 잘림은 G4가 막는다. 여기서 조용히 자르지 않는다
        )
        labels = enc["input_ids"].clone()
        pad = getattr(proc.tokenizer, "pad_token_id", None)
        if pad is not None:
            labels[labels == pad] = -100
        for i, r in enumerate(batch):  # 프롬프트 구간은 손실에서 제외
            n_prompt = len(proc.tokenizer(r["prompt"])["input_ids"])
            labels[i, :n_prompt] = -100
        enc["labels"] = labels
        return dict(enc)


def register(ref: str) -> type:
    """`hf:<경로 또는 id>`를 레지스트리에 올린다. config.json만 읽는다."""
    model_id = ref if ref.startswith(PREFIX) else PREFIX + ref
    path = find_config(ref)
    cls = type(
        "HFBackbone_" + model_id.replace("/", "_").replace(":", "_").replace(".", "_"),
        (HFBackbone,),
        {"model_ref": ref, "config_path": path, "spec_data": spec_from_config(path, model_id)},
    )
    return register_backbone(cls)
