"""모델 Export — 합친 한 덩어리 + 계약.

두 가지가 전부다. **합친 모델이 학습한 모델과 같은 답을 내야 하고**, **계약 없이는
내보내지 않아야 한다.** 첫째가 깨지면 내보낸 순간 다른 모델이 되고, 둘째가 깨지면
받은 사람이 프롬프트 형식을 몰라 아무 답도 못 얻는다 — 그리고 모델 탓을 한다.
"""

from __future__ import annotations

import json
import os

import pytest

from vlm_trainer.core import registry
from vlm_trainer.core.compiler import compile_project

torch = pytest.importorskip("torch")
nn = torch.nn

from vlm_trainer.plugins.base import BackboneAdapter  # noqa: E402
from vlm_trainer.train import checkpoint as ckpt_mod  # noqa: E402
from vlm_trainer.train.freeze import LoRALinear, inject_lora  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OPEN_SPEC = os.path.join(ROOT, "solutions", "vlm_open", "projects", "01_open", "project.yaml")


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.out = nn.Linear(8, 4)

    def forward(self, x):
        return self.out(self.q_proj(x))


class _Adapter(BackboneAdapter):
    @classmethod
    def build(cls, cfg, stage):
        torch.manual_seed(0)
        return _Net()

    @classmethod
    def lora_root(cls, model):
        return model


# ── 합치기 ──────────────────────────────────────────────────────────────
def test_merging_lora_changes_nothing_about_the_answer():
    """합친 모델이 다른 답을 내면, 내보낸 순간 그것은 학습한 모델이 아니다.
    `W' = W + (B @ A) * scale` — 학습 때 forward 가 더하던 것과 같은 값이어야 한다."""
    torch.manual_seed(0)
    model = _Net()
    inject_lora(model, ("q_proj",), r=2, alpha=4, dropout=0.0)
    with torch.no_grad():                       # b 가 0이면 항등이라 시험이 안 된다
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.b.weight.normal_(0, 0.1)

    x = torch.randn(3, 8)
    model.eval()
    before = model(x)
    merged = ckpt_mod.merge_lora(model)
    after = model(x)

    assert merged == ["q_proj"]
    assert torch.allclose(before, after, atol=1e-5), (before - after).abs().max()


def test_nothing_of_ours_survives_in_the_exported_model():
    """LoRA 모듈이 남으면 받은 사람이 `vlm_trainer.train.freeze` 를 임포트해야
    가중치를 읽는다. 그것은 모델이 아니라 우리 저장소에 묶인 물건이다."""
    torch.manual_seed(0)
    model = _Net()
    inject_lora(model, ("q_proj",), r=2, alpha=4, dropout=0.0)
    ckpt_mod.merge_lora(model)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    assert isinstance(model.q_proj, nn.Linear)


def test_merging_nothing_is_not_an_error():
    """LoRA 를 안 쓴 학습도 내보낼 수 있어야 한다."""
    assert ckpt_mod.merge_lora(_Net()) == []


# ── 계약 ────────────────────────────────────────────────────────────────
def _trained(tmp_path, *, with_contract: bool = True, lora: bool = True) -> str:
    """학습이 남긴 것처럼 생긴 폴더 하나."""
    d = tmp_path / "train"
    (d / "s1").mkdir(parents=True)
    torch.manual_seed(0)
    model = _Net()
    spec = {"targets": ["q_proj"], "r": 2, "alpha": 4, "dropout": 0.0}
    if lora:
        inject_lora(model, ("q_proj",), r=2, alpha=4, dropout=0.0)
    torch.save({"model": model.state_dict(), "lora": spec if lora else {}}, d / "s1" / "c.pt")
    if with_contract:
        (d / ckpt_mod.CONTRACT).write_text(
            json.dumps({"backbone": {"id": "fake", "dtype": "bf16"}, "prompt_template": "{q}"}),
            encoding="utf-8")
    return str(d)


def test_export_refuses_without_the_contract(tmp_path, monkeypatch):
    """프롬프트 형식을 모르면 가중치만 있어도 못 쓴다. 반쪽을 내보내느니 멈춘다."""
    monkeypatch.setattr("vlm_trainer.plugins.base.resolve_backbone", lambda ref: _Adapter)
    src = _trained(tmp_path, with_contract=False)
    with pytest.raises(ckpt_mod.CheckpointError, match="계약 없이는 내보내지 않는다"):
        ckpt_mod.export(src, "fake", str(tmp_path / "out"))
    assert not os.path.isdir(tmp_path / "out"), "거부해 놓고 폴더를 만들었다"


def test_export_writes_the_weights_and_the_contract(tmp_path, monkeypatch):
    monkeypatch.setattr("vlm_trainer.plugins.base.resolve_backbone", lambda ref: _Adapter)
    rep = ckpt_mod.export(_trained(tmp_path), "fake", str(tmp_path / "out"))

    assert rep.ok and len(rep.merged) == 1
    got = json.loads(open(rep.contract, encoding="utf-8").read())
    assert got["prompt_template"] == "{q}", "원래 계약의 내용이 사라졌다"
    assert got["exported"]["merged_lora_modules"] == 1
    # 받은 사람이 이 폴더를 무엇으로 열어야 하는지
    assert got["exported"]["format"] == "state_dict"
    # 그래프의 `<image>` 가 실제로 어떤 토큰으로 나가는지는 백본만 안다
    assert got["backbone"]["image_placeholder"] == "<image>"


def test_the_stop_tag_comes_from_the_schema_not_from_a_string_in_the_code():
    """코드에 태그를 적어 두면 도메인이 바뀐 날에도 그 글자가 남고, 받은 사람은
    답이 잘리거나 끝나지 않는 이유를 모른다."""
    from vlm_trainer.train import contract as contract_mod

    registry.load_builtin_nodes()
    cg = compile_project(OPEN_SPEC)
    spec_dir = os.path.dirname(OPEN_SPEC)
    data = contract_mod.build_contract(cg, _Cfg(), spec_dir)

    assert data["answer_schema"]["id"] == "open_parts_kind"
    assert data["answer_schema"]["steps"] == ["part_type", "orientation"]
    assert data["parser"]["stop"] == ["</orientation>"], "다른 도메인의 태그가 남아 있다"


class _Cfg:
    backbone = "tiny-vlm"
    dtype = "bf16"
    quantization = None
    sequence = None


# ── 버튼 ────────────────────────────────────────────────────────────────
def test_the_export_button_only_appears_on_a_box_that_trains():
    """다른 상자에 붙여 놓고 아무 일도 안 일어나게 하느니 없는 편이 낫다.
    학습 상자는 타입 이름이 아니라 `per_sample=False` 로 찾는다."""
    pytest.importorskip("PySide6", reason="편집기는 선택 사항이다")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from vlm_trainer.ui.api import Editor
    from vlm_trainer.ui.app.inspector import Inspector

    registry.load_builtin_nodes()
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ed = Editor.open(OPEN_SPEC)
    insp = Inspector()
    insp.bind(ed)

    assert insp._trains("n_train")
    assert not insp._trains("n_infer") and not insp._trains("n_img")
