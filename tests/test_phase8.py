"""Phase 8 — 확장 지점.

완료 조건 두 개를 고정한다.
  1. `distributed.enabled: true`가 인터페이스 수준에서 동작하되, 기본 프로파일은 단일 GPU다.
  2. `StorageBackend`를 교체해도 캐시 키 계산이 동일하다.
"""

from __future__ import annotations

import pytest

from vlm_trainer.engine.cache import CacheStore, sample_key_hash
from vlm_trainer.engine.storage import InMemory, LocalFiles, all_backends, resolve_backend
from vlm_trainer.train.config import Distributed, TrainerConfig

NODE_KEY = "b2:9f3c1a77bd4e0021aa55"
SAMPLE = sample_key_hash("s001", "fp", "space")


# ── 조건 2: 백엔드를 갈아도 키는 그대로 ─────────────────────────────────


def test_the_key_does_not_depend_on_the_backend(tmp_path):
    """키 계산은 CacheStore의 몫이다. 백엔드가 키에 손대면 저장소 교체가
    캐시를 통째로 무효로 만들거나, 더 나쁘게는 다른 계산이 같은 키를 공유한다."""
    local = CacheStore(root=str(tmp_path), backend="local")
    mem = CacheStore(root=str(tmp_path), backend="memory")

    assert local.key_for(NODE_KEY, SAMPLE) == mem.key_for(NODE_KEY, SAMPLE)
    assert local.key_for(NODE_KEY, SAMPLE).endswith(f"{SAMPLE}.pkl")


def test_every_backend_round_trips_the_same_values(tmp_path):
    value = {"a": [1, 2, 3], "b": "한글", "c": (4.5, None)}

    for name in all_backends():
        store = CacheStore(root=str(tmp_path / name), backend=name)
        assert store.get(NODE_KEY, SAMPLE) == (False, None)
        store.put(NODE_KEY, SAMPLE, value)

        hit, back = store.get(NODE_KEY, SAMPLE)
        assert hit and back == value, name
        assert store.stats == {"hits": 1, "misses": 1, "writes": 1}, name


def test_clearing_one_node_leaves_the_others(tmp_path):
    other = "b2:0000ffff11112222cccc"
    for name in all_backends():
        store = CacheStore(root=str(tmp_path / ("c" + name)), backend=name)
        store.put(NODE_KEY, SAMPLE, 1)
        store.put(other, SAMPLE, 2)

        assert store.clear(NODE_KEY) == 1
        assert store.get(NODE_KEY, SAMPLE)[0] is False
        assert store.get(other, SAMPLE) == (True, 2), name


def test_a_corrupt_blob_is_a_miss_not_a_crash(tmp_path):
    """캐시가 깨졌으면 다시 계산하면 된다. 실행을 죽일 이유가 없다."""
    store = CacheStore(root=str(tmp_path), backend="memory")
    store.store.blobs[store.key_for(NODE_KEY, SAMPLE)] = "pickle이 아니다".encode("utf-8")
    assert store.get(NODE_KEY, SAMPLE) == (False, None)
    assert store.misses == 1


def test_an_unknown_backend_says_what_exists(tmp_path):
    from vlm_trainer.core.errors import RegistrationError

    with pytest.raises(RegistrationError, match="local"):
        resolve_backend("s3", str(tmp_path))


def test_the_default_backend_is_local_files(tmp_path):
    store = CacheStore(root=str(tmp_path))
    assert isinstance(store.store, LocalFiles)
    store.put(NODE_KEY, SAMPLE, 7)
    assert (tmp_path / "9f" / "3c1a77bd4e0021aa").is_dir(), "로컬 백엔드는 키를 경로로 쓴다"


def test_memory_backend_keeps_nothing_on_disk(tmp_path):
    store = CacheStore(root=str(tmp_path), backend="memory")
    store.put(NODE_KEY, SAMPLE, 7)
    assert isinstance(store.store, InMemory)
    assert not any(tmp_path.iterdir())


# ── 조건 1: 다중 GPU는 확장 지점이지 기본값이 아니다 ────────────────────


def _cfg(**dist) -> TrainerConfig:
    c = TrainerConfig(backbone="tiny-vlm")
    c.distributed = Distributed(**dist)
    return c


def test_the_default_profile_still_refuses_multi_gpu():
    errs = _cfg(enabled=True, strategy="ddp", world_size=4).profile_errors("windows_single_gpu")
    assert errs and "단일 GPU" in "\n".join(errs)


def test_the_extension_profile_accepts_ddp():
    assert not _cfg(enabled=True, strategy="ddp", world_size=4).profile_errors("linux_multi_gpu")


def test_the_extension_profile_adds_no_other_restrictions():
    """확장 프로파일이 기본 프로파일과 다른 점은 다중 GPU 하나뿐이어야 한다."""
    c = _cfg()
    c.offload = "disk"
    c.attn_impl = "flash_attn2"
    assert c.profile_errors("windows_single_gpu")
    assert not c.profile_errors("linux_multi_gpu")


@pytest.mark.parametrize(
    "dist,needle",
    [
        (dict(enabled=False, world_size=4), "켜지 않은 다중 GPU는 없다"),
        (dict(enabled=False, strategy="ddp"), "strategy를 none으로"),
        (dict(enabled=True, strategy="ddp", world_size=1), "장치가 둘 이상"),
        (dict(enabled=True, strategy="fsdp", world_size=4), "아직 예산을 계산하지 못한다"),
        (dict(enabled=True, strategy="deepspeed", world_size=8), "아직 예산을 계산하지 못한다"),
        (dict(enabled=True, strategy="magic", world_size=2), "알 수 없는"),
    ],
)
def test_inconsistent_distributed_settings_are_refused(dist, needle):
    errs = _cfg(**dist).profile_errors("linux_multi_gpu")
    assert errs and needle in "\n".join(errs)


def test_an_unmodeled_strategy_is_refused_rather_than_guessed():
    """FSDP는 가중치를 쪼개 장치당 VRAM이 달라진다. 그 계산이 없는데 단일 GPU
    숫자로 통과시키면 G4가 존재할 이유가 사라진다."""
    errs = _cfg(enabled=True, strategy="fsdp", world_size=4).profile_errors("linux_multi_gpu")
    assert "장치당 VRAM이 달라지는데" in "\n".join(errs)


def test_world_size_changes_the_digest():
    """장치 수만 바꾼 다른 학습이 같은 해시를 쓰면 재현이 무너진다."""
    one = _cfg().digest()
    four = _cfg(enabled=True, strategy="ddp", world_size=4).digest()
    assert one != four
    assert four["distributed"]["world_size"] == 4


def test_the_effective_batch_multiplier_follows_the_switch():
    assert _cfg().distributed.effective_multiplier() == 1
    assert _cfg(enabled=True, strategy="ddp", world_size=4).distributed.effective_multiplier() == 4
    # 꺼져 있으면 world_size가 무엇이든 한 장이다 (그 조합은 위에서 이미 거부된다)
    assert _cfg(world_size=8).distributed.effective_multiplier() == 1


def test_the_config_round_trips_world_size():
    d = {"backbone": "tiny-vlm", "distributed": {"enabled": True, "strategy": "ddp", "world_size": 4}}
    c = TrainerConfig.from_dict(d)
    assert (c.distributed.enabled, c.distributed.strategy, c.distributed.world_size) == (True, "ddp", 4)


def test_the_budget_reports_the_device_count():
    from vlm_trainer.engine import budget as budget_mod

    res = budget_mod.estimate(_cfg(enabled=True, strategy="ddp", world_size=4), images=1)
    assert res.world_size == 4
    assert budget_mod.estimate(_cfg(), images=1).world_size == 1


def test_the_backend_is_reachable_from_the_cli():
    """파이썬에서만 고를 수 있는 기능은 반쯤 만든 것이다. CLI로 되어야 한다."""
    from vlm_trainer.cli.main import build_parser

    a = build_parser().parse_args(["run", "x.yaml", "--cache-backend", "memory"])
    assert a.cache_backend == "memory"
    assert build_parser().parse_args(["run", "x.yaml"]).cache_backend == "local"


def test_the_report_says_the_vram_is_per_device():
    """장치를 늘리면 VRAM이 준다고 읽히면 안 된다. DDP는 장치마다 모델을 통째로 든다."""
    from vlm_trainer.engine import budget as budget_mod

    out = budget_mod.render(
        budget_mod.estimate(_cfg(enabled=True, strategy="ddp", world_size=4), images=1)
    )
    assert "장치 하나 기준" in out and "유효 배치가 4배" in out
    assert "다중 GPU" not in budget_mod.render(budget_mod.estimate(_cfg(), images=1))
