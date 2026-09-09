"""실행 저널 — append-only 커밋 기록.

재개의 기준은 "무엇을 계산했나"가 아니라 "무엇을 커밋했나"다.
중간에 죽어도 저널에 남은 사실만 인정하고 나머지는 다시 만든다. 설계 문서 08 §8.7.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set

RUN_START = "run_start"
SHARD_COMMITTED = "shard_committed"
PHASE_DONE = "phase_done"


@dataclass
class Replay:
    spec_hash: str = ""
    done_keys: Set[str] = field(default_factory=set)
    shards: List[Dict[str, Any]] = field(default_factory=list)
    phases_done: Set[str] = field(default_factory=set)

    @property
    def committed_samples(self) -> int:
        return len(self.done_keys)


class Journal:
    """한 run의 커밋 사실만 담는다. 한 줄이 곧 하나의 되돌릴 수 없는 사실이다."""

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def append(self, event: str, **fields: Any) -> None:
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **fields}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())  # 크래시 직전 커밋도 살아남아야 한다

    def read(self) -> Iterator[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return iter(())
        def _gen() -> Iterator[Dict[str, Any]]:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue  # 크래시로 잘린 마지막 줄은 버린다
        return _gen()

    def replay(self) -> Replay:
        r = Replay()
        for rec in self.read():
            ev = rec.get("event")
            if ev == RUN_START:
                r.spec_hash = rec.get("spec_hash", r.spec_hash)
            elif ev == SHARD_COMMITTED:
                r.shards.append(rec)
                r.done_keys.update(rec.get("keys") or ())
            elif ev == PHASE_DONE:
                r.phases_done.add(rec.get("phase", ""))
        return r

    def exists(self) -> bool:
        return os.path.exists(self.path)
