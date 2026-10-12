"""용어집 — 노드 설명, 묶음 상자, 스테이션, 사용자 문서가 합의한 말만 쓴다.

AI 용어(프롬프트, 누설, 파인튜닝 …)는 제조 현장 사용자에게 뜻이 없다. 말이 어긋나면 같은 것이
두 이름으로 불린다. 금지어 표는 `vlm_trainer/ui/words.py` 한 곳에 있다.
"""

import glob
import os

import pytest
import yaml

from vlm_trainer.core import registry
from vlm_trainer.ui import words as W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 5노드와 상관없는 옛 말: 구 용어집(W) 중 이번 표에서 **코드 식별자로 남는** 것은 제외하지 않는다.
BANNED = [w for w in W.RENAMED if w not in ("Procedure", "shard", "Materialize")]


def _hits(text: str):
    return [w for w in BANNED if w in (text or "")]


@pytest.fixture(autouse=True)
def _nodes():
    registry.load_builtin_nodes()


def test_node_docs_and_port_texts_use_the_glossary():
    bad = []
    for d in registry.all_defs():
        doc = d.doc
        texts = [doc.label, doc.hint, doc.summary, doc.scenario, doc.ports]
        texts += [p.description for p in [*d.inputs.values(), *d.outputs.values()] if getattr(p, "description", "")]
        for t in texts:
            if _hits(t):
                bad.append((d.type, _hits(t), t[:30]))
    assert not bad, bad


def test_procedure_and_station_files_use_the_glossary():
    bad = []
    for f in glob.glob(os.path.join(ROOT, "solutions", "procedures", "*.yaml")) + \
            glob.glob(os.path.join(ROOT, "solutions", "stations", "*.yaml")):
        d = yaml.safe_load(open(f, encoding="utf-8"))
        for k in ("label", "hint", "summary", "description"):
            if _hits(str(d.get(k, ""))):
                bad.append((os.path.basename(f), k))
        for e in (d.get("exposed_params") or {}).values():
            if _hits(str(e.get("label", ""))):
                bad.append((os.path.basename(f), e))
    assert not bad, bad


@pytest.mark.parametrize("name", ["solutions/vlm_open/GRAPH.md", "solutions/vlm_open/README.md", "docs/line-guide.md"])
def test_user_documents_use_the_glossary(name):
    text = open(os.path.join(ROOT, name), encoding="utf-8").read()
    assert not _hits(text), (name, _hits(text))
