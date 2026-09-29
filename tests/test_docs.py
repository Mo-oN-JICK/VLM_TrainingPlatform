"""설계 문서가 현실과 어긋나는 것을 막는다.

문서가 한 번 낡은 이유는 아무도 읽지 않기 때문이고, 아무도 읽지 않으면 또 낡는다.
`09-directory.md` 는 없어진 `ui/server.py` · `ui/web/` 을 넉 달 동안 가리키고 있었고,
CLI 명령을 12개로 적어 두었는데 실제로는 20개였다.

**모든 것을 고정하지는 않는다.** 설계 문서의 트리 대부분은 *만들 것*의 의도이고
(`engine/scheduler.py`, `spec/lock.py` …), 그것을 코드에 맞추라고 강요하면 설계가
구현 요약으로 전락한다. `CLAUDE.md` 는 "구현은 이 설계를 따르며, 벗어난 지점은
README 표에 이유와 함께 기록한다" 고 말한다 — 그래서 여기서 고정하는 것은 둘뿐이다:

1. 문서가 **현재형으로 단언한** `ui/` · `cli/` 트리의 경로는 실재해야 한다
2. 문서가 세는 것(CLI 명령 수, 노드 수)은 실제와 맞아야 한다
"""

from __future__ import annotations

import io
import os
import re

import pytest

from vlm_trainer.core import registry

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DESIGN = os.path.join(ROOT, "docs", "design")


def _read(name: str) -> str:
    return io.open(os.path.join(DESIGN, name), encoding="utf-8").read()


def _block(text: str, header: str) -> str:
    """`header` 로 시작하는 들여쓴 트리 한 덩이. 다음 빈 줄까지."""
    i = text.index(header)
    out = []
    for line in text[i:].split("\n")[1:]:
        if not line.strip():
            break
        out.append(line)
    return "\n".join(out)


# ── 문서가 가리키는 경로는 실재해야 한다 ────────────────────────────────
@pytest.mark.parametrize("header", ["  cli/", "  ui/"])
def test_every_path_the_directory_doc_claims_here_exists(header):
    """`ui/server.py` 가 문서에만 남아 있던 것이 이 검사가 없어서다.
    새로 온 사람은 문서를 먼저 읽고, 없는 파일을 찾다 시간을 버린다."""
    doc = _read("09-directory.md")
    top = header.strip().rstrip("/")
    missing = []
    for line in _block(doc, header).split("\n"):
        name = line.strip().split()[0] if line.strip() else ""
        if not re.fullmatch(r"[a-z_]+(\.py|/|_\*\.py)", name):
            continue
        if name.endswith("_*.py"):                 # `editor_*.py` 같은 묶음 표기
            stem = name[:-5]
            d = os.path.join(ROOT, "vlm_trainer", top)
            if not any(f.startswith(stem) for f in os.listdir(d)):
                missing.append(f"{top}/{name}")
            continue
        p = os.path.join(ROOT, "vlm_trainer", top, name.rstrip("/"))
        if not os.path.exists(p):
            missing.append(f"{top}/{name}")
    assert not missing, f"문서가 가리키는데 없는 것: {missing}"


def test_the_doc_does_not_still_describe_the_web_editor():
    """웹 편집기는 걷어냈다. 문서가 그것을 현재형으로 적어 두면 읽는 사람이
    있지도 않은 서버를 띄우려 한다."""
    tree = _block(_read("09-directory.md"), "  ui/")
    for gone in ("server.py", "events.py", "web/"):
        assert gone not in tree, f"걷어낸 {gone} 가 트리에 남아 있다"


# ── 문서가 세는 것은 실제와 맞아야 한다 ─────────────────────────────────
def test_the_cli_command_count_matches_the_parser():
    """문서는 12개라 적어 두고 실제로는 20개였다. `pipeline` 도 `export` 도
    문서에 없으니, 설계만 읽은 사람에게는 존재하지 않는 기능이다."""
    from vlm_trainer.cli.main import build_parser

    real = len(build_parser()._subparsers._group_actions[0].choices)
    doc = _read("09-directory.md")
    m = re.search(r"하위 명령 (\d+)개", doc)
    assert m, "문서가 명령 수를 적지 않는다"
    assert int(m.group(1)) == real, f"문서 {m.group(1)}개 vs 실제 {real}개"


def test_every_command_the_doc_lists_exists():
    from vlm_trainer.cli.main import build_parser

    real = set(build_parser()._subparsers._group_actions[0].choices)
    named = set()
    for line in _block(_read("09-directory.md"), "  cli/").split("\n"):
        if "cmd_" in line or line.strip().startswith(("compile", "dryrun", "new", "pipeline")):
            named |= {w for w in line.split() if w in real or re.fullmatch(r"[a-z-]+", w)}
    ghosts = {w for w in named if "_" not in w and "." not in w and "/" not in w} - real
    # 설명 문구의 낱말이 아니라 명령처럼 보이는 것만 본다
    ghosts = {g for g in ghosts if g.islower() and len(g) > 2 and g.isalpha()} - {
        "cmd", "py", "등", "개"}
    assert not ghosts, f"문서에만 있는 명령: {sorted(ghosts)}"


# ── 노드 카탈로그 ───────────────────────────────────────────────────────
def test_every_registered_node_is_in_the_catalog():
    """반대 방향(카탈로그에 있는데 미구현)은 검사하지 않는다 — 그쪽은 **설계**이고,
    아직 안 만든 노드를 스펙에 쓰면 컴파일이 막는다. 위험한 것은 이쪽이다:
    만들어 놓고 문서에 안 적으면 아무도 그 노드가 있는 줄 모른다."""
    registry.load_builtin_nodes()
    doc = _read("03-node-catalog.md")
    named = set(re.findall(r"`([a-z_]+\.[a-z_]+)`", doc))
    # **저장소 안 노드만 본다.** `test.*`(fixture)와 `example.*`(문서용 예제)는
    # 설계 카탈로그에 들어갈 물건이 아니다. 구분은 구현 지문으로 한다 — 저장소 밖에서
    # 온 노드에만 붙으므로, 이름 규칙을 새로 외울 필요가 없다.
    real = {d.type for d in registry.all_defs() if not d.impl_fingerprint}
    missing = sorted(real - named)
    assert not missing, f"등록됐는데 카탈로그에 없는 노드: {missing}"


def test_the_catalog_says_it_is_a_design_not_a_listing():
    """72개를 적어 두고 29개가 등록돼 있다. 그 차이를 모르고 읽으면 없는 노드를
    스펙에 쓰게 된다."""
    head = _read("03-node-catalog.md")[:900]
    assert "vlmt nodes" in head, "실제 목록을 어디서 보는지 적혀 있지 않다"
