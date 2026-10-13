"""빈 Solution/Project 껍데기.

편집기는 스펙을 **편집하는** 뷰다. 그래서 열 파일이 하나는 있어야 시작한다.
이 모듈은 그 하나를 만든다 — 그 이상은 하지 않는다. 노드도 배선도 넣지 않는다.
어떤 그래프를 만들지는 편집기에서 정한다.

만들어진 프로젝트는 **아직 컴파일되지 않는다**(Output 노드가 없다). 그것이 정상이다.
편집기의 draft 모드가 그 상태를 다루고, 저장과 실행은 여전히 strict를 요구한다.
"""

from __future__ import annotations

import os
from typing import List, Tuple

from ..core.errors import SpecError

SOLUTION = """kind: Solution
spec_version: 1
id: {sid}
name: "{name}"
description: ""

# 노드 기본값. 프로젝트마다 반복해 쓰지 않기 위한 자리입니다.
defaults: {{}}

runtime_profile: {profile}
cache:
  max_gb: 8
"""

PROJECT = """spec_version: 1
kind: Project
id: "{pid}"
name: "{name}"

# 그래프 밖의 샘플 공간 선언. Input 노드는 배선 대신 여기서 값을 길어옵니다.
# 경로는 모두 이 파일이 있는 디렉터리 기준입니다.
#
# index 파일이 아직 없다면 먼저 만들어야 합니다 — 한 줄에 샘플 하나(JSONL),
# key로 지정한 열이 샘플을 가리키는 이름입니다.
sample_space:
  index: "{index}"
  key: sample_id
  filter: ""
  splits:
    strategy: random
    ratios: {{train: 0.8, val: 0.2}}
    seed: 20260910

# 편집기에서 노드를 놓고 배선합니다. 비어 있는 그래프는 아직 컴파일되지 않습니다 —
# Output 노드가 하나는 있어야 합니다.
nodes: []
edges: []

# 물질화 경계. 여기 적힌 노드까지가 한 번 계산해 두고 재사용하는 구간입니다.
materialize:
  boundary: []
"""


def _slug(s: str) -> str:
    keep = [c if (c.isalnum() or c in "._-") else "_" for c in s.strip()]
    out = "".join(keep).strip("_")
    return out or "untitled"


def new_project(
    root: str,
    *,
    solution_id: str = "",
    project_id: str = "",
    name: str = "",
    index: str = "data/index.jsonl",
    profile: str = "windows_single_gpu",
) -> Tuple[str, List[str]]:
    """Solution 하나와 그 안의 Project 하나를 만든다. (project.yaml 경로, 만든 파일들)

    이미 있는 파일은 덮지 않는다 — 껍데기를 만드는 명령이 남의 스펙을 지우면 안 된다.
    """
    root = os.path.abspath(root)
    sid = _slug(solution_id or os.path.basename(root))
    pid = _slug(project_id or "01_" + sid)
    label = name or sid

    sol_path = os.path.join(root, "solution.yaml")
    proj_dir = os.path.join(root, "projects", pid)
    proj_path = os.path.join(proj_dir, "project.yaml")

    for p in (sol_path, proj_path):
        if os.path.exists(p):
            raise SpecError(
                f"이미 있는 파일을 덮지 않습니다: {p}\n"
                "  다른 디렉터리를 쓰거나, 그 프로젝트를 직접 여세요:\n"
                f"    vlmt edit {proj_path}"
            )

    made: List[str] = []
    for d in (root, proj_dir, os.path.join(root, "procedures"), os.path.join(root, "schemas")):
        os.makedirs(d, exist_ok=True)

    # index는 프로젝트 파일 기준의 상대 경로여야 한다
    rel_index = os.path.relpath(os.path.join(root, index), proj_dir).replace(os.sep, "/")

    with open(sol_path, "w", encoding="utf-8") as fh:
        fh.write(SOLUTION.format(sid=sid, name=label, profile=profile))
    made.append(sol_path)

    with open(proj_path, "w", encoding="utf-8") as fh:
        fh.write(PROJECT.format(pid=pid, name=label, index=rel_index))
    made.append(proj_path)

    return proj_path, made


# ── 5노드(Line) ─────────────────────────────────────────────────────────
LINE = """spec_version: 1
kind: Line
id: "{lid}"
name: "{name}"

# 표(CSV)와 열 지정은 화면의 `데이터` 에서 정합니다. 아래 값은 그 전의 빈 자리입니다.
sample_space:
  index: "data/table.csv"
  key: sample_id
  filter: ""
  splits:
    strategy: random
    ratios: {{train: 0.9, val: 0.1}}
    seed: 20260921

stations:
  - id: s1
    ref: "image@1.0.0"
    params: {{image_column: "", columns: [], size: [448, 448], keep_aspect: true}}
  - id: s2
    ref: "question@1.0.0"
    params: {{template: "{{question}}", placeholder: "<image>", policy: prepend}}
  - id: s3
    ref: "answer@1.0.0"
    params:
      schema_path: "{lid}_answer.yaml"
      mapping: {{}}
      render: tagged
      guard_mode: exact
      on_leak: fail
  - id: s4
    ref: "train@1.0.0"
    params: {{config_path: "trainer.yaml", out_dir: "runs/{{run_id}}/train"}}
  - id: s5
    ref: "verify@1.0.0"
    params: {{max_new_tokens: 48, only_split: val, out_dir: "runs/{{run_id}}/infer"}}
"""

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SAMPLE = os.path.join(_REPO, "solutions")


def new_line(folder: str, *, line_id: str = "", name: str = "") -> Tuple[str, List[str]]:
    """빈 5노드 프로젝트 폴더. (line.yaml 경로, 만든 것들)

    스테이션·묶음 상자 정의는 폴더 위쪽 `stations/`·`procedures/` 에서 찾아지면 그것을 쓰고,
    안 찾아지면 폴더 안으로 복사한다 — 폴더 하나만 옮겨도 열리게 한다. 학습 설정도 한 벌 복사한다.
    표와 정답 항목은 만들지 않는다. 그것은 `데이터` 에서 정하고, 그 전에는 컴파일되지 않는다.
    """
    import shutil

    from .line import find_station

    folder = os.path.abspath(folder)
    lid = _slug(line_id or os.path.basename(folder))
    path = os.path.join(folder, "line.yaml")
    if os.path.exists(path):
        raise SpecError(f"이미 있는 파일을 덮지 않습니다: {path}\n  다른 폴더를 쓰거나 그 파일을 직접 여세요: vlmt edit {path}")
    os.makedirs(folder, exist_ok=True)
    made: List[str] = []
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(LINE.format(lid=lid, name=name or lid))
    made.append(path)

    try:
        find_station("image@1.0.0", folder)
        have_stations = True
    except SpecError:
        have_stations = False
    for sub, have in (("stations", have_stations), ("procedures", _has_up(folder, "procedures", "image_prep_resize.yaml"))):
        if not have:
            dst = os.path.join(folder, sub)
            shutil.copytree(os.path.join(_SAMPLE, sub), dst)
            made.append(dst + os.sep)
    trainer = os.path.join(folder, "trainer.yaml")
    shutil.copyfile(os.path.join(_SAMPLE, "vlm_open", "projects", "01_open", "trainer.yaml"), trainer)
    made.append(trainer)
    return path, made


def _has_up(folder: str, sub: str, probe: str) -> bool:
    d = folder
    for _ in range(4):
        if os.path.exists(os.path.join(d, sub, probe)):
            return True
        d = os.path.dirname(d)
    return False
