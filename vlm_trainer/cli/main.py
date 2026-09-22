"""vlmt — 커맨드라인 진입점.

UI 없이 스펙만으로 전부 할 수 있어야 한다. UI 전용 실행 경로는 존재하지 않는다.
Phase 2 범위: compile / decompile / nodes / show / dryrun / run / preview.
budget(G4) · materialize · sweep은 Phase 3 이후에 붙는다.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from typing import List, Optional

from ..core.compiler import CompileFailed
from ..core.errors import VlmtError
from ..engine import budget as budget_mod
from ..engine import materialize as materialize_mod
from ..engine import sweep as sweep_mod
from ..engine import dryrun as dryrun_mod
from ..engine import preview as preview_mod
from ..engine import samples as samples_mod
from ..spec import recipe as recipe_mod
from ..train import tokens as tokens_mod
from ..spec.decompile import decompile
from ..train import shards as shards_mod
from ..train import contract as contract_mod




# 여러 명령이 함께 쓰는 헬퍼. main 에 두면 명령 모듈이 main 을 되불러 순환이 된다.

# 명령 본문은 주제별 모듈에 있다. 파서는 그것을 가리키기만 한다.
from .cmd_graph import cmd_backbones, cmd_compile, cmd_decompile, cmd_infer_graph, cmd_nodes, cmd_show, cmd_view
from .cmd_execute import cmd_budget, cmd_dryrun, cmd_materialize, cmd_preview, cmd_run, cmd_sweep, cmd_train
from .cmd_project import cmd_edit, cmd_new, cmd_recipe


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vlmt", description="VLM Trainer 플랫폼 CLI")
    p.add_argument("--nodes", action="append", default=[], help="추가로 임포트할 노드 모듈")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compile", help="스펙을 컴파일하고 4중 게이트 중 G1/G2를 통과시킨다")
    c.add_argument("spec")
    c.add_argument("--set", action="append", default=[], help="파라미터 오버라이드 (노드id.파라미터=값)")
    c.add_argument("--out", help="compiled.json 저장 경로")
    c.add_argument("-v", "--verbose", action="store_true")
    c.set_defaults(func=cmd_compile)

    d = sub.add_parser("decompile", help="컴파일된 그래프를 스펙으로 되돌린다")
    d.add_argument("spec")
    d.add_argument("--flatten", action="store_true", help="Procedure를 펼친 채로 출력")
    d.add_argument("--out")
    d.set_defaults(func=cmd_decompile)

    n = sub.add_parser("nodes", help="노드 라이브러리를 카테고리별로 나열한다")
    n.set_defaults(func=cmd_nodes)

    s = sub.add_parser("show", help="노드 하나의 포트와 파라미터를 보여준다 (Node Quick Info)")
    s.add_argument("type")
    s.set_defaults(func=cmd_show)

    dr = sub.add_parser("dryrun", help="G3 — 실제 샘플 몇 건을 전 노드에 통과시켜 실측 검증한다")
    dr.add_argument("spec")
    dr.add_argument("--samples", type=int, default=3)
    dr.add_argument("--set", action="append", default=[])
    dr.add_argument("--cache-dir", default=".cache")
    dr.add_argument("--violation-threshold", type=float, default=0.02)
    dr.add_argument("--run-id", default="")
    dr.add_argument("-v", "--verbose", action="store_true")
    dr.set_defaults(func=cmd_dryrun)

    r = sub.add_parser("run", help="그래프를 실행한다 (Output 노드 포함)")
    r.add_argument("spec")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--split", default="")
    r.add_argument("--set", action="append", default=[])
    r.add_argument("--cache-dir", default=".cache")
    r.add_argument("--cache-backend", default="local",
                   help="캐시 저장 백엔드 (local | memory). 키 계산은 백엔드와 무관하다")
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--run-id", default="")
    r.add_argument("--device", default="", help="예산 프로파일 (rtx3060_12gb | rtx4090_24gb)")
    r.add_argument("--skip-budget", action="store_true",
                   help="예산 검사를 건너뛴다(Trainer 없는 그래프 전용)")
    r.add_argument("--view", default="", help="실행 상태를 칠한 그래프 HTML을 남긴다 ('-'면 runs/<id>/graph.html)")
    r.add_argument("--debug-output", action="store_true",
                   help="Debug Output 토글. 꺼져 있으면 미리보기를 생성조차 하지 않는다")
    r.add_argument("--trigger", default="cli", choices=("cli", "ui", "external"),
                   help="누가 이 실행을 띄웠나. Debug Output 규약이 여기에 달려 있다")
    r.add_argument("--progress", default="",
                   help="진행 상황 스냅샷 경로 (기본 runs/<run-id>/progress.json, 'off'면 남기지 않는다)")
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("budget", help="G4 — 학습 전에 단계별 VRAM과 시퀀스 길이를 산정한다")
    b.add_argument("spec")
    b.add_argument("--set", action="append", default=[])
    b.add_argument("--device", default="", help="rtx3060_12gb | rtx4090_24gb | a100_40gb")
    b.add_argument("--what-if", action="append", default=[],
                   help="images=2 tiles=4 max_len=2048 backbone=dummy-7b quantization=none per_device=4")
    b.add_argument("--no-measure", action="store_true", help="dry-run 실측 없이 가정값으로 계산")
    b.add_argument("--cache-dir", default=".cache")
    b.add_argument("--run-id", default="")
    b.set_defaults(func=cmd_budget)

    m = sub.add_parser("materialize", help="물질화 경계까지 구워 shard로 남긴다 (학습은 이것만 읽는다)")
    m.add_argument("spec")
    m.add_argument("--out-dir", default="", help="기본값 runs/<run_id>/materialized")
    m.add_argument("--shard-size", type=int, default=64)
    m.add_argument("--resume", action="store_true", help="커밋된 shard를 인정하고 남은 샘플만 굽는다")
    m.add_argument("--limit", type=int, default=0)
    m.add_argument("--split", default="")
    m.add_argument("--set", action="append", default=[])
    m.add_argument("--cache-dir", default=".cache")
    m.add_argument("--run-id", default="")
    m.set_defaults(func=cmd_materialize)

    nw = sub.add_parser("new", help="빈 Solution/Project 껍데기를 만든다")
    nw.add_argument("dir", help="만들 디렉터리 (없으면 만든다)")
    nw.add_argument("--solution-id", default="", help="기본값은 디렉터리 이름")
    nw.add_argument("--project-id", default="", help="기본값은 01_<solution-id>")
    nw.add_argument("--name", default="", help="사람이 읽는 이름")
    nw.add_argument("--index", default="data/index.jsonl",
                    help="샘플 인덱스 JSONL의 자리 (Solution 루트 기준)")
    nw.add_argument("--profile", default="windows_single_gpu",
                    help="실행 프로파일 (windows_single_gpu | linux_multi_gpu)")
    nw.add_argument("--edit", action="store_true", help="만든 뒤 편집기를 연다")
    nw.add_argument("--port", type=int, default=8770)
    nw.set_defaults(func=cmd_new)

    ed = sub.add_parser("edit", help="그래프 편집기를 연다 (네이티브 창)")
    ed.add_argument("spec")
    ed.set_defaults(func=cmd_edit)

    # `app` 은 이식하는 동안 쓰던 이름이다. 손가락이 기억하는 것을 끊지 않는다.
    ap2 = sub.add_parser("app", help="`edit` 과 같다")
    ap2.add_argument("spec")
    ap2.set_defaults(func=cmd_edit)

    bb = sub.add_parser("backbones", help="등록된 백본과 그 형상을 보여준다 (가중치는 열지 않는다)")
    bb.add_argument("--add", default="", help="hf:<경로 또는 모델 id>를 config.json만 읽어 등록한다")
    bb.set_defaults(func=cmd_backbones)

    vw = sub.add_parser("view", help="컴파일된 그래프를 한 장의 HTML로 그린다 (읽기 전용)")
    vw.add_argument("spec")
    vw.add_argument("--set", action="append", default=[])
    vw.add_argument("--out", default="")
    vw.add_argument("--open", action="store_true", help="브라우저로 연다")
    vw.set_defaults(func=cmd_view)

    rc = sub.add_parser("recipe", help="Parameter Recipe 관리 (list/show/diff/expand/set-active)")
    rc.add_argument("action", choices=["list", "show", "diff", "expand", "set-active"])
    rc.add_argument("spec")
    rc.add_argument("--recipe-id", default="1")
    rc.add_argument("--other", default="2", help="diff 대상")
    rc.add_argument("--sweep", default="", help="expand 할 스윕 이름")
    rc.add_argument("--device", default="")
    rc.set_defaults(func=cmd_recipe)

    sw = sub.add_parser("sweep", help="레시피 여러 개를 순차로 돌린다 (물질화는 지문이 같으면 공유)")
    sw.add_argument("spec")
    sw.add_argument("--recipes", default="", help="예: 1,2,3 또는 10-21")
    sw.add_argument("--sweep", default="", help="스윕 이름으로 전개해서 돌린다")
    sw.add_argument("--run-root", default="runs/sweeps")
    sw.add_argument("--trainer-config", default="")
    sw.add_argument("--limit", type=int, default=0)
    sw.add_argument("--shard-size", type=int, default=64)
    sw.add_argument("--device", default="", help="예산 프로파일")
    sw.add_argument("--device-torch", default="auto")
    sw.add_argument("--max-steps", type=int, default=0)
    sw.add_argument("--no-train", action="store_true", help="물질화와 예산까지만")
    sw.add_argument("--cache-dir", default=".cache")
    sw.set_defaults(func=cmd_sweep)

    ig = sub.add_parser("infer-graph", help="학습 그래프에서 정답 경로를 잘라낸 추론 그래프를 뽑는다")
    ig.add_argument("spec")
    ig.add_argument("--set", action="append", default=[])
    ig.add_argument("--out", default="")
    ig.set_defaults(func=cmd_infer_graph)

    tr = sub.add_parser("train", help="물질화된 shard로 다단계 학습을 실행한다")
    tr.add_argument("spec")
    tr.add_argument("--materialized", default="", help="기본값 runs/<run_id>/materialized")
    tr.add_argument("--set", action="append", default=[])
    tr.add_argument("--device", default="", help="예산 프로파일")
    tr.add_argument("--device-torch", default="auto", help="cuda | cpu | auto")
    tr.add_argument("--resume", action="store_true")
    tr.add_argument("--max-steps", type=int, default=0)
    tr.add_argument("--skip-budget", action="store_true")
    tr.add_argument("--what-if", action="append", default=[])
    tr.add_argument("--cache-dir", default=".cache")
    tr.add_argument("--run-id", default="")
    tr.set_defaults(func=cmd_train)

    pv = sub.add_parser("preview", help="노드 하나만 실행해 시각화 출력을 본다")
    pv.add_argument("spec")
    pv.add_argument("--node", required=True)
    pv.add_argument("--sample", default="")
    pv.add_argument("--set", action="append", default=[])
    pv.add_argument("--cache-dir", default=".cache")
    pv.add_argument("--no-cache", action="store_true")
    pv.add_argument("--save", action="store_true", help="이미지 미리보기를 runs/<id>/previews에 저장")
    pv.add_argument("--run-id", default="")
    pv.set_defaults(func=cmd_preview)
    return p


def _subparsers(parser: argparse.ArgumentParser) -> List[argparse.ArgumentParser]:
    out: List[argparse.ArgumentParser] = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            out.extend(action.choices.values())
    return out


def _utf8_output() -> None:
    """출력을 파일이나 파이프로 넘길 때 인코딩 때문에 죽지 않게 한다.

    Windows에서 파이썬은 진짜 콘솔에는 유니코드를 그대로 쓰지만, 리다이렉트되면
    로케일 인코딩(여기서는 cp949)을 탄다. 보고서에 쓰이는 `—`나 `·`는 cp949에
    없어서 `vlmt budget > log.txt` 한 줄이 UnicodeEncodeError로 끝났다.
    계산이 끝난 뒤 출력에서 죽는 것만큼 허탈한 실패도 없다.
    """
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", "") or "").lower()
        if enc.replace("-", "") in ("utf8", "utf8mb4"):
            continue
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError, ValueError):
            pass   # 다시 설정할 수 없는 스트림이면 그대로 둔다. 관찰이 실행을 막지 않는다


def main(argv: Optional[List[str]] = None) -> int:
    _utf8_output()
    parser = build_parser()
    for p in _subparsers(parser):
        if any(x.dest == "spec" for x in p._actions) and not any(x.dest == "recipe" for x in p._actions):
            p.add_argument("--recipe", default="", help="Parameter Recipe 번호로 값 오버레이")
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CompileFailed as e:
        print(str(e), file=sys.stderr)
        return 2
    except VlmtError as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        # 노드가 계약을 어기고 일반 예외를 던져도(주로 --set 으로 들어온 엉뚱한 값)
        # 사람에게는 트레이스백이 아니라 무엇이 잘못됐는지가 보여야 한다.
        # 편집기는 이미 이 경우를 잡는데 CLI만 날것으로 터지고 있었다.
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "  노드가 받아들일 수 없는 값이다. --set 으로 준 값의 형태를 확인하라 "
            "(예: 리스트는 --set n_plot.size=[640,320]).",
            file=sys.stderr,
        )
        print(
            "  전체 추적을 보려면 VLMT_TRACE=1 을 설정하고 다시 실행하라.",
            file=sys.stderr,
        )
        if os.environ.get("VLMT_TRACE"):
            raise
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
