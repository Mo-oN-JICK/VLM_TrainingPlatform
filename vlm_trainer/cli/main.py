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
from .cmd_graph import cmd_backbones, cmd_compile, cmd_expand, cmd_decompile, cmd_infer_graph, cmd_nodes, cmd_show, cmd_view
from .cmd_execute import cmd_budget, cmd_dryrun, cmd_export, cmd_materialize, cmd_pipeline, cmd_preview, cmd_run, cmd_sweep, cmd_train
from .cmd_project import cmd_schema_rows, cmd_schema_build, cmd_judge, cmd_judgments, cmd_row_set, cmd_row_exclude, cmd_new_line, cmd_table_template, cmd_check_node, cmd_edit, cmd_new, cmd_new_node, cmd_recipe


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vlmt", description="VLM Trainer 플랫폼 CLI")
    p.add_argument("--nodes", action="append", default=[], help="추가로 임포트할 노드 모듈")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compile", help="스펙을 컴파일하고 4중 게이트 중 G1/G2를 통과시킵니다")
    c.add_argument("spec")
    c.add_argument("--set", action="append", default=[], help="파라미터 오버라이드 (노드id.파라미터=값)")
    c.add_argument("--out", help="compiled.json 저장 경로")
    c.add_argument("-v", "--verbose", action="store_true")
    c.set_defaults(func=cmd_compile)

    d = sub.add_parser("decompile", help="컴파일된 그래프를 스펙으로 되돌립니다")
    d.add_argument("spec")
    d.add_argument("--flatten", action="store_true", help="Procedure를 펼친 채로 출력")
    d.add_argument("--out")
    d.set_defaults(func=cmd_decompile)

    ex0 = sub.add_parser("expand", help="Line(5노드) 파일을 11노드 Project 스펙으로 펼칩니다")
    ex0.add_argument("spec")
    ex0.add_argument("--out")
    ex0.set_defaults(func=cmd_expand)

    n = sub.add_parser("nodes", help="노드 라이브러리를 카테고리별로 나열합니다")
    n.set_defaults(func=cmd_nodes)

    s = sub.add_parser("show", help="노드 하나의 포트와 파라미터를 보여줍니다 (Node Quick Info)")
    s.add_argument("type")
    s.set_defaults(func=cmd_show)

    dr = sub.add_parser("dryrun", help="G3 — 실제 샘플 몇 건을 전 노드에 통과시켜 실측 검증합니다")
    dr.add_argument("spec")
    dr.add_argument("--samples", type=int, default=3)
    dr.add_argument("--set", action="append", default=[])
    dr.add_argument("--cache-dir", default=".cache")
    dr.add_argument("--violation-threshold", type=float, default=0.02)
    dr.add_argument("--run-id", default="")
    dr.add_argument("-v", "--verbose", action="store_true")
    dr.set_defaults(func=cmd_dryrun)

    r = sub.add_parser("run", help="그래프를 실행합니다 (Output 노드 포함)")
    r.add_argument("spec")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--split", default="")
    r.add_argument("--set", action="append", default=[])
    r.add_argument("--cache-dir", default=".cache")
    r.add_argument("--cache-backend", default="local",
                   help="캐시 저장 백엔드 (local | memory). 키 계산은 백엔드와 무관합니다")
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--run-id", default="")
    r.add_argument("--device", default="", help="예산 프로파일 (rtx3060_12gb | rtx4090_24gb)")
    r.add_argument("--skip-budget", action="store_true",
                   help="예산 검사를 건너뛴다(Trainer 없는 그래프 전용)")
    r.add_argument("--view", default="", help="실행 상태를 칠한 그래프 HTML을 남깁니다 ('-'면 runs/<id>/graph.html)")
    r.add_argument("--debug-output", action="store_true",
                   help="Debug Output 토글. 꺼져 있으면 미리보기를 생성조차 하지 않습니다")
    r.add_argument("--trigger", default="cli", choices=("cli", "ui", "external"),
                   help="누가 이 실행을 띄웠나. Debug Output 규약이 여기에 달려 있습니다")
    r.add_argument("--progress", default="",
                   help="진행 상황 스냅샷 경로 (기본 runs/<run-id>/progress.json, 'off'면 남기지 않습니다)")
    r.add_argument("--step-delay", type=float, default=0.0,
                   help="상자 하나를 지날 때마다 쉬는 시간(초). 어디를 도는지 눈으로 "
                        "보려는 용도입니다 — 결과는 바뀌지 않고 그만큼 느려집니다")
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("budget", help="G4 — 학습 전에 단계별 VRAM과 시퀀스 길이를 산정합니다")
    b.add_argument("spec")
    b.add_argument("--set", action="append", default=[])
    b.add_argument("--device", default="", help="rtx3060_12gb | rtx4090_24gb | a100_40gb")
    b.add_argument("--what-if", action="append", default=[],
                   help="images=2 tiles=4 max_len=2048 backbone=dummy-7b quantization=none per_device=4")
    b.add_argument("--no-measure", action="store_true", help="dry-run 실측 없이 가정값으로 계산")
    b.add_argument("--cache-dir", default=".cache")
    b.add_argument("--run-id", default="")
    b.set_defaults(func=cmd_budget)

    m = sub.add_parser("materialize", help="물질화 경계까지 구워 shard로 남깁니다 (학습은 이것만 읽습니다)")
    m.add_argument("spec")
    m.add_argument("--out-dir", default="", help="기본값 runs/<run_id>/materialized")
    m.add_argument("--shard-size", type=int, default=64)
    m.add_argument("--resume", action="store_true", help="커밋된 shard를 인정하고 남은 샘플만 굽습니다")
    m.add_argument("--limit", type=int, default=0)
    m.add_argument("--split", default="")
    m.add_argument("--set", action="append", default=[])
    m.add_argument("--cache-dir", default=".cache")
    m.add_argument("--run-id", default="")
    m.set_defaults(func=cmd_materialize)

    nw = sub.add_parser("new", help="빈 Solution/Project 껍데기를 만듭니다")
    nw.add_argument("dir", help="만들 디렉터리 (없으면 만듭니다)")
    nw.add_argument("--solution-id", default="", help="기본값은 디렉터리 이름")
    nw.add_argument("--project-id", default="", help="기본값은 01_<solution-id>")
    nw.add_argument("--name", default="", help="사람이 읽는 이름")
    nw.add_argument("--index", default="data/index.jsonl",
                    help="샘플 인덱스 JSONL의 자리 (Solution 루트 기준)")
    nw.add_argument("--profile", default="windows_single_gpu",
                    help="실행 프로파일 (windows_single_gpu | linux_multi_gpu)")
    nw.add_argument("--edit", action="store_true", help="만든 뒤 편집기를 엽니다")
    nw.set_defaults(func=cmd_new)

    nl = sub.add_parser("new-line", help="빈 5노드 프로젝트 폴더를 만듭니다")
    nl.add_argument("dir", help="만들 폴더")
    nl.add_argument("--id", default="", help="프로젝트 id (기본: 폴더 이름)")
    nl.add_argument("--name", default="", help="화면에 보일 이름")
    nl.add_argument("--edit", action="store_true", help="만든 뒤 편집기를 엽니다")
    nl.set_defaults(func=cmd_new_line)

    tt = sub.add_parser("table-template", help="사용자가 채울 빈 표(CSV)의 첫 줄을 만듭니다")
    tt.add_argument("--out", required=True)
    tt.add_argument("--key", default="sample_id", help="건 구분 열")
    tt.add_argument("--image", default="image_path", help="사진 열")
    tt.add_argument("--question", default="question", help="질문 열")
    tt.add_argument("--answers", required=True, help="정답 열들, 쉼표로 구분 (예: part_type,orientation)")
    tt.add_argument("--split", default="", help="학습/검증 구분 열 (구분 열 사용 모드일 때)")
    tt.add_argument("--group", default="", help="그룹 열 (그룹 기준 자동 나누기일 때)")
    tt.set_defaults(func=cmd_table_template)

    sb = sub.add_parser("schema-build", help="정답 형태 표(CSV)에서 스키마 파일을 만듭니다")
    sb.add_argument("rows", help="첫 줄이 항목,이름,종류,허용값 인 CSV")
    sb.add_argument("--id", required=True, help="정답 형태 이름")
    sb.add_argument("--render", default="tagged", choices=("tagged", "json"))
    sb.add_argument("--max-tokens", type=int, default=320)
    sb.add_argument("--out", default="")
    sb.set_defaults(func=cmd_schema_build)

    sr = sub.add_parser("schema-rows", help="스키마 파일을 정답 형태 표(CSV)로 되돌립니다")
    sr.add_argument("schema")
    sr.add_argument("--out", default="")
    sr.set_defaults(func=cmd_schema_rows)

    jg = sub.add_parser("judge", help="검증 건 하나에 사람 판정(ok | ng | none)을 남깁니다")
    jg.add_argument("run", help="실행 id(runs/ 아래) 또는 폴더")
    jg.add_argument("sample_id")
    jg.add_argument("verdict", choices=("ok", "ng", "none"))
    jg.set_defaults(func=cmd_judge)

    rs = sub.add_parser("row-set", help="표(CSV)의 한 건의 칸을 고칩니다 (처음 고치면 원본을 .bak.csv 로 남깁니다)")
    rs.add_argument("spec", help="line.yaml 또는 project.yaml")
    rs.add_argument("sample_id", help="건 구분 열의 값")
    rs.add_argument("edits", nargs="+", metavar="열=값")
    rs.set_defaults(func=cmd_row_set)

    rx = sub.add_parser("row-exclude", help="한 건을 표에서 제외합니다 (`제외` 열에 x — 지우지 않습니다)")
    rx.add_argument("spec", help="line.yaml 또는 project.yaml")
    rx.add_argument("sample_id")
    rx.add_argument("--undo", action="store_true", help="제외를 풀어 되살립니다")
    rx.set_defaults(func=cmd_row_exclude)

    jl = sub.add_parser("judgments", help="검증 건마다 항목별 O/X 와 사람 판정을 보입니다")
    jl.add_argument("run", help="실행 id(runs/ 아래) 또는 폴더")
    jl.set_defaults(func=cmd_judgments)

    nn = sub.add_parser(
        "new-node", help="새 노드의 뼈대와 그 테스트 파일을 만듭니다 (Custom Node)")
    nn.add_argument("path", help="만들 파일 (예: my_nodes/crop.py)")
    nn.add_argument("--type", required=True, help="노드 타입 (예: my.crop_by_rule)")
    nn.add_argument("--kind", default="processing", choices=("input", "processing", "output"))
    nn.add_argument("--category", default="Data Processing", help="라이브러리에서 묶일 자리")
    nn.add_argument("--label", default="", help="카드에 뜰 한국어 이름")
    nn.add_argument("--root", default="", help="임포트 이름을 셈할 기준 폴더 (기본: 파일의 폴더)")
    nn.set_defaults(func=cmd_new_node)

    cn = sub.add_parser(
        "check-node", help="노드가 계약을 지키는지 실제로 돌려 봅니다 (등록 검사가 못 보는 것)")
    cn.add_argument("module", help="임포트 이름 (예: my_nodes.crop)")
    cn.add_argument("--root", default="", help="모듈을 찾을 기준 폴더 (기본: 현재 폴더)")
    cn.set_defaults(func=cmd_check_node)

    ed = sub.add_parser("edit", help="그래프 편집기를 엽니다 (네이티브 창)")
    ed.add_argument("spec")
    ed.add_argument("--canvas", action="store_true", help="5노드를 캔버스 화면으로 엽니다 (단계 5 작업 중인 임시 스위치)")
    ed.set_defaults(func=cmd_edit)

    # `app` 은 이식하는 동안 쓰던 이름이다. 손가락이 기억하는 것을 끊지 않는다.
    ap2 = sub.add_parser("app", help="`edit` 과 같습니다")
    ap2.add_argument("spec")
    ap2.set_defaults(func=cmd_edit)

    bb = sub.add_parser("backbones", help="등록된 백본과 그 형상을 보여줍니다 (가중치는 열지 않습니다)")
    bb.add_argument("--add", default="", help="hf:<경로 또는 모델 id>를 config.json만 읽어 등록합니다")
    bb.add_argument("--fetch", default="", help="가중치까지 내려받습니다 (수 GB). 받은 뒤 등록합니다")
    bb.set_defaults(func=cmd_backbones)

    vw = sub.add_parser("view", help="컴파일된 그래프를 한 장의 HTML로 그립니다 (읽기 전용)")
    vw.add_argument("spec")
    vw.add_argument("--set", action="append", default=[])
    vw.add_argument("--out", default="")
    vw.add_argument("--open", action="store_true", help="브라우저로 엽니다")
    vw.set_defaults(func=cmd_view)

    rc = sub.add_parser("recipe", help="Parameter Recipe 관리 (list/show/diff/expand/set-active)")
    rc.add_argument("action", choices=["list", "show", "diff", "expand", "set-active"])
    rc.add_argument("spec")
    rc.add_argument("--recipe-id", default="1")
    rc.add_argument("--other", default="2", help="diff 대상")
    rc.add_argument("--sweep", default="", help="expand 할 스윕 이름")
    rc.add_argument("--device", default="")
    rc.set_defaults(func=cmd_recipe)

    sw = sub.add_parser("sweep", help="레시피 여러 개를 순차로 돌립니다 (물질화는 지문이 같으면 공유)")
    sw.add_argument("spec")
    sw.add_argument("--recipes", default="", help="예: 1,2,3 또는 10-21")
    sw.add_argument("--sweep", default="", help="스윕 이름으로 전개해서 돌립니다")
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

    ig = sub.add_parser("infer-graph", help="학습 그래프에서 정답 경로를 잘라낸 추론 그래프를 뽑습니다")
    ig.add_argument("spec")
    ig.add_argument("--set", action="append", default=[])
    ig.add_argument("--out", default="")
    ig.set_defaults(func=cmd_infer_graph)

    tr = sub.add_parser("train", help="물질화된 shard로 다단계 학습을 실행합니다")
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

    pl = sub.add_parser(
        "pipeline",
        help="준비 -> 굽기 -> 학습 -> 추론을 한 번에 돕니다 (그래프에 있는 단계까지만)")
    pl.add_argument("spec")
    pl.add_argument("--run-id", default="", help="같은 id 로 다시 부르면 이어 받을 수 있습니다")
    pl.add_argument("--resume", action="store_true", help="굽다 만 shard 를 인정하고 이어 받습니다")
    pl.add_argument("--fresh", action="store_true", help="굽다 만 것을 버리고 처음부터 굽습니다")
    pl.add_argument("--bake-split", default="auto",
                    help="굽기가 돌 split. auto=train 이 있으면 train 만, all=전부, "
                         "또는 split 이름. 추론 split 은 추론 노드가 선언한 것을 씁니다")
    pl.add_argument("--set", action="append", default=[])
    pl.add_argument("--device", default="", help="예산 프로파일 (rtx3060_12gb | rtx4090_24gb)")
    pl.add_argument("--device-torch", default="auto", help="cuda | cpu | auto")
    pl.add_argument("--skip-budget", action="store_true", help="준비 단계(G4)를 건너뜁니다")
    pl.add_argument("--what-if", action="append", default=[])
    pl.add_argument("--no-measure", action="store_true")
    pl.add_argument("--limit", type=int, default=0)
    pl.add_argument("--shard-size", type=int, default=64)
    pl.add_argument("--out-dir", default="", help="기본값 runs/<run_id>/materialized")
    pl.add_argument("--materialized", default="", help="기본값 runs/<run_id>/materialized")
    pl.add_argument("--max-steps", type=int, default=0)
    pl.add_argument("--cache-dir", default=".cache")
    pl.add_argument("--cache-backend", default="local")
    pl.add_argument("--no-cache", action="store_true")
    pl.add_argument("--view", default="")
    pl.add_argument("--debug-output", action="store_true")
    pl.add_argument("--trigger", default="cli", choices=("cli", "ui", "external"))
    pl.add_argument("--progress", default="")
    pl.add_argument("--step-delay", type=float, default=0.0,
                    help="상자 하나를 지날 때마다 쉬는 시간(초)")
    pl.set_defaults(func=cmd_pipeline)

    ex = sub.add_parser(
        "export", help="학습한 모델을 LoRA 까지 합쳐 한 덩어리로 내보냅니다 (계약 포함)")
    ex.add_argument("spec")
    ex.add_argument("--run-id", default="")
    ex.add_argument("--model-dir", default="", help="기본값 runs/<run_id>/train")
    ex.add_argument("--out", default="", help="기본값 runs/<run_id>/export")
    ex.add_argument("--set", action="append", default=[])
    ex.add_argument("--cache-dir", default=".cache")
    ex.set_defaults(func=cmd_export)

    pv = sub.add_parser("preview", help="노드 하나만 실행해 시각화 출력을 봅니다")
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
            "  노드가 받아들일 수 없는 값입니다. --set 으로 준 값의 형태를 확인하라 "
            "(예: 리스트는 --set n_plot.size=[640,320]).",
            file=sys.stderr,
        )
        print(
            "  전체 추적을 보려면 VLMT_TRACE=1 을 설정하고 다시 실행하세요.",
            file=sys.stderr,
        )
        if os.environ.get("VLMT_TRACE"):
            raise
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
