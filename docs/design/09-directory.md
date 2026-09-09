# 9. 디렉터리 구조와 파일별 책임

## 9.1 코드

```
vlm_trainer/
  core/
    types.py           PortType/BaseKind/Dim/Frame/TimeBase 정의, 표기 파서·프린터
    unify.py           타입 단일화, 제네릭 해소, 호환성 판정(2.3의 7규칙)
    registry.py        노드 등록 데코레이터, 3분류-포트형상 검사, entry point 스캔
    node.py            Node 기반 클래스, RunCtx, NodeError, 계약 C1~C9 정의
    graph.py           GraphModel(노드/배선/Procedure 인스턴스), 편집 커맨드 객체
    compiler.py        compile 9단계 파이프라인, CompiledGraph 산출
    checks/
      structure.py     미연결 포트, 도달 불가, Output 존재, 팬인
      policy.py        누설 taint 도달성, external_call 배치, 프로파일 허용 목록
      determinism.py   결정성 선언 검사
    errors.py          게이트별 에러 타입과 "안 잡혔다면" 메시지 렌더러
    fingerprint.py     blake3 지문, canonical 직렬화, 스펙 해시

  spec/
    schema/            *.json — Project/Procedure/Recipe/AnswerSchema/TrainerConfig 스키마
    loader.py          YAML 로드, 스키마 검증, 상속 병합(solution→project→procedure→node)
    canonical.py       canonical form 정규화 규칙(4.3)
    decompile.py       CompiledGraph/GraphModel → Spec, Procedure 되접기
    lock.py            graph.lock 읽기/쓰기, 버전 해소
    recipe.py          Parameter Recipe 오버레이 적용, 화이트리스트 검사, 스윕 전개

  engine/
    scheduler.py       위상 실행, 3분류별 정책, 노드 상태 머신
    cache.py           캐시 키 계산, 짧은 해시 경로, LRU GC
    materialize.py     물질화 경계 실행, shard 원자 커밋, 매니페스트
    worker.py          spawn 워커 풀, GPU 워커 배타 락, 크래시 감지
    journal.py         run 저널 append/replay, 재개 계획 수립
    preview.py         preview 렌더러 레지스트리(8.4 표), PreviewBus, Debug Output 게이팅
    budget.py          G4 정적 추정(7.3 공식), 민감도 분석, 거부 메시지
    dryrun.py          G3 실행, 실측 대조, 위반율 집계
    isolate.py         샘플 quarantine, 실패 격리 정책
    history.py         편집 저널, diff 스냅샷, 되감기

  nodes/
    source/  image.py timeseries.py metadata.py text_asset.py schema_define.py
    image/   crop.py tile.py overlay.py concat.py
    timeseries/ window.py resample.py stats.py plot.py detect.py
    expert/  propose.py classify.py caption.py embed.py     (플러그인 호출 껍데기)
    adapt/   image.py regions.py timebase.py tokenize.py
    text/    template.py join.py knowledge.py image_slots.py
    answer/  stepwise.py evidence.py validate.py leakage.py
    data/    sample.py table.py list_ops.py
    eval/    conformance.py stats.py leakage.py balance.py
    vis/     grid.py prompt.py answer.py timeseries.py
    io/      export.py report.py data_storage.py
    train/   trainer.py resume.py

  plugins/
    base.py            ExpertPlugin/BackboneAdapter/StorageBackend 프로토콜
    manifest.py        plugin.yaml 검증, requires/os 검사, lock 기록
    experts/           내장 예시 플러그인
    backbones/         백본 어댑터 구현(hf_* 등)

  train/
    loop.py            단계 실행, 체크포인트, 데이터로더 상태 저장
    freeze.py          trainable 정책 → requires_grad 적용, overrides 정규식
    shards.py          물질화 shard 리더(webdataset/arrow)
    contract.py        inference_contract.json / inference_graph.yaml 생성(6.5)

  cli/
    main.py            compile dryrun budget materialize run preview decompile
                       sweep recipe cache infer-graph
    render.py          터미널 표·에러 출력

  ui/                  (Phase 7)
    server.py          코어 API를 감싸는 로컬 서버. UI 전용 실행 경로 없음
    events.py          노드 상태·preview 스트림
    web/               7파티션 레이아웃, 캔버스, 토큰(문서 12)
```

## 9.2 사용자 데이터

```
solutions/<solution_name>/
  solution.yaml        과제 메타, 전역 defaults, 실행 프로파일, 캐시 상한
  procedures/*.yaml    재사용 서브그래프 (name@semver)
  schemas/*.yaml       AnswerSchema, EvidenceRules
  knowledge/*.md       도메인 지식 텍스트 자산
  data/index.jsonl     sample_space 인덱스
  projects/NN_name/
    project.yaml       그래프 스펙 (진실의 원천)
    trainer.yaml       TrainerConfig
    recipes.yaml       Parameter Recipe
    layout.json        노드 좌표 (의미 없음)
    graph.lock         해소된 버전·해시
    edit_history.jsonl History 탭 원천

runs/<run_id>/
  spec.snapshot.yaml   실행 시점 canonical 스펙 (재현의 기준)
  compiled.json        실행 계획
  journal.jsonl        커밋 사실 (재개의 기준)
  budget.json          G4 산출 표
  dryrun.json          G3 결과
  materialized/*.tar   물질화 shard + manifest.jsonl
  ckpt/<stage>/        가중치, optimizer, dl_state, inference_contract.json
  quarantine/          위반·실패 샘플
  previews/            CLI에서 --debug-output으로 떨어뜨린 렌더 결과
  logs/                jsonl 로그

.cache/<2자>/<16자>/   out.bin + meta.json  (짧은 해시 경로)
```

## 9.3 파일별 책임 한 줄 요약(핵심만)

| 파일 | 한 줄 책임 |
|---|---|
| `core/types.py` | 포트 타입이 무엇인지에 대한 유일한 정의. 다른 어떤 모듈도 타입 필드를 임의로 늘리지 않는다 |
| `core/unify.py` | "연결해도 되는가"에 답하는 유일한 곳. 자동 캐스팅 코드가 들어가면 안 되는 곳 |
| `core/registry.py` | 노드가 계약을 선언하는 입구이자 3분류-포트형상 정합을 강제하는 문지기 |
| `core/compiler.py` | G1/G2를 모아 실행 계획을 만드는 곳. 게이트 우회 옵션을 절대 추가하지 않는다 |
| `core/errors.py` | "이 오류가 안 잡혔다면 언제 어디서 터졌을지"를 만드는 곳. 게이트의 가치를 사용자에게 전달하는 유일한 통로 |
| `spec/canonical.py` | 실험 동일성의 정의(해시). 여기 규칙이 바뀌면 과거 실험과의 비교가 끊긴다 |
| `spec/recipe.py` | 값 오버레이만 허용하고 구조 변경을 거부하는 경계 |
| `engine/cache.py` | 무엇이 바뀌면 다시 계산해야 하는가에 대한 유일한 답 |
| `engine/materialize.py` | 전처리와 학습이 GPU를 두고 경쟁하지 않게 만드는 분리막 |
| `engine/budget.py` | GPU를 잡기 전 마지막 문. 추정이 틀리면 여기 상수를 dry-run 실측으로 보정한다 |
| `engine/worker.py` | spawn 전제, GPU 배타 락. Windows 단일 GPU 제약이 코드로 나타나는 곳 |
| `train/contract.py` | 학습과 추론의 전처리가 어긋나지 않게 묶는 곳 |
| `plugins/base.py` | 백본·전문가 모델 교체가 설정 변경으로 끝나게 만드는 추상 경계 |
