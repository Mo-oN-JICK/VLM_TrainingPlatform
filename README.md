# VLM Trainer

Mech-Vision의 규약을 모방한 노드 그래프 기반 파인튜닝 플랫폼. 설계 문서는 [`docs/design/`](docs/design/README.md).

현재 상태: **Phase 0–4 완료** — 타입 시스템 · 레지스트리 · 컴파일러(G1/G2) · decompile 왕복 ·
노드 카탈로그 26개 · 실행 엔진(캐시 · spawn 워커 · 격리) · dry-run(G3) · **자원 예산 게이트(G4)** ·
노드 단위 미리보기 · **물질화와 재개** · CLI 9개 명령. **4중 게이트가 전부 동작한다.**
합성 더미 데이터로 전 경로가 돌고, 예산을 넘는 설정은 학습 전에 거부되며, 중단된 물질화는 이어받는다.
다음은 Phase 5(Trainer 실학습과 추론 계약).

## 실행

```bash
python -m pytest tests -q

# 0) 합성 더미 데이터 생성 (한 번만, seed 고정이라 언제 돌려도 같은 데이터)
python tools/make_dummy_dataset.py --n 24

# 1) 컴파일 - G1(타입) . G2(구조/정책)
python -m vlm_trainer.cli.main compile solutions/dummy_ecg/projects/01_dummy/project.yaml -v

# 2) dry-run - G3. 샘플 3건을 전 노드에 통과시켜 실측 검증 + 결정성 감사
python -m vlm_trainer.cli.main dryrun solutions/dummy_ecg/projects/01_dummy/project.yaml --samples 3 -v

# 3) 노드 하나만 실행해 시각화 출력 보기 (상류만 계산한다)
python -m vlm_trainer.cli.main preview solutions/dummy_ecg/projects/01_dummy/project.yaml --node n_sample

# 4) 예산 - G4. 단계별 VRAM과 시퀀스 길이를 학습 전에 산정한다
python -m vlm_trainer.cli.main budget solutions/dummy_ecg/projects/01_dummy/project.yaml
python -m vlm_trainer.cli.main budget solutions/dummy_ecg/projects/01_dummy/project.yaml \
    --what-if backbone=dummy-7b --what-if quantization=none      # 거부되는 예
python -m vlm_trainer.cli.main budget solutions/dummy_ecg/projects/01_dummy/project.yaml \
    --device rtx4090_24gb                                        # 다른 PC 프로파일

# 5) 물질화 - 경계까지 구워 shard로 남긴다. 중단해도 --resume이 이어받는다
python -m vlm_trainer.cli.main materialize solutions/dummy_ecg/projects/01_dummy/project.yaml --run-id demo --shard-size 8
python -m vlm_trainer.cli.main materialize solutions/dummy_ecg/projects/01_dummy/project.yaml --run-id demo --resume

# 6) 실행 - Output 노드까지 (dataset export + train plan). G4에 걸리면 시작하지 않는다
python -m vlm_trainer.cli.main run solutions/dummy_ecg/projects/01_dummy/project.yaml --limit 8 --run-id demo

# 노드 라이브러리 / 노드 상세 / 스펙 되돌리기
python -m vlm_trainer.cli.main nodes
python -m vlm_trainer.cli.main show image.crop_by_regions
python -m vlm_trainer.cli.main decompile solutions/dummy_ecg/projects/01_dummy/project.yaml
```

테스트용 노드가 필요할 때만 `--nodes fixture_nodes`를 붙이고 `PYTHONPATH`에 `tests`를 넣는다
(Git Bash의 `$PWD`는 POSIX 경로라 Windows Python이 읽지 못하므로 Windows 경로로 지정할 것).

## 구성

| 경로 | 내용 |
|---|---|
| `vlm_trainer/core/` | 포트 타입, 단일화, 레지스트리, 그래프, 컴파일러, 게이트 에러 |
| `vlm_trainer/spec/` | YAML 로더, canonical form, decompile |
| `vlm_trainer/nodes/` | 노드 카탈로그 26개 (Data Acquisition · 2D · Time Series · Expert · Adapters · Prompt · Answer · Data · File · Training) |
| `vlm_trainer/engine/` | 실행 엔진 — 캐시, spawn 워커, 샘플 공간, dry-run(G3), 예산(G4), 물질화, 저널, 미리보기 |
| `vlm_trainer/train/` | 선언형 TrainerConfig, 물질화 산출물 리더 |
| `vlm_trainer/answer/` | 정답 Text 스키마 — 렌더러와 파서를 같은 정의에서 생성 |
| `vlm_trainer/plugins/` | 플러그인 규약 + 더미 전문가 모델 2종(이미지 영역 / 시계열 구간) |
| `vlm_trainer/cli/` | `vlmt` 커맨드 |
| `solutions/dummy_ecg/` | 합성 더미 데이터로 도는 예제 Solution (Procedure 포함) |
| `tools/` | 합성 더미 데이터 생성기 |
| `tests/` | 완료 조건 81개 + 예제 Solution |
| `docs/design/` | 설계 문서 13편 |

## 설계에서 구현으로 오며 바뀐 것

| 설계 문서 | 구현 | 이유 |
|---|---|---|
| blake3 해시 | `blake2b`(16바이트) | 표준 라이브러리만으로 돌리기 위해. 해시는 engine ABI에 포함되므로 나중에 바꾸면 캐시가 전부 무효화된다 |
| `ImageList` / `TextList` 를 별도 base kind로 | `list_of` 하나로만 표현하고 표시할 때만 `ImageList`로 렌더 | 같은 것을 두 방식으로 표현하면 반드시 어긋난다 |
| shape의 심볼 변수 | 심볼 변수 + **`DYN`(가변 차원)** 을 구분 | 원본 이미지 크기는 "미해결"이 아니라 "샘플마다 다름"으로 확정된 값이다. `DYN`은 compile을 통과하고, 고정 크기를 요구하는 포트에서만 거부되며 리사이즈를 요구한다 |
| 순환은 캔버스가 구조적으로 막으므로 검사 없음 | 손으로 쓴 YAML에 한해 위상 정렬 후 남은 노드를 구조 오류로 보고 | CLI 스펙은 캔버스의 보호를 받지 않는다 |
| 실행 환경 RTX 4090 24GB | 이 PC는 **RTX 3060 12GB** | 예산 게이트 기준값이 다르다. Phase 3에서 프로파일로 분리한다 |
| `list.map(procedure)` | 지금은 **단일 노드만** 매핑한다 | 서브그래프 실행은 엔진이 더 필요하다. 리스트 원소마다 리사이즈하는 실제 용도는 이것으로 충분하다 |
| `text.template`이 슬롯마다 포트를 만든다 | 슬롯 값은 `context: Table` 포트 하나로 받는다 | 노드 등록은 정적이라 인스턴스마다 포트를 바꿀 수 없다. 동적 포트는 UI(Phase 7)와 함께 다시 본다 |
| — | `adapt.image_frame` 신설 | crop을 원본과 한 리스트에 담으려면 좌표 기준을 다시 선언해야 한다. 암묵 변환을 금지했으므로 이 선언도 노드로 남는다 |
| 프롬프트의 이미지 자리표시자 개수를 타입에 싣는다 | 지금은 `sample.assemble`이 실행 시점에 실측 대조한다. 예산 산정은 `list.concat.max_n`(타입의 리스트 상한)을 쓴다 | 실제 개수는 샘플마다 달라 컴파일 시점에 확정되지 않는다. 예산은 상한으로 보수적으로 잡는다 |
| 예산 기준 장치가 4090 24GB | 기본 프로파일은 **`rtx3060_12gb`**, `rtx4090_24gb`는 전환만 하면 된다 | 지금 이 PC가 3060이다. 설정 한 줄로 바뀌므로 다른 PC로 옮길 때 그래프는 그대로다 |
| 물질화 shard 포맷이 webdataset tar | **디렉터리 + jsonl + PNG** | 검증 대상(원자 커밋·재개·매니페스트)은 같고 읽기가 훨씬 단순하다. tar 패킹은 학습 처리량이 실제로 문제될 때 바꾼다 |
| 텍스트 토큰 수를 토크나이저로 잰다 | 토크나이저가 없는 동안 `chars_per_token`(기본 2.5)으로 환산하고, dry-run 실측 문자 수를 쓴다 | 백본이 붙는 Phase 5에서 실제 토크나이저로 교체한다. 선언값이 실측보다 작으면 G4가 거부하므로 낙관적으로 기울지 않는다 |
