# VLM Trainer

Mech-Vision의 규약을 모방한 노드 그래프 기반 파인튜닝 플랫폼. 설계 문서는 [`docs/design/`](docs/design/README.md).

현재 상태: **Phase 0–8 완료** (Phase 8은 인터페이스 수준). 편집기로 한 바퀴가 돈다 — 타입 시스템 · 레지스트리 · 컴파일러(G1/G2) · decompile 왕복 ·
노드 카탈로그 26개 · 실행 엔진(캐시 · spawn 워커 · 격리) · dry-run(G3) · **자원 예산 게이트(G4)** ·
노드 단위 미리보기 · 물질화와 재개 · **다단계 학습(LoRA·freeze·체크포인트·재개)** ·
추론 계약 · **Parameter Recipe와 스윕** · CLI 13개 명령. **4중 게이트가 전부 동작한다.**
합성 더미 데이터로 전 경로가 GPU에서 돌고, 추론 그래프가 만든 프롬프트는 학습 때와 바이트 단위로 같으며,
그래프 하나 위에서 레시피만 바꾼 실험 여러 개가 물질화를 공유하며 순차로 돈다.
`vlmt view`가 컴파일된 그래프를 한 장의 HTML로 그리고, `vlmt run --view --debug-output`은
거기에 실행 상태와 노드별 Debug Output을 함께 칠한다. `vlmt edit`는 로컬 편집기를 연다 —
**타입이 맞지 않는 배선은 드롭 자체가 되지 않고**, 노드를 고르면 파라미터를 그 자리에서 고치며,
Node Library에서 노드를 누르거나 캔버스로 끌어다 놓아 추가하고, Parameter Recipe를 골라
값을 덮어 보고 새 조합을 레시피로 담으며, History 항목을 누르면 그 시점으로 되감는다.
Run·Materialize·Train은 각각 같은 이름의 CLI 명령을 그대로 하위 프로세스로 띄우고,
그 진행 파일을 폴링해 카드 상태와 loss를 칠한다 — UI 전용 실행 경로는 없다.
`vlmt new`가 빈 껍데기를 만들고, Sample Space도 패널에서 고친다
(프레임워크 없이 표준 라이브러리만 쓴다).
실물 백본은 `backbone: hf:<모델 경로>` 한 줄로 들어온다 — 어댑터가 `config.json`만 읽어
예산을 답하므로 가중치 없이도 G4가 돈다. 남은 것은 모델 id 결정과 다운로드,
그리고 실제 다중 GPU 실행과 원격 실행.

실행 환경은 `.venv`(Python 3.12 + torch 2.14.0+cu130)다. `python` 대신 `.venv\\Scripts\\python.exe`를 쓴다.

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

# 7) 학습 - 물질화된 shard로 다단계 학습 (G4를 먼저 통과해야 시작한다)
python -m vlm_trainer.cli.main train solutions/dummy_ecg/projects/01_dummy/project.yaml --run-id demo --set n_train.config_path=trainer_tiny.yaml

# 8) 추론 그래프 - 학습 그래프에서 정답 경로를 잘라낸 서브그래프
python -m vlm_trainer.cli.main infer-graph solutions/dummy_ecg/projects/01_dummy/project.yaml --out infer.yaml

# 9) Parameter Recipe - 그래프는 그대로 두고 값만 바꾼다
python -m vlm_trainer.cli.main recipe list solutions/dummy_ecg/projects/01_dummy/project.yaml
python -m vlm_trainer.cli.main recipe diff solutions/dummy_ecg/projects/01_dummy/project.yaml --recipe-id 1 --other 3
python -m vlm_trainer.cli.main recipe expand solutions/dummy_ecg/projects/01_dummy/project.yaml --sweep spike_x_crop
python -m vlm_trainer.cli.main dryrun solutions/dummy_ecg/projects/01_dummy/project.yaml --recipe 2

# 10) 스윕 - 단일 GPU 순차 큐. 전처리 지문이 같은 레시피는 물질화를 공유한다
python -m vlm_trainer.cli.main sweep solutions/dummy_ecg/projects/01_dummy/project.yaml --recipes 1,3,4 --limit 6 --max-steps 3

# 11) 그래프 뷰어 - 캔버스 규약을 눈으로 확인한다 (읽기 전용)
python -m vlm_trainer.cli.main view solutions/dummy_ecg/projects/01_dummy/project.yaml --open

# 12) 실행 상태 + Debug Output을 칠한 뷰 (토글이 꺼져 있으면 미리보기를 만들지도 않는다)
python -m vlm_trainer.cli.main run solutions/dummy_ecg/projects/01_dummy/project.yaml --run-id vw --limit 3 --debug-output --view -

# 13) 편집기 - 배선을 끌어 놓는다. 타입이 안 맞으면 놓이지 않는다
python -m vlm_trainer.cli.main edit solutions/dummy_ecg/projects/01_dummy/project.yaml --open

# 14) 백본 - 등록된 것과 그 형상 (가중치는 열지 않는다)
python -m vlm_trainer.cli.main backbones
python -m vlm_trainer.cli.main backbones --add hf:D:/models/my-2b-vlm

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
| `vlm_trainer/spec/` | YAML 로더, canonical form, decompile, Parameter Recipe |
| `vlm_trainer/nodes/` | 노드 카탈로그 26개 (Data Acquisition · 2D · Time Series · Expert · Adapters · Prompt · Answer · Data · File · Training) |
| `vlm_trainer/engine/` | 실행 엔진 — 캐시, spawn 워커, 샘플 공간, dry-run(G3), 예산(G4), 물질화, 저널, 미리보기, 스윕 |
| `vlm_trainer/train/` | 선언형 TrainerConfig, shard 리더, freeze 정책, 학습 루프, 추론 계약 |
| `vlm_trainer/answer/` | 정답 Text 스키마 — 렌더러와 파서를 같은 정의에서 생성 |
| `vlm_trainer/plugins/` | 플러그인 규약 + 더미 전문가 모델 2종(이미지 영역 / 시계열 구간) |
| `vlm_trainer/ui/` | 디자인 토큰(문서 12의 실측값), 그래프 뷰어, 편집기 API·로컬 서버 |
| `vlm_trainer/cli/` | `vlmt` 커맨드 |
| `solutions/dummy_ecg/` | 합성 더미 데이터로 도는 예제 Solution (Procedure 포함) |
| `tools/` | 합성 더미 데이터 생성기 |
| `tests/` | 완료 조건 159개 + 예제 Solution |
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
| Phase 5의 백본이 실물 2B | **로컬 소형 백본 `tiny-vlm`**(2.3M 파라미터, 다운로드 없음)으로 학습 경로 전체를 검증 | 비전 타워·프로젝터·LoRA·다단계 freeze·체크포인트·재개는 모델 크기와 무관하게 같은 코드다. 실물 백본은 `BackboneAdapter`를 채우고 `backbone:` 한 줄을 바꾸면 된다 |
| 프로파일 미지원 옵션은 G2가 거부 | **G4가 함께 본다** | Trainer 설정이 로드되는 지점이 G4다. 학습 시작 전이라는 성질은 같다 |
| Phase 7이 편집 가능한 UI | 읽기 전용 뷰어 → 그 위에 편집기 | 캔버스 규약을 먼저 눈과 테스트로 고정했다. 편집기는 같은 렌더러에 스크립트만 얹는다 |
| History가 직전 스펙과의 **diff**를 저장 | 저널에는 diff를, 되감기에는 **스냅샷**을 쓴다 | 스펙이 작아서 스냅샷이 싸다. diff를 되감기에 쓰려면 패치 적용기를 직접 짜야 하는데 그 값어치가 없다. 저널 파일은 여전히 diff라 사람이 읽는다 |
| 입력 포트가 이미 찼으면 연결 거부 | **연결은 교체다** — 원자적으로 갈아끼운다 | 팬인 금지는 그대로다. 다만 갈아끼우기를 disconnect+connect 두 단계로 만들면 중간 상태가 컴파일되지 않아 편집기가 쓸모없어진다 |
| 편집 중에도 그래프는 언제나 컴파일된다 | 편집기는 **draft 컴파일**로 미완성 상태를 허용한다. 타입(G1)과 정책 검사는 그대로 돌고, 미루는 것은 완결성뿐이다 — 미연결 필수 포트·도달 불가 노드·Output 없음·미해결 제네릭 | 노드를 놓고 배선을 잇기까지 사이에 반드시 미완성 순간이 있다. 그 순간을 금지하면 UI로는 그래프를 만들 수 없다. 대신 **저장과 실행은 언제나 strict 경로를 지난다** — 게이트 우회 스위치가 아니라 미완성을 디스크에 남기지 않는 장치다 |
| 라이브러리에서 캔버스로 끌어 놓은 **위치**가 노드의 자리 | 놓은 위치를 쓰지 않는다. 캔버스는 위상 순서로 스스로 정렬한다 | 자리를 사람이 정하면 스펙에 좌표가 생기고, 좌표는 spec_hash에 들어가거나 아니면 스펙과 화면이 어긋난다. 둘 다 손해다 |
| 물질화 경계를 안 적으면 그 검사만 건너뛴다 | 학습 노드가 있는 그래프에서 경계가 비면 **거부한다** | `if boundary:`라 비어 있으면 external_call 배치 검사가 통째로 사라졌다. 검사를 끄는 방법이 "경계를 안 적는 것"이면 게이트가 아니다. 새 프로젝트의 기본값이 빈 경계라 이 구멍은 기본 경로에 있었다 |
| 다중 GPU를 켜면 예산이 알아서 나눠 계산한다 | ddp만 계산하고 **fsdp/deepspeed는 거부한다**. 보고서는 "이 VRAM은 장치 하나 기준"이라고 적는다 | fsdp는 가중치와 옵티마이저를 장치에 쪼개 장치당 VRAM 자체가 달라지는데 그 모델이 없다. 모르는 것을 단일 GPU 숫자로 통과시키면 G4가 존재할 이유가 사라진다. 장치 수를 늘려 VRAM이 준다고 읽히는 것도 같은 종류의 거짓말이다 |
| History는 세션 안에서만 되감는다 | 저널 옆에 **내용 주소 스냅샷 저장소**(`edit_history/`)를 두어 어제의 시점으로도 되감는다 | 되감기에는 스펙 본문이 필요한데 저널에 본문을 섞으면 사람이 읽는 기록이라는 성질을 잃는다. 해시로 이름 붙이면 같은 상태로 돌아와도 파일이 늘지 않고, 저널이 가리키지 않는 본문은 여는 순간 지워진다 |
| 편집기가 실행 상태를 직접 계산해 그린다 | 실행은 **하위 프로세스의 `vlmt run`**이고, 편집기는 그 프로세스가 남기는 `runs/<id>/progress.json`을 폴링한다 | UI 전용 실행 경로를 만들지 않는다는 규칙이 여기서 시험대에 오른다. 프로세스를 나누면 실행이 편집기를 붙잡지 않고, 같은 파일을 다른 터미널에서도 볼 수 있으며, 편집기를 닫아도 실행은 계속된다 |
| 레시피 편집이 프로젝트의 파라미터 값을 바꾼다 | 편집기는 레시피를 **오버레이로만** 적용한다. 프로젝트 스펙은 그대로 저장된다 | Procedure가 노출한 파라미터는 프로젝트 스펙이 아니라 다른 파일 안의 노드를 가리킨다. 스펙에 써 넣으려면 Procedure를 고쳐야 하고, 그러면 그 Procedure를 쓰는 다른 프로젝트가 같이 바뀐다. 화면의 값과 저장될 값이 다르다는 사실은 감추지 않고 띠와 두 개의 spec_hash로 말한다 |
| `recipes.yaml`은 사람이 쓴 파일이니 편집기가 건드리지 않는다 | Save가 `project.yaml`과 함께 원자 교체한다. **주석은 사라진다** | 레시피를 UI에서 만들 수 없으면 스윕의 절반이 손으로 남는다. 주석 손실은 CLI의 `recipe set-active`도 이미 가진 성질이라 새로 생긴 함정이 아니다 |
| 레시피가 `stages[1].optimizer.lr` 같은 깊은 경로를 덮는다 | 오버라이드는 **`노드id.파라미터` 한 단계**만. Trainer 설정 변주는 `n_train.config_path`로 파일을 바꿔 표현한다 | 깊은 경로 오버레이는 화이트리스트 검사가 복잡해진다. 지금 형태로도 스윕은 성립하고, 필요해지면 그때 확장한다 |
| 텍스트 토큰 수를 토크나이저로 잰다 | **토크나이저가 있으면 센다.** 없으면 `chars_per_token`으로 환산하되 안전 여유(×1.15)를 얹고, 예산 보고서가 `(토크나이저)`인지 `(문자 환산)`인지 이름을 밝힌다 | 비율은 언어와 토크나이저에 따라 두 배씩 틀린다. 셀 수 있으면 세는 쪽이 언제나 낫고, 못 셀 때 낙관적으로 기울면 G4가 통과시킨 학습이 컨텍스트를 넘겨 터진다. 센 것과 환산한 것을 같은 말로 적으면 읽는 사람이 그 차이를 알 수 없다 |
