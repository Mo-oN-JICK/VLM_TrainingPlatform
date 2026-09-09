# VLM Trainer — 플랫폼 설계

Mech-Vision의 규약을 의도적으로 모방한 범용 VLM 파인튜닝 플랫폼의 초기 설계 문서.

## 설계 의도 (3줄)

1. **모방이 기본값이다.** Solution/Project/Procedure/Node 4계층, 기능별 Node Library, 7파티션 UI, Debug Output·단일 노드 실행·History·Data Storage 디버깅 규약, Parameter Recipe — Mech-Vision에 이미 있는 개념은 이름과 구조를 그대로 가져오고 새 개념을 발명하지 않는다.
2. **딱 한 축에서만 더 엄격하다.** 비전 파이프라인은 오배선을 수 초 만에 눈으로 보지만 우리는 GPU 수십 시간을 태운 뒤에 안다. 그래서 포트 타입을 이름표가 아닌 구조체로 올리고, 편집·compile·dry-run·자원예산의 4중 게이트를 모두 통과해야만 Trainer가 GPU를 잡는다. 검증 비용 ≪ 학습 실패 비용이므로 편집기는 항상 보수적으로 거부한다.
3. **그래프는 전처리까지, 학습은 선언형이다.** 데이터 로딩~정답 Text 생성은 수직 DAG로, 다단계 학습·freeze·LoRA·optimizer는 말단 Trainer 노드의 선언형 설정으로 받는다. 양쪽 모두 정규화된 텍스트 스펙으로 compile/decompile되어 UI 없이 CLI로 재현되며, 1급 실행 프로파일은 단일 RTX 4090 24GB / Windows다.

## 근거 표기 규약

문서 전체에서 다음 표기를 사용한다.

| 표기 | 의미 |
|---|---|
| `[문서확인]` | docs.mech-mind.net 공개 문서에서 확인한 사실 |
| `[이미지확인]` | 첨부된 Mech-Vision 화면 캡처에서 직접 관찰·측정한 사실 |
| `[추정]` | 문서·이미지로 확정하지 못한 Mech-Vision 동작에 대한 추정 |
| `[각색]` | Mech-Vision에 대응 개념이 있으나 우리 도메인에 맞춰 바꾼 것 (이유 명시) |
| `[신설]` | Mech-Vision에 대응 개념이 없어 새로 도입한 것 (이유 명시) |

## 확인한 1차 근거

**문서 (docs.mech-mind.net)**
- 계층: Solution > Project > Procedure > Step. Step은 "a minimum functional unit in a project", Procedure는 "a collection of multiple Steps". `[문서확인]`
- Procedure 생성: Step Library에서 Procedure 추가 → 더블클릭해 진입 → Step 드래그 → Step의 포트를 더블클릭해 Procedure 입출력으로 승격(클릭 순서대로) → Navigate Up으로 빠져나옴. 내부 Step의 파라미터를 Procedure 수준 파라미터로 노출 가능. `[문서확인]`
- UI 파티션: Menu Bar / Toolbar / Project List / Step Library / Project Toolbar / Graphical Programming Workspace / Project Configuration Pane / Log. `[문서확인]`
- Parameter Recipe: "sets of parameter settings that need to be adjusted according to different situations for the same project", 동일 로직 프로젝트를 여러 벌 만들 필요를 없앰. Project Assistant에서 진입, Parameter Recipe Editor에서 Add Recipe, 새 레시피는 현재 프로젝트 값을 기본 상속, 드롭다운으로 전환, 프로젝트를 직접 수정하면 **Customized**로 전환, 레시피마다 ID 보유. `[문서확인]`
- 외부 전환: `Switch_Recipe` 신호 + `Vision_Recipe_Num`(1~99 양의 정수), `Vision_Proj_Num`은 Project List의 프로젝트 이름 앞 번호. Start Mech-Vision Project 이전에 실행. `[문서확인]`
- 디버깅: Project Toolbar의 Debug Output을 켜면 실행 중 Step 출력이 패널에 표시됨. 개별 Step 단독 실행 아이콘 존재. Step 실행 시간은 Step 박스에 표시. "Statuses of Steps" 절 존재. Debug Output 패널은 메인 화면 우상단. `[문서확인]`
- Step 포트: 입력 포트는 선행 Step에서 지정 타입·용도의 데이터를 받고, 출력 포트는 후행 Step으로 내보냄. `[문서확인]`
- Step Library 카테고리(최신 문서 기준): Data Acquisition, Preprocessing, Recognition, AI Tools, Locating, Postprocessing, Measurement, Pose Processing, Path Planning, Trajectory Processing, Evaluation, Data Processing, File, Transmission, System, Tools, Common Procedure, Advanced Step. `[문서확인]`

**이미지 (MechVisionImage/)**
- `mechvision_full.png`: 좌측 상단 Project List(Solution `Test` 아래 `2 Beams / 3 convert / 4 sleeves / 5 Compressors / 6 Track_Links / 7 Crankshaft (Small)` — 번호 접두 확인), 좌측 하단 Step Library(검색창 + 카테고리 트리: Capture, Preprocessing, Recognition, Pose Processing, Path Planning, Evaluation, Data Processing, File, Transmission, System, Advanced Step), 중앙 워크스페이스(상단에 Run / Continuous Run / Debug Output 토글), 우상단 Debug Output 패널, 우하단 Step Parameters 패널 + 탭 스트립 `Step Parameters | Project Assistant | Step Quick Info | History | Step Comment List`. `[이미지확인]`
- `mechvision_canvas.png`: 그래프가 **위에서 아래로 수직 흐름**. 노드 카드 상단에 입력 포트 칩, 하단에 출력 포트 칩. 포트 칩은 2줄(윗줄 `<타입>`, 아랫줄 포트 이름)이고 **타입별로 색이 다르다**. 배선은 포트 칩 하단에서 다음 노드 상단 칩으로 이어지는 곡선이며 **선 색이 소스 포트 타입 색을 따른다**. 노드 헤더에 Step 이름 + 인스턴스 번호, 우상단에 접기/단독실행 아이콘, 본문에 `Time: 2ms`와 파라미터 요약, 우하단에 시각화(눈) 아이콘. `[이미지확인]`
- `mechvision_params.png`: Step Parameters는 `라벨 | 위젯` 2열 행, 접히는 그룹(`Execution Flags` 등), 우측 상단에 Step Quick Info(Step description / Usage scenario / Input / Output). `[이미지확인]`

## 목차

| # | 문서 | 내용 |
|---|---|---|
| 1 | [01-layers.md](01-layers.md) | 플랫폼 레이어 구조와 데이터 흐름, 4계층 저장·버전·상속 |
| 2 | [02-node-interface.md](02-node-interface.md) | 노드 인터페이스, 강타입 포트 구조체, 3분류, 4중 게이트 |
| 3 | [03-node-catalog.md](03-node-catalog.md) | 카테고리별 초기 노드 카탈로그 |
| 4 | [04-project-spec.md](04-project-spec.md) | Project Spec 스키마, compile/decompile, Triad 예시 |
| 5 | [05-expert-plugin.md](05-expert-plugin.md) | 외부 전문가 모델 플러그인 인터페이스 |
| 6 | [06-answer-schema.md](06-answer-schema.md) | Output 설계 모듈: 정답 Text 스키마·검증·누설 차단·추론 계약 |
| 7 | [07-trainer-config.md](07-trainer-config.md) | Trainer 선언형 설정 스키마와 VRAM 산출 |
| 8 | [08-engine.md](08-engine.md) | 그래프 실행 엔진, 캐시·물질화·디버깅·Windows 워커 |
| 9 | [09-directory.md](09-directory.md) | 디렉터리 구조와 파일별 책임 |
| 10 | [10-roadmap.md](10-roadmap.md) | 단계별 구축 순서와 완료 조건 |
| 11 | [11-mapping.md](11-mapping.md) | Mech-Vision ↔ 우리 플랫폼 개념 대응표 |
| 12 | [12-ui-theme.md](12-ui-theme.md) | UI 파티션 레이아웃과 디자인 토큰 |
| 13 | [13-parameter-recipe.md](13-parameter-recipe.md) | Parameter Recipe 시스템과 스윕 |

## 용어 충돌 주의

"레시피"가 두 곳에서 쓰인다. 문서 전체에서 아래로 고정한다.

- **Project Spec** = 하나의 학습 레시피 = 그래프 + Trainer 설정 (문서 4)
- **Parameter Recipe** = Mech-Vision과 동일한 파라미터 오버레이 묶음, 번호로 전환 (문서 13)
