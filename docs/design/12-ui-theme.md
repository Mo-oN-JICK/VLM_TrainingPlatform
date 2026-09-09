# 12. UI 파티션 레이아웃과 테마 토큰

범위: 파티션 구조와 디자인 토큰까지만 다룬다. 컴포넌트 스타일링, 픽셀 목업, 인터랙션 디테일은 다루지 않는다.

1차 근거는 첨부 캡처다. `mechvision_full.png`에서 배치·상대 크기·도킹을, `mechvision_canvas.png`에서 노드 해부와 배선을, `mechvision_params.png`에서 파라미터 행 구성을 읽었다.

## 12.1 파티션 레이아웃 (텍스트 와이어프레임)

`mechvision_full.png`에서 관찰된 배치와 상대 크기를 그대로 따른다. 괄호 안 비율은 캡처에서 측정한 폭·높이 비율이다. `[이미지확인]`

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ Title Bar                                                    ─ □ ×            │
├───────────────────────────────────────────────────────────────────────────────┤
│ Menu Bar   File  Edit  View  Solution  Node  Plugins  Settings  Help          │
├───────────────────────────────────────────────────────────────────────────────┤
│ Toolbar    [New Project] [Node Registry] [Editing Mode ▾] [Budget] [Profile ▾] │
├──────────────────┬────────────────────────────────┬───────────────────────────┤
│ Projects List    │ ┌ Project Tabs ──────────────┐ │ Debug Output              │
│ (폭 18%)         │ │ 02_triad_ecg × │ 03_… ×    │ │ (폭 43%, 우측 상단 70%)   │
│  ▾ Solution      │ ├────────────────────────────┤ │                           │
│    2 baseline    │ │ Project Toolbar            │ │  선택 노드의 시각화 출력  │
│    3 triad_ecg ◀ │ │ Save Undo Redo │ Compile   │ │  (문서 8.4의 preview)     │
│    4 sweep_lora  │ │ Dry-run Budget │ ▶Run ■Stop│ │                           │
│  (높이 35%)      │ │ │ ◉ Debug Output          │ │  탭: 노드별 출력 스택     │
├──────────────────┤ ├────────────────────────────┤ ├───────────────────────────┤
│ Node Library     │ │                            │ │ Project Configuration     │
│ (폭 18%, 높이 65%)│ │  Graphical Programming     │ │ Panel (우측 하단 30%)     │
│  🔍 검색          │ │  Workspace                 │ │                           │
│  ▸ Data Acquisition│ │                          │ │  Node Name  [n_crop     ] │
│  ▸ 2D General Proc│ │   (수직 흐름 DAG)          │ │  ▸ Execution Flags        │
│  ▸ Time Series   │ │                            │ │  ▾ Crop                   │
│  ▸ Expert Models │ │    Input 레인 = 최상단      │ │    padding_ratio [0.15  ] │
│  ▸ Adapters      │ │        ↓                   │ │    square_pad    [✓]      │
│  ▸ Prompt Assembly│ │    Processing 레인          │ │  ⚙ Config wizard          │
│  ▸ Answer Design │ │        ↓                   │ │                           │
│  ▸ Data Processing│ │    Output 레인 = 최하단     │ ├───────────────────────────┤
│  ▸ Evaluation    │ │                            │ │ Node Params │ Project     │
│  ▸ Visualization │ │                            │ │ Assistant │ Node Quick    │
│  ▸ File          │ │                            │ │ Info │ History │ Node      │
│  ▸ Training      │ │                            │ │ Comment List │ Debug Out  │
│  ▸ System        │ │                            │ │                           │
├──────────────────┴─┴────────────────────────────┴─┴───────────────────────────┤
│ Log   [All ▾] [Errors] [Node] [Worker] [Trainer]          (기본 접힘)          │
└───────────────────────────────────────────────────────────────────────────────┘
```

`mechvision_full.png`에서 Log 패널은 열려 있지 않다. 문서에는 파티션으로 존재한다. `[문서확인]` `[추정: 기본 접힘 여부]`

## 12.2 파티션별 책임

| 파티션 | Mech-Vision 대응 | 우리 플랫폼에서 들어가는 것 |
|---|---|---|
| **Menu Bar** | 동일 `[문서확인]` | File(Solution 열기/저장), Edit(Undo/Redo/찾기), View(패널 토글), Solution(프로젝트 추가·번호 변경), Node(레지스트리 새로고침), Plugins(전문가·백본 플러그인 관리), Settings(실행 프로파일·캐시 상한), Help |
| **Toolbar** | 동일 `[문서확인/이미지확인]` | 새 Project, Node Registry 뷰어, 편집 모드, 전역 Budget 확인, 실행 프로파일 선택(`windows_single_gpu`) |
| **Projects List** | Project List `[문서확인/이미지확인]` | Solution 트리 + 번호 접두 Project 목록. 각 Project 옆에 마지막 run 상태(성공/실패/실행 중)와 활성 Parameter Recipe 번호 뱃지 |
| **Node Library** | Step Library `[문서확인/이미지확인]` | 검색창 + 문서 3의 카테고리 트리. 각 노드 항목에 3분류 뱃지(I/P/O)와 포트 타입 요약. 드래그해서 캔버스에 놓는다 |
| **Project Toolbar** | 동일 `[이미지확인]` | Save / Undo / Redo / **Compile / Dry-run / Budget** / Run / Stop / **Debug Output 토글** / 활성 Recipe 드롭다운 |
| **Graphical Programming Workspace** | 동일 `[문서확인/이미지확인]` | 수직 DAG 캔버스. 노드 카드, 포트 칩, 배선, 상태 표시, 레인 정렬 |
| **Debug Output** | 동일 `[문서확인/이미지확인]` | 문서 8.4의 노드 타입별 시각화 출력. 토글 off면 아무것도 생성·표시하지 않음. 외부 트리거 실행에서는 항상 비표시 |
| **Project Configuration Panel** | Project Configuration Pane `[문서확인]` | 아래 탭 구성 |
| └ Node Parameters | Step Parameters `[이미지확인]` | 라벨-위젯 2열, 접히는 그룹, Config wizard. 타입에 영향 주는 파라미터는 표식 |
| └ Project Assistant | 동일 `[문서확인]` | **Parameter Recipe**(편집기 진입), **Data Storage**, sample_space 설정, 물질화 경계, 실행 프로파일 |
| └ Node Quick Info | Step Quick Info `[이미지확인]` | 노드 기능 / 사용 시나리오 / **입출력 포트 설명 + 포트 타입 구조체 전체 표기** |
| └ History | 동일 `[이미지확인]` | 편집 이력. 항목 클릭 시 그 시점으로 되감기. 좌표 변경 항목은 "의미 없음" 표시 |
| └ Node Comment List | Step Comment List `[이미지확인]` | 노드 주석 목록 |
| └ Debug Output(탭) | `[이미지확인: canvas 캡처의 하단 탭에 존재]` | 패널이 접혔을 때의 대체 표시 |
| **Log** | 동일 `[문서확인]` | 워커·엔진·Trainer 로그 통합. 게이트 실패 메시지는 여기와 캔버스 양쪽에 표시 |

## 12.3 디자인 토큰

값은 첨부 캡처에서 **픽셀 단위로 측정**한 것이다. 두 캡처의 UI 버전이 달라 값이 다른 항목은 둘 다 기록하고, 우리 기본값은 노드 카드 해부가 선명한 `canvas` 계열을 택한다.

### 표면

| 역할 | 토큰명 | 값 | 출처 | 적용처 |
|---|---|---|---|---|
| 앱 크롬 배경 | `--surface-chrome` | `#121212` | full.png 타이틀/메뉴바 | 타이틀 바, 메뉴 바 |
| 툴바 배경 | `--surface-toolbar` | `#282828` | canvas.png 툴바 | Toolbar, Project Toolbar |
| 패널 배경 | `--surface-panel` | `#1F1F1F` | canvas.png 좌/우 패널 | Projects List, Node Library, Configuration Panel, Log |
| 패널 배경(대체 관측) | `--surface-panel-alt` | `#202020` | full.png Project List | 신버전 대응값. 기본값 아님 |
| 캔버스 배경 | `--surface-canvas` | `#1A1A1A` | canvas.png 캔버스 | Workspace 배경 |
| 캔버스 배경(대체 관측) | `--surface-canvas-alt` | `#121212` | full.png 캔버스 | 신버전 대응값 |
| 패널 구분선 | `--border-panel` | **확인 불가** | 1px 경계가 안티앨리어싱으로 분리 측정 불가 | 파티션 사이 구분선, 스플리터 |

### 노드 카드

| 역할 | 토큰명 | 값 | 출처 | 적용처 |
|---|---|---|---|---|
| 노드 본문 배경 | `--node-bg` | `#262A2F` | canvas.png 노드 본문 | 모든 노드 카드 본문 |
| 노드 테두리(기본) | `--node-border` | `#14C2AA` | canvas.png 카드 외곽(측정 범위 `#0FA78C`~`#0DC8AD`) | 노드 카드 1px 외곽 |
| 노드 강조 배경(선택/호버) | `--node-bg-selected` | `#3B5067` | canvas.png Instance Segmentation 카드 | 선택된 노드 본문 |
| 노드 강조 테두리 | `--node-border-selected` | `#57F7E6` | canvas.png 선택 카드 외곽 | 선택된 노드 외곽 |
| 노드 강조 내부선 | `--node-border-selected-inner` | `#447F8F` | canvas.png 선택 카드 2번째 픽셀 | 선택 시 이중 테두리 안쪽 |
| 노드 제목 텍스트 | `--text-node-title` | `#FFFFFF` | full.png 노드 헤더 | 노드 이름 + 인스턴스 번호 |
| 노드 본문 보조 텍스트 | `--text-muted` | `#6F7478` | canvas.png `Time: 2ms` 행 | 실행 시간, 파라미터 요약 |
| 아이콘 강조(단독 실행) | `--accent` | `#0DC7AB` | canvas.png 노드 우상단 실행 아이콘 | 단일 액센트. 실행 버튼, 활성 토글, 포커스 링 |

### 카테고리별 노드 헤더 톤

`mechvision_full.png`(신버전)에서는 카테고리마다 노드 카드 톤이 다르다. `[이미지확인]` `canvas.png`(구버전)에서는 전부 `#262A2F` 단색이다.

| 관측된 색 | 관측된 노드 | 우리 매핑 |
|---|---|---|
| `#514343` (웜 그레이) | Capture Images from Camera | `--cat-data-acquisition` |
| `#414C5B` (블루 그레이) | 3D Target Object Recognition, Adjust Poses V2 | `--cat-processing-generic` |
| `#54576B` (퍼플 그레이) | Output | `--cat-output` |

그 외 카테고리의 헤더 색은 캡처에 등장하지 않는다 → **확인 불가**. 관측된 세 톤의 채도·명도 대역(채도 낮음, 명도 30~35%) 안에서만 확장하고, 임의의 고채도 색을 도입하지 않는다.

### 포트 타입 색

`canvas.png`에서 포트 칩은 2줄(윗줄 `<타입>`, 아랫줄 포트 이름)이고 타입마다 색이 다르다. `[이미지확인]`

| 관측된 색 | 관측된 타입 | 우리 매핑 |
|---|---|---|
| `#379B90` | `<Image>` | `--port-image` |
| `#33999B` | `<Image/Color/Mask []>` (리스트/복합) | `--port-image-list` |
| `#7F4164` | `<NumberList/ScaleParam>`, `<NumberList/Roi>` | `--port-numeric` → 우리는 `--port-table` (수치/표 계열) |
| `#7D4B2E` | `<StringList>` | `--port-text` |

우리 타입 중 위 4계열에 대응하지 않는 것(`TimeSeries`, `Regions`, `Schema`, `Sample`, `Embedding`, `Report`, `Tokens`)의 색은 **확인 불가**. 위 4색과 같은 채도·명도 대역에서 배정하되 값은 미정으로 둔다.

### 배선

| 역할 | 토큰명 | 값 | 출처 | 적용처 |
|---|---|---|---|---|
| 배선(Image 계열) | `--wire-image` | `#3CB0A7` (구간 `#39A398`~`#3CB0A7`) | canvas.png 배선 스캔 | Image 출력에서 나가는 곡선 |
| 배선(Numeric 계열) | `--wire-numeric` | `#693A56` (구간 `#623347`~`#693649`) | canvas.png 배선 스캔 | NumberList 출력에서 나가는 곡선 |
| 배선 강조 | `--wire-selected` | **확인 불가** | 선택된 배선이 캡처에 없음 | 선택·호버된 배선 |

**관측된 규칙: 배선 색 = 소스 포트 타입 색을 약간 어둡게 한 값.** `[이미지확인]` 우리도 그대로 따른다. 타입 색이 배선까지 이어지면, 잘못된 계열의 값이 어디로 흘러가는지 한눈에 보인다.

### 노드 상태 색

| 역할 | 토큰명 | 값 | 비고 |
|---|---|---|---|
| 대기 | `--state-pending` | **확인 불가** | 캡처에 상태 구분이 나타나지 않음 |
| 실행 중 | `--state-running` | **확인 불가** | 동일 |
| 성공 | `--state-success` | **확인 불가** | 캡처의 노드는 모두 완료 상태이며 별도 색 없이 `Time: 2ms` 텍스트로만 표시됨 |
| 캐시 히트 | `--state-cached` | **확인 불가** | Mech-Vision에 대응 개념 없음 |
| 실패 | `--state-failed` | **확인 불가** | 캡처에 실패 노드 없음 |
| 부분 성공 | `--state-partial` | **확인 불가** | Mech-Vision에 대응 개념 없음 |

값은 비워두되 **적용 규칙은 확정한다.** 상태는 (1) 노드 카드 좌측 4px 스트라이프, (2) 카드 헤더 우측 상태 점, (3) Log 패널 행 색의 세 곳에 동시에 반영한다. 색만으로 구분하지 않고 항상 아이콘·텍스트를 동반한다(색각 접근성).

### 텍스트

| 역할 | 토큰명 | 값 | 출처 |
|---|---|---|---|
| 기본 텍스트 | `--text-primary` | `#FFFFFF` | full.png 노드 제목 |
| 보조/비활성 텍스트 | `--text-muted` | `#6F7478` | canvas.png 노드 본문 |
| 패널 라벨 텍스트 | `--text-panel` | **확인 불가** | 캡처 해상도에서 안티앨리어싱으로 분리 측정 불가 |
| 액센트 텍스트 | `--accent` | `#0DC7AB` | 위와 동일 토큰 재사용 |

## 12.4 캔버스 규약

### 수직 흐름

- 입력 포트는 노드 **상단**, 출력 포트는 노드 **하단**. `[이미지확인]`
- 배선은 하단 출력 → 상단 입력 방향으로만 생성된다. 반대 방향 드래그는 드롭 자체가 되지 않는다.
- 이 규칙이 순환을 구조적으로 금지하므로 순환 검사 UI(경고, 하이라이트)는 존재하지 않는다.

### 포트 칩 해부

`canvas.png`에서 관찰된 구조를 그대로 따른다. `[이미지확인]`

```
        ┌──────────────────────────────────────────────┐
        │ Crop By Regions (1)              [▾] [▶]     │   ← 헤더: 이름+번호, 접기, 단독 실행
        ├──────────────────────────────────────────────┤
        │ Time: 12ms                            👁      │   ← 본문: 실행 시간, 파라미터 요약,
        │ padding=0.15, max_n=3                        │      우하단 시각화 토글
        └──────────────────────────────────────────────┘
         ┌─────────────┐ ┌──────────────────┐
         │ <ImageList> │ │ <Regions>        │              ← 출력 포트 칩(2줄, 타입 색)
         │ crops       │ │ regions          │
         └──────┬──────┘ └────────┬─────────┘
                │                 │                        ← 배선: 소스 타입 색
```

입력 포트 칩은 같은 형태로 카드 **위**에 붙는다.

### 레인 자동 정렬

- 위상 순서에 따라 위에서 아래로 레인을 쌓는다. 같은 위상 깊이의 노드는 같은 레인에 좌우로 배치.
- **Input 노드는 항상 최상단 레인**, **Output 노드는 항상 최하단 레인**에 고정된다. 위상 깊이와 무관하게 강제한다.
- 레인 내 좌우 순서는 배선 교차를 줄이는 방향으로 정렬하되(barycenter 등), 사용자가 수동으로 옮긴 좌표는 `layout.json`에 남아 자동 정렬을 덮는다.
- 자동 정렬은 `layout.json`만 바꾸므로 스펙 해시에 영향이 없다.

### 3분류의 시각적 구분

색에만 의존하지 않는다. 형태로 먼저 구분된다.

| 분류 | 카드 형태 | 포트 | 부가 표식 |
|---|---|---|---|
| **Input** | 상단 모서리 둥글게, **상단 변에 포트 없음** | 하단만 | 카드 상단 중앙에 소스 아이콘. 카테고리 톤은 `--cat-data-acquisition` |
| **Processing** | 사각 카드 | 상하 모두 | 기본 형태 |
| **Output** | 하단 모서리 둥글게, **하단 변에 포트 없음** | 상단만 | 카드 하단에 종결 표식. 카테고리 톤은 `--cat-output` |

포트가 있는 변이 곧 분류의 표식이므로, 태그를 읽지 않아도 형태만으로 그래프의 시작·중간·끝을 알 수 있다.

### 타입 거부의 시각화

- 드래그 중, 호환되지 않는 입력 포트는 **비활성 처리**되고 드롭이 불가능하다(연결 후 경고가 아니다).
- 호환되는 포트만 강조된다.
- 드롭 실패 시 문서 2.7의 에러 메시지가 툴팁으로 뜬다: 불일치 필드, 안 잡혔다면 언제 터졌을지, 필요한 Adapter 체인.
- Adapter 제안을 클릭하면 노드가 삽입되지만, **삽입은 사용자의 명시적 행위**로 History에 기록된다.
