# VLM Trainer

Mech-Vision의 규약을 모방한 노드 그래프 기반 파인튜닝 플랫폼. 설계 문서는 [`docs/design/`](docs/design/README.md).

현재 상태: **Phase 0–1 완료** (타입 시스템 · 레지스트리 · 스펙 · 컴파일러 G1/G2 · decompile 왕복 · CLI).
Phase 2(노드 카탈로그 · 실행 엔진 · 합성 더미 데이터 dry-run)부터가 다음 작업이다.

## 실행

```bash
python -m pytest tests -q

# 예제 스펙 컴파일 (테스트용 노드 모듈을 함께 로드)
set PYTHONPATH=%CD%;%CD%\tests
python -m vlm_trainer.cli.main --nodes fixture_nodes compile tests/data/solution/projects/01_demo/project.yaml -v

# 노드 라이브러리 / 노드 상세
python -m vlm_trainer.cli.main --nodes fixture_nodes nodes
python -m vlm_trainer.cli.main --nodes fixture_nodes show test.crop

# 컴파일된 그래프를 스펙으로 되돌리기 (Procedure 되접기 / 펼치기)
python -m vlm_trainer.cli.main --nodes fixture_nodes decompile tests/data/solution/projects/01_demo/project.yaml
```

## 구성

| 경로 | 내용 |
|---|---|
| `vlm_trainer/core/` | 포트 타입, 단일화, 레지스트리, 그래프, 컴파일러, 게이트 에러 |
| `vlm_trainer/spec/` | YAML 로더, canonical form, decompile |
| `vlm_trainer/cli/` | `vlmt` 커맨드 |
| `tests/` | Phase 0–1 완료 조건 51개 + 예제 Solution |
| `docs/design/` | 설계 문서 13편 |

## 설계에서 구현으로 오며 바뀐 것

| 설계 문서 | 구현 | 이유 |
|---|---|---|
| blake3 해시 | `blake2b`(16바이트) | 표준 라이브러리만으로 돌리기 위해. 해시는 engine ABI에 포함되므로 나중에 바꾸면 캐시가 전부 무효화된다 |
| `ImageList` / `TextList` 를 별도 base kind로 | `list_of` 하나로만 표현하고 표시할 때만 `ImageList`로 렌더 | 같은 것을 두 방식으로 표현하면 반드시 어긋난다 |
| shape의 심볼 변수 | 심볼 변수 + **`DYN`(가변 차원)** 을 구분 | 원본 이미지 크기는 "미해결"이 아니라 "샘플마다 다름"으로 확정된 값이다. `DYN`은 compile을 통과하고, 고정 크기를 요구하는 포트에서만 거부되며 리사이즈를 요구한다 |
| 순환은 캔버스가 구조적으로 막으므로 검사 없음 | 손으로 쓴 YAML에 한해 위상 정렬 후 남은 노드를 구조 오류로 보고 | CLI 스펙은 캔버스의 보호를 받지 않는다 |
| 실행 환경 RTX 4090 24GB | 이 PC는 **RTX 3060 12GB** | 예산 게이트 기준값이 다르다. Phase 3에서 프로파일로 분리한다 |
