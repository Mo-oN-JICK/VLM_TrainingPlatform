# CLAUDE.md

## 먼저 읽을 것

**세션을 시작하면 [`STATUS.md`](STATUS.md)를 읽어라.** 무엇이 끝났고 무엇이 다음인지, 이미 확정된 결정과
실측으로 확인된 함정이 거기 있다. 작업을 끝낼 때마다 STATUS.md를 갱신한다.

설계 문서는 [`docs/design/`](docs/design/README.md) 13편. 구현은 이 설계를 따르며, 벗어난 지점은
`README.md`의 표에 이유와 함께 기록한다.

## 이 프로젝트의 성격

Mech-Vision의 규약을 의도적으로 모방한 노드 그래프 기반 VLM 파인튜닝 **학습 도구**.
응용 서비스나 최종 사용자 워크플로로 확장하지 않는다.

## 절대 하지 말 것

- **자동 타입 캐스팅을 넣지 마라.** 리사이즈·정규화·레이아웃·색공간·dtype·좌표계·토크나이즈는 전부 명시적
  `adapt.*` 노드로만 표현된다. `core/unify.py`에 완화 코드가 들어가는 순간 이 플랫폼의 존재 이유가 사라진다.
- **게이트 우회 옵션을 만들지 마라.** 경고 등급, `--allow-unsafe`, strict 해제 스위치 전부 금지. 타입 위반은 이진이다.
- **UI 전용 실행 경로를 만들지 마라.** UI는 스펙을 편집하는 뷰일 뿐이고, CLI로 안 되는 기능은 존재하지 않는다.
- **부작용을 Output 노드 밖에 두지 마라.** Processing은 순수 함수다(파일·네트워크·전역 RNG·시계 접근 금지).
- 새 의존성을 가볍게 추가하지 마라. 코어는 표준 라이브러리 + `yaml`만 쓴다.

## 코드 규약

- 에러 메시지는 한국어로, 항상 네 가지를 담는다: 불일치 필드 / 안 잡혔다면 언제 어디서 터졌을지 / 추정 낭비 /
  구체적 해결 배선. 이 형식이 게이트를 유지시키는 장치다.
- 노드 구현은 반드시 모듈 최상위에 둔다(Windows spawn 워커가 임포트해야 한다).
- 문서·주석·커밋 메시지는 평이한 산문으로 쓴다.

## 자주 쓰는 명령

```bash
python -m pytest tests -q

# CLI (테스트용 노드를 함께 로드할 때는 Windows 경로로 PYTHONPATH 지정)
PYTHONPATH="C:\Users\wnsgu\Work\VLM_Trainer;C:\Users\wnsgu\Work\VLM_Trainer\tests" \
  python -m vlm_trainer.cli.main --nodes fixture_nodes compile tests/data/solution/projects/01_demo/project.yaml -v
```

## 실행 환경

RTX 3060 12GB / Windows / Python 3.14 / torch 미설치. 4090 24GB는 다른 PC에 있다.
단일 GPU와 Windows spawn은 부차 조건이 아니라 설계의 1급 제약이다.
