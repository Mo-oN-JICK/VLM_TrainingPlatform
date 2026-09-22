"""편집기가 프로젝트 옆에 두는 파일 이름들.

api.py 와 믹스인이 같은 값을 봐야 해서 따로 둔다. 값이 두 벌이 되면
이력을 쓰는 곳과 읽는 곳이 어긋난다.
"""

from __future__ import annotations

HISTORY_FILE = "edit_history.jsonl"
LAYOUT_FILE = "layout.yaml"   # 캔버스 자리. 스펙이 아니므로 spec_hash 에 들어가지 않는다
HISTORY_STORE = "edit_history"   # 시점별 스펙 본문. 내용 주소라 같은 상태로 돌아와도 파일이 늘지 않는다
HISTORY_WINDOW = 200             # 열 때 되살릴 시점의 최대 개수
