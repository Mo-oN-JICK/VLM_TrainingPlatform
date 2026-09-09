import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm_trainer.core import registry  # noqa: E402

registry.load_builtin_nodes()  # 내장 노드 카탈로그

import fixture_nodes  # noqa: F401,E402  (임포트 자체가 테스트용 노드를 등록한다)
