import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fixture_nodes  # noqa: F401,E402  (임포트 자체가 노드를 등록한다)
