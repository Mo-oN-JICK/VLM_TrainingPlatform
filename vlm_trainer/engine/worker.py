"""워커 프로세스 모델.

Windows는 fork가 없고 spawn만 있다. 그래서 워커에 넘기는 것은 직렬화 가능한
스펙 조각과 값뿐이고, 노드 구현은 모듈 최상위에서 임포트 가능해야 한다
(레지스트리가 등록 시점에 검사한다).

외부 모델을 호출하는 노드는 격리 실행한다. 워커가 죽어도 엔진은 살아 있고
그 노드만 failed가 된다.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from typing import Any, Dict, Optional, Tuple

from ..core.node import NodeError, RunCtx

_POOL: Optional[ProcessPoolExecutor] = None


def _child_execute(
    node_ref: str, params: Dict[str, Any], inputs: Dict[str, Any], ctx: RunCtx, modules: Tuple[str, ...]
) -> Dict[str, Any]:
    """자식 프로세스에서 실행된다. 모듈 최상위 함수여야 spawn이 임포트할 수 있다."""
    import importlib

    from ..core import registry

    registry.load_builtin_nodes()
    for m in modules:
        importlib.import_module(m)
    d = registry.resolve(node_ref)
    impl = d.impl()
    return impl.run(ctx, d.build_params(params), **inputs)


def _pool() -> ProcessPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
    return _POOL


def shutdown() -> None:
    global _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=False, cancel_futures=True)
        _POOL = None


def run_isolated(
    node_ref: str,
    params: Dict[str, Any],
    inputs: Dict[str, Any],
    ctx: RunCtx,
    modules: Tuple[str, ...] = (),
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """별도 프로세스에서 노드를 실행한다. 크래시는 NodeError로 바뀐다."""
    try:
        fut = _pool().submit(_child_execute, node_ref, params, inputs, ctx, modules)
        return fut.result(timeout=timeout)
    except (BrokenExecutor, mp.ProcessError) as e:
        shutdown()  # 깨진 풀은 버리고 다음 노드에서 새로 만든다
        raise NodeError(
            ctx.node_id,
            f"워커 프로세스가 죽었다 ({type(e).__name__})",
            sample_key=ctx.sample_key,
            hint="네이티브 크래시이거나 메모리 부족이다. 이 노드만 failed로 격리한다.",
        ) from None
    except NodeError:
        raise
    except Exception as e:  # 자식에서 올라온 일반 예외
        raise NodeError(ctx.node_id, f"{type(e).__name__}: {e}", sample_key=ctx.sample_key) from None
