from __future__ import annotations

from collections.abc import Callable

from . import rmsnorm as _rmsnorm  # noqa: F401 — パターン登録の副作用
from . import softmax as _softmax  # noqa: F401 — パターン登録の副作用
from .registry import graph_op_counts, match_counts


def identify(fn: Callable[..., object]) -> str | None:
    """純粋関数を torch.fx で trace し、既知の op_type にパターンマッチする。

    trace 不能（動的制御フロー等）や未知のパターンの場合は None を返す。呼び出し側は
    None を eager フォールバックとして扱う。
    """
    try:
        import torch.fx

        graph = torch.fx.symbolic_trace(fn).graph
    except Exception:  # noqa: BLE001 — trace 失敗は単に「最適化不可」
        return None
    return match_counts(graph_op_counts(graph))


__all__ = ["identify"]
