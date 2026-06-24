from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OpPattern:
    """既知の演算を torch.fx グラフの call_function 名の多重集合で識別する。

    MVP の素朴なパターンマッチ。標準的な式の形のみを認識し、それ以外は None を返して
    eager 実行にフォールバックする（任意グラフのコンパイラ化は範囲外 — 総評参照）。
    """

    op_type: str
    # 期待する call_function の __name__ → 個数（例: rmsnorm = {mul:3, mean:1, add:1, rsqrt:1}）
    op_counts: dict[str, int]

    def matches(self, counts: Counter[str]) -> bool:
        return counts == Counter(self.op_counts)


_REGISTRY: dict[str, OpPattern] = {}


def register(pattern: OpPattern) -> None:
    _REGISTRY[pattern.op_type] = pattern


def match_counts(counts: Counter[str]) -> str | None:
    for pattern in _REGISTRY.values():
        if pattern.matches(counts):
            return pattern.op_type
    return None


def graph_op_counts(graph: Any) -> Counter[str]:
    """torch.fx Graph から call_function ノードの __name__ 多重集合を作る。"""
    names: list[str] = []
    for node in graph.nodes:
        if node.op == "call_function":
            names.append(getattr(node.target, "__name__", str(node.target)))
    return Counter(names)


def _matcher(op_type: str) -> Callable[[Counter[str]], bool]:
    return _REGISTRY[op_type].matches
