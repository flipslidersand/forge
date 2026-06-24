"""lowering の CPU テスト（torch.fx は CPU で動くため GPU 不要）。"""

import torch

from forge.lowering import identify
from forge.lowering.registry import OpPattern, graph_op_counts, match_counts


def rmsnorm(x, weight, eps=1e-6):
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight


class TestIdentify:
    def test_recognizes_rmsnorm(self) -> None:
        assert identify(rmsnorm) == "rmsnorm"

    def test_unknown_op_returns_none(self) -> None:
        def softmax(x):
            return torch.softmax(x, dim=-1)

        assert identify(softmax) is None

    def test_unrelated_fn_returns_none(self) -> None:
        def add(x, y):
            return x + y

        assert identify(add) is None

    def test_untraceable_returns_none(self) -> None:
        # データ依存の制御フローは symbolic_trace 不能 → None
        def dynamic(x):
            if x.sum() > 0:
                return x * 2
            return x

        assert identify(dynamic) is None


class TestRegistry:
    def test_graph_op_counts_rmsnorm(self) -> None:
        graph = torch.fx.symbolic_trace(rmsnorm).graph
        counts = graph_op_counts(graph)
        assert counts == {"mul": 3, "mean": 1, "add": 1, "rsqrt": 1}

    def test_match_counts_hit(self) -> None:
        from collections import Counter

        assert match_counts(Counter({"mul": 3, "mean": 1, "add": 1, "rsqrt": 1})) == "rmsnorm"

    def test_match_counts_miss(self) -> None:
        from collections import Counter

        assert match_counts(Counter({"softmax": 1})) is None

    def test_pattern_matches_exact_multiset(self) -> None:
        from collections import Counter

        p = OpPattern(op_type="x", op_counts={"mul": 2})
        assert p.matches(Counter({"mul": 2}))
        assert not p.matches(Counter({"mul": 3}))  # 個数が違えば不一致
