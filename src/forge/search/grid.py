from __future__ import annotations

from forge.ir.kernel_spec import KernelSpec

from .params import SearchParams
from .space import SearchSpace


class GridSearch:
    """探索空間を全列挙するベースライン探索器。

    Phase 5 で導入する CandidateGenerator Protocol の最も単純な実装に相当する。
    LLM 等の探索効果は、この全列挙との比較で測る。
    """

    def __init__(self, space: SearchSpace | None = None) -> None:
        self.space = space or SearchSpace()

    def generate(
        self,
        spec: KernelSpec,
        compute_capability: str,
        budget: int | None = None,
    ) -> list[SearchParams]:
        candidates = list(self.space.enumerate(spec, compute_capability))
        if budget is not None:
            candidates = candidates[:budget]
        return candidates
