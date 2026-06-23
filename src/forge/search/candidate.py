from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from forge.ir.kernel_spec import KernelSpec

from .params import SearchParams


@dataclass(frozen=True)
class HistoryEntry:
    """探索済み候補の結果。LLM へのフィードバックに使う。"""

    params: SearchParams
    correct: bool
    median_us: float | None

    def summary(self) -> str:
        if not self.correct:
            outcome = "INCORRECT"
        elif self.median_us is not None:
            outcome = f"{self.median_us:.1f}us"
        else:
            outcome = "ok"
        return (
            f"variant={self.params.variant} block={self.params.block_size} "
            f"warps={self.params.num_warps} stages={self.params.num_stages} "
            f"acc={self.params.acc_dtype} rows={self.params.rows_per_program} -> {outcome}"
        )


@runtime_checkable
class CandidateGenerator(Protocol):
    """探索候補を生成する戦略の共通インターフェース。

    GridSearch / RandomSearch / LLMGenerator が実装する。history は過去の
    実験結果で、LLM 等のフィードバック型生成器が利用する（grid/random は無視）。
    """

    def generate(
        self,
        spec: KernelSpec,
        compute_capability: str,
        budget: int | None = None,
        history: list[HistoryEntry] | None = None,
    ) -> list[SearchParams]: ...
