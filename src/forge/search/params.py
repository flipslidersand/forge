from __future__ import annotations

from dataclasses import asdict, dataclass

# Phase 2 がサポートするバリアント（実装方式）。Phase 4 で拡張する。
SUPPORTED_VARIANTS = ("single_row",)
SUPPORTED_ACC_DTYPES = ("fp32", "fp16")


@dataclass(frozen=True)
class SearchParams:
    """1 つの探索候補を表すパラメータ。

    block_size は tl.arange の制約上 2 のべき乗かつ hidden_size 以上である必要がある
    （検証は SearchSpace 側 / Phase 3 で行う）。

    num_stages は Volta+ の非同期パイプライン向け。Pascal (cc 6.x) では効果がない
    か、値によってはコンパイルに失敗しうる。worker 側でエラーを捕捉する。
    """

    block_size: int
    num_warps: int
    num_stages: int
    acc_dtype: str = "fp32"
    variant: str = "single_row"

    def __post_init__(self) -> None:
        if self.variant not in SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported variant: {self.variant!r}. Must be one of {SUPPORTED_VARIANTS}"
            )
        if self.acc_dtype not in SUPPORTED_ACC_DTYPES:
            raise ValueError(
                f"Unsupported acc_dtype: {self.acc_dtype!r}. Must be one of {SUPPORTED_ACC_DTYPES}"
            )
        if self.block_size <= 0 or (self.block_size & (self.block_size - 1)) != 0:
            raise ValueError(f"block_size must be a positive power of 2, got {self.block_size}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> SearchParams:
        return cls(**d)  # type: ignore[arg-type]
