from __future__ import annotations

from dataclasses import asdict, dataclass

# サポートする実装バリアント（コード構造）。
#   single_row  : 1 program = 1 行、BLOCK_SIZE >= N（reduction op）
#   multi_row   : 1 program = ROWS 行、BLOCK_SIZE >= N（小さい N で occupancy 改善）
#   two_pass    : BLOCK_SIZE をタイルとして N をループ。BLOCK_SIZE < N を許容（大きい N 向け）
#   elementwise : flat に numel をタイル分割。BLOCK_SIZE は N に縛られない（elementwise op）
SUPPORTED_VARIANTS = ("single_row", "multi_row", "two_pass", "elementwise")
SUPPORTED_ACC_DTYPES = ("fp32", "fp16")


@dataclass(frozen=True)
class SearchParams:
    """1 つの探索候補を表すパラメータ。

    block_size は tl.arange の制約上 2 のべき乗。single_row/multi_row では N 以上、
    two_pass ではタイルサイズとして N 未満も可（検証は SearchSpace 側）。

    rows_per_program は multi_row のみ意味を持つ（他は 1）。

    num_stages は Volta+ の非同期パイプライン向け。Pascal (cc 6.x) では効果がない
    か、値によってはコンパイルに失敗しうる。worker 側でエラーを捕捉する。
    """

    block_size: int
    num_warps: int
    num_stages: int
    acc_dtype: str = "fp32"
    variant: str = "single_row"
    rows_per_program: int = 1

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
        if self.rows_per_program < 1:
            raise ValueError(f"rows_per_program must be >= 1, got {self.rows_per_program}")
        if self.variant != "multi_row" and self.rows_per_program != 1:
            raise ValueError("rows_per_program > 1 is only valid for variant='multi_row'")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> SearchParams:
        return cls(**d)  # type: ignore[arg-type]
