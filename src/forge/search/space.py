from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from forge.ir.kernel_spec import KernelSpec

from .params import SearchParams

# Volta (cc 7.0) 未満では num_stages による非同期パイプラインが効かない。
_MIN_CC_FOR_PIPELINING = 70


def _cc_to_int(compute_capability: str) -> int:
    """'8.9' -> 89, '6.1' -> 61。"""
    major, _, minor = compute_capability.partition(".")
    return int(major) * 10 + int(minor or 0)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


@dataclass
class SearchSpace:
    """探索する各軸の候補値。spec・GPU によって不可能な組み合わせは enumerate で除外する。

    variant ごとに block_size の意味と制約が異なる:
      single_row / multi_row : block_size は N 以上（行全体を 1 タイルで処理）
      two_pass               : block_size はタイルサイズで N 未満も可
    """

    block_sizes: list[int] = field(default_factory=lambda: [512, 1024, 2048, 4096, 8192])
    num_warps: list[int] = field(default_factory=lambda: [4, 8, 16])
    num_stages: list[int] = field(default_factory=lambda: [1, 2, 3])
    acc_dtypes: list[str] = field(default_factory=lambda: ["fp32", "fp16"])
    variants: list[str] = field(default_factory=lambda: ["single_row", "multi_row", "two_pass"])
    rows_per_program: list[int] = field(default_factory=lambda: [2, 4])

    def _blocks_for_variant(self, variant: str, n: int) -> list[int]:
        if variant == "two_pass":
            tiles = sorted({b for b in self.block_sizes if b <= n})
            return tiles or [min(self.block_sizes)]
        # single_row / multi_row: 行全体を 1 タイルに収める
        blocks = sorted({b for b in self.block_sizes if b >= n})
        return blocks or [_next_pow2(n)]

    def _rows_for_variant(self, variant: str) -> list[int]:
        return self.rows_per_program if variant == "multi_row" else [1]

    def enumerate(self, spec: KernelSpec, compute_capability: str) -> Iterator[SearchParams]:
        """spec と GPU に対して有効な SearchParams を列挙する。

        Pascal 等 cc<7.0 では num_stages を [1] に制限する。
        """
        n = spec.input_specs[0].shape[-1]
        cc = _cc_to_int(compute_capability)
        stages = self.num_stages if cc >= _MIN_CC_FOR_PIPELINING else [1]

        seen: set[tuple] = set()
        for variant in self.variants:
            for block in self._blocks_for_variant(variant, n):
                for rows in self._rows_for_variant(variant):
                    for warps in self.num_warps:
                        for stage in stages:
                            for acc in self.acc_dtypes:
                                key = (variant, block, rows, warps, stage, acc)
                                if key in seen:
                                    continue
                                seen.add(key)
                                yield SearchParams(
                                    block_size=block,
                                    num_warps=warps,
                                    num_stages=stage,
                                    acc_dtype=acc,
                                    variant=variant,
                                    rows_per_program=rows,
                                )
