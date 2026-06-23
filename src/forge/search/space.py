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
    """探索する各軸の候補値。GPU・spec によって不可能な組み合わせは enumerate で除外する。"""

    block_sizes: list[int] = field(default_factory=lambda: [1024, 2048, 4096, 8192])
    num_warps: list[int] = field(default_factory=lambda: [4, 8, 16])
    num_stages: list[int] = field(default_factory=lambda: [1, 2, 3])
    acc_dtypes: list[str] = field(default_factory=lambda: ["fp32", "fp16"])
    variants: list[str] = field(default_factory=lambda: ["single_row"])

    def enumerate(self, spec: KernelSpec, compute_capability: str) -> Iterator[SearchParams]:
        """spec と GPU に対して有効な SearchParams を列挙する。

        - single_row variant では BLOCK_SIZE >= hidden_size(N) が必須（tl.arange 制約）。
          N 未満の block_size は除外し、どれも満たさなければ次の 2 のべき乗を補う。
        - Pascal 等 cc<7.0 では num_stages を [1] に制限する。
        """
        n = spec.input_specs[0].shape[-1]
        cc = _cc_to_int(compute_capability)

        valid_blocks = sorted({b for b in self.block_sizes if b >= n})
        if not valid_blocks:
            valid_blocks = [_next_pow2(n)]

        stages = self.num_stages if cc >= _MIN_CC_FOR_PIPELINING else [1]

        seen: set[tuple] = set()
        for variant in self.variants:
            for block in valid_blocks:
                for warps in self.num_warps:
                    for stage in stages:
                        for acc in self.acc_dtypes:
                            key = (variant, block, warps, stage, acc)
                            if key in seen:
                                continue
                            seen.add(key)
                            yield SearchParams(
                                block_size=block,
                                num_warps=warps,
                                num_stages=stage,
                                acc_dtype=acc,
                                variant=variant,
                            )
