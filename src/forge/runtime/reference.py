from __future__ import annotations

from collections.abc import Callable

import torch

# op_type -> 正解（ground truth）となる純粋 PyTorch 実装。
# 高速化候補はこの出力と比較して正確性を検証し、ベンチマークの baseline にもなる。
# fp32 で縮約してから入力 dtype に戻す（数値的に最も正確な参照）。


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    rms = torch.rsqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + eps)
    return (x32 * rms * weight.float()).to(x.dtype)


REFERENCE_IMPLS: dict[str, Callable[..., torch.Tensor]] = {
    "rmsnorm": rmsnorm_reference,
}


def get_reference(op_type: str) -> Callable[..., torch.Tensor]:
    if op_type not in REFERENCE_IMPLS:
        raise ValueError(f"No reference implementation for op_type={op_type!r}")
    return REFERENCE_IMPLS[op_type]
