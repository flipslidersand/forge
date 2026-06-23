from __future__ import annotations

from collections.abc import Callable

import torch

# op_type -> 正解（ground truth）となる純粋 PyTorch 実装。
# 候補はこの出力と比較して正確性を検証する。fp32 で縮約してから入力 dtype に戻す
# （数値的に最も正確な参照）。


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    rms = torch.rsqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + eps)
    return (x32 * rms * weight.float()).to(x.dtype)


REFERENCE_IMPLS: dict[str, Callable[..., torch.Tensor]] = {
    "rmsnorm": rmsnorm_reference,
}


def get_reference(op_type: str) -> Callable[..., torch.Tensor]:
    """正確性検証用の ground truth 実装を返す。"""
    if op_type not in REFERENCE_IMPLS:
        raise ValueError(f"No reference implementation for op_type={op_type!r}")
    return REFERENCE_IMPLS[op_type]


# 速度比較用の「公平な」baseline。素朴な fp32-upcast 実装ではなく、PyTorch の
# 最適化済み組み込み演算と比べる（Phase 2 申し送り）。組み込みが無い環境では
# reference にフォールバックする。


def _rmsnorm_baseline(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if hasattr(torch.nn.functional, "rms_norm"):
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps=eps)
    return rmsnorm_reference(x, weight, eps)


BASELINE_IMPLS: dict[str, Callable[..., torch.Tensor]] = {
    "rmsnorm": _rmsnorm_baseline,
}


def get_baseline(op_type: str) -> Callable[..., torch.Tensor]:
    """速度比較の対抗馬（PyTorch 最適化実装）を返す。"""
    if op_type not in BASELINE_IMPLS:
        raise ValueError(f"No baseline implementation for op_type={op_type!r}")
    return BASELINE_IMPLS[op_type]


def baseline_name(op_type: str) -> str:
    if op_type == "rmsnorm" and hasattr(torch.nn.functional, "rms_norm"):
        return "F.rms_norm"
    return "fp32_reference"
