from __future__ import annotations

from typing import Any

from forge.ir.kernel_spec import KernelSpec


def _rmsnorm_inputs(
    m: int,
    n: int,
    dtype: str,
    x_init: str = "randn",
    x_scale: float = 1.0,
    w_init: str = "ones",
    seed: int = 0,
) -> list[dict[str, Any]]:
    return [
        {"shape": [m, n], "dtype": dtype, "init": x_init, "scale": x_scale, "seed": seed},
        {"shape": [n], "dtype": dtype, "init": w_init, "scale": 1.0, "seed": seed + 1},
    ]


def primary_input(spec: KernelSpec) -> list[dict[str, Any]]:
    """ベンチマークに使う代表入力（spec の shape そのまま、x=randn / weight=ones）。"""
    x = spec.input_specs[0]
    m, n = x.shape
    return _rmsnorm_inputs(m, n, x.dtype_str())


def correctness_cases(spec: KernelSpec) -> list[dict[str, Any]]:
    """正確性検証のエッジケース群。

    block_size は hidden_size(N) に対して調律されるため N は固定し、行数 M と
    数値分布を変える。各ケースは worker が tensor を組み立てる input_specs を持つ。
    """
    if spec.op_type != "rmsnorm":
        raise ValueError(f"correctness_cases only supports rmsnorm, got {spec.op_type!r}")

    x = spec.input_specs[0]
    _, n = x.shape
    dt = x.dtype_str()

    return [
        {"name": "basic", "input_specs": _rmsnorm_inputs(2048, n, dt)},
        {"name": "single_row", "input_specs": _rmsnorm_inputs(1, n, dt)},
        {"name": "odd_rows", "input_specs": _rmsnorm_inputs(7, n, dt, seed=3)},
        {"name": "weight_randn", "input_specs": _rmsnorm_inputs(64, n, dt, w_init="randn", seed=5)},
        {"name": "large_values", "input_specs": _rmsnorm_inputs(64, n, dt, x_scale=100.0, seed=7)},
        {"name": "small_values", "input_specs": _rmsnorm_inputs(64, n, dt, x_scale=1e-3, seed=9)},
        {"name": "zeros", "input_specs": _rmsnorm_inputs(8, n, dt, x_init="zeros")},
    ]
