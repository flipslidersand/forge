from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from forge.ir.kernel_spec import KernelSpec
from forge.search.params import SearchParams

from .dtypes import acc_dtype_to_tl, torch_dtype_str_to_tl

_TEMPLATE_DIR = Path(__file__).parent / "templates"


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )


# op_type -> テンプレートファイル名。Phase 4 でバリアント別に分岐する。
_TEMPLATES = {
    "rmsnorm": "rmsnorm.py.jinja",
}


def generate(spec: KernelSpec, params: SearchParams) -> str:
    """KernelSpec + SearchParams から完全な standalone Triton モジュールを生成する。

    生成コードは ``kernel_fn(x, weight, eps)`` を公開し、subprocess worker が
    一時 .py ファイルとして実行する（@triton.jit はソースをファイルから読むため
    インライン exec では動かない — Issue #3 参照）。
    """
    if spec.op_type not in _TEMPLATES:
        raise ValueError(f"No codegen template for op_type={spec.op_type!r}")

    out_dtype_str = spec.output_specs[0].dtype_str()
    template = _env().get_template(_TEMPLATES[spec.op_type])
    return template.render(
        variant=params.variant,
        block_size=params.block_size,
        num_warps=params.num_warps,
        num_stages=params.num_stages,
        acc_dtype=params.acc_dtype,
        acc_tl=acc_dtype_to_tl(params.acc_dtype),
        out_tl=torch_dtype_str_to_tl(out_dtype_str),
    )


def generate_rmsnorm(spec: KernelSpec, params: SearchParams) -> str:
    """rmsnorm 専用のショートカット（テストや examples 用）。"""
    if spec.op_type != "rmsnorm":
        raise ValueError(f"Expected rmsnorm spec, got {spec.op_type!r}")
    return generate(spec, params)
