from __future__ import annotations

# torch dtype の文字列表現 → Triton language の型名。
# TensorSpec.dtype_str() の出力に対応させる。
_TORCH_TO_TL = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float64": "float64",
}

_ACC_TO_TL = {
    "fp32": "float32",
    "fp16": "float16",
}


def torch_dtype_str_to_tl(dtype_str: str) -> str:
    if dtype_str not in _TORCH_TO_TL:
        raise ValueError(f"Unsupported dtype for codegen: {dtype_str!r}")
    return _TORCH_TO_TL[dtype_str]


def acc_dtype_to_tl(acc_dtype: str) -> str:
    if acc_dtype not in _ACC_TO_TL:
        raise ValueError(f"Unsupported acc_dtype: {acc_dtype!r}")
    return _ACC_TO_TL[acc_dtype]
