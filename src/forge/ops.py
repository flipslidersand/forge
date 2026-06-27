from __future__ import annotations

from dataclasses import dataclass

# 演算ごとのメタデータ。SearchSpace（block 制約）と validation（入力構成）が参照する。
#   reduction:   行ごとに last-dim を縮約。BLOCK は N 以上（single/multi_row）。出力は入力同形。
#   elementwise: 要素ごと。flat に numel をタイル分割するため BLOCK は N に縛られない。


@dataclass(frozen=True)
class OpInfo:
    kind: str  # "reduction" | "elementwise"
    n_tensor_inputs: int  # kernel_fn に渡す tensor 入力数


OP_INFO: dict[str, OpInfo] = {
    "rmsnorm": OpInfo(kind="reduction", n_tensor_inputs=2),  # x, weight
    "softmax": OpInfo(kind="reduction", n_tensor_inputs=1),  # x
    "layernorm": OpInfo(kind="reduction", n_tensor_inputs=3),  # x, weight, bias
    "gelu": OpInfo(kind="elementwise", n_tensor_inputs=1),  # x
}


def get_op_info(op_type: str) -> OpInfo:
    if op_type not in OP_INFO:
        raise ValueError(f"Unknown op_type: {op_type!r}")
    return OP_INFO[op_type]


def is_elementwise(op_type: str) -> bool:
    return get_op_info(op_type).kind == "elementwise"
