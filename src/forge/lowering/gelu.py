from __future__ import annotations

from .registry import OpPattern, register

# F.gelu(x) を torch.fx で trace すると {gelu:1}（tanh 近似も同じ target 名）。
# codegen は exact(erf) を生成するため、tanh 近似の関数は許容誤差を超えて
# eager フォールバックする可能性がある。
GELU_PATTERN = OpPattern(
    op_type="gelu",
    op_counts={"gelu": 1},
)

register(GELU_PATTERN)
