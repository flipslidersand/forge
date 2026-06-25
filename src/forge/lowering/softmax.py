from __future__ import annotations

from .registry import OpPattern, register

# torch.softmax(x, dim=-1) を torch.fx で trace すると call_function は {softmax:1}。
SOFTMAX_PATTERN = OpPattern(
    op_type="softmax",
    op_counts={"softmax": 1},
)

register(SOFTMAX_PATTERN)
