from __future__ import annotations

from .registry import OpPattern, register

# x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight
# を torch.fx で trace すると call_function は {mul:3, mean:1, add:1, rsqrt:1} になる。
RMSNORM_PATTERN = OpPattern(
    op_type="rmsnorm",
    op_counts={"mul": 3, "mean": 1, "add": 1, "rsqrt": 1},
)

register(RMSNORM_PATTERN)
