from __future__ import annotations

from .registry import OpPattern, register

# F.layer_norm(x, (N,), weight, bias, eps) を torch.fx で trace すると {layer_norm:1}。
LAYERNORM_PATTERN = OpPattern(
    op_type="layernorm",
    op_counts={"layer_norm": 1},
)

register(LAYERNORM_PATTERN)
