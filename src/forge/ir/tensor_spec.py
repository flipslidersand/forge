from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype
    is_contiguous: bool

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> TensorSpec:
        return cls(
            shape=tuple(t.shape),
            dtype=t.dtype,
            is_contiguous=t.is_contiguous(),
        )

    def dtype_str(self) -> str:
        _map = {
            torch.float32: "float32",
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float64: "float64",
        }
        return _map.get(self.dtype, repr(self.dtype))
