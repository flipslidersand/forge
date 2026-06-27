from __future__ import annotations

from dataclasses import dataclass

from .tensor_spec import TensorSpec


@dataclass(frozen=True)
class KernelSpec:
    op_type: str
    input_specs: tuple[TensorSpec, ...]
    output_specs: tuple[TensorSpec, ...]
    constants: dict[str, object]
    graph_hash: str
    constraints: tuple[str, ...]

    def validate(self) -> None:
        from forge.ops import OP_INFO

        if self.op_type not in OP_INFO:
            raise ValueError(
                f"Unsupported op_type: {self.op_type!r}. Must be one of {set(OP_INFO)}"
            )
        if not self.input_specs:
            raise ValueError("input_specs must not be empty")
