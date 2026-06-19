from __future__ import annotations

import hashlib
import json
from typing import Any

from .kernel_spec import KernelSpec


def hash_constants(constants: dict[str, Any]) -> str:
    serialized = json.dumps(constants, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def hash_kernel_spec(spec: KernelSpec) -> str:
    data = {
        "op_type": spec.op_type,
        "inputs": [
            {"shape": s.shape, "dtype": s.dtype_str(), "contiguous": s.is_contiguous}
            for s in spec.input_specs
        ],
        "outputs": [
            {"shape": s.shape, "dtype": s.dtype_str(), "contiguous": s.is_contiguous}
            for s in spec.output_specs
        ],
        "constants": hash_constants(spec.constants),
        "constraints": list(spec.constraints),
    }
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:24]
