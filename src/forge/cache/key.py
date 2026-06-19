from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from forge.ir.kernel_spec import KernelSpec

__version__ = "0.1.0"


@dataclass(frozen=True)
class CacheKey:
    graph_hash: str
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    constants_hash: str
    compute_capability: str
    torch_version: str
    triton_version: str
    cuda_version: str
    library_version: str

    @classmethod
    def from_spec_and_env(cls, spec: KernelSpec) -> CacheKey:
        import torch

        try:
            import triton

            triton_ver = triton.__version__
        except ImportError:
            triton_ver = "none"

        cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)

        from forge.ir.hashing import hash_constants

        return cls(
            graph_hash=spec.graph_hash,
            shapes=tuple(s.shape for s in spec.input_specs),
            dtypes=tuple(s.dtype_str() for s in spec.input_specs),
            constants_hash=hash_constants(spec.constants),
            compute_capability=f"{cc[0]}.{cc[1]}",
            torch_version=torch.__version__,
            triton_version=triton_ver,
            cuda_version=torch.version.cuda or "none",
            library_version=__version__,
        )

    def digest(self) -> str:
        data = json.dumps(
            {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()},
            sort_keys=True,
        )
        return hashlib.sha256(data.encode()).hexdigest()
