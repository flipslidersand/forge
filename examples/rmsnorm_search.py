"""
Phase 3 完成後に動作するデモスクリプト。
現時点では構造確認用のプレースホルダー。
"""

import torch

from forge.cache.key import CacheKey
from forge.cache.repository import KernelRepository
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec


def main() -> None:
    spec = KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec(shape=(2048, 4096), dtype=torch.float16, is_contiguous=True),
            TensorSpec(shape=(4096,), dtype=torch.float16, is_contiguous=True),
        ),
        output_specs=(TensorSpec(shape=(2048, 4096), dtype=torch.float16, is_contiguous=True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_v1",
        constraints=(),
    )

    key = CacheKey.from_spec_and_env(spec)
    print(f"CacheKey digest: {key.digest()}")
    print(f"  compute_capability: {key.compute_capability}")
    print(f"  torch_version:      {key.torch_version}")
    print(f"  shapes:             {key.shapes}")

    repo = KernelRepository()
    cached = repo.get(key)
    if cached:
        print(f"\nCache HIT — best params: {cached.params}")
    else:
        print("\nCache MISS — run search (Phase 3 未実装)")
    repo.close()


if __name__ == "__main__":
    main()
