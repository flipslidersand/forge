import tempfile
from pathlib import Path

import torch

from forge.cache.key import CacheKey
from forge.cache.repository import CachedKernel, KernelRepository
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec


def make_spec() -> KernelSpec:
    return KernelSpec(
        op_type="rmsnorm",
        input_specs=(TensorSpec(shape=(2048, 4096), dtype=torch.float16, is_contiguous=True),),
        output_specs=(TensorSpec(shape=(2048, 4096), dtype=torch.float16, is_contiguous=True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_v1",
        constraints=(),
    )


class TestCacheKey:
    def test_digest_deterministic(self) -> None:
        key = CacheKey(
            graph_hash="abc",
            shapes=((2048, 4096),),
            dtypes=("float16",),
            constants_hash="deadbeef",
            compute_capability="8.9",
            torch_version="2.3.0",
            triton_version="3.0.0",
            cuda_version="12.1",
            library_version="0.1.0",
        )
        assert key.digest() == key.digest()

    def test_digest_differs_on_shape(self) -> None:
        def make(shape: tuple[int, int]) -> CacheKey:
            return CacheKey(
                graph_hash="abc",
                shapes=(shape,),
                dtypes=("float16",),
                constants_hash="deadbeef",
                compute_capability="8.9",
                torch_version="2.3.0",
                triton_version="3.0.0",
                cuda_version="12.1",
                library_version="0.1.0",
            )

        assert make((2048, 4096)).digest() != make((1024, 4096)).digest()

    def test_digest_differs_on_cuda_version(self) -> None:
        base = dict(
            graph_hash="abc",
            shapes=((2048, 4096),),
            dtypes=("float16",),
            constants_hash="deadbeef",
            compute_capability="8.9",
            torch_version="2.3.0",
            triton_version="3.0.0",
            library_version="0.1.0",
        )
        k1 = CacheKey(**base, cuda_version="12.1")
        k2 = CacheKey(**base, cuda_version="12.4")
        assert k1.digest() != k2.digest()


class TestKernelRepository:
    def _make_key(self) -> CacheKey:
        return CacheKey(
            graph_hash="abc",
            shapes=((2048, 4096),),
            dtypes=("float16",),
            constants_hash="deadbeef",
            compute_capability="8.9",
            torch_version="2.3.0",
            triton_version="3.0.0",
            cuda_version="12.1",
            library_version="0.1.0",
        )

    def _make_kernel(self, key: CacheKey) -> CachedKernel:
        from datetime import datetime, timezone

        return CachedKernel(
            cache_key=key,
            params={"block_size": 1024, "num_warps": 8},
            kernel_code="def rmsnorm(): pass",
            benchmark_json={"median_us": 42.0, "p20_us": 40.0, "p80_us": 44.0},
            created_at=datetime.now(timezone.utc),
        )

    def test_get_miss(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "cache.db")
            assert repo.get(self._make_key()) is None
            repo.close()

    def test_put_and_get(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "cache.db")
            key = self._make_key()
            kernel = self._make_kernel(key)
            repo.put(key, kernel)

            result = repo.get(key)
            assert result is not None
            assert result.kernel_code == "def rmsnorm(): pass"
            assert result.params["block_size"] == 1024
            repo.close()

    def test_put_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "cache.db")
            key = self._make_key()

            k1 = self._make_kernel(key)
            k1.kernel_code = "def v1(): pass"
            repo.put(key, k1)

            k2 = self._make_kernel(key)
            k2.kernel_code = "def v2(): pass"
            repo.put(key, k2)

            result = repo.get(key)
            assert result is not None
            assert result.kernel_code == "def v2(): pass"
            repo.close()

    def test_persists_across_instances(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.db"
            key = self._make_key()

            repo1 = KernelRepository(path)
            repo1.put(key, self._make_kernel(key))
            repo1.close()

            repo2 = KernelRepository(path)
            result = repo2.get(key)
            assert result is not None
            repo2.close()
