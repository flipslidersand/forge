from dataclasses import FrozenInstanceError

import pytest
import torch

from forge.ir.hashing import hash_constants, hash_kernel_spec
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec


def make_rmsnorm_spec(shape: tuple[int, int] = (2048, 4096)) -> KernelSpec:
    return KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec(shape=shape, dtype=torch.float16, is_contiguous=True),
            TensorSpec(shape=(shape[1],), dtype=torch.float16, is_contiguous=True),
        ),
        output_specs=(TensorSpec(shape=shape, dtype=torch.float16, is_contiguous=True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_v1",
        constraints=(),
    )


class TestTensorSpec:
    def test_from_tensor(self) -> None:
        t = torch.zeros(2048, 4096, dtype=torch.float16)
        spec = TensorSpec.from_tensor(t)
        assert spec.shape == (2048, 4096)
        assert spec.dtype == torch.float16
        assert spec.is_contiguous is True

    def test_from_tensor_noncontiguous(self) -> None:
        t = torch.zeros(2048, 4096 * 2)[:, ::2]
        spec = TensorSpec.from_tensor(t)
        assert spec.is_contiguous is False

    def test_dtype_str(self) -> None:
        spec = TensorSpec(shape=(1,), dtype=torch.float16, is_contiguous=True)
        assert spec.dtype_str() == "float16"

    def test_frozen(self) -> None:
        spec = TensorSpec(shape=(1,), dtype=torch.float32, is_contiguous=True)
        with pytest.raises(FrozenInstanceError):
            spec.shape = (2,)  # type: ignore[misc]


class TestKernelSpec:
    def test_validate_ok(self) -> None:
        spec = make_rmsnorm_spec()
        spec.validate()  # should not raise

    def test_validate_unknown_op(self) -> None:
        spec = make_rmsnorm_spec()
        bad = KernelSpec(
            op_type="unknown_op",
            input_specs=spec.input_specs,
            output_specs=spec.output_specs,
            constants=spec.constants,
            graph_hash="x",
            constraints=(),
        )
        with pytest.raises(ValueError, match="Unsupported op_type"):
            bad.validate()

    def test_validate_empty_inputs(self) -> None:
        with pytest.raises(ValueError, match="input_specs must not be empty"):
            KernelSpec(
                op_type="rmsnorm",
                input_specs=(),
                output_specs=(),
                constants={},
                graph_hash="x",
                constraints=(),
            ).validate()

    def test_frozen(self) -> None:
        spec = make_rmsnorm_spec()
        with pytest.raises(FrozenInstanceError):
            spec.op_type = "softmax"  # type: ignore[misc]


class TestHashing:
    def test_hash_constants_deterministic(self) -> None:
        h1 = hash_constants({"eps": 1e-6, "dim": -1})
        h2 = hash_constants({"dim": -1, "eps": 1e-6})
        assert h1 == h2

    def test_hash_constants_different(self) -> None:
        assert hash_constants({"eps": 1e-6}) != hash_constants({"eps": 1e-5})

    def test_hash_kernel_spec_deterministic(self) -> None:
        spec = make_rmsnorm_spec()
        assert hash_kernel_spec(spec) == hash_kernel_spec(spec)

    def test_hash_kernel_spec_shape_sensitive(self) -> None:
        h1 = hash_kernel_spec(make_rmsnorm_spec((2048, 4096)))
        h2 = hash_kernel_spec(make_rmsnorm_spec((1024, 4096)))
        assert h1 != h2
