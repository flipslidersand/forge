import torch

from forge.codegen.triton_codegen import generate_rmsnorm
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.search.params import SearchParams


def make_spec(out_dtype: torch.dtype = torch.float16) -> KernelSpec:
    return KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec(shape=(2048, 4096), dtype=torch.float16, is_contiguous=True),
            TensorSpec(shape=(4096,), dtype=torch.float16, is_contiguous=True),
        ),
        output_specs=(TensorSpec(shape=(2048, 4096), dtype=out_dtype, is_contiguous=True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_v1",
        constraints=(),
    )


def default_params(**kw: object) -> SearchParams:
    base = dict(block_size=4096, num_warps=8, num_stages=1, acc_dtype="fp32", variant="single_row")
    base.update(kw)
    return SearchParams(**base)  # type: ignore[arg-type]


class TestCodegen:
    def test_generates_valid_python(self) -> None:
        code = generate_rmsnorm(make_spec(), default_params())
        # 構文として valid であること（GPU なしで検証可能）
        compile(code, "<generated>", "exec")

    def test_contains_kernel_fn(self) -> None:
        code = generate_rmsnorm(make_spec(), default_params())
        assert "def kernel_fn(" in code
        assert "@triton.jit" in code

    def test_block_size_embedded(self) -> None:
        code = generate_rmsnorm(make_spec(), default_params(block_size=2048))
        assert "BLOCK_SIZE=2048" in code

    def test_num_warps_embedded(self) -> None:
        code = generate_rmsnorm(make_spec(), default_params(num_warps=16))
        assert "num_warps=16" in code

    def test_acc_dtype_fp32(self) -> None:
        code = generate_rmsnorm(make_spec(), default_params(acc_dtype="fp32"))
        assert "tl.float32" in code

    def test_acc_dtype_fp16(self) -> None:
        code = generate_rmsnorm(make_spec(), default_params(acc_dtype="fp16"))
        assert ".to(tl.float16)" in code

    def test_output_dtype_respected(self) -> None:
        code = generate_rmsnorm(make_spec(out_dtype=torch.bfloat16), default_params())
        assert "out.to(tl.bfloat16)" in code

    def test_deterministic(self) -> None:
        a = generate_rmsnorm(make_spec(), default_params())
        b = generate_rmsnorm(make_spec(), default_params())
        assert a == b


class TestSearchParams:
    def test_rejects_non_power_of_2_block_size(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="power of 2"):
            default_params(block_size=3000)

    def test_rejects_unknown_variant(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="variant"):
            default_params(variant="nonexistent")

    def test_roundtrip_dict(self) -> None:
        p = default_params()
        assert SearchParams.from_dict(p.to_dict()) == p
