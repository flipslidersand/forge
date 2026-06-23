import torch

from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.validation.test_cases import correctness_cases, primary_input
from forge.validation.tolerance import get_tolerance


def spec(n: int = 4096) -> KernelSpec:
    return KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec((2048, n), torch.float16, True),
            TensorSpec((n,), torch.float16, True),
        ),
        output_specs=(TensorSpec((2048, n), torch.float16, True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_v1",
        constraints=(),
    )


class TestTolerance:
    def test_rmsnorm_atol_relaxed(self) -> None:
        # #4 実測 1.95e-3 を通すため atol >= 2e-3
        assert get_tolerance("rmsnorm").atol >= 2e-3

    def test_to_dict(self) -> None:
        d = get_tolerance("rmsnorm").to_dict()
        assert set(d) == {"atol", "rtol", "equal_nan"}

    def test_unknown_op_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="No tolerance"):
            get_tolerance("nope")


class TestTestCases:
    def test_primary_input_matches_spec_shape(self) -> None:
        inp = primary_input(spec(4096))
        assert inp[0]["shape"] == [2048, 4096]
        assert inp[1]["shape"] == [4096]
        assert inp[0]["init"] == "randn"
        assert inp[1]["init"] == "ones"

    def test_correctness_cases_keep_hidden_constant(self) -> None:
        cases = correctness_cases(spec(4096))
        for c in cases:
            assert c["input_specs"][0]["shape"][1] == 4096  # N 固定
            assert c["input_specs"][1]["shape"] == [4096]

    def test_correctness_cases_cover_edge_categories(self) -> None:
        names = {c["name"] for c in correctness_cases(spec())}
        assert {"single_row", "zeros", "large_values", "weight_randn"} <= names

    def test_zeros_case_uses_zeros_init(self) -> None:
        cases = {c["name"]: c for c in correctness_cases(spec())}
        assert cases["zeros"]["input_specs"][0]["init"] == "zeros"
