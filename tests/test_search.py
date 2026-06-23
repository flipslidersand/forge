import torch

from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.search.grid import GridSearch
from forge.search.space import SearchSpace, _cc_to_int, _next_pow2


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


class TestHelpers:
    def test_cc_to_int(self) -> None:
        assert _cc_to_int("8.9") == 89
        assert _cc_to_int("6.1") == 61
        assert _cc_to_int("7.0") == 70

    def test_next_pow2(self) -> None:
        assert _next_pow2(4096) == 4096
        assert _next_pow2(4000) == 4096
        assert _next_pow2(5000) == 8192


class TestSearchSpace:
    def test_block_size_at_least_hidden(self) -> None:
        params = list(SearchSpace().enumerate(spec(4096), "8.9"))
        assert all(p.block_size >= 4096 for p in params)

    def test_no_valid_block_falls_back_to_next_pow2(self) -> None:
        # hidden=6000 → block_sizes [1024..8192] のうち >=6000 は 8192 のみ
        params = list(SearchSpace().enumerate(spec(6000), "8.9"))
        assert all(p.block_size == 8192 for p in params)

    def test_pascal_restricts_num_stages(self) -> None:
        # cc 6.1 では num_stages=1 のみ
        params = list(SearchSpace().enumerate(spec(4096), "6.1"))
        assert {p.num_stages for p in params} == {1}

    def test_volta_allows_pipelining(self) -> None:
        params = list(SearchSpace().enumerate(spec(4096), "8.9"))
        assert {p.num_stages for p in params} == {1, 2, 3}

    def test_no_duplicates(self) -> None:
        params = list(SearchSpace().enumerate(spec(4096), "8.9"))
        keys = [(p.variant, p.block_size, p.num_warps, p.num_stages, p.acc_dtype) for p in params]
        assert len(keys) == len(set(keys))


class TestGridSearch:
    def test_budget_caps_results(self) -> None:
        cands = GridSearch().generate(spec(4096), "8.9", budget=5)
        assert len(cands) == 5

    def test_no_budget_returns_all(self) -> None:
        all_c = GridSearch().generate(spec(4096), "8.9")
        # cc 8.9: blocks{4096,8192} x warps{4,8,16} x stages{1,2,3} x acc{fp32,fp16} = 36
        assert len(all_c) == 36

    def test_pascal_fewer_candidates(self) -> None:
        cands = GridSearch().generate(spec(4096), "6.1")
        # blocks{4096,8192} x warps{4,8,16} x stages{1} x acc{fp32,fp16} = 12
        assert len(cands) == 12
