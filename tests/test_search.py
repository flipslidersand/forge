import torch

from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.search.grid import GridSearch
from forge.search.random_search import RandomSearch
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


def single(variant: str, **kw) -> SearchSpace:
    return SearchSpace(variants=[variant], **kw)


def gelu_spec(n: int = 4096) -> KernelSpec:
    return KernelSpec(
        op_type="gelu",
        input_specs=(TensorSpec((2048, n), torch.float16, True),),
        output_specs=(TensorSpec((2048, n), torch.float16, True),),
        constants={},
        graph_hash="gelu_v1",
        constraints=(),
    )


class TestElementwiseSpace:
    def test_gelu_uses_elementwise_variant_only(self) -> None:
        params = list(SearchSpace().enumerate(gelu_spec(4096), "8.9"))
        assert params and {p.variant for p in params} == {"elementwise"}

    def test_gelu_block_not_tied_to_hidden(self) -> None:
        # elementwise は N=4096 でも小さい block(256 等)を許す
        params = list(SearchSpace().enumerate(gelu_spec(4096), "8.9"))
        assert any(p.block_size < 4096 for p in params)
        assert all(p.rows_per_program == 1 for p in params)


class TestHelpers:
    def test_cc_to_int(self) -> None:
        assert _cc_to_int("8.9") == 89
        assert _cc_to_int("6.1") == 61
        assert _cc_to_int("7.0") == 70

    def test_next_pow2(self) -> None:
        assert _next_pow2(4096) == 4096
        assert _next_pow2(4000) == 4096
        assert _next_pow2(5000) == 8192


class TestSearchSpaceBlocks:
    def test_single_row_block_at_least_hidden(self) -> None:
        params = list(single("single_row").enumerate(spec(4096), "8.9"))
        assert params and all(p.block_size >= 4096 for p in params)

    def test_two_pass_allows_smaller_tiles(self) -> None:
        params = list(single("two_pass").enumerate(spec(4096), "8.9"))
        # two_pass のタイルは N 以下
        assert params and all(p.block_size <= 4096 for p in params)
        assert any(p.block_size < 4096 for p in params)

    def test_multi_row_has_rows_gt_1(self) -> None:
        params = list(single("multi_row").enumerate(spec(4096), "8.9"))
        assert params and {p.rows_per_program for p in params} == {2, 4}

    def test_non_multi_row_keeps_rows_1(self) -> None:
        for v in ("single_row", "two_pass"):
            params = list(single(v).enumerate(spec(4096), "8.9"))
            assert all(p.rows_per_program == 1 for p in params)

    def test_no_valid_block_falls_back_to_next_pow2(self) -> None:
        params = list(single("single_row").enumerate(spec(6000), "8.9"))
        assert all(p.block_size == 8192 for p in params)


class TestSearchSpaceGpu:
    def test_pascal_restricts_num_stages(self) -> None:
        params = list(SearchSpace().enumerate(spec(4096), "6.1"))
        assert {p.num_stages for p in params} == {1}

    def test_volta_allows_pipelining(self) -> None:
        params = list(SearchSpace().enumerate(spec(4096), "8.9"))
        assert {p.num_stages for p in params} == {1, 2, 3}

    def test_no_duplicates(self) -> None:
        params = list(SearchSpace().enumerate(spec(4096), "8.9"))
        keys = [
            (p.variant, p.block_size, p.rows_per_program, p.num_warps, p.num_stages, p.acc_dtype)
            for p in params
        ]
        assert len(keys) == len(set(keys))

    def test_all_three_variants_present(self) -> None:
        params = list(SearchSpace().enumerate(spec(4096), "6.1"))
        assert {p.variant for p in params} == {"single_row", "multi_row", "two_pass"}


class TestGridSearch:
    def test_budget_caps_results(self) -> None:
        cands = GridSearch().generate(spec(4096), "8.9", budget=5)
        assert len(cands) == 5

    def test_single_row_pascal_count(self) -> None:
        # single_row, cc6.1: blocks{4096,8192} x warps{4,8,16} x stages{1} x acc{fp32,fp16} = 12
        cands = GridSearch(single("single_row")).generate(spec(4096), "6.1")
        assert len(cands) == 12

    def test_variants_expand_space(self) -> None:
        single_only = GridSearch(single("single_row")).generate(spec(4096), "6.1")
        all_variants = GridSearch().generate(spec(4096), "6.1")
        assert len(all_variants) > len(single_only)


class TestRandomSearch:
    def test_deterministic_with_seed(self) -> None:
        a = RandomSearch(seed=42).generate(spec(4096), "8.9", budget=10)
        b = RandomSearch(seed=42).generate(spec(4096), "8.9", budget=10)
        assert a == b

    def test_different_seed_differs(self) -> None:
        a = RandomSearch(seed=1).generate(spec(4096), "8.9", budget=10)
        b = RandomSearch(seed=2).generate(spec(4096), "8.9", budget=10)
        assert a != b

    def test_budget_caps(self) -> None:
        cands = RandomSearch().generate(spec(4096), "8.9", budget=7)
        assert len(cands) == 7

    def test_samples_are_valid_subset(self) -> None:
        full = set(SearchSpace().enumerate(spec(4096), "8.9"))
        sample = RandomSearch(seed=3).generate(spec(4096), "8.9", budget=15)
        assert set(sample) <= full
