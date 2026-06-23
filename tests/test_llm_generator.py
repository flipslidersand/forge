"""LLMGenerator のオフラインテスト。propose_fn を注入し、ネットワーク/API キー不要。"""

import torch

from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.search.candidate import CandidateGenerator, HistoryEntry
from forge.search.grid import GridSearch
from forge.search.llm_generator import LLMGenerator, build_prompt
from forge.search.params import SearchParams
from forge.search.random_search import RandomSearch


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


def cand(**kw) -> dict:
    base = dict(
        base_variant="single_row",
        block_size=4096,
        num_warps=8,
        num_stages=1,
        acc_dtype="fp32",
        rows_per_program=1,
        hypothesis="test",
    )
    base.update(kw)
    return base


class TestProtocolConformance:
    def test_all_generators_satisfy_protocol(self) -> None:
        assert isinstance(GridSearch(), CandidateGenerator)
        assert isinstance(RandomSearch(), CandidateGenerator)
        assert isinstance(LLMGenerator(propose_fn=lambda *a: []), CandidateGenerator)


class TestLLMGeneratorCoercion:
    def _gen(self, raw: list[dict]) -> LLMGenerator:
        return LLMGenerator(propose_fn=lambda s, cc, n, h: raw)

    def test_valid_candidates_pass_through(self) -> None:
        out = self._gen([cand(), cand(num_warps=16)]).generate(spec(), "8.9")
        assert len(out) == 2
        assert all(isinstance(p, SearchParams) for p in out)

    def test_invalid_block_size_dropped(self) -> None:
        # 3000 は 2 のべき乗でない → SearchParams が弾く
        out = self._gen([cand(block_size=3000)]).generate(spec(), "8.9")
        assert out == []

    def test_single_row_block_below_hidden_dropped(self) -> None:
        # single_row で block < N(4096) は無効
        out = self._gen([cand(block_size=2048)]).generate(spec(4096), "8.9")
        assert out == []

    def test_two_pass_small_tile_allowed(self) -> None:
        out = self._gen([cand(base_variant="two_pass", block_size=1024)]).generate(
            spec(4096), "8.9"
        )
        assert len(out) == 1
        assert out[0].variant == "two_pass"

    def test_unknown_variant_dropped(self) -> None:
        out = self._gen([cand(base_variant="bogus")]).generate(spec(), "8.9")
        assert out == []

    def test_bad_rows_for_non_multi_row_dropped(self) -> None:
        # rows>1 は multi_row のみ許可
        out = self._gen([cand(base_variant="single_row", rows_per_program=4)]).generate(
            spec(), "8.9"
        )
        assert out == []

    def test_multi_row_with_rows_ok(self) -> None:
        out = self._gen([cand(base_variant="multi_row", rows_per_program=4)]).generate(
            spec(), "8.9"
        )
        assert len(out) == 1
        assert out[0].rows_per_program == 4

    def test_missing_key_dropped_not_crash(self) -> None:
        bad = {"base_variant": "single_row", "block_size": 4096}  # warps/stages 欠落
        out = self._gen([bad, cand()]).generate(spec(), "8.9")
        assert len(out) == 1  # 壊れた候補は捨て、正常な 1 件のみ

    def test_dedup(self) -> None:
        out = self._gen([cand(), cand(), cand()]).generate(spec(), "8.9")
        assert len(out) == 1

    def test_budget_caps(self) -> None:
        raw = [cand(num_warps=w) for w in (4, 8, 16)] + [cand(block_size=8192)]
        out = self._gen(raw).generate(spec(), "8.9", budget=2)
        assert len(out) == 2


class TestProposeFnContract:
    def test_propose_fn_receives_history_and_n(self) -> None:
        captured = {}

        def fake(s, cc, n, h):
            captured["cc"] = cc
            captured["n"] = n
            captured["history"] = h
            return [cand()]

        hist = [HistoryEntry(SearchParams(4096, 8, 1, "fp16"), correct=False, median_us=None)]
        LLMGenerator(propose_fn=fake, default_n=7).generate(spec(), "6.1", history=hist)
        assert captured["cc"] == "6.1"
        assert captured["n"] == 7  # budget 未指定なら default_n
        assert captured["history"] == hist

    def test_budget_overrides_n(self) -> None:
        captured = {}

        def fake(s, cc, n, h):
            captured["n"] = n
            return []

        LLMGenerator(propose_fn=fake).generate(spec(), "8.9", budget=20)
        assert captured["n"] == 20


class TestHistoryEntry:
    def test_summary_incorrect(self) -> None:
        e = HistoryEntry(SearchParams(4096, 8, 1, "fp16"), correct=False, median_us=None)
        assert "INCORRECT" in e.summary()

    def test_summary_with_time(self) -> None:
        e = HistoryEntry(SearchParams(4096, 8, 1, "fp32"), correct=True, median_us=42.3)
        assert "42.3us" in e.summary()


class TestBuildPrompt:
    def test_includes_shape_and_constraints(self) -> None:
        p = build_prompt(spec(4096), "6.1", 12, [])
        assert "6.1" in p
        assert "(2048, 4096)" in p
        assert "two_pass" in p

    def test_includes_history(self) -> None:
        hist = [HistoryEntry(SearchParams(4096, 8, 1, "fp32"), correct=True, median_us=60.0)]
        p = build_prompt(spec(), "8.9", 12, hist)
        assert "60.0us" in p
