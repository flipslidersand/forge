"""Orchestrator の GPU end-to-end テスト。CUDA 非対応環境ではスキップ。"""

import tempfile
from pathlib import Path

import pytest
import torch

from forge.cache.repository import KernelRepository
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.orchestrator import Orchestrator
from forge.search.grid import GridSearch
from forge.search.space import SearchSpace

pytestmark = pytest.mark.gpu
_SKIP = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")


def _spec() -> KernelSpec:
    return KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec((512, 4096), torch.float16, True),
            TensorSpec((4096,), torch.float16, True),
        ),
        output_specs=(TensorSpec((512, 4096), torch.float16, True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_v1",
        constraints=(),
    )


def _orch(repo: KernelRepository) -> Orchestrator:
    return Orchestrator(repo=repo, warmup=5, repeat=30)


def _search(acc_dtypes: list[str] | None = None) -> GridSearch:
    # 計測軸を絞って探索を高速化
    space = SearchSpace(num_warps=[4, 8], acc_dtypes=acc_dtypes or ["fp32"])
    return GridSearch(space)


@_SKIP
def test_search_finds_correct_faster_kernel_and_caches() -> None:
    with tempfile.TemporaryDirectory() as d:
        repo = KernelRepository(Path(d) / "cache.db")
        orch = _orch(repo)
        spec = _spec()

        result = orch.optimize(spec, budget=10, search=_search())
        assert not result.cache_hit
        assert result.best_params is not None
        assert result.best_params.acc_dtype == "fp32"
        assert result.best_benchmark is not None
        # 全候補が正確性を通ること（fp32 のみなので）
        assert all(e.correct for e in result.experiments)
        # 融合カーネルは F.rms_norm baseline より速い
        assert result.speedup is not None and result.speedup > 1.0
        repo.close()


@_SKIP
def test_second_run_is_cache_hit() -> None:
    with tempfile.TemporaryDirectory() as d:
        repo = KernelRepository(Path(d) / "cache.db")
        orch = _orch(repo)
        spec = _spec()

        first = orch.optimize(spec, budget=6, search=_search())
        assert not first.cache_hit

        second = orch.optimize(spec, budget=6, search=_search())
        assert second.cache_hit
        assert second.experiments == []  # 探索していない
        assert second.best_params == first.best_params
        repo.close()


@_SKIP
def test_fp16_accumulator_rejected_as_incorrect() -> None:
    # fp16 accumulator は縮約精度不足で tolerance 超過 → 不採用になるはず
    with tempfile.TemporaryDirectory() as d:
        repo = KernelRepository(Path(d) / "cache.db")
        space = SearchSpace(num_warps=[8], acc_dtypes=["fp16"])
        orch = Orchestrator(repo=repo, warmup=5, repeat=20)
        result = orch.optimize(_spec(), budget=4, search=GridSearch(space))
        assert result.best_params is None  # fp16 acc は全滅
        assert all(not e.correct for e in result.experiments)
        repo.close()
