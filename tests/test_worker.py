"""GPU を要する worker の end-to-end テスト。

CUDA 非対応環境ではスキップ。実行は .venv/bin/python（torch+triton+cuda）で:
    .venv/bin/python -m pytest tests/test_worker.py -v
"""

import pytest
import torch

from forge.codegen.triton_codegen import generate_rmsnorm
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.runtime.worker import run_in_worker
from forge.search.params import SearchParams

pytestmark = pytest.mark.gpu

_CUDA = torch.cuda.is_available()
_SKIP = pytest.mark.skipif(not _CUDA, reason="requires CUDA GPU")

INPUT_SPECS = [
    {"shape": [2048, 4096], "dtype": "float16", "init": "randn", "seed": 0},
    {"shape": [4096], "dtype": "float16", "init": "ones", "seed": 1},
]


def _spec() -> KernelSpec:
    return KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec((2048, 4096), torch.float16, True),
            TensorSpec((4096,), torch.float16, True),
        ),
        output_specs=(TensorSpec((2048, 4096), torch.float16, True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_v1",
        constraints=(),
    )


def _run(params: SearchParams, task: str = "full", **kw: object):
    code = generate_rmsnorm(_spec(), params)
    return run_in_worker(
        code,
        "rmsnorm",
        INPUT_SPECS,
        {"eps": 1e-6},
        task=task,
        warmup=10,
        repeat=50,
        **kw,  # type: ignore[arg-type]
    )


@_SKIP
def test_valid_candidate_is_correct_and_benchmarked() -> None:
    r = _run(SearchParams(block_size=4096, num_warps=8, num_stages=1, acc_dtype="fp32"))
    assert r.success
    assert r.correct
    assert r.max_abs_diff is not None and r.max_abs_diff < 2e-3
    assert r.candidate is not None and r.candidate.median_us > 0
    assert r.baseline is not None and r.baseline.median_us > 0


@_SKIP
def test_undersized_block_is_incorrect_not_crash() -> None:
    # block_size=2048 < N=4096 → 末尾の要素が落ちて不正確だが crash はしない
    r = _run(SearchParams(block_size=2048, num_warps=4, num_stages=1, acc_dtype="fp32"))
    assert r.success  # プロセスは正常終了
    assert r.correct is False  # 正確性チェックで弾かれる


@_SKIP
def test_correctness_only_skips_benchmark() -> None:
    r = _run(
        SearchParams(block_size=4096, num_warps=8, num_stages=1, acc_dtype="fp32"),
        task="correctness",
    )
    assert r.success and r.correct
    assert r.candidate is None  # benchmark は走らない


@_SKIP
def test_timeout_returns_failure() -> None:
    r = _run(
        SearchParams(block_size=4096, num_warps=8, num_stages=1, acc_dtype="fp32"),
        timeout_s=0.001,
    )
    assert r.success is False
    assert r.error is not None and "timeout" in r.error
