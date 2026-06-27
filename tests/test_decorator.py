"""@forge.optimize の end-to-end テスト。GPU 必須。"""

import tempfile
from pathlib import Path

import pytest
import torch

import forge
from forge.cache.repository import KernelRepository
from forge.search.grid import GridSearch
from forge.search.space import SearchSpace

pytestmark = pytest.mark.gpu
_SKIP = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")


def _ref(x, weight, eps=1e-6):
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight


def _fast_search() -> GridSearch:
    # 探索を軽く: fp32 single_row のみ、warps 2 通り
    return GridSearch(
        SearchSpace(num_warps=[4, 8], num_stages=[1], acc_dtypes=["fp32"], variants=["single_row"])
    )


@_SKIP
def test_decorated_matches_eager() -> None:
    with tempfile.TemporaryDirectory() as d:
        repo = KernelRepository(Path(d) / "cache.db")

        @forge.optimize(budget=6, repo=repo, search=_fast_search())
        def rmsnorm(x, weight, eps=1e-6):
            return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight

        x = torch.randn(256, 4096, dtype=torch.float16, device="cuda")
        w = torch.ones(4096, dtype=torch.float16, device="cuda")

        out = rmsnorm(x, w)
        ref = _ref(x, w)
        assert torch.allclose(out.float(), ref.float(), atol=2e-3, rtol=1e-2)
        repo.close()


@_SKIP
def test_second_call_uses_inprocess_cache() -> None:
    with tempfile.TemporaryDirectory() as d:
        repo = KernelRepository(Path(d) / "cache.db")

        @forge.optimize(budget=6, repo=repo, search=_fast_search())
        def rmsnorm(x, weight, eps=1e-6):
            return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight

        x = torch.randn(256, 4096, dtype=torch.float16, device="cuda")
        w = torch.ones(4096, dtype=torch.float16, device="cuda")

        rmsnorm(x, w)  # 初回: 探索
        compiled = rmsnorm._forge_compiled  # type: ignore[attr-defined]
        assert len(compiled) == 1
        key = next(iter(compiled))

        rmsnorm(x, w)  # 2 回目: 同 shape → 再探索しない
        assert len(compiled) == 1
        assert next(iter(compiled)) == key
        repo.close()


@_SKIP
def test_cpu_tensor_falls_back_to_eager() -> None:
    @forge.optimize(budget=6)
    def rmsnorm(x, weight, eps=1e-6):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight

    x = torch.randn(8, 4096, dtype=torch.float32)  # CPU
    w = torch.ones(4096, dtype=torch.float32)
    out = rmsnorm(x, w)
    assert torch.allclose(out, _ref(x, w))
    assert rmsnorm._forge_compiled == {}  # type: ignore[attr-defined]  # 探索していない


@_SKIP
def test_unrecognized_op_falls_back_to_eager() -> None:
    @forge.optimize(budget=6)
    def not_rmsnorm(x, y):
        return x + y  # 既知パターンでない

    x = torch.randn(8, 8, device="cuda")
    y = torch.randn(8, 8, device="cuda")
    out = not_rmsnorm(x, y)
    assert torch.allclose(out, x + y)
    assert not_rmsnorm._forge_compiled == {}  # type: ignore[attr-defined]


@_SKIP
def test_decorated_layernorm_matches_eager() -> None:
    # 3 入力(x, weight, bias)の reduction op
    import torch.nn.functional as F

    space = SearchSpace(
        num_warps=[4, 8], num_stages=[1], acc_dtypes=["fp32"], variants=["single_row"]
    )
    with tempfile.TemporaryDirectory() as d:
        repo = KernelRepository(Path(d) / "cache.db")

        @forge.optimize(budget=4, repo=repo, search=GridSearch(space))
        def layernorm(x, weight, bias, eps=1e-5):
            return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)

        x = torch.randn(256, 4096, dtype=torch.float16, device="cuda")
        w = torch.randn(4096, dtype=torch.float16, device="cuda")
        b = torch.randn(4096, dtype=torch.float16, device="cuda")
        out = layernorm(x, w, b)
        ref = F.layer_norm(x.float(), (4096,), w.float(), b.float(), 1e-5).to(x.dtype)
        assert torch.allclose(out.float(), ref.float(), atol=2e-3, rtol=1e-2)
        repo.close()


@_SKIP
def test_decorated_gelu_matches_eager() -> None:
    # elementwise op（block が N に縛られない）
    import torch.nn.functional as F

    with tempfile.TemporaryDirectory() as d:
        repo = KernelRepository(Path(d) / "cache.db")

        @forge.optimize(budget=6, repo=repo)
        def gelu(x):
            return F.gelu(x)

        x = torch.randn(512, 2048, dtype=torch.float16, device="cuda")
        out = gelu(x)
        ref = F.gelu(x.float()).to(x.dtype)
        assert torch.allclose(out.float(), ref.float(), atol=2e-3, rtol=1e-2)
        # elementwise variant が選ばれていること
        kfns = gelu._forge_compiled  # type: ignore[attr-defined]
        assert len(kfns) == 1 and next(iter(kfns.values())) is not None
        repo.close()


@_SKIP
def test_decorated_softmax_matches_eager() -> None:
    # アーキの汎用性: 同じデコレータで softmax も最適化される
    space = SearchSpace(
        num_warps=[4, 8], num_stages=[1], acc_dtypes=["fp32"], variants=["single_row"]
    )
    with tempfile.TemporaryDirectory() as d:
        repo = KernelRepository(Path(d) / "cache.db")

        @forge.optimize(budget=4, repo=repo, search=GridSearch(space))
        def softmax(x):
            return torch.softmax(x, dim=-1)

        x = torch.randn(256, 4096, dtype=torch.float16, device="cuda")
        out = softmax(x)
        ref = torch.softmax(x.float(), dim=-1).to(x.dtype)
        assert torch.allclose(out.float(), ref.float(), atol=2e-3, rtol=1e-2)
        # softmax は単一行の確率分布 → 各行の和が ~1
        assert torch.allclose(out.float().sum(-1), torch.ones(256, device="cuda"), atol=1e-2)
        repo.close()
