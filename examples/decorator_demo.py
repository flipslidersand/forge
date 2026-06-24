"""@forge.optimize のデモ。デコレータ 1 つで RMSNorm を自動最適化する。

GPU + venv で実行:
    .venv/bin/python examples/decorator_demo.py
"""

from __future__ import annotations

import time

import torch

import forge


@forge.optimize(budget=12, progress=print)
def rmsnorm(x, weight, eps=1e-6):
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight


def main() -> None:
    x = torch.randn(2048, 4096, dtype=torch.float16, device="cuda")
    w = torch.ones(4096, dtype=torch.float16, device="cuda")

    print("=== first call: searches + compiles (slow) ===")
    t0 = time.perf_counter()
    out = rmsnorm(x, w)
    print(f"first call: {time.perf_counter() - t0:.1f}s")

    # 正確性チェック
    ref = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6) * w
    print(f"correct: {torch.allclose(out.float(), ref.float(), atol=2e-3, rtol=1e-2)}")

    print("\n=== second call: in-process cache (fast) ===")
    t0 = time.perf_counter()
    rmsnorm(x, w)
    torch.cuda.synchronize()
    print(f"second call: {(time.perf_counter() - t0) * 1e3:.2f}ms")


if __name__ == "__main__":
    main()
