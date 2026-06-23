"""RMSNorm 自動最適化の end-to-end デモ。

GPU + venv で実行:
    .venv/bin/python examples/rmsnorm_search.py
"""

from __future__ import annotations

import argparse

import torch

from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.orchestrator import Orchestrator


def build_spec(m: int, n: int) -> KernelSpec:
    return KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec((m, n), torch.float16, True),
            TensorSpec((n,), torch.float16, True),
        ),
        output_specs=(TensorSpec((m, n), torch.float16, True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_v1",
        constraints=(),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    spec = build_spec(args.rows, args.hidden)
    orch = Orchestrator(progress=print)

    result = orch.optimize(spec, budget=args.budget, use_cache=not args.no_cache)

    print("\n" + "=" * 60)
    if result.cache_hit:
        print(f"CACHE HIT — best params: {result.best_params}")
        print(f"  cached median: {result.best_benchmark.median_us:.1f}us")
    elif result.best_params is None:
        print("No correct+faster candidate found.")
    else:
        print(f"Best: {result.best_params}")
        print(f"  candidate median: {result.best_benchmark.median_us:.1f}us")
        if result.baseline_benchmark:
            print(
                f"  baseline ({result.baseline_name}): "
                f"{result.baseline_benchmark.median_us:.1f}us"
            )
        if result.speedup:
            print(f"  speedup vs {result.baseline_name}: {result.speedup:.2f}x")
        n_ok = sum(1 for e in result.experiments if e.correct)
        print(f"  candidates: {len(result.experiments)} tried, {n_ok} correct")


if __name__ == "__main__":
    main()
