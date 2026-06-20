from __future__ import annotations

from collections.abc import Callable

import torch

from .statistics import BenchmarkResult


def measure(
    fn: Callable[[], object],
    warmup: int = 25,
    repeat: int = 200,
) -> BenchmarkResult:
    """GPU カーネルのレイテンシを CUDA Event で計測する。

    各サンプルごとに synchronize するため、非同期実行のオーバーラップを排除した
    純粋なカーネル時間を測る。戻り値は µs 単位。
    """
    if not torch.cuda.is_available():
        raise RuntimeError("measure() requires CUDA")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples_us: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)  # ms -> µs

    return BenchmarkResult.from_samples(samples_us, warmup=warmup, repeat=repeat)
