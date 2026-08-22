from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import TypedDict

_logger = logging.getLogger(__name__)


class _BenchmarkResultDictRequired(TypedDict):
    median_us: float
    p20_us: float
    p80_us: float


class BenchmarkResultDict(_BenchmarkResultDictRequired, total=False):
    p95_us: float
    samples_us: list[float]
    warmup_count: int
    measure_count: int


@dataclass
class BenchmarkResult:
    samples_us: list[float] = field(default_factory=list)
    median_us: float = 0.0
    p20_us: float = 0.0
    p80_us: float = 0.0
    p95_us: float = 0.0
    warmup_count: int = 0
    measure_count: int = 0

    @classmethod
    def from_samples(cls, samples_us: list[float], warmup: int, repeat: int) -> BenchmarkResult:
        if not samples_us:
            raise ValueError("samples_us must not be empty")
        s = sorted(samples_us)
        n = len(s)
        return cls(
            samples_us=s,
            median_us=statistics.median(s),
            p20_us=s[int(n * 0.2)],
            p80_us=s[min(int(n * 0.8), n - 1)],
            p95_us=s[min(int(n * 0.95), n - 1)],
            warmup_count=warmup,
            measure_count=repeat,
        )

    def to_dict(self) -> BenchmarkResultDict:
        return BenchmarkResultDict(
            median_us=self.median_us,
            p20_us=self.p20_us,
            p80_us=self.p80_us,
            p95_us=self.p95_us,
            warmup_count=self.warmup_count,
            measure_count=self.measure_count,
        )

    @classmethod
    def from_dict(cls, d: BenchmarkResultDict) -> BenchmarkResult:
        return cls(
            samples_us=list(d.get("samples_us", [])),
            median_us=float(d["median_us"]),
            p20_us=float(d["p20_us"]),
            p80_us=float(d["p80_us"]),
            p95_us=float(d.get("p95_us", 0.0)),
            warmup_count=int(d.get("warmup_count", 0)),
            measure_count=int(d.get("measure_count", 0)),
        )


def is_improvement(
    candidate: BenchmarkResult,
    baseline: BenchmarkResult,
    min_speedup: float = 1.03,
) -> bool:
    """候補が baseline より統計的に有意に速いか判定する。

    候補の遅い側 (p80) が baseline の速い側 (p20) を min_speedup 倍上回って
    なお速い場合のみ True。測定ノイズ (#5 実測 ~0.7%) に対し min_speedup=1.03
    は約 4 倍のマージン（ADR-004）。
    """
    if baseline.p20_us <= 0:
        _logger.warning(
            "is_improvement: baseline.p20_us=%s is invalid (<=0), treating as no improvement",
            baseline.p20_us,
        )
        return False
    return candidate.p80_us < baseline.p20_us / min_speedup
