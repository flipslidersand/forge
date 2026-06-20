import pytest

from forge.benchmark.statistics import BenchmarkResult, is_improvement


def make_result(samples: list[float]) -> BenchmarkResult:
    return BenchmarkResult.from_samples(samples, warmup=25, repeat=len(samples))


class TestBenchmarkResult:
    def test_from_samples_percentiles(self) -> None:
        r = make_result([float(i) for i in range(1, 101)])  # 1..100
        assert r.median_us == 50.5
        assert r.p20_us == 21.0  # samples[20]
        assert r.p80_us == 81.0  # samples[80]

    def test_from_samples_sorts(self) -> None:
        r = make_result([5.0, 1.0, 3.0, 2.0, 4.0])
        assert r.samples_us == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert r.median_us == 3.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            make_result([])

    def test_dict_roundtrip(self) -> None:
        r = make_result([10.0, 20.0, 30.0])
        r2 = BenchmarkResult.from_dict(r.to_dict())
        assert r2.median_us == r.median_us
        assert r2.p20_us == r.p20_us
        assert r2.p80_us == r.p80_us


class TestIsImprovement:
    def test_clear_improvement(self) -> None:
        # candidate ~40µs, baseline ~80µs
        cand = make_result([float(x) for x in range(38, 43)])
        base = make_result([float(x) for x in range(78, 83)])
        assert is_improvement(cand, base) is True

    def test_no_improvement_same_speed(self) -> None:
        cand = make_result([float(x) for x in range(40, 45)])
        base = make_result([float(x) for x in range(40, 45)])
        assert is_improvement(cand, base) is False

    def test_marginal_improvement_below_threshold(self) -> None:
        # 約 2% しか速くない → 3% 閾値で棄却されるべき
        cand = make_result([98.0] * 10)
        base = make_result([100.0] * 10)
        assert is_improvement(cand, base, min_speedup=1.03) is False

    def test_candidate_slower_rejected(self) -> None:
        cand = make_result([100.0] * 10)
        base = make_result([50.0] * 10)
        assert is_improvement(cand, base) is False

    def test_overlapping_distributions_rejected(self) -> None:
        # 中央値は candidate が速いが分布が重なる → 棄却（保守的判定）
        cand = make_result([float(x) for x in range(40, 90)])  # p80 高い
        base = make_result([float(x) for x in range(50, 100)])  # p20 低い
        assert is_improvement(cand, base) is False

    def test_zero_baseline_rejected(self) -> None:
        cand = make_result([10.0])
        base = make_result([0.0])
        assert is_improvement(cand, base) is False
