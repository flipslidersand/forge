from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from forge.benchmark.statistics import BenchmarkResult, is_improvement
from forge.cache.key import CacheKey
from forge.cache.repository import CachedKernel, KernelRepository
from forge.codegen.triton_codegen import generate
from forge.ir.kernel_spec import KernelSpec
from forge.runtime.worker import WorkerResult, run_in_worker
from forge.search.grid import GridSearch
from forge.search.params import SearchParams
from forge.validation.test_cases import correctness_cases, primary_input
from forge.validation.tolerance import get_tolerance


@dataclass
class ExperimentResult:
    params: SearchParams
    success: bool
    correct: bool
    median_us: float | None
    error: str | None
    is_best: bool = False


@dataclass
class SearchResult:
    spec: KernelSpec
    cache_hit: bool
    best_params: SearchParams | None
    best_benchmark: BenchmarkResult | None
    baseline_benchmark: BenchmarkResult | None
    baseline_name: str | None
    experiments: list[ExperimentResult]

    @property
    def speedup(self) -> float | None:
        if self.best_benchmark and self.baseline_benchmark and self.best_benchmark.median_us > 0:
            return self.baseline_benchmark.median_us / self.best_benchmark.median_us
        return None


class Orchestrator:
    """KernelSpec を受け取り、探索 → 検証 → ベンチマーク → キャッシュの一連を回す。

    再現可能なパイプラインが中心で、探索器 (GridSearch) は差し替え可能な一要素。
    """

    def __init__(
        self,
        repo: KernelRepository | None = None,
        python_executable: str | None = None,
        min_speedup: float = 1.03,
        warmup: int = 25,
        repeat: int = 200,
        timeout_s: float = 60.0,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.repo = repo or KernelRepository()
        self.python_executable = python_executable
        self.min_speedup = min_speedup
        self.warmup = warmup
        self.repeat = repeat
        self.timeout_s = timeout_s
        self._progress = progress or (lambda _msg: None)

    def optimize(
        self,
        spec: KernelSpec,
        budget: int = 50,
        search: GridSearch | None = None,
        use_cache: bool = True,
    ) -> SearchResult:
        spec.validate()
        key = CacheKey.from_spec_and_env(spec)

        if use_cache and (cached := self.repo.get(key)) is not None:
            self._progress(f"cache HIT: {cached.params}")
            bench = BenchmarkResult.from_dict(cached.benchmark_json)
            return SearchResult(
                spec=spec,
                cache_hit=True,
                best_params=SearchParams.from_dict(cached.params),
                best_benchmark=bench,
                baseline_benchmark=None,
                baseline_name=None,
                experiments=[],
            )

        search = search or GridSearch()
        candidates = search.generate(spec, key.compute_capability, budget=budget)
        self._progress(f"searching {len(candidates)} candidates (cc {key.compute_capability})")

        bench_input = primary_input(spec)
        cases = correctness_cases(spec)
        tol = get_tolerance(spec.op_type).to_dict()

        experiments: list[ExperimentResult] = []
        best_params: SearchParams | None = None
        best_bench: BenchmarkResult | None = None
        baseline_bench: BenchmarkResult | None = None
        baseline_name: str | None = None

        for i, params in enumerate(candidates, 1):
            code = generate(spec, params)
            wr: WorkerResult = run_in_worker(
                code,
                spec.op_type,
                bench_input,
                spec.constants,
                correctness_cases=cases,
                task="full",
                warmup=self.warmup,
                repeat=self.repeat,
                tolerance=tol,
                timeout_s=self.timeout_s,
                python_executable=self.python_executable,
            )

            label = f"[{i}/{len(candidates)}] {params.block_size}/{params.num_warps}"
            if not wr.success:
                experiments.append(ExperimentResult(params, False, False, None, wr.error))
                self._progress(f"{label} FAIL: {wr.error}")
                continue
            if not wr.correct:
                experiments.append(ExperimentResult(params, True, False, None, "incorrect"))
                self._progress(f"{label} INCORRECT")
                continue

            assert wr.candidate is not None and wr.baseline is not None
            baseline_bench = wr.baseline
            baseline_name = wr.baseline_name

            improved = best_bench is None or is_improvement(
                wr.candidate, best_bench, self.min_speedup
            )
            tag = ""
            if improved:
                best_params, best_bench = params, wr.candidate
                tag = " BEST"
            experiments.append(
                ExperimentResult(params, True, True, wr.candidate.median_us, None, is_best=improved)
            )
            self._progress(f"{label}/{params.acc_dtype} -> {wr.candidate.median_us:.1f}us{tag}")

        if best_params is not None and best_bench is not None:
            code = generate(spec, best_params)
            self.repo.put(
                key,
                CachedKernel(
                    cache_key=key,
                    params=best_params.to_dict(),
                    kernel_code=code,
                    benchmark_json=best_bench.to_dict(),
                    created_at=datetime.now(UTC),
                ),
            )
            self._progress(f"cached best: {best_params} ({best_bench.median_us:.1f}us)")

        return SearchResult(
            spec=spec,
            cache_hit=False,
            best_params=best_params,
            best_benchmark=best_bench,
            baseline_benchmark=baseline_bench,
            baseline_name=baseline_name,
            experiments=experiments,
        )
