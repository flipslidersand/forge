from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

from forge.benchmark.statistics import BenchmarkResult


@dataclass
class WorkerResult:
    success: bool
    correct: bool = False
    max_abs_diff: float | None = None
    failures: list[dict[str, Any]] = field(default_factory=list)
    candidate: BenchmarkResult | None = None
    baseline: BenchmarkResult | None = None
    baseline_name: str | None = None
    error: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkerResult:
        if not d.get("success"):
            return cls(success=False, error=d.get("error"))
        return cls(
            success=True,
            correct=bool(d.get("correct", False)),
            max_abs_diff=d.get("max_abs_diff"),
            failures=list(d.get("failures", [])),
            candidate=BenchmarkResult.from_dict(d["candidate"]) if "candidate" in d else None,
            baseline=BenchmarkResult.from_dict(d["baseline"]) if "baseline" in d else None,
            baseline_name=d.get("baseline_name"),
        )


def run_in_worker(
    kernel_code: str,
    op_type: str,
    benchmark_input: list[dict[str, Any]],
    constants: dict[str, Any],
    correctness_cases: list[dict[str, Any]] | None = None,
    task: str = "full",
    warmup: int = 25,
    repeat: int = 200,
    tolerance: dict[str, Any] | None = None,
    timeout_s: float = 60.0,
    python_executable: str | None = None,
) -> WorkerResult:
    """生成カーネルを使い捨て subprocess で実行する。

    benchmark_input はタイミング計測に使う代表入力。correctness_cases を渡すと
    各ケースで正確性を検証する（省略時は benchmark_input で 1 回検証）。
    CUDA エラー・タイムアウト・コンパイル失敗のいずれも WorkerResult(success=False)
    として返し、親プロセスは生き残る。python_executable 省略時は現在のインタプリタ。
    """
    payload = json.dumps(
        {
            "kernel_code": kernel_code,
            "op_type": op_type,
            "benchmark_input": benchmark_input,
            "correctness_cases": correctness_cases,
            "constants": constants,
            "task": task,
            "warmup": warmup,
            "repeat": repeat,
            "tolerance": tolerance or {"atol": 2e-3, "rtol": 1e-2, "equal_nan": False},
        }
    )
    exe = python_executable or sys.executable
    try:
        proc = subprocess.run(
            [exe, "-m", "forge.runtime._worker_entry"],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return WorkerResult(success=False, error=f"timeout after {timeout_s}s")

    if proc.returncode != 0:
        # CUDA abort 等で JSON を出す前に死んだケース
        err = (
            proc.stderr.strip().splitlines()[-1]
            if proc.stderr.strip()
            else f"exit {proc.returncode}"
        )
        return WorkerResult(success=False, error=f"worker crashed: {err}")

    try:
        return WorkerResult.from_dict(json.loads(proc.stdout.strip().splitlines()[-1]))
    except (json.JSONDecodeError, IndexError):
        return WorkerResult(success=False, error=f"bad worker output: {proc.stdout[:200]!r}")
