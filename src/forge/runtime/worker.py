from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from forge.benchmark.statistics import BenchmarkResult


@dataclass
class WorkerResult:
    success: bool
    correct: bool = False
    max_abs_diff: float | None = None
    candidate: BenchmarkResult | None = None
    baseline: BenchmarkResult | None = None
    error: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkerResult:
        if not d.get("success"):
            return cls(success=False, error=d.get("error"))
        return cls(
            success=True,
            correct=bool(d.get("correct", False)),
            max_abs_diff=d.get("max_abs_diff"),
            candidate=BenchmarkResult.from_dict(d["candidate"]) if "candidate" in d else None,
            baseline=BenchmarkResult.from_dict(d["baseline"]) if "baseline" in d else None,
        )


def run_in_worker(
    kernel_code: str,
    op_type: str,
    input_specs: list[dict[str, Any]],
    constants: dict[str, Any],
    task: str = "full",
    warmup: int = 25,
    repeat: int = 200,
    tolerance: dict[str, float] | None = None,
    timeout_s: float = 60.0,
    python_executable: str | None = None,
) -> WorkerResult:
    """生成カーネルを使い捨て subprocess で実行する。

    CUDA エラー・タイムアウト・コンパイル失敗のいずれも WorkerResult(success=False)
    として返し、親プロセスは生き残る。python_executable を省略すると現在の
    インタプリタ (.venv 推奨) を使う。
    """
    payload = json.dumps(
        {
            "kernel_code": kernel_code,
            "op_type": op_type,
            "input_specs": input_specs,
            "constants": constants,
            "task": task,
            "warmup": warmup,
            "repeat": repeat,
            "tolerance": tolerance or {"atol": 2e-3, "rtol": 1e-2},
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
