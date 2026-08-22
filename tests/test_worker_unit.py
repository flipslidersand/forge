"""run_in_worker / run_extended_baseline_in_worker のエラーパスを CPU mock でテスト。

GPU 不要。subprocess.run を patch して全ケースを CI で常時実行する。
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from forge.runtime.worker import WorkerResult, run_extended_baseline_in_worker, run_in_worker

_DUMMY_KERNEL = "def kernel(): pass"
_DUMMY_OP = "rmsnorm"
_DUMMY_INPUT = [{"shape": [512, 4096], "dtype": "float16", "init": "randn", "seed": 0}]
_DUMMY_CONSTANTS: dict = {"eps": 1e-6}


def _proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    p = MagicMock()
    p.returncode = returncode
    p.stdout = stdout
    p.stderr = stderr
    return p


def _ok_output() -> str:
    return json.dumps(
        {
            "success": True,
            "correct": True,
            "max_abs_diff": 1e-4,
            "failures": [],
            "candidate": {"median_us": 42.0, "p20_us": 40.0, "p80_us": 44.0, "p95_us": 46.0},
            "baseline": {"median_us": 60.0, "p20_us": 58.0, "p80_us": 62.0, "p95_us": 65.0},
            "baseline_name": "torch.rmsnorm",
        }
    )


class TestRunInWorkerErrorPaths:
    """subprocess エラーパス 3種を CPU mock で検証。"""

    def test_timeout_returns_failure(self) -> None:
        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="python", timeout=5.0)
        ):
            r = run_in_worker(
                _DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS, timeout_s=5.0
            )

        assert r.success is False
        assert r.error is not None
        assert "timeout" in r.error.lower()
        assert "5.0" in r.error

    def test_nonzero_returncode_returns_failure(self) -> None:
        proc = _proc(returncode=1, stderr="CUDA error: device-side assert triggered\nAborted\n")
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.success is False
        assert r.error is not None
        assert "worker crashed" in r.error
        assert "Aborted" in r.error  # stderr の最終行

    def test_nonzero_returncode_empty_stderr(self) -> None:
        proc = _proc(returncode=139, stderr="")
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.success is False
        assert r.error is not None
        assert "worker crashed" in r.error
        assert "139" in r.error  # exit コードを含む

    def test_bad_json_stdout_returns_failure(self) -> None:
        proc = _proc(returncode=0, stdout="this is not json at all")
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.success is False
        assert r.error is not None
        assert "bad worker output" in r.error

    def test_empty_stdout_returns_failure(self) -> None:
        proc = _proc(returncode=0, stdout="")
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.success is False
        assert r.error is not None
        assert "bad worker output" in r.error

    def test_success_path_parsed_correctly(self) -> None:
        proc = _proc(returncode=0, stdout=_ok_output())
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.success is True
        assert r.correct is True
        assert r.candidate is not None
        assert r.candidate.median_us == pytest.approx(42.0)
        assert r.baseline is not None
        assert r.baseline_name == "torch.rmsnorm"

    def test_worker_reported_failure_propagated(self) -> None:
        stdout = json.dumps(
            {"success": False, "error": "ValueError: unsupported op", "traceback": "..."}
        )
        proc = _proc(returncode=0, stdout=stdout)
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.success is False
        assert r.error == "ValueError: unsupported op"

    def test_only_last_stdout_line_is_parsed(self) -> None:
        """ワーカーが debug print を出した後に JSON を吐くケース。"""
        stdout = "debug: starting\nwarning: slow path\n" + _ok_output()
        proc = _proc(returncode=0, stdout=stdout)
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.success is True


class TestRunExtendedBaselineInWorkerErrorPaths:
    """run_extended_baseline_in_worker のエラーパスも CPU mock でカバー。"""

    def test_timeout_returns_empty_list(self) -> None:
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="python", timeout=180.0),
        ):
            result = run_extended_baseline_in_worker(_DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert result == []

    def test_nonzero_returncode_returns_empty_list(self) -> None:
        proc = _proc(returncode=1, stderr="crash")
        with patch("subprocess.run", return_value=proc):
            result = run_extended_baseline_in_worker(_DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert result == []

    def test_bad_json_returns_empty_list(self) -> None:
        proc = _proc(returncode=0, stdout="not json")
        with patch("subprocess.run", return_value=proc):
            result = run_extended_baseline_in_worker(_DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert result == []

    def test_success_false_in_payload_returns_empty_list(self) -> None:
        proc = _proc(returncode=0, stdout=json.dumps({"success": False}))
        with patch("subprocess.run", return_value=proc):
            result = run_extended_baseline_in_worker(_DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert result == []

    def test_success_path_returns_baselines(self) -> None:
        bench = {"median_us": 80.0, "p20_us": 78.0, "p80_us": 82.0, "p95_us": 85.0}
        payload = json.dumps(
            {
                "success": True,
                "task": "extended_baseline",
                "baselines": [
                    {"name": "torch.compile(reference)", "benchmark": bench, "compile_time_s": 3.2}
                ],
            }
        )
        proc = _proc(returncode=0, stdout=payload)
        with patch("subprocess.run", return_value=proc):
            result = run_extended_baseline_in_worker(_DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert len(result) == 1
        assert result[0].name == "torch.compile(reference)"
        assert result[0].benchmark.median_us == pytest.approx(80.0)
        assert result[0].compile_time_s == pytest.approx(3.2)
        assert result[0].failed is False


class TestWorkerResultFromDict:
    """WorkerResult.from_dict の境界ケース。"""

    def test_success_false_ignores_other_fields(self) -> None:
        r = WorkerResult.from_dict({"success": False, "error": "boom"})
        assert r.success is False
        assert r.error == "boom"
        assert r.correct is False

    def test_success_without_candidate_baseline(self) -> None:
        r = WorkerResult.from_dict(
            {"success": True, "correct": True, "max_abs_diff": 0.0, "failures": []}
        )
        assert r.success is True
        assert r.candidate is None
        assert r.baseline is None

    def test_failures_list_preserved(self) -> None:
        r = WorkerResult.from_dict(
            {
                "success": True,
                "correct": False,
                "max_abs_diff": 0.05,
                "failures": [{"case": "primary", "max_diff": 0.05}],
            }
        )
        assert r.correct is False
        assert len(r.failures) == 1
        assert r.failures[0]["case"] == "primary"
