"""subprocess worker のエントリポイント。

親から stdin に JSON を受け取り、生成カーネルをコンパイル・複数ケースで正確性
検証・ベンチマークして結果を stdout に JSON で返す。CUDA illegal memory access 等で
このプロセスが死んでも親は生き残る（ADR-003）。

@triton.jit は inspect でソースをファイルから読むため、カーネルコードは必ず実在する
一時 .py ファイルとして import する（Issue #3 の申し送り）。

実行: <python> -m forge.runtime._worker_entry  < payload.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any


def _build_tensor(spec: dict[str, Any], torch: Any):
    dtype = getattr(torch, spec["dtype"])
    shape = tuple(spec["shape"])
    init = spec.get("init", "randn")
    scale = float(spec.get("scale", 1.0))
    seed = int(spec.get("seed", 0))
    if init == "randn":
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return torch.randn(shape, dtype=dtype, device="cuda", generator=gen) * scale
    if init == "ones":
        return torch.ones(shape, dtype=dtype, device="cuda") * scale
    if init == "zeros":
        return torch.zeros(shape, dtype=dtype, device="cuda")
    raise ValueError(f"unknown init: {init}")


def _load_kernel_fn(code: str):
    tmp_dir = Path(tempfile.gettempdir()) / "forge_workers"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mod_path = tmp_dir / f"kernel_{uuid.uuid4().hex}.py"
    mod_path.write_text(code)
    spec = importlib.util.spec_from_file_location(mod_path.stem, str(mod_path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.kernel_fn


def main() -> None:
    payload = json.loads(sys.stdin.read())
    try:
        import torch

        from forge.benchmark.timer import measure
        from forge.runtime.reference import baseline_name, get_baseline, get_reference

        op_type = payload["op_type"]
        constants = payload.get("constants", {})
        task = payload.get("task", "full")
        tol = payload.get("tolerance", {"atol": 2e-3, "rtol": 1e-2, "equal_nan": False})

        kernel_fn = _load_kernel_fn(payload["kernel_code"])
        reference = get_reference(op_type)

        # --- 正確性検証（複数ケース） ---
        cases = payload.get("correctness_cases") or [
            {"name": "primary", "input_specs": payload["benchmark_input"]}
        ]
        failures: list[dict[str, Any]] = []
        max_abs_diff = 0.0
        for case in cases:
            tensors = [_build_tensor(s, torch) for s in case["input_specs"]]
            out_c = kernel_fn(*tensors, **constants)
            out_r = reference(*tensors, **constants)
            torch.cuda.synchronize()
            diff = (out_c.float() - out_r.float()).abs().max().item()
            max_abs_diff = max(max_abs_diff, diff)
            ok = bool(
                torch.allclose(
                    out_c.float(),
                    out_r.float(),
                    atol=float(tol["atol"]),
                    rtol=float(tol["rtol"]),
                    equal_nan=bool(tol.get("equal_nan", False)),
                )
            )
            if not ok:
                failures.append({"case": case["name"], "max_diff": diff})

        correct = len(failures) == 0
        result: dict[str, Any] = {
            "success": True,
            "correct": correct,
            "max_abs_diff": max_abs_diff,
            "failures": failures,
        }

        # --- ベンチマーク（正確な候補のみ） ---
        if task in ("benchmark", "full") and correct:
            warmup = int(payload.get("warmup", 25))
            repeat = int(payload.get("repeat", 200))
            bench_tensors = [_build_tensor(s, torch) for s in payload["benchmark_input"]]
            baseline = get_baseline(op_type)
            cand = measure(lambda: kernel_fn(*bench_tensors, **constants), warmup, repeat)
            base = measure(lambda: baseline(*bench_tensors, **constants), warmup, repeat)
            result["candidate"] = cand.to_dict()
            result["baseline"] = base.to_dict()
            result["baseline_name"] = baseline_name(op_type)

        print(json.dumps(result))
    except Exception as e:  # noqa: BLE001 — worker は何が起きても JSON を返す
        import traceback

        print(
            json.dumps(
                {
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
            )
        )


if __name__ == "__main__":
    main()
