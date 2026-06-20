"""subprocess worker のエントリポイント。

親プロセスから stdin に JSON を受け取り、生成カーネルをコンパイル・検証・
ベンチマークして結果を stdout に JSON で返す。CUDA illegal memory access 等で
このプロセスが死んでも親は生き残る（ADR-003）。

@triton.jit は inspect でソースをファイルから読むため、カーネルコードは必ず
実在する一時 .py ファイルとして import する（Issue #3 の申し送り）。

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
    seed = int(spec.get("seed", 0))
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if init == "randn":
        return torch.randn(shape, dtype=dtype, device="cuda", generator=gen)
    if init == "ones":
        return torch.ones(shape, dtype=dtype, device="cuda")
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
        from forge.runtime.reference import get_reference

        tensors = [_build_tensor(s, torch) for s in payload["input_specs"]]
        constants = payload.get("constants", {})
        task = payload.get("task", "full")

        kernel_fn = _load_kernel_fn(payload["kernel_code"])
        reference = get_reference(payload["op_type"])

        out_candidate = kernel_fn(*tensors, **constants)
        out_reference = reference(*tensors, **constants)
        torch.cuda.synchronize()

        tol = payload.get("tolerance", {"atol": 2e-3, "rtol": 1e-2})
        max_abs_diff = (out_candidate.float() - out_reference.float()).abs().max().item()
        correct = bool(
            torch.allclose(
                out_candidate.float(),
                out_reference.float(),
                atol=float(tol["atol"]),
                rtol=float(tol["rtol"]),
            )
        )

        result: dict[str, Any] = {
            "success": True,
            "correct": correct,
            "max_abs_diff": max_abs_diff,
        }

        if task in ("benchmark", "full") and correct:
            warmup = int(payload.get("warmup", 25))
            repeat = int(payload.get("repeat", 200))
            cand = measure(lambda: kernel_fn(*tensors, **constants), warmup, repeat)
            base = measure(lambda: reference(*tensors, **constants), warmup, repeat)
            result["candidate"] = cand.to_dict()
            result["baseline"] = base.to_dict()

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
