from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

from forge.cache.repository import KernelRepository
from forge.codegen.triton_codegen import generate
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.lowering import identify
from forge.orchestrator import Orchestrator
from forge.runtime.loader import load_kernel_fn
from forge.search.candidate import CandidateGenerator


def optimize(
    budget: int = 50,
    backend: str = "triton",
    objective: str = "latency",
    *,
    repo: KernelRepository | None = None,
    search: CandidateGenerator | None = None,
    min_speedup: float = 1.03,
    python_executable: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """純粋な PyTorch 演算を最速の Triton 実装へ自動置換するデコレータ。

    初回（shape ごと）: torch.fx で op_type を判定 → KernelSpec を組み立て →
    探索・検証・ベンチ・キャッシュ（Orchestrator）→ 最速カーネルを in-process ロード。
    2 回目以降（同 shape）: in-process キャッシュから直接実行。新しい shape は
    再探索するが SQLite キャッシュにヒットすれば即座に返る。

    判定不能・最適化で速くならない場合は元の eager 関数にフォールバックする。
    backend/objective は将来拡張用の予約引数（現状は triton/latency 固定）。
    """

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(fn)
        # shape シグネチャ -> in-process カーネル (or None=eager)
        compiled: dict[tuple, Callable[..., Any] | None] = {}
        op_type_box: list[
            str | None
        ] = []  # 一度だけ判定（[] 未判定 / [None] 不可 / [str] 判定済み）

        def _resolve_op_type() -> str | None:
            if not op_type_box:
                op_type_box.append(identify(fn) if backend == "triton" else None)
            return op_type_box[0]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import torch

            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            tensors = [v for v in bound.arguments.values() if isinstance(v, torch.Tensor)]
            constants = {
                k: v for k, v in bound.arguments.items() if not isinstance(v, torch.Tensor)
            }

            # GPU テンソルが無い / op 判定不能 → eager
            op_type = _resolve_op_type()
            if not tensors or op_type is None or not tensors[0].is_cuda:
                return fn(*args, **kwargs)

            key = tuple((tuple(t.shape), str(t.dtype)) for t in tensors)
            if key not in compiled:
                compiled[key] = _build(
                    op_type,
                    tensors,
                    constants,
                    budget,
                    repo,
                    search,
                    min_speedup,
                    python_executable,
                    progress,
                )
            kfn = compiled[key]
            if kfn is None:
                return fn(*args, **kwargs)
            return kfn(*tensors, **constants)

        wrapper._forge_compiled = compiled  # type: ignore[attr-defined]  # テスト/内省用
        return wrapper

    return deco


def _build(
    op_type: str,
    tensors: list[Any],
    constants: dict[str, Any],
    budget: int,
    repo: KernelRepository | None,
    search: CandidateGenerator | None,
    min_speedup: float,
    python_executable: str | None,
    progress: Callable[[str], None] | None,
) -> Callable[..., Any] | None:
    """この shape に対して最速カーネルを探索し、in-process 実行関数を返す（無ければ None）。"""
    input_specs = tuple(TensorSpec.from_tensor(t) for t in tensors)
    out = TensorSpec.from_tensor(tensors[0])
    spec = KernelSpec(
        op_type=op_type,
        input_specs=input_specs,
        output_specs=(out,),
        constants=constants,
        graph_hash=f"{op_type}_v1",
        constraints=(),
    )

    orch = Orchestrator(
        repo=repo,
        min_speedup=min_speedup,
        python_executable=python_executable,
        progress=progress or (lambda _m: None),
    )
    result = orch.optimize(spec, budget=budget, search=search)
    if result.best_params is None:
        return None
    code = generate(spec, result.best_params)
    return load_kernel_fn(code)
