from __future__ import annotations

import importlib.util
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path


def load_kernel_fn(code: str) -> Callable[..., object]:
    """生成された Triton モジュール文字列を一時 .py として import し kernel_fn を返す。

    @triton.jit は inspect でソースをファイルから読むため、実在するファイル経由で
    import する必要がある（インライン exec は不可 — Issue #3）。worker（subprocess）と
    デコレータ（in-process）の両方がこれを使う。in-process 実行は、キャッシュ済みの
    検証通過カーネルのみを対象とすること。
    """
    tmp_dir = Path(tempfile.gettempdir()) / "forge_kernels"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mod_path = tmp_dir / f"kernel_{uuid.uuid4().hex}.py"
    mod_path.write_text(code)
    spec = importlib.util.spec_from_file_location(mod_path.stem, str(mod_path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.kernel_fn
