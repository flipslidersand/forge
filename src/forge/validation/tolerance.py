from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float
    equal_nan: bool = False

    def to_dict(self) -> dict[str, float | bool]:
        return {"atol": self.atol, "rtol": self.rtol, "equal_nan": self.equal_nan}


# op_type ごとの許容誤差。
# rmsnorm の atol は #3/#4 の実測（fp16 で max_diff ≈ 1.95e-3）に基づき 2e-3 とする。
# 1e-3 では正しい fp32-accumulate 実装を誤って落とすため厳しすぎる。
TOLERANCE: dict[str, Tolerance] = {
    "rmsnorm": Tolerance(atol=2e-3, rtol=1e-2, equal_nan=False),
    # softmax 出力は [0,1]、fp16 分解能 ~1e-3。fp32 縮約実装を通すため 2e-3。
    "softmax": Tolerance(atol=2e-3, rtol=1e-2, equal_nan=False),
    # layernorm は rmsnorm 同様 fp16 縮約誤差で 2e-3。gelu は exact(erf) で誤差小。
    "layernorm": Tolerance(atol=2e-3, rtol=1e-2, equal_nan=False),
    "gelu": Tolerance(atol=2e-3, rtol=1e-2, equal_nan=False),
}


def get_tolerance(op_type: str) -> Tolerance:
    if op_type not in TOLERANCE:
        raise ValueError(f"No tolerance defined for op_type={op_type!r}")
    return TOLERANCE[op_type]
