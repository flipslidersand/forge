from __future__ import annotations

from collections.abc import Callable
from typing import Any

from forge.ir.kernel_spec import KernelSpec

from .candidate import HistoryEntry
from .params import SUPPORTED_ACC_DTYPES, SUPPORTED_VARIANTS, SearchParams

# LLM に候補を出させる際の戻り値型: 構造化された dict のリスト。
ProposeFn = Callable[[KernelSpec, str, int, list[HistoryEntry]], list[dict[str, Any]]]

DEFAULT_MODEL = "claude-opus-4-8"

_SYSTEM = (
    "You are a GPU kernel autotuning assistant. You propose Triton kernel "
    "configurations for an RMSNorm kernel and a one-line hypothesis for each. "
    "You do NOT write code — only structured parameters. Favor configurations "
    "likely to be both correct and fast, and learn from the provided history."
)


class LLMGenerator:
    """Claude に構造化された探索候補を出させる CandidateGenerator 実装。

    LLM には自由な Triton コードを書かせず、変更命令（variant + パラメータ + 仮説）
    のみを JSON で出力させる（ADR-005）。実際のコード生成はテンプレートが担う。

    API 呼び出しは ``propose_fn`` で差し替え可能。テストや非ネットワーク環境では
    canned な dict を返す関数を注入する。省略時は Anthropic SDK を遅延 import して
    ``claude-opus-4-8`` を呼ぶ（ANTHROPIC_API_KEY が必要）。
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client: Any | None = None,
        default_n: int = 12,
        propose_fn: ProposeFn | None = None,
    ) -> None:
        self.model = model
        self.client = client
        self.default_n = default_n
        self._propose_fn = propose_fn

    def generate(
        self,
        spec: KernelSpec,
        compute_capability: str,
        budget: int | None = None,
        history: list[HistoryEntry] | None = None,
    ) -> list[SearchParams]:
        n = budget or self.default_n
        propose = self._propose_fn or self._propose_via_claude
        raw = propose(spec, compute_capability, n, history or [])

        out: list[SearchParams] = []
        seen: set[SearchParams] = set()
        for d in raw:
            params = self._coerce(d, spec)
            if params is not None and params not in seen:
                seen.add(params)
                out.append(params)
        if budget is not None:
            out = out[:budget]
        return out

    # --- 構造化 dict -> SearchParams（無効な候補は捨てる） ---

    @staticmethod
    def _coerce(d: dict[str, Any], spec: KernelSpec) -> SearchParams | None:
        try:
            params = SearchParams(
                block_size=int(d["block_size"]),
                num_warps=int(d["num_warps"]),
                num_stages=int(d["num_stages"]),
                acc_dtype=str(d.get("acc_dtype", "fp32")),
                variant=str(d.get("base_variant", d.get("variant", "single_row"))),
                rows_per_program=int(d.get("rows_per_program", 1)),
            )
        except (KeyError, ValueError, TypeError):
            return None
        if not _valid_block(params, spec.input_specs[0].shape[-1]):
            return None
        return params

    # --- 実際の Claude 呼び出し（遅延 import） ---

    def _propose_via_claude(
        self,
        spec: KernelSpec,
        compute_capability: str,
        n: int,
        history: list[HistoryEntry],
    ) -> list[dict[str, Any]]:
        import anthropic
        from pydantic import BaseModel

        class Candidate(BaseModel):
            base_variant: str
            block_size: int
            num_warps: int
            num_stages: int
            acc_dtype: str
            rows_per_program: int
            hypothesis: str

        class Proposal(BaseModel):
            candidates: list[Candidate]

        client = self.client or anthropic.Anthropic()
        prompt = build_prompt(spec, compute_capability, n, history)
        resp = client.messages.parse(
            model=self.model,
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_format=Proposal,
        )
        proposal = resp.parsed_output
        return [c.model_dump() for c in proposal.candidates]


def _valid_block(params: SearchParams, n: int) -> bool:
    """variant ごとの block_size 制約（SearchSpace と同じルール）。"""
    if params.variant == "two_pass":
        return params.block_size <= n
    # single_row / multi_row は行全体を 1 タイルに収める
    return params.block_size >= n


def build_prompt(
    spec: KernelSpec,
    compute_capability: str,
    n: int,
    history: list[HistoryEntry],
) -> str:
    x = spec.input_specs[0]
    lines = [
        f"Propose {n} distinct Triton RMSNorm kernel configurations.",
        "",
        f"GPU compute capability: {compute_capability}",
        f"Input shape (rows x hidden): {x.shape}, dtype: {x.dtype_str()}",
        f"eps: {spec.constants.get('eps')}",
        "",
        "Each candidate must specify:",
        f"- base_variant: one of {list(SUPPORTED_VARIANTS)}",
        "- block_size: power of 2. For single_row/multi_row it must be >= hidden "
        f"size ({x.shape[-1]}); for two_pass it is a tile and must be <= hidden size.",
        "- num_warps: typically 4, 8, or 16",
        "- num_stages: 1 on pre-Volta (cc < 7.0), else 1-3",
        f"- acc_dtype: one of {list(SUPPORTED_ACC_DTYPES)}. Note: fp16 accumulation "
        "in the reduction is usually NOT numerically correct for single_row/multi_row.",
        "- rows_per_program: >1 only for multi_row, else 1",
        "- hypothesis: one line explaining why this config may be fast",
    ]
    if history:
        lines += ["", "History of already-evaluated configs (learn from these):"]
        lines += [f"  {h.summary()}" for h in history[-30:]]
    return "\n".join(lines)
