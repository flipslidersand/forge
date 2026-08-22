from __future__ import annotations

from typing import TYPE_CHECKING, Any

from forge.ir.kernel_spec import KernelSpec
from forge.search._base_generator import _BaseGenerator
from forge.search._proposal_models import Proposal
from forge.search.llm_generator import build_prompt

if TYPE_CHECKING:
    from forge.search.candidate import HistoryEntry

_DEFAULT_MODEL = "qwen2.5-coder:latest"
_DEFAULT_HOST = "http://localhost:11434"

_SYSTEM = (
    "You are a GPU kernel autotuning assistant. "
    "Propose Triton kernel configurations as structured JSON. "
    "Do NOT write code — only structured parameters."
)


class OllamaGenerator(_BaseGenerator):
    """ローカル ollama サーバーを使った CandidateGenerator。

    ANTHROPIC_API_KEY 不要。`qwen2.5-coder` など JSON 出力を得意とするモデルを推奨。
    ollama の ``format`` パラメータで Pydantic スキーマを強制するため、
    JSON パース失敗のリスクが低い。

    ``generate()`` のシグネチャは ``LLMGenerator`` と同一なので、
    ``Orchestrator.optimize_rounds()`` にそのまま渡せる。

    Args:
        model: ollama モデル名（``ollama list`` で確認）。
        host: ollama サーバーアドレス。デフォルトは ``http://localhost:11434``。
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        host: str = _DEFAULT_HOST,
    ) -> None:
        self.model = model
        self.host = host

    def _propose(
        self,
        spec: KernelSpec,
        compute_capability: str,
        n: int,
        history: list[HistoryEntry],
    ) -> list[dict[str, Any]]:
        import ollama  # type: ignore[import]  # ollama は py.typed 未対応

        prompt = build_prompt(spec, compute_capability, n, history)
        try:
            resp = ollama.Client(host=self.host).chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                format=Proposal.model_json_schema(),
            )
            content = resp.message.content or ""
            proposal = Proposal.model_validate_json(content)
            return [c.model_dump() for c in proposal.candidates]
        except Exception:  # noqa: BLE001 — ollama connection/JSON parse failure → return empty candidates
            return []
