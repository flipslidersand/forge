# forge

PyTorch 演算に対し、Triton カーネルの実装方式とパラメータを自動探索し、
正確性・測定ノイズ・環境差を考慮した上で最速実装をキャッシュ再利用する
GPU カーネル自動最適化エンジン。

```python
import forge
import torch


@forge.optimize(budget=50)
def rmsnorm(x, weight, eps=1e-6):
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight


x = torch.randn(2048, 4096, dtype=torch.float16, device="cuda")
w = torch.ones(4096, dtype=torch.float16, device="cuda")
y = rmsnorm(x, w)  # 初回: 探索してキャッシュ / 2回目以降: 最速カーネルを即実行
```

## 対応演算

RMSNorm / Softmax / LayerNorm / GELU

## 仕組み

1. `@forge.optimize` が関数を torch.fx で trace し op を判定（lowering）
2. 入力テンソルから `KernelSpec` を構築
3. 探索器（Grid / Random / LLM）が候補を生成
4. 各候補を **使い捨て subprocess** でコンパイル・正確性検証・ベンチマーク
   （CUDA エラーで親プロセスが死なない）
5. 統計的に最速（`p80 < p20/1.03`）かつ正確な実装を SQLite にキャッシュ
6. 2 回目以降は環境込み `CacheKey`（torch/triton/cuda/compute-capability）で
   ヒット → 探索ゼロ

## 必要環境

- NVIDIA GPU（compute capability は問わないが、Triton は 7.0+ が公式サポート）
- Python 3.11+ / PyTorch 2.x（CUDA 対応ビルド）/ Triton 3.x

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"   # コア + 開発ツール（pytest, ruff, pyright）
pip install -e ".[llm]"   # LLM 候補生成を使う場合（任意。anthropic + pydantic）
```

> システム Python が externally-managed の場合は venv 必須。
> PyTorch は使用 GPU のドライバに合う CUDA ビルドを入れること
> （例: ドライバが CUDA 12.x なら
> `pip install torch --index-url https://download.pytorch.org/whl/cu121`）。

## 使い方

### デコレータ

```python
@forge.optimize(budget=50)  # budget = 探索候補数の上限
def softmax(x):
    return torch.softmax(x, dim=-1)  # dim は定数で書く（trace 要件）
```

判定できない・最適化で速くならない場合は、元の eager 関数にフォールバックする。

### 探索 API（直接）

```python
from forge.ir.kernel_spec import KernelSpec
from forge.orchestrator import Orchestrator

# KernelSpec を組み立てて orch.optimize(spec, budget=50)
```

動かせる例は `examples/decorator_demo.py` / `examples/rmsnorm_search.py` を参照。

```bash
.venv/bin/python examples/decorator_demo.py
```

### LLM 候補生成（任意）

`forge.search.llm_generator.LLMGenerator` は Claude（`claude-opus-4-8`）に
構造化された候補パラメータを出させる探索器。実 API 利用には `ANTHROPIC_API_KEY`
が必要（`pip install -e ".[llm]"`）。テストは `propose_fn` 注入でオフライン実行できる。

## テスト

```bash
.venv/bin/python -m pytest tests/ -m "not gpu"   # GPU 不要（CPU のみ）
.venv/bin/python -m pytest tests/                # GPU を含む全テスト
.venv/bin/ruff check src/ tests/                 # Lint
```

## ディレクトリ構成

```
src/forge/
  ops.py            op メタデータ（reduction / elementwise）
  ir/               TensorSpec / KernelSpec / hashing
  lowering/         torch.fx グラフ → op_type 判定
  codegen/          KernelSpec + params → Triton コード（Jinja2 テンプレート）
  search/           SearchSpace / GridSearch / RandomSearch / LLMGenerator
  runtime/          subprocess worker / kernel ローダ / 参照実装
  validation/       正確性スイート / 許容誤差
  benchmark/        CUDA Event タイマー / 統計的採用判定
  cache/            CacheKey / SQLite リポジトリ
  orchestrator.py   探索 → 検証 → ベンチ → キャッシュの統括
  decorator.py      @forge.optimize
docs/               spec / data-model / implementation-guide / adr/
examples/           実行デモ
tests/              CPU テスト + GPU テスト（@pytest.mark.gpu）
```

設計判断は `docs/adr/` を参照（Triton 採用、SQLite、subprocess 隔離、
統計的ベンチ判定、LLM 構造化生成）。

## 既知の制約

- 判定できる演算は上記 4 種のみ。未対応・trace 不能（動的 dim 等）は eager フォールバック
- GELU は exact（erf）のみ。tanh 近似の関数は許容誤差を超えて eager になり得る
- 演算は標準的な式の形のみ認識（torch.fx の call_function 多重集合でマッチ）
- 開発・検証は GTX 1080（compute capability 6.1、Triton 公式サポート外）で実施

## ロードマップ

GitHub Issues を参照:

- #9 baseline 拡張（torch.compile / `@triton.autotune` を公平比較に追加）
- #10 ライブ LLM 反復探索（history フィードバックループ）
- #11 探索コスト考慮の採用判定
