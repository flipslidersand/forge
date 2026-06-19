# Implementation Guide — Forge

## Phase 1 — IR とキャッシュ基盤（Week 1）

**ゴール**: KernelSpec → CacheKey → SQLite の往復が動く。コード生成・GPU 不要。

### 実装ステップ

1. `src/forge/ir/tensor_spec.py` — `TensorSpec` dataclass
2. `src/forge/ir/kernel_spec.py` — `KernelSpec` dataclass + `validate()`
3. `src/forge/ir/hashing.py` — `hash_constants()` / `hash_kernel_spec()`
4. `src/forge/cache/key.py` — `CacheKey.from_spec_and_env()`
5. `src/forge/cache/repository.py` — `KernelRepository` (get / put / list)
6. `tests/test_ir.py` — TensorSpec / KernelSpec の単体テスト
7. `tests/test_cache.py` — put → get のラウンドトリップテスト

### 完成確認コマンド

```bash
pytest tests/test_ir.py tests/test_cache.py -v
# → 全テスト PASSED
```

### 難所

- `CacheKey` のハッシュは JSON シリアライズ → SHA256。`torch.dtype` を文字列化する際に `str(torch.float16)` → `"torch.float16"` を使う（`__repr__` に依存しないこと）
- SQLite の `created_at` は UTC ISO 8601 で統一する

---

## Phase 2 — コード生成とベンチマーク（Week 2）

**ゴール**: RMSNorm の Triton コードを生成してベンチマークが動く。GPU 必須。

### 実装ステップ

1. `src/forge/codegen/triton_codegen.py` — `generate_rmsnorm(spec, params) -> str`
2. `src/forge/codegen/templates/rmsnorm.py.jinja` — Triton テンプレート
3. `src/forge/benchmark/timer.py` — `measure(fn, *args) -> BenchmarkResult`
4. `src/forge/benchmark/statistics.py` — `is_improvement(candidate, baseline)`
5. `src/forge/runtime/worker.py` — subprocess ワーカー（コンパイル・実行）
6. `src/forge/runtime/launcher.py` — ワーカー管理・タイムアウト
7. `tests/test_codegen.py` — 生成コードが構文エラーなしで import できること
8. `tests/test_benchmark.py` — measure() が BenchmarkResult を返すこと

### 完成確認コマンド

```bash
python -c "
from forge.codegen.triton_codegen import generate_rmsnorm
from forge.ir.kernel_spec import KernelSpec, TensorSpec
import torch

spec = KernelSpec(
    op_type='rmsnorm',
    input_specs=(TensorSpec((2048, 4096), torch.float16, True),),
    output_specs=(TensorSpec((2048, 4096), torch.float16, True),),
    constants={'eps': 1e-6},
    graph_hash='rmsnorm_v1',
    constraints=(),
)
params = {'block_size': 1024, 'num_warps': 8, 'num_stages': 2, 'acc_dtype': 'fp32', 'variant': 'single_row'}
code = generate_rmsnorm(spec, params)
print(code[:200])
"
```

### 難所

- subprocess ワーカー内で CUDA illegal memory access が起きても親プロセスが死なない設計が必須
- ワーカーは JSON で入出力。Tensor は shape/dtype/値のリストで受け渡す
- `torch.cuda.Event` を使った GPU タイマーは `synchronize()` を忘れると測定値が0になる

---

## Phase 3 — 探索とバリデーション（Week 3）

**ゴール**: 50 候補を探索し、PyTorch Eager 比 +20% 以上の実装をキャッシュできる。

### 実装ステップ

1. `src/forge/search/space.py` — `SearchSpace` + `filter_by_gpu()`
2. `src/forge/search/grid.py` — `GridSearch.generate(space) -> list[SearchParams]`
3. `src/forge/validation/test_cases.py` — エッジケース生成（NaN / Inf / 非連続 / 端数 shape）
4. `src/forge/validation/correctness.py` — `validate(candidate_fn, reference_fn, test_cases)`
5. `src/forge/orchestrator.py` — パイプライン統括
6. `examples/rmsnorm_search.py` — 動作デモ
7. `tests/test_correctness.py`

### 完成確認コマンド

```bash
python examples/rmsnorm_search.py
# → 出力例:
# Searching 48 candidates...
# [12/48] block_size=1024 num_warps=8 → 42.3µs (baseline: 68.1µs) ✓ BEST
# ...
# Best: block_size=1024 num_warps=8 num_stages=2 acc_dtype=fp32
# Speedup vs PyTorch Eager: 1.38x
# Cached to ~/.forge/cache.db
```

### 難所

- `is_improvement()` は `p80(candidate) < p20(baseline) / 1.03` で判定。単純な median 比較は避ける
- 非連続 Tensor のテストは `x = torch.randn(2048, 4096 * 2)[:, ::2]` のように作る
- RMSNorm の縮約誤差は fp16 と fp32 で出やすい。`atol=1e-3` が現実的な許容値

---

## Phase 4 — 複数バリアント（Week 4〜5）

**ゴール**: `autotune` 相当以上の性能。実装方式の違いが探索できる。

### バリアント一覧

| variant        | 内容                                        |
| -------------- | ------------------------------------------- |
| `single_row`   | 1 program = 1 行（基本実装）                |
| `multi_row`    | 1 program = N 行（小さい hidden_size 向け） |
| `two_pass`     | 2 パス reduction（大きい hidden_size 向け） |
| `fused_weight` | weight 乗算を kernel 内で融合               |
| `fp32_acc`     | FP32 accumulator（精度優先）                |

### 実装ステップ

1. `src/forge/codegen/templates/rmsnorm_multi_row.py.jinja`
2. `src/forge/codegen/templates/rmsnorm_two_pass.py.jinja`
3. `src/forge/search/space.py` に `variant` 軸を追加
4. `src/forge/search/random_search.py` — ランダムサーチ（グリッドが大きくなるため）

### 完成確認

```bash
python examples/rmsnorm_search.py --variants all --budget 100
# → バリアントを含む探索が動き、最速バリアントを報告
```

---

## Phase 5 — LLM 候補生成（Week 6+）

**ゴール**: LLM が構造化命令を出し、探索候補を拡張できる。

### 設計方針

LLM には自由なコードを書かせず、以下の JSON を出力させる：

```json
{
  "base_variant": "single_row",
  "params": { "block_size": 4096, "num_warps": 8, "acc_dtype": "fp32" },
  "hypothesis": "hidden_size=4096 では block_size=4096 が L2 キャッシュに収まる"
}
```

### 実装ステップ

1. `src/forge/search/candidate.py` — `CandidateGenerator` Protocol 定義
2. `src/forge/search/llm_generator.py` — `LLMGenerator(CandidateGenerator)`
3. Grid/Random を `CandidateGenerator` に統一リファクタ

---

## Phase 6 — `@forge.optimize` デコレータ（Week 7+）

```python
@forge.optimize(budget=50)
def rmsnorm(x, weight, eps=1e-6):
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight
```

1. `src/forge/decorator.py` — `optimize()` デコレータ
2. `src/forge/lowering/registry.py` — `torch.fx` グラフ → `op_type` パターンマッチ
3. `src/forge/lowering/rmsnorm.py` — RMSNorm パターン → `KernelSpec`

---

## 実装順序の根拠

```
Phase 1 (IR + Cache)
  理由: データ構造が固まらないとコード生成・検証・ベンチマーク全部が場当たり的になる

Phase 2 (Codegen + Benchmark)
  理由: GPU が必要なコンポーネント。早めに動かして測定ノイズの実態を把握する

Phase 3 (Search + Validation)
  理由: Phase 1〜2 の上に乗せるだけ。ここで初めて end-to-end が動く

Phase 4 (Variants)
  理由: パラメータ探索だけでは限界がある。実装方式の違いが差別化になる

Phase 5〜6 (LLM + Decorator)
  理由: 基盤が安定してから載せる。最初から LLM 前提にすると検証が難しい
```

---

## リスクと対策

| リスク                                        | 対策                                                 |
| --------------------------------------------- | ---------------------------------------------------- |
| GPU 測定ノイズで誤判定                        | p80 < p20/1.03 の統計判定。3% 未満改善は採用しない   |
| CUDA illegal memory access で親プロセスが死ぬ | 候補ごとに subprocess ワーカーを使い捨て             |
| キャッシュキー衝突                            | 環境情報 (Triton/CUDA/torch バージョン) を全部含める |
| RMSNorm の縮約誤差                            | `atol=1e-3, rtol=1e-2` で演算順序の差を許容          |
| 探索時間が長すぎる                            | GPU 制約で不可能な候補を事前除外。`filter_by_gpu()`  |
