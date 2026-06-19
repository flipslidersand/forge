# Spec — Forge

## 目的

PyTorch で書いた演算（RMSNorm / Softmax 等）に対して、Triton カーネルの実装方式とパラメータを自動探索し、正確性・測定ノイズ・環境差を考慮した上で最速実装をキャッシュ再利用するライブラリ。

## 解決する問題

- 手書き Triton カーネルは速いが、`BLOCK_SIZE` や `num_warps` の最適値は GPU・dtype・shape によって異なる
- `@triton.autotune` は手動でヒントを書く必要があり、実装方式の探索はできない
- LLM に自由コード生成させると検証が難しく再現性がない
- GPU ベンチマークは測定ノイズが大きく、1 回比較では信頼できない

## MVP の境界線

### やること（MVP 1〜2）

- RMSNorm を対象とした Triton カーネルの自動探索
- `BLOCK_SIZE` / `num_warps` / `num_stages` のグリッドサーチ
- 複数の実装バリアント（single_row / multi_row / fp32_acc 等）の探索
- 正確性検証（エッジケース含む）
- 統計的に安全なベンチマーク（warmup + 中央値 + p20/p80）
- 最速実装の SQLite キャッシュと再利用
- subprocess による安全な実行隔離

### やらないこと（MVP 1〜2）

- 任意の Python 関数の自動 GPU 化
- PyTorch モデル全体の最適化
- LLM による候補コード生成
- Backward / 勾配計算
- マルチ GPU
- CUDA C++ 直接生成

### MVP 3 以降（将来）

- LLM による構造化候補生成
- `@forge.optimize` デコレータ
- Softmax 等への対応拡張
- Bayesian / Successive Halving 探索

## ユーザーが使うコマンドのイメージ

```python
# MVP 1: 直接 API
from forge import optimize_rmsnorm

result = optimize_rmsnorm(
    shapes=[(2048, 4096), (1024, 4096)],
    dtype=torch.float16,
    search_budget=50,
)
print(result.best_params)      # {"block_size": 2048, "num_warps": 8, ...}
print(result.speedup_vs_eager) # 1.34x

# MVP 4: デコレータ（将来）
@forge.optimize(budget=50)
def rmsnorm(x, weight, eps=1e-6):
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight
```

```bash
# CLI（将来）
forge search rmsnorm --shapes 2048x4096 --dtype fp16 --budget 50
forge cache list
forge cache clear
```

## 成功条件

| Phase        | 条件                                                                        |
| ------------ | --------------------------------------------------------------------------- |
| Phase 1 完了 | KernelSpec → CacheKey → SQLite の往復が動く。テスト全通過                   |
| Phase 2 完了 | RMSNorm 50 候補を探索し、PyTorch Eager 比 +20% 以上の実装をキャッシュできる |
| Phase 3 完了 | 複数バリアントを含む探索で、autotune 相当以上の性能が出る                   |
| Phase 4 完了 | `@forge.optimize` デコレータが 2 回目以降ゼロ探索で動く                     |
