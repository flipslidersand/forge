# ADR-003: 生成カーネルの実行を subprocess で隔離する

- **日付**: 2026-06-19
- **状態**: Accepted

## 背景

LLM またはテンプレートで生成した Triton コードを実行する際、CUDA illegal memory access 等のエラーが
発生するとプロセスの CUDA コンテキスト全体が壊れる。また、無限ループや過剰なメモリ確保も起こりうる。

## 決定

候補カーネルの **コンパイル・正確性検証・ベンチマーク**を、毎回使い捨ての subprocess ワーカーで実行する。

```
Orchestrator (親プロセス)
  └─ Worker subprocess (子プロセス・使い捨て)
       ├─ triton.compile
       ├─ correctness_check
       ├─ benchmark
       └─ exit → JSON 結果を stdout に出力
```

## 理由

- CUDA エラーで子プロセスが死んでも親は生き残る
- タイムアウト (`subprocess.run(..., timeout=30)`) で無限ループを確実に終了できる
- 子プロセスごとに CUDA コンテキストがリセットされるため、前の候補の状態が混入しない
- LLM 生成コードの `exec()` を親プロセスで行う必要がなく、セキュリティ面でも分離できる

## トレードオフ

- subprocess 起動コストが候補ごとに発生する（PyTorch import で ~1 秒）
  → 検証フェーズとベンチマークフェーズをまとめて1プロセスで実行することで緩和
- デバッグ時に子プロセスの状態が見えにくい
  → `--debug` フラグ時は subprocess を使わず直接実行するモードを設ける（将来）
