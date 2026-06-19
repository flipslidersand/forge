# ADR-001: Triton を CUDA C++ より優先する

- **日付**: 2026-06-19
- **状態**: Accepted

## 背景

GPU カーネルを自動生成する際、記述言語として CUDA C++、Triton、OpenAI の cuTile 等が候補になる。
LLM による候補生成・検証・デバッグのしやすさが重要な評価軸。

## 決定

カーネル記述言語として **Triton** を採用する。

## 理由

- Triton は Python で書けるため、LLM が生成・編集しやすい
- コンパイル結果が PTX / SASS に変換されるため、性能は CUDA C++ と同等レベルに達する
- `triton.compile` の API が安定しており、subprocess 内でのコンパイルが容易
- PyTorch 2.x の `torch.compile` バックエンドとして公式採用済みで、将来の統合が自然
- CUDA C++ は Jinja テンプレートの複雑さと LLM 生成の検証コストが Triton の 3 倍以上になる

## トレードオフ

- Triton では表現できない低レベル最適化（warp shuffle の細かい制御、PTX 直書き等）は使えない
- cuBLAS / cuDNN のような高度にチューニングされたライブラリには基本的に勝てない（GEMM 等）
- Triton のバージョン差で API が変わりやすい（compute capability ごとの挙動差に注意）
