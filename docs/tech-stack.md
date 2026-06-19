# Tech Stack — Forge

## 言語・バージョン

| 項目    | バージョン |
| ------- | ---------- |
| Python  | 3.11+      |
| PyTorch | 2.3+       |
| Triton  | 3.0+       |
| CUDA    | 12.1+      |

## 主要ライブラリと選定理由

| ライブラリ   | 用途                         | 選定理由                                             |
| ------------ | ---------------------------- | ---------------------------------------------------- |
| `triton`     | GPU カーネル記述・コンパイル | Python で書けるため LLM 生成に向く。PTX より高レベル |
| `torch.fx`   | PyTorch グラフ取得           | 公式サポート。`torch.export` より安定している        |
| `jinja2`     | Triton コードテンプレート    | ロジックとテンプレートを分離できる                   |
| `sqlite3`    | カーネルキャッシュ           | 標準ライブラリ。外部サービス不要。単一ファイル管理   |
| `pytest`     | テストフレームワーク         | デファクトスタンダード                               |
| `hypothesis` | プロパティベーステスト       | 正確性検証のエッジケース自動生成                     |
| `ruff`       | Linter / Formatter           | 高速。black + isort + flake8 を一括代替              |
| `pyright`    | 型チェック                   | strict モードで dataclass の型安全性を保証           |

## ビルドツール・実行環境

```
pyproject.toml (hatchling)
  └─ src/forge/       # パッケージ本体
  └─ tests/           # pytest
  └─ examples/        # 実行デモ

仮想環境: venv または uv venv
GPU 環境: NVIDIA GPU (compute capability 7.0+)
```

## 開発ツール

```bash
uv pip install -e ".[dev]"   # 開発用インストール
ruff check src/              # Lint
ruff format src/             # Format
pyright src/                 # 型チェック
pytest tests/ -v             # テスト
```

## 依存関係の構造

```
forge
├── torch          (PyTorch - グラフ取得・参照実装・ベンチマーク)
├── triton         (カーネルコンパイル・実行)
├── jinja2         (コード生成テンプレート)
└── sqlite3        (標準ライブラリ - キャッシュ)

dev
├── pytest
├── hypothesis
├── ruff
└── pyright
```
