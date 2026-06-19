# Data Model — Forge

## コアデータ構造

### TensorSpec

```python
@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype
    is_contiguous: bool
```

### KernelSpec（IR 中間表現）

```python
@dataclass(frozen=True)
class KernelSpec:
    op_type: str                         # "rmsnorm" | "softmax"
    input_specs: tuple[TensorSpec, ...]
    output_specs: tuple[TensorSpec, ...]
    constants: dict[str, object]         # eps, dim 等
    graph_hash: str                      # 演算構造の識別子
    constraints: tuple[str, ...]         # "hidden_size % 128 == 0" 等
```

### SearchParams（候補パラメータ）

```python
@dataclass(frozen=True)
class SearchParams:
    block_size: int       # 512 | 1024 | 2048 | 4096
    num_warps: int        # 4 | 8 | 16
    num_stages: int       # 1 | 2 | 3 | 4
    acc_dtype: str        # "fp32" | "fp16"
    variant: str          # "single_row" | "multi_row" | "two_pass"
```

### CacheKey（キャッシュの一意キー）

```python
@dataclass(frozen=True)
class CacheKey:
    graph_hash: str
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    constants_hash: str
    compute_capability: str   # "8.9"
    torch_version: str
    triton_version: str
    cuda_version: str
    library_version: str      # forge 自身のバージョン
```

キャッシュキーは `sha256(json.dumps(asdict(key), sort_keys=True))` でハッシュ化して SQLite の PRIMARY KEY に使う。

### BenchmarkResult

```python
@dataclass
class BenchmarkResult:
    samples_us: list[float]   # µs 単位の生データ（200 サンプル）
    median_us: float
    p20_us: float
    p80_us: float
    warmup_count: int
    measure_count: int
```

### ValidationResult

```python
@dataclass
class ValidationResult:
    passed: bool
    failures: list[dict]      # [{"case": "nan_input", "max_diff": 0.003}, ...]
    checked_cases: int
```

### CachedKernel（SQLite に保存するエンティティ）

```python
@dataclass
class CachedKernel:
    cache_key: CacheKey
    params: SearchParams
    kernel_code: str          # Triton ソースコード（文字列）
    benchmark: BenchmarkResult
    created_at: datetime
```

### ExperimentResult（探索ログ）

```python
@dataclass
class ExperimentResult:
    params: SearchParams
    validation: ValidationResult
    benchmark: BenchmarkResult | None   # 検証失敗時は None
    is_best: bool
```

---

## 状態遷移

### 候補の評価フロー

```
PENDING
  │
  ├─[コード生成成功]──→ GENERATED
  │                        │
  │                        ├─[正確性OK]──→ VALIDATED
  │                        │                  │
  │                        │                  ├─[改善あり]──→ BEST_CANDIDATE
  │                        │                  └─[改善なし]──→ REJECTED
  │                        │
  │                        └─[正確性NG]──→ CORRECTNESS_FAILED
  │
  └─[コード生成失敗]──→ CODEGEN_FAILED
```

### キャッシュヒットフロー

```
呼び出し
  │
  ├─[CacheKey ヒット]──→ CACHE_HIT ──→ 最適実装を直接実行
  │
  └─[ミス]──→ SEARCH ──→ VALIDATED ──→ CACHE_STORE ──→ 最適実装を実行
```

---

## SQLite スキーマ

```sql
CREATE TABLE kernels (
    cache_key_hash  TEXT     PRIMARY KEY,
    cache_key_json  TEXT     NOT NULL,
    params_json     TEXT     NOT NULL,
    kernel_code     TEXT     NOT NULL,
    benchmark_json  TEXT     NOT NULL,
    created_at      TEXT     NOT NULL   -- ISO 8601
);

CREATE TABLE experiments (
    id              INTEGER  PRIMARY KEY AUTOINCREMENT,
    cache_key_hash  TEXT     NOT NULL,
    params_json     TEXT     NOT NULL,
    validation_json TEXT     NOT NULL,
    benchmark_json  TEXT,               -- NULL = 検証失敗
    is_best         INTEGER  NOT NULL,  -- 0 or 1
    created_at      TEXT     NOT NULL
);

CREATE INDEX idx_experiments_key ON experiments(cache_key_hash);
```

---

## パイプライン全体のデータフロー

```
PyTorch 関数
    │
    ▼
KernelSpec          ← ir/ で定義
    │
    ├──→ CacheKey   ← cache/key.py
    │       │
    │       └──→ SQLite 検索
    │               │
    │          [ヒット] → CachedKernel → 実行
    │          [ミス]  ↓
    │
    ├──→ SearchSpace → [SearchParams, ...]
    │
    └──→ (SearchParams × KernelSpec)
              │
              ▼
          Triton コード文字列
              │
              ├──→ subprocess Worker → ValidationResult
              │
              └──→ subprocess Worker → BenchmarkResult
                                           │
                                     is_improvement?
                                           │
                                    [Yes] → CachedKernel → SQLite 保存
```
