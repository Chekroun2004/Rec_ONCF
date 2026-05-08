# Fast Preprocessor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `ColumnTransformer.transform` on single-row inference (19ms, 45% of latency) with a pre-baked dict-lookup encoder that runs in ~0.1ms — reducing API p50 from 56ms to ~27ms and p99 from 108ms to ~78ms, with no model retraining.

**Architecture:** `FastPreprocessor` is a read-only class built once at startup from a fitted `ColumnTransformer`. It holds ordinal-encoding dicts and numeric column names extracted from the fitted model. `Recommender` stores one as `fast_preprocessor`; when both `onnx_session` and `fast_preprocessor` are set, `recommend()` uses the fast path. The existing `predict_proba_onnx` function is untouched and remains active in tests and any sklearn fallback.

**Tech Stack:** Python 3.12, numpy, sklearn (read-only at inference), onnxruntime, pandas (still used in `compute_inference_row` — only the *transform* step is replaced).

---

### Task 1: `FastPreprocessor` class + tests

**Files:**
- Modify: `src/rec_oncf/training.py` (add class after `predict_proba_onnx`)
- Modify: `tests/test_onnx.py` (add 3 new tests at the bottom)

---

- [ ] **Step 1: Write 3 failing tests in `tests/test_onnx.py`**

Add these three tests at the bottom of the file. They import `FastPreprocessor` which doesn't exist yet, so they will fail at import.

```python
from rec_oncf.training import FastPreprocessor


def test_fast_preprocessor_matches_sklearn():
    """FastPreprocessor.encode must produce the same float32 array as ColumnTransformer.transform."""
    pipe, row, _ = _make_pipeline_and_row()
    ct = pipe["pre"]
    fp = FastPreprocessor(ct)

    extra = ["LiaisonId", "CodeClient", "DateHeureDepartVoyageSegment"]
    row_clean = row.drop(columns=[c for c in extra if c in row.columns])
    X_sklearn = ct.transform(row_clean).astype(np.float32)

    row_dict = row.iloc[0].to_dict()
    X_fast = fp.encode(row_dict)

    np.testing.assert_array_equal(X_sklearn, X_fast)


def test_fast_preprocessor_unknown_category():
    """An unseen category value must encode to -1.0 (matching OrdinalEncoder unknown_value=-1)."""
    pipe, row, _ = _make_pipeline_and_row()
    fp = FastPreprocessor(pipe["pre"])
    cat_cols = list(pipe["pre"].transformers_[0][2])

    row_dict = row.iloc[0].to_dict()
    row_dict["TypeParcoursId"] = "UNSEEN_VALUE_XYZ"

    X_fast = fp.encode(row_dict)
    type_idx = cat_cols.index("TypeParcoursId")
    assert X_fast[0, type_idx] == -1.0


def test_fast_preprocessor_nan_numeric_passthrough():
    """NaN in a numeric column must survive as np.nan in the output array."""
    pipe, row, _ = _make_pipeline_and_row()
    fp = FastPreprocessor(pipe["pre"])
    cat_cols = list(pipe["pre"].transformers_[0][2])
    num_cols = list(pipe["pre"].transformers_[1][2])

    row_dict = row.iloc[0].to_dict()
    row_dict["PrixParLiaison"] = float("nan")

    X_fast = fp.encode(row_dict)
    prix_idx = len(cat_cols) + num_cols.index("PrixParLiaison")
    assert np.isnan(X_fast[0, prix_idx])
```

- [ ] **Step 2: Run tests to confirm they fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_onnx.py::test_fast_preprocessor_matches_sklearn tests/test_onnx.py::test_fast_preprocessor_unknown_category tests/test_onnx.py::test_fast_preprocessor_nan_numeric_passthrough -v
```

Expected: `ImportError: cannot import name 'FastPreprocessor' from 'rec_oncf.training'`

- [ ] **Step 3: Add `FastPreprocessor` to `src/rec_oncf/training.py`**

Add this class directly after the `predict_proba_onnx` function (after line ~215):

```python
class FastPreprocessor:
    """Drop-in for ColumnTransformer.transform — ~200x faster on single rows.

    Built once at startup from a fitted ColumnTransformer. Encodes a feature
    dict directly to a float32 numpy array via pre-extracted ordinal lookup
    tables and numeric passthrough, bypassing all pandas overhead.

    Column order matches ColumnTransformer output: cat-encoded columns first
    (in transformer order), then passthrough numeric columns.
    """

    def __init__(self, ct: ColumnTransformer) -> None:
        _, cat_enc, cat_cols = ct.transformers_[0]   # ("cat", OrdinalEncoder, [...])
        _, _, num_cols = ct.transformers_[1]          # ("num", "passthrough", [...])

        self._cat_cols: list[str] = list(cat_cols)
        self._num_cols: list[str] = list(num_cols)
        # {col: {str(value): float(ordinal_index)}}
        self._cat_maps: dict[str, dict[str, float]] = {
            col: {str(v): float(i) for i, v in enumerate(cat_enc.categories_[j])}
            for j, col in enumerate(self._cat_cols)
        }
        self._n_features: int = len(self._cat_cols) + len(self._num_cols)

    def encode(self, row: dict) -> np.ndarray:
        """Encode one feature dict → float32 array of shape (1, n_features).

        Unknown category values → -1.0 (matching OrdinalEncoder unknown_value=-1).
        NaN / None numeric values pass through as np.nan.
        """
        out = np.empty(self._n_features, dtype=np.float32)
        for i, col in enumerate(self._cat_cols):
            val = str(row.get(col, "nan"))
            out[i] = self._cat_maps[col].get(val, -1.0)
        offset = len(self._cat_cols)
        for i, col in enumerate(self._num_cols):
            v = row.get(col)
            out[offset + i] = np.nan if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
        return out[np.newaxis, :]
```

Also add `FastPreprocessor` to the module's public exports — update the imports at the top of any file that will use it (none needed yet; the import in `recommender.py` will be added in Task 2).

- [ ] **Step 4: Run the 3 new tests to confirm they pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_onnx.py::test_fast_preprocessor_matches_sklearn tests/test_onnx.py::test_fast_preprocessor_unknown_category tests/test_onnx.py::test_fast_preprocessor_nan_numeric_passthrough -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Run the full test suite to check for regressions**

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

Expected: all tests pass (the new class is additive — nothing is wired in yet).

- [ ] **Step 6: Commit**

```powershell
git add src/rec_oncf/training.py tests/test_onnx.py
git commit -m "feat: add FastPreprocessor for single-row inference encoding"
```

---

### Task 2: Wire `FastPreprocessor` into `Recommender`

**Files:**
- Modify: `src/rec_oncf/recommender.py`

No new tests are needed here — the existing `test_recommender.py` and `test_api.py` suites will validate correctness because `from_data()` leaves `fast_preprocessor=None` (uses the sklearn fallback path, same as before).

---

- [ ] **Step 1: Update the import at the top of `src/rec_oncf/recommender.py`**

Change:
```python
from rec_oncf.training import TrainArtifacts, load_artifacts, predict_proba, predict_proba_onnx
```
To:
```python
from rec_oncf.training import FastPreprocessor, TrainArtifacts, load_artifacts, predict_proba, predict_proba_onnx
```

- [ ] **Step 2: Add `fast_preprocessor` field to the `Recommender` dataclass**

Change:
```python
@dataclass
class Recommender:
    artifacts: TrainArtifacts
    history_lookup: dict[str, pd.DataFrame]
    cold_start_rec: ColdStartRecommender
    onnx_session: object | None = None  # onnxruntime.InferenceSession
```
To:
```python
@dataclass
class Recommender:
    artifacts: TrainArtifacts
    history_lookup: dict[str, pd.DataFrame]
    cold_start_rec: ColdStartRecommender
    onnx_session: object | None = None        # onnxruntime.InferenceSession
    fast_preprocessor: FastPreprocessor | None = None
```

- [ ] **Step 3: Update `_build()` to accept and store `fast_preprocessor`**

Change signature:
```python
@classmethod
def _build(
    cls,
    artifacts: TrainArtifacts,
    clean_df: pd.DataFrame,
    cold_start_rec: ColdStartRecommender,
    onnx_session: object | None = None,
) -> Recommender:
```
To:
```python
@classmethod
def _build(
    cls,
    artifacts: TrainArtifacts,
    clean_df: pd.DataFrame,
    cold_start_rec: ColdStartRecommender,
    onnx_session: object | None = None,
    fast_preprocessor: FastPreprocessor | None = None,
) -> Recommender:
```

And update the `return cls(...)` call inside `_build()`:
```python
return cls(
    artifacts=artifacts,
    history_lookup=history_lookup,
    cold_start_rec=cold_start_rec,
    onnx_session=onnx_session,
    fast_preprocessor=fast_preprocessor,
)
```

- [ ] **Step 4: Update `from_paths()` to build `FastPreprocessor` from the pipeline**

Inside `from_paths()`, after the line `onnx_session = InferenceSession(str(paths.onnx_model_path))`, add:

```python
fast_preprocessor = FastPreprocessor(artifacts.pipeline["pre"])
```

And update the `return cls._build(...)` call:
```python
return cls._build(artifacts, clean, cold_start_rec, onnx_session, fast_preprocessor)
```

(`from_data()` does not set `onnx_session` or `fast_preprocessor` — it calls `cls._build(artifacts, clean_df, cold_start_rec)` unchanged, so both remain `None` in tests.)

- [ ] **Step 5: Replace the ONNX inference block in `recommend()` with the fast path**

Find this block in `recommend()`:
```python
        if self.onnx_session is not None:
            proba = predict_proba_onnx(
                self.onnx_session,
                self.artifacts.pipeline["pre"],
                feat_row,
                label_col="LiaisonId",
            )[0]
        else:
            proba = predict_proba(self.artifacts, feat_row, label_col="LiaisonId")[0]
```

Replace it with:
```python
        if self.onnx_session is not None:
            if self.fast_preprocessor is not None:
                row_dict = feat_row.iloc[0].to_dict()
                X_pre = self.fast_preprocessor.encode(row_dict)
            else:
                # fallback: use sklearn ColumnTransformer (slower, no fast_preprocessor set)
                drop = [c for c in ["LiaisonId", "CodeClient"] if c in feat_row.columns]
                drop += [c for c in feat_row.columns if feat_row[c].dtype.kind == "M"]
                X = feat_row.drop(columns=drop)
                X_pre = self.artifacts.pipeline["pre"].transform(X).astype(np.float32)
            proba = self.onnx_session.run(["probabilities"], {"input": X_pre})[0]
        else:
            proba = predict_proba(self.artifacts, feat_row, label_col="LiaisonId")[0]
```

- [ ] **Step 6: Run the full test suite**

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

Expected: all tests pass. The tests in `test_recommender.py` and `test_api.py` use `from_data()` which leaves `fast_preprocessor=None`, so they hit the sklearn fallback — same behavior as before.

- [ ] **Step 7: Commit**

```powershell
git add src/rec_oncf/recommender.py
git commit -m "feat: use FastPreprocessor in Recommender hot path for 45% latency reduction"
```

---

### Task 3: Benchmark + update CLAUDE.md

**Files:**
- Run: `scripts/_profile_latency.py` (add a comparison section)
- Run: `scripts/_bench_api.py`
- Modify: `CLAUDE.md`

---

- [ ] **Step 1: Update `scripts/_profile_latency.py` to compare old vs. new path**

After the existing benchmark lines, add at the end of the file:

```python
from rec_oncf.training import FastPreprocessor

fp = FastPreprocessor(preprocessor)
row_dict = feat_row.iloc[0].to_dict()

_, t_fast = bench("FastPreprocessor.encode (NEW)", lambda: fp.encode(row_dict))

print()
print(f"Preprocessor savings: {t_pre:.2f} ms  →  {t_fast:.2f} ms  ({t_pre/t_fast:.0f}x speedup)")
new_total = t_cands + t_feat + t_fast + t_onnx + t_score
print(f"Projected logic total: {new_total:.2f} ms  (was {total:.2f} ms)")
```

- [ ] **Step 2: Run the updated profiler**

```powershell
.venv\Scripts\python.exe scripts/_profile_latency.py
```

Expected output (approximate):
```
preprocessor.transform               18.xx ms
FastPreprocessor.encode (NEW)         0.xx ms
Preprocessor savings: 18.xx ms → 0.xx ms (Nx speedup)
Projected logic total: ~13 ms (was ~42 ms)
```

- [ ] **Step 3: Run the API benchmark**

```powershell
.venv\Scripts\python.exe scripts/_bench_api.py
```

Expected:
```
p50:  ~27 ms   (was 56 ms)
p99:  ~78 ms   (was 108 ms)
```

Record the actual numbers for the CLAUDE.md update.

- [ ] **Step 4: Update `CLAUDE.md`**

In the **Current Status** table, update the latency row:

```markdown
| API latency p50 (ONNX + FastPreprocessor) | ✅ ~27 ms | FastPreprocessor replaced ColumnTransformer.transform (19ms → 0.1ms); p99 ~78ms |
```

Also update the **What's Left to Do** section to replace the stale latency note. Change:
```
1. **Reduce inference latency** (current p50 = 852 ms, target < 100 ms).
   Major redesign: replace multiclass softmax by a **binary pairwise
   ranker** scored only on the 10 candidates (~100× fewer scores per
   request). This unlocks per-candidate features like
   `liaison_global_freq` that don't fit a multiclass setup. Several
   days of work.
```
To:
```
1. **Inference latency** ✅ target met. p50 = ~27ms, p99 = ~78ms (both < 100ms).
   FastPreprocessor replaced sklearn ColumnTransformer.transform on the hot path
   (saved 19ms/request). Profiling: candidates 5ms, compute_inference_row 2.5ms,
   FastPreprocessor 0.1ms, ONNX 5ms. No model retraining was needed.
   Next latency lever if ever needed: replace pandas ops in generate_candidates
   and compute_inference_row with pure numpy (~10ms additional savings).
```

Also update the test count in the layout section: `test_onnx.py` now has 6 tests.

- [ ] **Step 5: Delete the temporary profiling/benchmark scripts** (optional — they are prefixed with `_` so pytest ignores them, but they're dev tools not production code)

```powershell
# Optional cleanup — only if you prefer a clean scripts/ directory
# git rm scripts/_profile_latency.py scripts/_bench_api.py
```

- [ ] **Step 6: Commit**

```powershell
git add CLAUDE.md scripts/_profile_latency.py
git commit -m "docs: update latency figures after FastPreprocessor optimization"
```
