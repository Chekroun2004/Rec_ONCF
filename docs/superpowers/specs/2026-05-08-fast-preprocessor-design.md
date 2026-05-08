# Fast Preprocessor — Inference Latency Optimization

**Date:** 2026-05-08  
**Status:** Approved

---

## Context

After the ONNX export (script 06), the p50 end-to-end API latency is **56ms**, already under the <100ms target. The p99 is **108ms**, slightly over. Profiling reveals the breakdown:

| Step | Time | % |
|---|---|---|
| `generate_candidates` | 5.22 ms | 12% |
| `compute_inference_row` | 2.54 ms | 6% |
| `preprocessor.transform` | **18.76 ms** | **45%** |
| `onnx session.run` | 5.26 ms | 12% |
| filter + argsort | ~0 ms | 0% |
| **Total logic** | **42 ms** | |
| FastAPI/uvicorn overhead | ~14 ms | |

The `ColumnTransformer.transform` call on a single-row DataFrame dominates at 45% of logic time. It is slow due to pandas type-checking, column ordering, and per-call OrdinalEncoder overhead — none of which is necessary when the schema is fixed and known at startup.

The goal is **no model retraining, no ONNX re-export, no accuracy change**.

---

## Design

### FastPreprocessor (training.py)

A lightweight class built once at startup from a fitted `ColumnTransformer`. It extracts the ordinal mappings and column order at construction time and encodes a single feature row in ~0.1ms via dict lookups and direct float reads.

```python
class FastPreprocessor:
    """Single-row encoder, ~200x faster than ColumnTransformer.transform."""
    def __init__(self, ct: ColumnTransformer):
        # Extract cat transformer and its column list
        # Extract num passthrough and its column list
        # Build {col: {str(value): ordinal_idx}} dicts from categories_
        ...

    def encode(self, row: dict) -> np.ndarray:
        """Return (1, n_features) float32 array. Unknown cat values → -1."""
        ...
```

**Column order** — same as ColumnTransformer output: all cat-encoded columns first (in transformer order), then all passthrough numeric columns. This must exactly match the ONNX model's expected input.

**Unknown category handling** — `-1`, same as `OrdinalEncoder(unknown_value=-1)`.

**NaN handling for numerics** — preserved as `np.nan` (same as passthrough).

---

### Recommender changes

`FastPreprocessor` is added as an optional field:

```python
@dataclass
class Recommender:
    ...
    fast_preprocessor: FastPreprocessor | None = None
```

- `from_paths()` builds `FastPreprocessor` from `pipeline["pre"]` (always available with ONNX).
- `from_data()` (test path, no ONNX session) leaves it `None`.
- `_build()` accepts it as an optional argument.

In `recommend()`, when both `onnx_session` and `fast_preprocessor` are available, the fast path is used:

```python
if self.onnx_session is not None and self.fast_preprocessor is not None:
    row_dict = feat_row.iloc[0].to_dict()
    X_pre = self.fast_preprocessor.encode(row_dict)
    proba = self.onnx_session.run(["probabilities"], {"input": X_pre})[0]
else:
    proba = predict_proba(self.artifacts, feat_row, label_col="LiaisonId")[0]
```

The existing `predict_proba_onnx` function is untouched — it remains the path for tests and any fallback scenario.

---

## Testing

No new test fixtures needed. Correctness is verified by:

1. **Parity test** — run both `preprocessor.transform` and `FastPreprocessor.encode` on the same input; assert that the output arrays are equal element-wise within float32 tolerance.
2. **Unknown value test** — assert that an unseen category value encodes to `-1.0`.
3. **NaN passthrough test** — assert that NaN numerics pass through unchanged.

These go in `tests/test_onnx.py` (already tests ONNX parity) or a new `tests/test_fast_preprocessor.py`.

---

## Files Changed

| File | Change |
|---|---|
| `src/rec_oncf/training.py` | Add `FastPreprocessor` class |
| `src/rec_oncf/recommender.py` | Add `fast_preprocessor` field; use fast path in `recommend()` |
| `tests/test_onnx.py` | Add parity + edge-case tests for `FastPreprocessor` |
| `scripts/_profile_latency.py` | Update to benchmark both paths |
| `CLAUDE.md` | Update latency figures |

---

## Expected Outcome

| Metric | Before | After |
|---|---|---|
| p50 | 56 ms | ~27 ms |
| p99 | 108 ms | ~78 ms |

Both p50 and p99 land comfortably under 100ms with no model changes.
