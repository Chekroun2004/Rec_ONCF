# ONNX Runtime Inference — Design Spec

**Goal:** Reduce `/recommend` p50 latency from ~100ms to <50ms by replacing XGBoost sklearn inference with ONNX Runtime, without retraining or changing model accuracy.

**Date:** 2026-05-07

---

## Context

Profiling revealed that `predict_proba()` accounts for 95% of request time (~98ms). Root cause: XGBoost multiclass with 1 011 classes uses `n_estimators × n_classes = 200 × 1 011 = 202 200` trees per prediction. ONNX Runtime is a heavily optimized C++ inference engine that runs the same computation 5–10× faster.

The sklearn `ColumnTransformer` preprocessor takes ~2ms — not the bottleneck, not touched.

---

## Architecture

```
POST /recommend
  │
  ├── generate_candidates()         ~3ms   (unchanged)
  ├── compute_inference_row()       ~2ms   (unchanged)
  ├── ColumnTransformer.transform() ~2ms   (sklearn, unchanged)
  └── onnx_session.run()            ~10ms  (NEW — replaces predict_proba)
                                    ─────
                              Total ~17ms  (vs ~103ms before)
```

**Key design decision:** Export only the XGBoost classifier to ONNX (via `XGBClassifier.save_model("file.onnx")`), not the full sklearn Pipeline. XGBoost 2.x supports native ONNX export. The sklearn preprocessor stays as-is.

**Backward compatibility:** `from_data()` (used in all unit tests) does not load any ONNX session. Tests remain untouched. `from_paths()` (production) requires the ONNX file to exist; raises `RuntimeError` with a clear message if missing.

---

## Files

| Action | File | Responsibility |
|---|---|---|
| Modify | `requirements.txt` | Add `onnxruntime>=1.18` |
| Modify | `src/rec_oncf/config.py` | Add `onnx_model_path: Path` to `Paths`; default = `models/xgb_ranker.onnx` |
| Modify | `src/rec_oncf/training.py` | Add `export_onnx(pipeline, path)` and `predict_proba_onnx(session, preprocessor, feat_row)` |
| Create | `scripts/06_export_onnx.py` | Load joblib model → export ONNX → benchmark sklearn vs ONNX |
| Modify | `src/rec_oncf/recommender.py` | Add `onnx_session` field (optional); `from_paths` loads it; `recommend()` uses `predict_proba_onnx` when available |
| Create | `tests/test_onnx.py` | 3 tests: file creation, proba parity, output shape |

---

## API Contracts

### `export_onnx(pipeline: Pipeline, path: Path) -> None`
Extracts `pipeline["clf"]` (XGBClassifier), calls `clf.save_model(str(path))`. Creates parent dirs if needed.

### `predict_proba_onnx(session: InferenceSession, preprocessor: ColumnTransformer, feat_row: pd.DataFrame) -> np.ndarray`
1. Drops `LiaisonId` column if present (label col, not a feature)
2. Runs `preprocessor.transform(feat_row)` → `X` (numpy float64)
3. Casts to float32 (ONNX Runtime requirement)
4. Calls `session.run(["probabilities"], {"input": X})[0]`
5. Returns array of shape `(1, n_classes)` — same contract as existing `predict_proba()`

### `Recommender` changes
- New field: `onnx_session: InferenceSession | None = None`
- `from_paths()`: loads ONNX session from `paths.onnx_model_path`; raises `RuntimeError` if file missing
- `from_data()`: `onnx_session=None` (tests use sklearn path as fallback)
- `recommend()`: uses `predict_proba_onnx()` when `onnx_session is not None`, else falls back to `predict_proba()`

---

## Testing

**`tests/test_onnx.py`** (3 tests, TDD):

1. `test_export_creates_onnx_file` — call `export_onnx(pipeline, tmp_path/"m.onnx")`, assert file exists and size > 0
2. `test_onnx_probas_match_sklearn` — run both paths on same feat_row, assert `np.allclose(proba_onnx, proba_sklearn, atol=1e-4)`
3. `test_onnx_output_shape` — assert output shape is `(1, n_classes)`

Existing tests (`test_recommender.py`, `test_api.py`) require no changes — they use `from_data()` which has `onnx_session=None`.

---

## Script `06_export_onnx.py`

1. Load artifacts from `paths.xgb_model_path`
2. Call `export_onnx(pipeline, paths.onnx_model_path)`
3. Benchmark: 20 iterations of sklearn `predict_proba` vs ONNX `predict_proba_onnx`, print p50 of each
4. Assert ONNX p50 < 30ms (fail loudly if not)
5. Print speedup ratio

---

## Success Criteria

- ONNX inference p50 < 30ms (measured in script 06)
- Proba outputs match sklearn to within 1e-4
- All 53 existing tests still pass
- Ruff clean

---

## Out of Scope

- Converting the sklearn ColumnTransformer to ONNX (not the bottleneck)
- Pairwise binary ranker (separate future feature)
- GPU inference (not available at acceptable cost for this use case)
