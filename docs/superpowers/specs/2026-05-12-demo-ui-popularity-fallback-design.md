# Design — Popularity fallback + ONCF demo UI

**Date:** 2026-05-12
**Status:** Approved (design phase)
**Author:** session 2026-05-12

## Context

The recommender service is functionally complete (model, ONNX, cold-start CF, A/B
framework, schedule scraping, structured logging — see `CLAUDE.md`). Two gaps remain
before it is presentable to ONCF:

1. **Unknown users get an empty list.** `Recommender.recommend()` returns
   `_COLD_START` (`mode: "cold_start"`, `recommendations: []`) for any user with no
   history, fewer than 3 trips without a co-occurrence match, or no generated
   candidates. A user-facing app would then show *nothing*.
2. **No way to demo the model.** There is only the JSON API. We need a polished,
   ONCF-styled web page to show the top-3 recommendations live while waiting for
   ONCF's go-ahead to integrate it into their mobile app.

## Goals

- Never return an empty recommendation list — fall back to global popularity.
- Ship a single, polished, self-contained web page served by the existing FastAPI
  app that lets someone type a `code_client` and see the top-k recommended routes
  (with station names, the recommendation mode, latency, and optionally schedules).

## Non-goals (deferred, explicitly out of scope)

- Running a Redis server (the code already supports it; nothing to build).
- Training / promoting a challenger model.
- A `.exe` one-shot deployment installer for the ONCF machine.
- Production uvicorn config (workers, no `--reload`, reverse proxy).
- Sample-client picker, client-history view, or hit/miss evaluation mode in the UI.

---

## Part A — Popularity fallback (back-end)

### A.1 New artifact: `models/popularity.joblib`

A new pipeline script `scripts/08_build_popularity.py`:

- Reads `data/processed/oncf_clean.parquet` (cancellations are already removed
  upstream by `cleaning.py`, so a plain `LiaisonId.value_counts()` is correct).
- Produces `models/popularity.joblib`: a Python `list[str]` of `LiaisonId` ordered by
  descending global frequency. (List, not Series — keeps the artifact tiny and the
  load trivial.)
- Prints the top-10 to stdout for a sanity check.

`config.py` gains `popularity_path = models_dir / "popularity.joblib"`.

### A.2 `Recommender` changes

- `Recommender` dataclass gains `popularity: list[str]` (default `[]`).
- `from_paths()` loads `popularity.joblib` if present; if absent, falls back to an
  empty list (and the old `_COLD_START` behaviour is preserved — no crash).
- `from_data()` gains an optional `popularity` argument for tests.
- New private helper `_popularity_rec(self, k)` → `{"mode": "popularity",
  "recommendations": self.popularity[:k]}`. If `self.popularity` is empty it returns
  the existing `_COLD_START` constant (so behaviour is unchanged when the artifact is
  missing).
- Every site in `recommend()` that currently returns `_COLD_START` because the user
  cannot be served by the model **but the user object/history exists or is simply
  unknown** now returns `_popularity_rec(k)` instead:
  - `code_client` not in `history_lookup`
  - `len(history) < 3` *and* cold-start CF produced nothing
  - `generate_candidates()` returned empty
  - no candidate is known to the label encoder
  Note: the cold-start CF path (`mode: "cold_start_cf"`) is unchanged and still takes
  priority for 1-2-trip users when it has a result.

### A.3 `labels` enrichment on `/recommend`

- A liaison → label lookup `dict[str, str]` of the form
  `{"14358": "CASA VOYAGEURS → MEKNES"}` is built once at `Recommender` construction
  from `oncf_clean.parquet`'s `LiaisonId` / `DesignationFrGareDepart` /
  `DesignationFrGareArrive` columns (drop-duplicates on `LiaisonId`).
- `recommend()`'s return dict gains `"labels": {lid: label for lid in
  recommendations if lid in lookup}` for every mode (model, cold_start_cf,
  popularity). Unknown ids are simply omitted from the dict — the UI falls back to
  showing the raw id.
- This is additive; no existing field changes, no existing test breaks.

### A.4 Tests (TDD — write first)

- `tests/test_recommender.py`: unknown `code_client` → `mode == "popularity"`,
  `len(recommendations) == k`, list never empty; `labels` present and correct for a
  known liaison; when `popularity` is empty the result is still the old `_COLD_START`.
- `tests/test_popularity.py`: `08_build_popularity.py` logic — given a small fake
  clean frame, the produced list is ordered by descending frequency and contains all
  liaisons.

### A.5 Docs

- `CLAUDE.md`: new `popularity` mode, new script 08, new artifact, new `labels` field,
  bump test count, update "How to run everything".

---

## Part B — Demo UI

### B.1 Serving

- Static assets live in `apps/api/static/` (`index.html`, `app.js`, `styles.css`).
- `apps/api/main.py`:
  - `app.mount("/static", StaticFiles(directory=...), name="static")`
  - `GET /` → `FileResponse(.../static/index.html)` (so the root URL is the demo).
- No new Python dependency (`StaticFiles` / `FileResponse` ship with FastAPI/Starlette).

### B.2 Page structure (single page, vanilla HTML/CSS/JS — no build step)

- **Header**: ONCF-style bar — left: wordmark "ONCF" in the ONCF orange on a dark
  slate bar, with subtitle "Recommandation de trajet — démo". The visual language
  follows the ONCF mobile app: orange (`#F37021`-ish ONCF orange) as the accent on a
  near-white background with a dark slate (`#1B2A4A`-ish) for headers/text. The
  `frontend-design` skill drives the exact polish; the palette above is the anchor.
- **Form card**: text input "Code client", a segmented control / select for `k`
  (1 / 2 / 3, default 3), a checkbox "Afficher les horaires", and a primary
  "Recommander" button.
- **Results area**:
  - One **route card** per recommendation: large `GARE DEP → GARE ARR` (from
    `labels`; falls back to `LiaisonId` if missing), a rank pill (`#1 / #2 / #3`), and
    — if "Afficher les horaires" was checked and the API returned `schedules` for that
    id — a compact list of next departures (time, train number).
  - A footer line under the cards: a colored **mode badge** (`model` → green,
    `cold_start_cf` → amber, `cold_start` → grey, `popularity` → grey), the
    `request_id`, and the round-trip latency measured client-side in ms.
- **States**: empty input → inline validation message, no request sent; in-flight →
  button disabled + spinner; network/HTTP error → error banner with the message;
  `mode === "cold_start"` with empty list (only possible if `popularity.joblib` is
  absent) → friendly "aucune recommandation disponible" message.

### B.3 Data flow

1. User fills the form, clicks "Recommander".
2. JS records `t0 = performance.now()`, `POST /recommend` with
   `{code_client, k, include_schedule}` (always via POST body — `code_client` never
   in the URL/query, never written to `localStorage`).
3. On `200`: render cards from `recommendations` + `labels` (+ `schedules` if
   present); show `mode`, `request_id`, and `performance.now() - t0`.
4. On error: show the error banner; keep the form usable.

### B.4 Loi 09-08 / CNDP

- `code_client` is sent only in the request body, never in the URL, never persisted
  client-side. No analytics, no third-party scripts. The page makes no other network
  calls. (The API side already never logs `code_client`.)

### B.5 Tests

- `tests/test_api.py`: `GET /` → `200` and `content-type` starts with `text/html`;
  `POST /recommend` response JSON includes a `labels` key (dict).

---

## File-level summary

**New files**
- `scripts/08_build_popularity.py`
- `tests/test_popularity.py`
- `apps/api/static/index.html`
- `apps/api/static/app.js`
- `apps/api/static/styles.css`

**Modified files**
- `src/rec_oncf/config.py` — `popularity_path`
- `src/rec_oncf/recommender.py` — `popularity` field, `from_paths`/`from_data`,
  `_popularity_rec`, label lookup, `labels` in `recommend()` output, replace the
  relevant `_COLD_START` returns
- `apps/api/main.py` — `StaticFiles` mount + `GET /`
- `tests/test_recommender.py` — popularity + labels tests
- `tests/test_api.py` — `GET /` + `labels` tests
- `CLAUDE.md` — docs

## Risks / open points

- **Liaison label coverage**: a recommended `LiaisonId` that never appears in
  `oncf_clean.parquet` would have no label — handled by falling back to the raw id in
  the UI. In practice popularity ids are by definition from the data, and model ids
  come from the label encoder which was fit on the same data, so misses should be
  near-zero.
- **ONCF brand exactness**: we approximate the ONCF app palette; if ONCF later
  provides brand assets the CSS variables make it a one-file change.
- **Startup time** unchanged (popularity.joblib is tiny; label lookup is a cheap
  drop-duplicates on an already-loaded frame).
