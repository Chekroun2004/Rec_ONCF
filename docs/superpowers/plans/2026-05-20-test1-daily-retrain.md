# Intégration test1.csv + Simulation retrain quotidien — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Intégrer `test1.csv` (voyages 2021) au pipeline ONCF et simuler 7 retrains quotidiens consécutifs avec fenêtre glissante de 365 jours, en commençant par valider que le `horaire.csv` enrichi améliore la couverture O/D.

**Architecture:** 4 phases d'exécution (tests horaire → extraction 7j → baseline post-test1 → simulation jour par jour) + 2 transverses (tests TDD au fil de l'eau, mises à jour rapport/CLAUDE.md). Chaque phase est indépendante, persiste ses artefacts sur disque, et peut être reprise sans perte de contexte. La simulation produit des modèles isolés dans `models/sim/day_N/` sans toucher au prod servi par l'API.

**Tech Stack:** Python 3.12 (`.venv\Scripts\python.exe`), pandas 3.x, XGBoost 3.2.0, scikit-learn, pytest, joblib, FastAPI (tests d'intégration seulement). Windows 11, PowerShell. Spec : `docs/superpowers/specs/2026-05-20-test1-daily-retrain-design.md`.

---

## Files Map

**Create (nouveaux fichiers) :**
- `scripts/12a_extract_test1_days.py` — extraction des 7 derniers jours de test1.csv
- `scripts/12_simulate_daily_retrain.py` — CLI orchestrateur, `--day N`
- `src/rec_oncf/simulation.py` — logique réutilisable de la simulation (testable sans CLI)
- `tests/test_extract_days.py` — tests TDD de l'extraction
- `tests/test_simulate_daily.py` — tests TDD de la simulation
- `tests/test_dataset_extra_csv.py` — tests TDD de l'option `--extra-csv`
- `data/raw/daily/` — dossier de sortie des 7 CSVs jour-par-jour (créé par le script 12a)
- `data/raw/test1_base.csv` — test1.csv sans les 7 derniers jours (créé par 12a)
- `reports/simulation_daily.json` — log cumulatif des 7 jours (créé par 12)
- `models/sim/day_N/*` — modèles de simulation, un dossier par jour (créé par 12)
- `models/archive/20260520T_pre_test1/*` — snapshot manuel du prod avant baseline

**Modify (fichiers existants) :**
- `scripts/01_make_dataset.py` — ajout argument `--extra-csv <path>` pour concaténer un CSV à oncf_data.csv avant nettoyage
- `src/rec_oncf/training.py` — ajout de paramètres optionnels à `train_xgb_multiclass` pour overrider les hyperparamètres (default = challenger : max_depth=8, n_estimators=250)
- `CLAUDE.md` — section "État actuel" mise à jour après chaque phase

**Pas modifié (intentionnel) :**
- `apps/api/main.py` et tout `apps/api/` — aucun changement API dans ce cycle
- `src/rec_oncf/cleaning.py` — la concaténation se fait avant `make_clean_dataset`, le module reste inchangé
- `src/rec_oncf/local_schedule.py` — rebuild de l'index seulement, pas de modif du code
- `scripts/07_retrain.py` — garde son guardrail à 5pp pour la prod

---

## Task 1 : Phase 1 — Rebuild index horaire et tests

**Files:**
- Run: `scripts/11_build_schedule_index.py` (existant, pas de modif)
- Run tests: `tests/test_local_schedule.py`, `tests/test_schedule.py` (existants)

- [ ] **Step 1.1 : Rebuild de l'index O/D avec le horaire.csv enrichi**

Run:
```powershell
.venv\Scripts\python.exe scripts\11_build_schedule_index.py
```

Expected stdout (couverture cible) :
```
Loading C:\Users\omarc\Desktop\horaire.csv...
  XXXX stops  |  XXX trains  |  XXX distinct stations
Building O/D index...
  XXXX O/D pairs generated
Index saved -> ...\models\schedule_index.joblib
Coverage: NNN/1067 LiaisonIds covered by index
```

Critère : `NNN/1067` doit être ≥ **800** (cible >75%). Noter le chiffre exact.

- [ ] **Step 1.2 : Si couverture < 800, arrêter et investiguer**

Si `NNN < 800` :
```powershell
.venv\Scripts\python.exe -c "import joblib; from rec_oncf.config import default_paths; from rec_oncf.io import read_parquet; from rec_oncf.schedule import build_liaison_station_map; p=default_paths(); idx=joblib.load(p.schedule_index_path); m=build_liaison_station_map(read_parquet(p.processed_dataset_parquet)); missing=[(lid,o,d) for lid,(o,d) in m.items() if (o.strip().upper(),d.strip().upper()) not in idx]; [print(f'{lid}: {o} -> {d}') for lid,o,d in missing[:30]]"
```

Cette commande imprime les 30 premiers LiaisonIds non couverts. **Stop ici** et rapporter à l'utilisateur ; ne pas continuer Phase 2 tant que la couverture n'est pas validée.

- [ ] **Step 1.3 : Tests unitaires local_schedule**

Run:
```powershell
.venv\Scripts\python.exe -m pytest tests\test_local_schedule.py tests\test_schedule.py -v
```

Expected: **35 tests PASSED** (21 + 14). Si un échec, arrêter et investiguer (probablement un changement de format dans horaire.csv qui casse le parser).

- [ ] **Step 1.4 : Smoke test live de 3 routes LGV connues**

Lancer l'API en arrière-plan dans un terminal séparé :
```powershell
.venv\Scripts\python.exe -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

Dans un autre terminal :
```powershell
# Trouver les LiaisonId de 3 routes LGV (Tanger ↔ Casa, Tanger ↔ Rabat, Tanger ↔ Kenitra)
.venv\Scripts\python.exe -c "from rec_oncf.config import default_paths; from rec_oncf.io import read_parquet; from rec_oncf.schedule import build_liaison_station_map; m=build_liaison_station_map(read_parquet(default_paths().processed_dataset_parquet)); [print(lid,o,d) for lid,(o,d) in m.items() if 'TANGER' in o.upper() or 'TANGER' in d.upper()][:6]"
```

Puis tester via curl pour chacun des 3 LiaisonId trouvés :
```powershell
curl http://127.0.0.1:8000/schedule/<liaison_id>
```

Expected : `{"liaison_id":"<id>","schedule":[{"depart":"HH:MM","arrive":"HH:MM"},...]}` avec **au moins 1 entrée non vide** pour les routes LGV désormais couvertes.

Si l'une renvoie `"schedule": []` : noter le LiaisonId concerné, arrêter l'API (Ctrl+C), retourner au step 1.2 pour investiguer.

- [ ] **Step 1.5 : Arrêter le serveur API**

Ctrl+C dans le terminal uvicorn.

- [ ] **Step 1.6 : Mettre à jour CLAUDE.md (section État actuel)**

Editer `CLAUDE.md` pour ajouter sous "État actuel" une ligne :

```markdown
- **2026-05-20 (Phase 1)** : horaire.csv enrichi (LGV + Marrakech) → couverture O/D : `NNN/1067` LiaisonIds (XX%, était 57%). 35 tests local_schedule + schedule passent. Smoke test LGV OK.
```

(remplacer `NNN` et `XX%` par les valeurs réelles)

- [ ] **Step 1.7 : Commit Phase 1**

```powershell
git add models\schedule_index.joblib CLAUDE.md
git commit -m "feat(schedule): rebuild index with enriched horaire.csv (LGV + Marrakech)"
```

---

## Task 2 : Phase 2 — Test TDD du module d'extraction

**Files:**
- Create: `tests/test_extract_days.py`
- Create: `src/rec_oncf/extract_days.py` (logique réutilisable, importée par script 12a)

- [ ] **Step 2.1 : Écrire le test (TDD — fail first)**

Create `tests/test_extract_days.py` :

```python
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rec_oncf.extract_days import extract_last_n_days


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """30 rows over 10 distinct departure dates (3 rows/day), 2021-12-22 → 2021-12-31."""
    rows = []
    for day_idx, day in enumerate(pd.date_range("2021-12-22", "2021-12-31", freq="D")):
        for hour in (8, 12, 18):
            rows.append({
                "CodeClient": 1000 + day_idx,
                "DateHeureDepartVoyageSegment": day + pd.Timedelta(hours=hour),
                "LiaisonVoyageurSegmentIdSTG": f"L{day_idx % 4}",
                "PrixParLiaison": 100.0,
            })
    return pd.DataFrame(rows)


def test_extract_returns_base_and_seven_days(sample_df):
    base, days = extract_last_n_days(sample_df, n=7, date_col="DateHeureDepartVoyageSegment")
    assert isinstance(base, pd.DataFrame)
    assert isinstance(days, dict)
    assert len(days) == 7


def test_extract_keys_are_dates(sample_df):
    _, days = extract_last_n_days(sample_df, n=7, date_col="DateHeureDepartVoyageSegment")
    expected_dates = [
        "2021-12-25", "2021-12-26", "2021-12-27", "2021-12-28",
        "2021-12-29", "2021-12-30", "2021-12-31",
    ]
    assert sorted(days.keys()) == expected_dates


def test_extract_preserves_total_rows(sample_df):
    base, days = extract_last_n_days(sample_df, n=7, date_col="DateHeureDepartVoyageSegment")
    total = len(base) + sum(len(d) for d in days.values())
    assert total == len(sample_df)


def test_extract_base_has_only_early_dates(sample_df):
    base, _ = extract_last_n_days(sample_df, n=7, date_col="DateHeureDepartVoyageSegment")
    base_dates = base["DateHeureDepartVoyageSegment"].dt.date.unique()
    # base should contain only the 3 earliest dates: 12-22, 12-23, 12-24
    assert set(str(d) for d in base_dates) == {"2021-12-22", "2021-12-23", "2021-12-24"}


def test_extract_day_csv_has_consistent_rows(sample_df):
    _, days = extract_last_n_days(sample_df, n=7, date_col="DateHeureDepartVoyageSegment")
    for date_str, day_df in days.items():
        unique = day_df["DateHeureDepartVoyageSegment"].dt.date.astype(str).unique()
        assert list(unique) == [date_str]
```

- [ ] **Step 2.2 : Vérifier que les tests échouent**

Run:
```powershell
.venv\Scripts\python.exe -m pytest tests\test_extract_days.py -v
```

Expected: `ModuleNotFoundError: No module named 'rec_oncf.extract_days'` ou similaire — **5 tests en ERROR**.

- [ ] **Step 2.3 : Implémenter le module**

Create `src/rec_oncf/extract_days.py` :

```python
from __future__ import annotations

import pandas as pd


def extract_last_n_days(
    df: pd.DataFrame,
    *,
    n: int = 7,
    date_col: str = "DateHeureDepartVoyageSegment",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Splits df into a base DataFrame (everything except the last n calendar days)
    and a dict mapping each of those last n days (YYYY-MM-DD) to its rows.

    The "day" is the calendar date of `date_col` (the departure date for ONCF data).
    Rows are bucketed by date, not by timestamp; rows with the same date are kept
    in their original order.
    """
    if date_col not in df.columns:
        raise ValueError(f"Missing column: {date_col}")

    parsed = pd.to_datetime(df[date_col], errors="coerce")
    if parsed.isna().any():
        raise ValueError(
            f"{int(parsed.isna().sum())} rows have unparseable {date_col} values"
        )

    df = df.copy()
    df[date_col] = parsed
    df["_day"] = parsed.dt.date.astype(str)

    unique_days = sorted(df["_day"].unique())
    if len(unique_days) < n:
        raise ValueError(
            f"Need at least {n} distinct days; got {len(unique_days)}"
        )

    last_days = unique_days[-n:]
    base = df[~df["_day"].isin(last_days)].drop(columns="_day").reset_index(drop=True)
    days = {
        day: df[df["_day"] == day].drop(columns="_day").reset_index(drop=True)
        for day in last_days
    }
    return base, days
```

- [ ] **Step 2.4 : Vérifier que les tests passent**

Run:
```powershell
.venv\Scripts\python.exe -m pytest tests\test_extract_days.py -v
```

Expected: **5 PASSED**.

- [ ] **Step 2.5 : Commit du module**

```powershell
git add src\rec_oncf\extract_days.py tests\test_extract_days.py
git commit -m "feat(extract): add extract_last_n_days for daily-retrain simulation prep"
```

---

## Task 3 : Phase 2 — Script CLI `12a_extract_test1_days.py`

**Files:**
- Create: `scripts/12a_extract_test1_days.py`

- [ ] **Step 3.1 : Écrire le script CLI**

Create `scripts/12a_extract_test1_days.py` :

```python
# scripts/12a_extract_test1_days.py
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rec_oncf.config import default_paths
from rec_oncf.extract_days import extract_last_n_days
from rec_oncf.io import read_csv, write_csv


def main() -> None:
    paths = default_paths()
    test1_path = paths.desktop / "test1.csv"
    if not test1_path.exists():
        raise FileNotFoundError(f"Missing: {test1_path}")

    print(f"Loading {test1_path}...")
    df = read_csv(test1_path)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    base, days = extract_last_n_days(df, n=7, date_col="DateHeureDepartVoyageSegment")

    out_dir = PROJECT_ROOT / "data" / "raw" / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = PROJECT_ROOT / "data" / "raw" / "test1_base.csv"

    write_csv(base, base_path, sep=",")
    print(f"  base: {len(base):,} rows -> {base_path}")

    for day, day_df in days.items():
        day_path = out_dir / f"test1_day_{day}.csv"
        write_csv(day_df, day_path, sep=",")
        print(f"  {day}: {len(day_df):,} rows -> {day_path}")

    total_check = len(base) + sum(len(d) for d in days.values())
    print(f"\nTotal preserved: {total_check:,} == {len(df):,} ? {total_check == len(df)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.2 : Lancer le script sur test1.csv réel**

Run:
```powershell
.venv\Scripts\python.exe scripts\12a_extract_test1_days.py
```

Expected stdout :
```
Loading C:\Users\omarc\Desktop\test1.csv...
  XXX,XXX rows, 18 columns
  base: XXX,XXX rows -> ...\data\raw\test1_base.csv
  YYYY-MM-DD: N rows -> ...\data\raw\daily\test1_day_YYYY-MM-DD.csv
  ... (7 lines total for the days)
Total preserved: XXX,XXX == XXX,XXX ? True
```

**Vérifier** : la dernière ligne doit afficher `True`. Si `False`, arrêter et investiguer (probablement valeurs NaN dans `DateHeureDepartVoyageSegment`).

Vérifier aussi les fichiers créés :
```powershell
dir data\raw\daily\
dir data\raw\test1_base.csv
```

- [ ] **Step 3.3 : Commit du script**

```powershell
git add scripts\12a_extract_test1_days.py
git commit -m "feat(scripts): add 12a_extract_test1_days.py — splits test1 into base + 7 daily CSVs"
```

---

## Task 4 : Phase 3 — Test TDD de l'option `--extra-csv` (script 01)

**Files:**
- Create: `tests/test_dataset_extra_csv.py`

- [ ] **Step 4.1 : Écrire les tests (TDD — fail first)**

Create `tests/test_dataset_extra_csv.py` :

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_main_csv(tmp_path) -> Path:
    """Minimal valid oncf-shaped CSV with 2 rows."""
    df = pd.DataFrame({
        "TrajetAllerRetour": ["A", "A"],
        "TypeParcoursId": [1, 1],
        "CodeClient": [100, 100],
        "ClassificationId": [1, 1],
        "ClassePhysiqueId": [1, 1],
        "NiveauPrixId": [1, 1],
        "TrainAutocarId": [1, 1],
        "LiaisonVoyageurSegmentIdSTG": ["L1", "L1"],
        "CarteClientId": [1, 1],
        "PrixParLiaison": [100.0, 100.0],
        "NbrVoySegment": [1, 1],
        "DatePaiement": ["2020-01-01", "2020-01-02"],
        "DateHeureDepartVoyageSegment": ["2020-01-10 08:00", "2020-01-15 09:00"],
        "DelaiAnticipation": [9, 13],
    })
    path = tmp_path / "main.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def fake_extra_csv(tmp_path) -> Path:
    """Same schema, 2 extra rows."""
    df = pd.DataFrame({
        "TrajetAllerRetour": ["A", "A"],
        "TypeParcoursId": [1, 1],
        "CodeClient": [100, 100],
        "ClassificationId": [1, 1],
        "ClassePhysiqueId": [1, 1],
        "NiveauPrixId": [1, 1],
        "TrainAutocarId": [1, 1],
        "LiaisonVoyageurSegmentIdSTG": ["L1", "L1"],
        "CarteClientId": [1, 1],
        "PrixParLiaison": [100.0, 100.0],
        "NbrVoySegment": [1, 1],
        "DatePaiement": ["2021-01-01", "2021-01-02"],
        "DateHeureDepartVoyageSegment": ["2021-01-10 08:00", "2021-01-15 09:00"],
        "DelaiAnticipation": [9, 13],
    })
    path = tmp_path / "extra.csv"
    df.to_csv(path, index=False)
    return path


def test_concatenate_main_and_extra(fake_main_csv, fake_extra_csv):
    """Direct unit-test of the concat helper, without running the full pipeline."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    # Import the helper exposed by the modified 01_make_dataset.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "make_dataset", PROJECT_ROOT / "scripts" / "01_make_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    combined = module.load_and_concat(fake_main_csv, fake_extra_csv)
    assert len(combined) == 4
    assert list(combined.columns) == list(pd.read_csv(fake_main_csv).columns)


def test_concatenate_no_extra(fake_main_csv):
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "make_dataset", PROJECT_ROOT / "scripts" / "01_make_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    combined = module.load_and_concat(fake_main_csv, None)
    assert len(combined) == 2


def test_concatenate_schema_mismatch_raises(fake_main_csv, tmp_path):
    """Extra CSV with different columns must raise."""
    bad = pd.DataFrame({"foo": [1], "bar": [2]})
    bad_path = tmp_path / "bad.csv"
    bad.to_csv(bad_path, index=False)

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "make_dataset", PROJECT_ROOT / "scripts" / "01_make_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(ValueError, match="schema"):
        module.load_and_concat(fake_main_csv, bad_path)
```

- [ ] **Step 4.2 : Vérifier que les tests échouent**

Run:
```powershell
.venv\Scripts\python.exe -m pytest tests\test_dataset_extra_csv.py -v
```

Expected: **3 ERRORS** (`AttributeError: module 'make_dataset' has no attribute 'load_and_concat'`).

- [ ] **Step 4.3 : Modifier `scripts/01_make_dataset.py` pour exposer `load_and_concat` + `--extra-csv`**

Replace the entire content of `scripts/01_make_dataset.py` with :

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rec_oncf.cleaning import make_clean_dataset
from rec_oncf.config import default_paths
from rec_oncf.io import read_csv, write_csv, write_parquet


def load_and_concat(main_path: Path, extra_path: Path | None) -> pd.DataFrame:
    """Load main CSV, optionally append extra CSV, return concatenated DataFrame.

    The extra CSV must share exactly the same columns as the main CSV (order
    irrelevant). A schema mismatch raises ValueError. If extra_path is None,
    returns the main CSV unchanged.
    """
    main = read_csv(main_path)
    if extra_path is None:
        return main

    extra = read_csv(extra_path)
    if set(extra.columns) != set(main.columns):
        only_main = set(main.columns) - set(extra.columns)
        only_extra = set(extra.columns) - set(main.columns)
        raise ValueError(
            f"schema mismatch between {main_path.name} and {extra_path.name}: "
            f"only_in_main={sorted(only_main)}, only_in_extra={sorted(only_extra)}"
        )
    extra = extra[main.columns.tolist()]
    return pd.concat([main, extra], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean raw ONCF bookings; optionally concatenate an extra CSV (same schema)."
    )
    parser.add_argument(
        "--extra-csv",
        type=Path,
        default=None,
        help="Optional path to an extra CSV (same schema as oncf_data.csv) to concatenate before cleaning.",
    )
    args = parser.parse_args()

    paths = default_paths()
    if not paths.raw_oncf_data.exists():
        raise FileNotFoundError(f"Missing file on Desktop: {paths.raw_oncf_data}")
    if not paths.raw_liaison.exists():
        raise FileNotFoundError(f"Missing file on Desktop: {paths.raw_liaison}")
    if args.extra_csv is not None and not args.extra_csv.exists():
        raise FileNotFoundError(f"Missing extra CSV: {args.extra_csv}")

    oncf = load_and_concat(paths.raw_oncf_data, args.extra_csv)
    liaison = read_csv(paths.raw_liaison)
    print(f"Loaded {len(oncf):,} input rows ({len(oncf.columns)} columns)")

    clean, report, provenance = make_clean_dataset(oncf, liaison)

    write_parquet(clean, paths.processed_dataset_parquet)
    write_csv(clean, paths.processed_dataset_csv)
    prov_path = paths.project_root / "reports" / "cleaning_provenance.parquet"
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(provenance, prov_path)
    report_path = paths.project_root / "reports" / "cleaning_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {paths.processed_dataset_parquet}")
    print(f"Wrote: {paths.processed_dataset_csv}")
    print(f"Rows: {len(clean):,}")
    print(f"Distinct liaisons: {clean['LiaisonId'].nunique():,}")
    print(f"Report: {report_path}")
    print(f"Provenance: {prov_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.4 : Vérifier que les tests passent**

Run:
```powershell
.venv\Scripts\python.exe -m pytest tests\test_dataset_extra_csv.py -v
```

Expected: **3 PASSED**.

- [ ] **Step 4.5 : Vérifier que les tests existants passent toujours**

Run:
```powershell
.venv\Scripts\python.exe -m pytest tests\test_cleaning.py -v
```

Expected: **5 PASSED** (régression).

- [ ] **Step 4.6 : Commit**

```powershell
git add scripts\01_make_dataset.py tests\test_dataset_extra_csv.py
git commit -m "feat(dataset): add --extra-csv flag to concatenate a second CSV before cleaning"
```

---

## Task 5 : Phase 3 — Snapshot prod + baseline post-test1

**Files:**
- Create (manuel) : `models/archive/20260520T_pre_test1/`
- Modify (run scripts) : `models/*.json|onnx|joblib`, `data/processed/*.parquet`

- [ ] **Step 5.1 : Snapshot manuel du prod actuel**

```powershell
New-Item -ItemType Directory -Force "models\archive\20260520T_pre_test1"
Copy-Item -Path "models\*.json","models\*.onnx","models\*.joblib" -Destination "models\archive\20260520T_pre_test1\" -Force
dir "models\archive\20260520T_pre_test1\"
```

Expected: ~8 fichiers (`xgb_ranker.json`, `xgb_ranker.onnx`, `label_encoder.joblib`, `cold_start.joblib`, `popularity.joblib`, `schedule_index.joblib`, `xgb_ranker_challenger.{json,onnx}`, `label_encoder_challenger.joblib`, `xgb_ranker.meta.json`).

- [ ] **Step 5.2 : Cleaning sur oncf + test1_base**

```powershell
.venv\Scripts\python.exe scripts\01_make_dataset.py --extra-csv data\raw\test1_base.csv
```

Expected stdout (durée ~3 min) :
```
Loaded XXX,XXX input rows (18 columns)
Wrote: ...\data\processed\oncf_clean.parquet
Wrote: ...\data\processed\oncf_clean.csv
Rows: ~570,000   (expected range: 540k–600k)
Distinct liaisons: ~1,011–1,067
Report: ...
Provenance: ...
```

Si `Rows < 500,000` ou `> 700,000` : arrêter et investiguer (test1.csv probablement très différent en distribution).

- [ ] **Step 5.3 : Build features**

```powershell
.venv\Scripts\python.exe scripts\02_build_features.py
```

Expected stdout (durée ~1 min) :
```
Wrote: ...\data\processed\features.parquet
Rows: ~570,000
Users: ~70,000–80,000
Classes (liaisons): ~1,011–1,067
```

- [ ] **Step 5.4 : Train baseline post-test1**

```powershell
.venv\Scripts\python.exe scripts\03_train_ranker.py
```

Expected durée : ~50-70 min (sur ~570k lignes vs 491k précédemment). Stdout final inclura :
```
hit_rate@1 = 0.XXXX
hit_rate@3 = 0.XXXX
mrr@3      = 0.XXXX
```

Noter ces 3 valeurs ; ce sont les **métriques baseline post-test1**, à comparer ensuite avec le prod archivé (HR@1 = 0.7691).

- [ ] **Step 5.5 : Rebuild cold-start, ONNX, popularité, schedule index**

```powershell
.venv\Scripts\python.exe scripts\05_build_cold_start.py
.venv\Scripts\python.exe scripts\06_export_onnx.py
.venv\Scripts\python.exe scripts\08_build_popularity.py
.venv\Scripts\python.exe scripts\11_build_schedule_index.py
```

Expected durée totale : ~5 min. Vérifier qu'aucun n'erreur.

- [ ] **Step 5.6 : Régression complète des tests**

```powershell
.venv\Scripts\python.exe -m pytest tests\ -v
```

Expected: **≥ 143 PASSED** (135 actuels + 5 extract_days + 3 dataset_extra_csv = 143). Si un échec lié au volume (par exemple test attendant exactement 491,680 lignes), corriger le test pour qu'il accepte le nouveau volume.

- [ ] **Step 5.7 : Mettre à jour CLAUDE.md**

Ajouter sous "État actuel" :

```markdown
- **2026-05-20 (Phase 3)** : Baseline post-test1 entraîné — oncf + test1_base (sans 7j) → ~570k lignes. Nouveau prod : HR@1=0.XXXX, HR@3=0.XXXX, MRR@3=0.XXXX. Ancien prod archivé dans `models/archive/20260520T_pre_test1/`.
```

- [ ] **Step 5.8 : Commit**

```powershell
git add CLAUDE.md models\xgb_ranker.meta.json reports\offline_metrics.json
git commit -m "feat(model): baseline post-test1 trained on oncf + test1_base (sliding window prep)"
```

---

## Task 6 : Phase 4 — Étendre `training.py` pour hyperparamètres overridables

**Files:**
- Modify: `src/rec_oncf/training.py` — function `train_xgb_multiclass`

- [ ] **Step 6.1 : Écrire le test (TDD — fail first)**

Append to `tests/test_training.py` :

```python
def test_train_xgb_accepts_hyperparam_overrides(sample_features_df):
    from rec_oncf.training import train_xgb_multiclass

    arts = train_xgb_multiclass(
        sample_features_df,
        label_col="LiaisonId",
        time_col="DateHeureDepartVoyageSegment",
        n_estimators=10,
        max_depth=3,
    )
    clf = arts.pipeline.named_steps["clf"]
    assert clf.n_estimators == 10
    assert clf.max_depth == 3
```

(check the existing `tests/test_training.py` for the `sample_features_df` fixture — re-use it. If absent, define a minimal one inline.)

- [ ] **Step 6.2 : Run test — expected FAIL**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_training.py::test_train_xgb_accepts_hyperparam_overrides -v
```

Expected: `TypeError: train_xgb_multiclass() got an unexpected keyword argument 'n_estimators'`.

- [ ] **Step 6.3 : Modifier `train_xgb_multiclass`**

Replace the function signature and the `XGBClassifier(...)` call in `src/rec_oncf/training.py`:

```python
def train_xgb_multiclass(
    df_train: pd.DataFrame,
    *,
    label_col: str,
    time_col: str,
    n_estimators: int = 250,
    learning_rate: float = 0.06,
    max_depth: int = 8,
    subsample: float = 0.85,
    colsample_bytree: float = 0.75,
    reg_lambda: float = 1.5,
) -> TrainArtifacts:
    df_train = df_train.sort_values(time_col)

    y_raw = df_train[label_col].astype(str).to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    _drop = [c for c in [label_col, time_col, "CodeClient"] if c in df_train.columns]
    X = df_train.drop(columns=_drop)

    cat_cols = [
        c for c in X.columns
        if not pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_datetime64_any_dtype(X[c])
    ]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        n_jobs=-1,
        random_state=42,
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)

    return TrainArtifacts(pipeline=pipe, label_encoder=le)
```

**Note** : les défauts sont maintenant les **hyperparams challenger** (= prod actuel). Le script `03_train_ranker.py` produira donc le même modèle qu'avant ce changement (challenger). Les anciens defaults Sprint 2 sont disponibles via passage explicite.

- [ ] **Step 6.4 : Vérifier le nouveau test + non-régression**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_training.py tests\test_retrain.py -v
```

Expected: **tous PASSED** (≥ 19 tests).

- [ ] **Step 6.5 : Commit**

```powershell
git add src\rec_oncf\training.py tests\test_training.py
git commit -m "feat(training): expose XGB hyperparameters as kwargs; defaults = challenger (prod)"
```

---

## Task 7 : Phase 4 — Module `simulation.py` (logique réutilisable)

**Files:**
- Create: `src/rec_oncf/simulation.py`
- Create: `tests/test_simulate_daily.py`

- [ ] **Step 7.1 : Écrire les tests (TDD — fail first)**

Create `tests/test_simulate_daily.py` :

```python
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rec_oncf.simulation import (
    append_day_to_clean,
    filter_sliding_window,
    eval_on_next_day,
    log_simulation_entry,
    load_simulation_log,
)


@pytest.fixture
def clean_df() -> pd.DataFrame:
    rows = []
    for day in pd.date_range("2020-01-01", "2021-12-30", freq="D"):
        for offset in (8, 14):
            rows.append({
                "CodeClient": 100,
                "DateHeureDepartVoyageSegment": day + pd.Timedelta(hours=offset),
                "LiaisonId": "L1",
                "PrixParLiaison": 100.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def day_csv_df() -> pd.DataFrame:
    return pd.DataFrame({
        "CodeClient": [200, 200],
        "DateHeureDepartVoyageSegment": [
            pd.Timestamp("2021-12-31 08:00"),
            pd.Timestamp("2021-12-31 18:00"),
        ],
        "LiaisonId": ["L2", "L2"],
        "PrixParLiaison": [120.0, 120.0],
    })


def test_append_day_concatenates_correctly(clean_df, day_csv_df):
    appended = append_day_to_clean(clean_df, day_csv_df)
    assert len(appended) == len(clean_df) + len(day_csv_df)
    # last 2 rows are the new ones
    last_two = appended.tail(2).reset_index(drop=True)
    assert (last_two["CodeClient"] == 200).all()


def test_filter_sliding_window_exact_365_days(clean_df, day_csv_df):
    """After appending the day, window of 365 days ending at 2021-12-31 keeps only that span."""
    appended = append_day_to_clean(clean_df, day_csv_df)
    window = filter_sliding_window(
        appended, end_date="2021-12-31", window_days=365
    )
    starts = window["DateHeureDepartVoyageSegment"]
    assert starts.min() >= pd.Timestamp("2020-12-31")
    assert starts.max() <= pd.Timestamp("2021-12-31 23:59:59")


def test_filter_sliding_window_empty_when_no_data(clean_df):
    """End date far before data start → empty."""
    window = filter_sliding_window(
        clean_df, end_date="2019-01-01", window_days=365
    )
    assert len(window) == 0


def test_log_entry_appends_to_json(tmp_path):
    log_path = tmp_path / "simulation_daily.json"
    entry1 = {"day": 1, "date": "2021-12-25", "hr@1": 0.75}
    entry2 = {"day": 2, "date": "2021-12-26", "hr@1": 0.76}
    log_simulation_entry(log_path, entry1)
    log_simulation_entry(log_path, entry2)
    log = load_simulation_log(log_path)
    assert len(log) == 2
    assert log[0]["day"] == 1
    assert log[1]["day"] == 2


def test_log_replaces_same_day(tmp_path):
    """Re-running --day N overwrites the entry for that day, doesn't duplicate."""
    log_path = tmp_path / "simulation_daily.json"
    log_simulation_entry(log_path, {"day": 1, "hr@1": 0.70})
    log_simulation_entry(log_path, {"day": 1, "hr@1": 0.78})
    log = load_simulation_log(log_path)
    assert len(log) == 1
    assert log[0]["hr@1"] == 0.78


def test_eval_on_next_day_returns_metrics_structure():
    """Lightweight stub-based test: just check the dict shape."""
    # eval_on_next_day signature : (recommender, next_day_df) -> {"hr@1": float, ...}
    # We use a minimal mock to validate the contract only.
    class FakeRecommender:
        def recommend(self, code_client, k=3):
            return {
                "mode": "model",
                "recommendations": ["L1", "L2", "L3"],
                "labels": {},
            }

    next_day_df = pd.DataFrame({
        "CodeClient": [100, 100],
        "LiaisonId": ["L1", "L4"],
    })
    metrics = eval_on_next_day(FakeRecommender(), next_day_df)
    assert "hr@1" in metrics
    assert "hr@3" in metrics
    assert "mrr@3" in metrics
    assert "n_eval" in metrics
    assert metrics["n_eval"] == 2
    # Row 0 has L1 at rank 1 → HR@1=1; row 1 has L4 not in top 3 → HR@1=0
    # Mean HR@1 = 0.5
    assert metrics["hr@1"] == 0.5
```

- [ ] **Step 7.2 : Run tests — expected FAIL**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_simulate_daily.py -v
```

Expected: 6 ERRORS (`ModuleNotFoundError: No module named 'rec_oncf.simulation'`).

- [ ] **Step 7.3 : Implémenter `src/rec_oncf/simulation.py`**

Create `src/rec_oncf/simulation.py` :

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def append_day_to_clean(
    clean_df: pd.DataFrame, day_df: pd.DataFrame
) -> pd.DataFrame:
    """Concatenates the new day's rows to the existing clean DataFrame.

    Both DataFrames must share the same columns. Returns a new DataFrame
    (does not mutate inputs). Index is reset.
    """
    if set(clean_df.columns) != set(day_df.columns):
        only_clean = set(clean_df.columns) - set(day_df.columns)
        only_day = set(day_df.columns) - set(clean_df.columns)
        raise ValueError(
            f"Schema mismatch: only_in_clean={sorted(only_clean)}, "
            f"only_in_day={sorted(only_day)}"
        )
    day_df = day_df[clean_df.columns.tolist()]
    return pd.concat([clean_df, day_df], ignore_index=True)


def filter_sliding_window(
    df: pd.DataFrame,
    *,
    end_date: str | pd.Timestamp,
    window_days: int = 365,
    time_col: str = "DateHeureDepartVoyageSegment",
) -> pd.DataFrame:
    """Keep only rows whose time_col falls in [end_date - window_days, end_date]
    (inclusive on both sides, end_date pushed to 23:59:59 to include the full day).
    """
    end_ts = pd.Timestamp(end_date)
    end_inclusive = end_ts.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    start_ts = end_ts.normalize() - pd.Timedelta(days=window_days - 1)
    mask = (df[time_col] >= start_ts) & (df[time_col] <= end_inclusive)
    return df.loc[mask].reset_index(drop=True)


def eval_on_next_day(recommender, next_day_df: pd.DataFrame) -> dict[str, Any]:
    """Evaluate recommender on next_day_df rows.

    For each row, recommender.recommend(code_client, k=3) is called; the row's
    true LiaisonId is compared to the returned top-3.
    Returns {"hr@1": float, "hr@3": float, "mrr@3": float, "n_eval": int}.
    """
    if len(next_day_df) == 0:
        return {"hr@1": 0.0, "hr@3": 0.0, "mrr@3": 0.0, "n_eval": 0}

    hits_1 = 0
    hits_3 = 0
    rr_sum = 0.0
    for _, row in next_day_df.iterrows():
        true_lid = str(row["LiaisonId"])
        result = recommender.recommend(str(row["CodeClient"]), k=3)
        recs = result.get("recommendations", [])
        recs = [str(r) for r in recs]
        if recs and recs[0] == true_lid:
            hits_1 += 1
        if true_lid in recs[:3]:
            hits_3 += 1
            rank = recs.index(true_lid) + 1
            rr_sum += 1.0 / rank

    n = len(next_day_df)
    return {
        "hr@1": hits_1 / n,
        "hr@3": hits_3 / n,
        "mrr@3": rr_sum / n,
        "n_eval": n,
    }


def load_simulation_log(path: Path) -> list[dict[str, Any]]:
    """Load list of entries; returns [] if file is missing or empty."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return json.loads(text)


def log_simulation_entry(path: Path, entry: dict[str, Any]) -> None:
    """Append-or-replace an entry in the log file based on entry['day'].

    If an entry with the same 'day' value already exists, it is replaced
    in-place (idempotent on re-runs). Otherwise the entry is appended.
    """
    log = load_simulation_log(path)
    log = [e for e in log if e.get("day") != entry.get("day")]
    log.append(entry)
    log.sort(key=lambda e: e.get("day", 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
```

- [ ] **Step 7.4 : Run tests — expected PASS**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_simulate_daily.py -v
```

Expected: **6 PASSED**.

- [ ] **Step 7.5 : Commit**

```powershell
git add src\rec_oncf\simulation.py tests\test_simulate_daily.py
git commit -m "feat(simulation): add sliding-window + next-day eval + cumulative JSON log"
```

---

## Task 8 : Phase 4 — Script CLI `12_simulate_daily_retrain.py`

**Files:**
- Create: `scripts/12_simulate_daily_retrain.py`

- [ ] **Step 8.1 : Écrire le script CLI**

Create `scripts/12_simulate_daily_retrain.py` :

```python
# scripts/12_simulate_daily_retrain.py
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rec_oncf.cleaning import make_clean_dataset
from rec_oncf.config import default_paths
from rec_oncf.features import build_training_rows
from rec_oncf.io import read_csv, read_parquet, write_parquet
from rec_oncf.recommender import Recommender
from rec_oncf.retrain import check_guardrail, evaluate_model, load_current_metrics
from rec_oncf.simulation import (
    append_day_to_clean,
    eval_on_next_day,
    filter_sliding_window,
    log_simulation_entry,
)
from rec_oncf.training import (
    TrainArtifacts,
    build_metadata,
    export_onnx,
    fingerprint_dataframe,
    save_artifacts,
    temporal_split,
    train_xgb_multiclass,
)


WINDOW_DAYS = 365
LOG_PATH = PROJECT_ROOT / "reports" / "simulation_daily.json"
DAILY_DIR = PROJECT_ROOT / "data" / "raw" / "daily"


def _find_day_csv(day_index: int) -> Path:
    """Return the path of the N-th day CSV (1-indexed, sorted by date)."""
    files = sorted(DAILY_DIR.glob("test1_day_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No daily CSVs in {DAILY_DIR}. Run scripts/12a_extract_test1_days.py first."
        )
    if day_index < 1 or day_index > len(files):
        raise ValueError(f"--day must be in [1, {len(files)}], got {day_index}")
    return files[day_index - 1]


def _date_from_filename(path: Path) -> str:
    """Extract YYYY-MM-DD from test1_day_YYYY-MM-DD.csv"""
    stem = path.stem  # test1_day_YYYY-MM-DD
    return stem.replace("test1_day_", "")


def _next_day_csv(day_index: int) -> Path | None:
    files = sorted(DAILY_DIR.glob("test1_day_*.csv"))
    if day_index >= len(files):
        return None
    return files[day_index]  # 0-indexed access of the next file


def _clean_day_csv(day_csv_path: Path, liaison_path: Path) -> pd.DataFrame:
    """Run the same cleaning logic as scripts/01 but on a single day."""
    raw = read_csv(day_csv_path)
    liaison = read_csv(liaison_path)
    clean, _, _ = make_clean_dataset(raw, liaison)
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate one day of the rolling-window daily retrain (365j)."
    )
    parser.add_argument("--day", type=int, required=True, help="Day index in [1..N]")
    args = parser.parse_args()

    paths = default_paths()
    if not paths.processed_dataset_parquet.exists() or not paths.features_parquet.exists():
        raise FileNotFoundError(
            "Run Phase 3 (baseline post-test1) before launching the simulation."
        )

    day_csv = _find_day_csv(args.day)
    day_date = _date_from_filename(day_csv)
    next_day_csv = _next_day_csv(args.day)

    print(f"\n=== SIMULATE DAILY RETRAIN — Day {args.day} ({day_date}) ===")
    print(f"  Input CSV : {day_csv}")
    print(f"  Next day  : {next_day_csv.name if next_day_csv else '(none, last day)'}")
    print(f"  Window    : {WINDOW_DAYS} days ending {day_date}")
    print()

    t0 = time.time()

    # 1. Load clean.parquet, clean the new day, append, save back
    print("[1/5] Cleaning new day + appending to oncf_clean.parquet...")
    clean = read_parquet(paths.processed_dataset_parquet)
    day_clean = _clean_day_csv(day_csv, paths.raw_liaison)
    print(f"      {len(clean):,} existing + {len(day_clean):,} new = {len(clean) + len(day_clean):,}")
    clean = append_day_to_clean(clean, day_clean)
    write_parquet(clean, paths.processed_dataset_parquet)

    # 2. Rebuild features.parquet
    print("[2/5] Rebuilding features.parquet...")
    feats = build_training_rows(clean)
    write_parquet(feats, paths.features_parquet)

    # 3. Filter to sliding window of WINDOW_DAYS ending at day_date
    print(f"[3/5] Filtering to sliding window of {WINDOW_DAYS} days...")
    window = filter_sliding_window(
        feats, end_date=day_date, window_days=WINDOW_DAYS
    )
    print(f"      Window rows: {len(window):,}")
    if len(window) < 10000:
        print(f"WARNING: only {len(window)} rows in window. Skipping training.")
        sys.exit(2)

    # 4. Train on window (challenger hyperparams by default in train_xgb_multiclass)
    print("[4/5] Training (challenger hyperparams, ~20 min CPU)...")
    df_train, _ = temporal_split(window, time_col="DateHeureDepartVoyageSegment")
    arts = train_xgb_multiclass(
        df_train,
        label_col="LiaisonId",
        time_col="DateHeureDepartVoyageSegment",
    )
    internal_metrics = evaluate_model(arts, window)
    print(f"      Internal split metrics: HR@1={internal_metrics['hit_rate@1']:.4f}, "
          f"HR@3={internal_metrics['hit_rate@3']:.4f}, MRR@3={internal_metrics['mrr@3']:.4f}")

    # 5. Save artifacts to models/sim/day_N/
    sim_dir = paths.models_dir / "sim" / f"day_{args.day}"
    sim_dir.mkdir(parents=True, exist_ok=True)
    model_path = sim_dir / paths.xgb_model_path.name
    le_path = sim_dir / paths.label_encoder_path.name
    onnx_path = sim_dir / paths.onnx_model_path.name

    metadata = build_metadata(
        arts,
        train_rows=len(df_train),
        test_rows=internal_metrics["test_rows"],
        metrics={k: v for k, v in internal_metrics.items() if k != "test_rows"},
        dataset_fingerprint=fingerprint_dataframe(window),
    )
    save_artifacts(arts, model_path=model_path, label_encoder_path=le_path, metadata=metadata)
    export_onnx(arts.pipeline, onnx_path)

    # 6. Eval on next day (J+1) if available — build a Recommender from sim model
    next_day_metrics = None
    if next_day_csv is not None:
        print(f"[5/5] Evaluating on J+1 ({_date_from_filename(next_day_csv)})...")
        next_day_clean = _clean_day_csv(next_day_csv, paths.raw_liaison)
        # Need to also pass the (now-updated) clean parquet so user histories include up through day N
        sim_paths = paths  # reuse default paths; Recommender.from_paths reads models/, not models/sim/
        # Build Recommender from the simulation artifacts
        rec = Recommender.from_paths(
            sim_paths,
            xgb_model_path=model_path,
            label_encoder_path=le_path,
            onnx_model_path=onnx_path,
        )
        next_day_metrics = eval_on_next_day(rec, next_day_clean)
        print(f"      J+1 metrics: HR@1={next_day_metrics['hr@1']:.4f}, "
              f"HR@3={next_day_metrics['hr@3']:.4f}, MRR@3={next_day_metrics['mrr@3']:.4f}, "
              f"n_eval={next_day_metrics['n_eval']}")
    else:
        print("[5/5] No J+1 available (last day) — skipping next-day eval.")

    # 7. Guardrail vs baseline post-test1 (the current prod meta)
    baseline_metrics = load_current_metrics(paths.xgb_model_path.with_suffix(".meta.json"))
    passes, reason = check_guardrail(baseline_metrics, internal_metrics, threshold=0.05)

    # 8. Log entry
    duration = time.time() - t0
    entry = {
        "day": args.day,
        "date": day_date,
        "window_days": WINDOW_DAYS,
        "train_rows": len(df_train),
        "internal_split_metrics": {
            "hr@1": internal_metrics["hit_rate@1"],
            "hr@3": internal_metrics["hit_rate@3"],
            "mrr@3": internal_metrics["mrr@3"],
            "test_rows": internal_metrics["test_rows"],
        },
        "next_day_metrics": next_day_metrics,
        "duration_seconds": round(duration, 1),
        "guardrail_passes": passes,
        "guardrail_reason": reason,
        "model_dir": str(sim_dir.relative_to(PROJECT_ROOT)),
    }
    log_simulation_entry(LOG_PATH, entry)
    print()
    print(f"=== DONE — duration {duration:.1f}s — log: {LOG_PATH.relative_to(PROJECT_ROOT)} ===")
    print(f"    guardrail: {'OK' if passes else 'FLAG'} — {reason}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.2 : Vérifier que les imports résolvent (smoke compile)**

```powershell
.venv\Scripts\python.exe -c "import importlib.util; s=importlib.util.spec_from_file_location('m','scripts\\12_simulate_daily_retrain.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print('imports OK')"
```

Expected: `imports OK` (pas d'erreur d'import).

Si erreur `ImportError: cannot import name 'X' from 'rec_oncf.recommender'` ou similaire (notamment sur `Recommender.from_paths` signature) : ouvrir `src/rec_oncf/recommender.py`, vérifier que la signature accepte les overrides `xgb_model_path`, `label_encoder_path`, `onnx_model_path`. Si pas, **ajouter ces paramètres optionnels** à `from_paths` (default = `paths.xgb_model_path` etc.) ; refaire un commit dédié dans `recommender.py` puis revenir ici.

- [ ] **Step 8.3 : Commit**

```powershell
git add scripts\12_simulate_daily_retrain.py
git commit -m "feat(scripts): add 12_simulate_daily_retrain.py — sliding-window 365j simulation"
```

---

## Task 9 : Phase 5 — Lancer Jour 1 (validation pipeline complet)

**Files:**
- Run: `scripts/12_simulate_daily_retrain.py --day 1`
- Modifies: `data/processed/{oncf_clean,features}.parquet`, `models/sim/day_1/*`, `reports/simulation_daily.json`

- [ ] **Step 9.1 : Backup l'état pré-Jour-1**

(au cas où on doit relancer du même point de départ)

```powershell
New-Item -ItemType Directory -Force "data\processed\pre_sim"
Copy-Item -Path "data\processed\oncf_clean.parquet","data\processed\features.parquet" -Destination "data\processed\pre_sim\" -Force
```

- [ ] **Step 9.2 : Lancer Jour 1**

```powershell
.venv\Scripts\python.exe scripts\12_simulate_daily_retrain.py --day 1
```

Expected stdout (durée estimée 20-30 min) :
```
=== SIMULATE DAILY RETRAIN — Day 1 (YYYY-MM-DD) ===
  Input CSV : ...test1_day_YYYY-MM-DD.csv
  Next day  : test1_day_YYYY-MM-DD.csv
  Window    : 365 days ending YYYY-MM-DD

[1/5] Cleaning new day + appending to oncf_clean.parquet...
      XXX,XXX existing + N new = XXX,XXX
[2/5] Rebuilding features.parquet...
[3/5] Filtering to sliding window of 365 days...
      Window rows: ~80,000
[4/5] Training (challenger hyperparams, ~20 min CPU)...
      Internal split metrics: HR@1=0.XXXX, HR@3=0.XXXX, MRR@3=0.XXXX
[5/5] Evaluating on J+1 (YYYY-MM-DD)...
      J+1 metrics: HR@1=0.XXXX, HR@3=0.XXXX, MRR@3=0.XXXX, n_eval=N

=== DONE — duration XXXXs — log: reports\simulation_daily.json ===
    guardrail: OK — HR@1 current=0.XXXX, new=0.XXXX, drop=0.XXXX
```

- [ ] **Step 9.3 : Vérifier le log**

```powershell
type reports\simulation_daily.json
```

Expected : 1 entrée bien formée avec `day=1` et toutes les clés (`date`, `window_days`, `train_rows`, `internal_split_metrics`, `next_day_metrics`, `duration_seconds`, `guardrail_passes`).

Vérifier aussi `models/sim/day_1/` :
```powershell
dir models\sim\day_1\
```

Expected : `xgb_ranker.json`, `xgb_ranker.onnx`, `label_encoder.joblib`, `xgb_ranker.meta.json` (4 fichiers).

- [ ] **Step 9.4 : Mettre à jour CLAUDE.md**

Ajouter sous "État actuel" :

```markdown
- **2026-05-20 (Phase 4 — Jour 1)** : Premier retrain simulé OK. Window 365j, train_rows=XX,XXX, durée XXs.
  - Split interne : HR@1=0.XXXX (vs baseline 0.XXXX, drop=X.XXXX) — guardrail OK/FLAG
  - J+1 : HR@1=0.XXXX (n_eval=XXX)
  - Artefacts : `models/sim/day_1/`. Prochain : `python scripts/12_simulate_daily_retrain.py --day 2`
```

- [ ] **Step 9.5 : Commit**

```powershell
git add reports\simulation_daily.json CLAUDE.md
git commit -m "feat(simulation): day 1 results — sliding window 365j, HR@1=0.XXXX"
```

---

## Task 10 : Phase 5 — Lancer Jours 2 à 7 (à la demande, un par un)

**À lancer manuellement par l'utilisateur quand il a du temps CPU disponible. Pas automatisé volontairement.**

Pour chacun de N ∈ [2, 3, 4, 5, 6, 7] :

- [ ] **Step 10.N.1 : Lancer le jour N**

```powershell
.venv\Scripts\python.exe scripts\12_simulate_daily_retrain.py --day N
```

Attendre la fin (~20-30 min).

- [ ] **Step 10.N.2 : Vérifier l'entrée ajoutée dans simulation_daily.json**

```powershell
.venv\Scripts\python.exe -c "import json; d=json.loads(open('reports/simulation_daily.json',encoding='utf-8').read()); print(json.dumps(d[-1], indent=2, ensure_ascii=False))"
```

- [ ] **Step 10.N.3 : Update CLAUDE.md + commit**

```powershell
# Ajouter dans CLAUDE.md une ligne par jour
git add reports\simulation_daily.json CLAUDE.md
git commit -m "feat(simulation): day N results — HR@1=0.XXXX, J+1=0.XXXX"
```

**Note** : à `--day 7`, `next_day_metrics` sera `null` (pas de J+1 disponible).

---

## Task 11 : Phase 6 — Mises à jour rapport et CLAUDE.md final

**Files:**
- Modify: `rapport_pfa_v2.tex`
- Modify: `scripts/generate_extra_figures.py` (ajout d'une figure)
- Modify: `CLAUDE.md`

- [ ] **Step 11.1 : Ajouter une figure simulation à `generate_extra_figures.py`**

Ouvrir `scripts/generate_extra_figures.py`, ajouter à la fin (avant `if __name__`) la fonction :

```python
def fig_simulation_daily(out_path: Path) -> None:
    """Plot HR@1 over the 7 simulation days, internal split + J+1."""
    import json
    log = json.loads((Path(__file__).resolve().parents[1] / "reports" / "simulation_daily.json").read_text(encoding="utf-8"))
    days = [e["day"] for e in log]
    internal = [e["internal_split_metrics"]["hr@1"] for e in log]
    next_day = [e["next_day_metrics"]["hr@1"] if e["next_day_metrics"] else None for e in log]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(days, internal, marker="o", label="HR@1 split interne", color="#1f77b4")
    ax.plot([d for d, v in zip(days, next_day) if v is not None],
            [v for v in next_day if v is not None],
            marker="s", label="HR@1 sur J+1", color="#ff7f0e")
    ax.set_xlabel("Jour de simulation")
    ax.set_ylabel("HR@1")
    ax.set_title("Stabilité du modèle sur 7 retrains quotidiens (fenêtre 365j)")
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```

Puis l'invoquer dans `main()` (suivre le pattern existant des autres `fig_*` dans le fichier).

- [ ] **Step 11.2 : Générer la figure**

```powershell
.venv\Scripts\python.exe scripts\generate_extra_figures.py
```

Vérifier que `pic/simulation_daily_hr1.png` (ou nom équivalent selon le pattern existant) est bien créé.

- [ ] **Step 11.3 : Modifier `rapport_pfa_v2.tex` (sous-section dédiée)**

Ajouter dans le chapitre Phase 3 (ou Sprint 2, selon emplacement le plus logique) une nouvelle sous-section :

```latex
\subsection{Intégration des données 2021 et simulation de retrain quotidien}

\subsubsection{Enrichissement de l'index horaires}

Le fichier \texttt{horaire.csv} fourni par l'ONCF a été enrichi en cours de stage
pour intégrer les trajets de la LGV Al Boraq (Tanger \(\leftrightarrow\) Casablanca,
Rabat, Kénitra) ainsi que les retours Marrakech manquants. La couverture de
l'index O/D est passée de 57\,\% à XX\,\% des LiaisonId présents dans le jeu
d'entraînement.

\subsubsection{Baseline post-2021}

Le jeu de données 2021 (\texttt{test1.csv}) a été ajouté à \texttt{oncf\_data.csv}
après exclusion des 7 derniers jours. Le modèle ré-entraîné sur cette base élargie
constitue le point de départ de la simulation.

\begin{table}[H]
\centering
\caption{Comparaison baseline avant et après intégration de 2021}
\begin{tabular}{lccc}
\toprule
\textbf{Modèle} & \textbf{HR@1} & \textbf{HR@3} & \textbf{MRR@3} \\
\midrule
Prod 2018--2020 (challenger) & 0.7691 & 0.9100 & 0.8333 \\
Baseline post-test1 (+2021)  & 0.XXXX & 0.XXXX & 0.XXXX \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Simulation quotidienne (fenêtre glissante 365j)}

Pour reproduire le scénario de production (Task Scheduler quotidien à 2h du
matin), nous simulons 7 retrains consécutifs sur les 7 derniers jours retenus
de 2021. Chaque jour, la fenêtre d'entraînement est décalée d'un jour : un
nouveau jour est ajouté et le plus ancien (J\(-\)365) est retiré. Ce choix
garantit un temps d'entraînement constant et capture une saisonnalité annuelle
complète.

\begin{table}[H]
\centering
\caption{Résultats des 7 retrains quotidiens (chaque modèle évalué sur split
interne 80/20 et sur J\(+\)1 lorsque disponible)}
\begin{tabular}{lcccccc}
\toprule
\textbf{Jour} & \textbf{Date} & \textbf{Train rows} &
\textbf{HR@1 int.} & \textbf{HR@1 J+1} & \textbf{Durée} & \textbf{Guardrail} \\
\midrule
J1 & YYYY-MM-DD & XX,XXX & 0.XXXX & 0.XXXX & XXs & OK \\
\ldots & \ldots & \ldots & \ldots & \ldots & \ldots & \ldots \\
J7 & YYYY-MM-DD & XX,XXX & 0.XXXX & --- & XXs & OK \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{pic/simulation_daily_hr1.png}
\caption{Évolution de HR@1 sur 7 retrains quotidiens (split interne et évaluation
sur le jour suivant). La stabilité observée valide le scénario de fenêtre
glissante 365j.}
\end{figure}
```

(Remplir les `XX,XXX`, `0.XXXX`, dates et durées à partir de `reports/simulation_daily.json`.)

- [ ] **Step 11.4 : Mise à jour CLAUDE.md (état final)**

Mettre à jour la section "État actuel" pour refléter que la simulation est terminée :

```markdown
- **2026-05-20 (Phase 4 — Jours 1 à 7 complétés)** : Simulation quotidienne terminée.
  - 7 modèles dans `models/sim/day_1/` à `models/sim/day_7/`
  - Log complet : `reports/simulation_daily.json`
  - HR@1 médian split interne : 0.XXXX ; HR@1 médian J+1 : 0.XXXX
  - Aucun guardrail déclenché (toutes les chutes restent < 5pp).
- **2026-05-20 (Phase 6)** : Rapport mis à jour avec nouvelle sous-section "Intégration 2021 + simulation quotidienne" + figure `simulation_daily_hr1.png`.
```

Mettre aussi à jour la table principale "État actuel — Intégration horaire.csv (2026-05-19)" pour étendre le statut à la nouvelle session, et ajouter le détail des nouveaux scripts dans la table "Fichiers clés créés / modifiés".

- [ ] **Step 11.5 : Commit final**

```powershell
git add rapport_pfa_v2.tex scripts\generate_extra_figures.py pic\simulation_daily_hr1.png CLAUDE.md
git commit -m "docs(rapport): add test1 + daily simulation section, figure and CLAUDE.md update"
```

- [ ] **Step 11.6 : Vérification finale — tous les tests verts**

```powershell
.venv\Scripts\python.exe -m pytest tests\ -v
```

Expected: **≥ 149 tests PASSED** (135 actuels + 5 extract_days + 3 dataset_extra_csv + 6 simulate_daily + 1 training override = 150). Si un test échoue, le corriger avant tout commit.

---

## Self-Review (à faire après lecture complète du plan, avant exécution)

**1. Spec coverage** : chaque exigence du spec est-elle adressée ?

| Section spec | Task(s) du plan |
|---|---|
| Phase 1 (tests horaire) | Task 1 |
| Phase 2 (extraction 7 jours) | Tasks 2, 3 |
| Phase 3 (baseline post-test1) | Tasks 4, 5 |
| Phase 4 (simulation jour par jour) | Tasks 6, 7, 8, 9, 10 |
| Phase 5 (tests) | Tasks 2.1, 4.1, 6.1, 7.1 (TDD intégré) + step 5.6 (régression) + step 11.6 (final) |
| Phase 6 (rapport) | Task 11 |

✅ Toutes les phases couvertes.

**2. Hyperparam consistency** : les hyperparams "challenger" sont harmonisés (Task 6 fait défauts = challenger, ce qui rend `03_train_ranker.py` et `12_simulate_daily_retrain.py` cohérents avec le prod actuel).

**3. Path/name consistency** :
- `models/sim/day_N/` cohérent partout (Task 7, Task 8, Task 9)
- `data/raw/daily/test1_day_YYYY-MM-DD.csv` cohérent (Task 3, Task 8)
- `reports/simulation_daily.json` cohérent (Task 7, Task 8, Task 11)
- `models/archive/20260520T_pre_test1/` cohérent (Task 5)

**4. Risques identifiés dans le spec** :
- Couverture horaire < 75% : géré par Task 1.2 (stop + investigate)
- test1.csv format différent : géré par read_csv qui détecte le séparateur + ValueError dans extract_last_n_days si dates non parseables
- Baseline HR@1 chute > 5pp : Task 5.4 le détectera (métrique stdout), décision à l'utilisateur car Option A acceptée
- Durée > 30 min : Task 9.2 valide empiriquement sur le Jour 1

✅ Plan auto-cohérent et complet.

---

## Execution Handoff

Plan complet et sauvegardé à `docs/superpowers/plans/2026-05-20-test1-daily-retrain.md`. Deux options d'exécution :

**1. Subagent-Driven (recommandé)** — Je dispatch un subagent frais par Task, review entre chaque Task, itération rapide.

**2. Inline Execution** — J'exécute les Tasks dans cette session avec executing-plans, batch d'exécutions et checkpoints pour ta review.

**Laquelle préfères-tu ?**
