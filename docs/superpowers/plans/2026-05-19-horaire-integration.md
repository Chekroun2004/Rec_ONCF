# Horaire CSV Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remplacer le scraper oncf.ma par un index local construit depuis `horaire.csv` pour servir `GET /schedule/{liaison_id}` en mode offline.

**Architecture:** `horaire.csv` est parsé en un index dict `{(origin, dest): [{depart, arrive}, ...]}` sauvegardé dans `models/schedule_index.joblib`. L'API charge l'index au startup et sert les prochains départs après l'heure courante (Casablanca TZ). La signature de `GET /schedule/{liaison_id}` reste inchangée.

**Tech Stack:** pandas (parsing CSV), joblib (sérialisation index), FastAPI (endpoint existant), ZoneInfo (Casablanca TZ), pytest + patch (tests)

---

## Fichiers

| Action | Fichier | Rôle |
|---|---|---|
| MODIFY | `src/rec_oncf/config.py` | Ajouter `horaire_csv_path`, `schedule_index_path` dans `Paths` |
| CREATE | `src/rec_oncf/local_schedule.py` | `parse_horaire_csv`, `build_od_index`, `get_local_schedule`, `save/load_schedule_index` |
| CREATE | `scripts/11_build_schedule_index.py` | Build + sauvegarde de l'index depuis horaire.csv |
| MODIFY | `apps/api/main.py` | Import + startup + 2 call sites (`schedule_endpoint` + `include_schedule`) |
| CREATE | `tests/test_local_schedule.py` | 15 tests TDD pour `local_schedule.py` |
| MODIFY | `tests/test_api.py` | Mise à jour de 4 tests qui mockaient `get_schedule` |

---

## Task 1 — Ajouter les chemins dans config.py

**Files:**
- Modify: `src/rec_oncf/config.py`

- [ ] **Step 1.1: Ajouter les deux champs dans la dataclass `Paths`**

```python
# src/rec_oncf/config.py  — ajouter après popularity_path
horaire_csv_path: Path
schedule_index_path: Path
```

La dataclass complète après modification :

```python
@dataclass(frozen=True)
class Paths:
    project_root: Path
    desktop: Path
    raw_oncf_data: Path
    raw_liaison: Path

    processed_dir: Path
    processed_dataset_parquet: Path
    processed_dataset_csv: Path
    features_parquet: Path

    models_dir: Path
    xgb_model_path: Path
    label_encoder_path: Path
    cold_start_path: Path
    onnx_model_path: Path
    popularity_path: Path

    horaire_csv_path: Path        # ← nouveau
    schedule_index_path: Path     # ← nouveau
```

- [ ] **Step 1.2: Initialiser les nouveaux chemins dans `default_paths()`**

```python
# src/rec_oncf/config.py  — dans default_paths(), après popularity_path =
horaire_csv_path    = desktop / "horaire.csv"
schedule_index_path = models_dir / "schedule_index.joblib"
```

Et les ajouter dans le `return Paths(...)` :

```python
return Paths(
    project_root=project_root,
    desktop=desktop,
    raw_oncf_data=raw_oncf_data,
    raw_liaison=raw_liaison,
    processed_dir=processed_dir,
    processed_dataset_parquet=processed_dataset_parquet,
    processed_dataset_csv=processed_dataset_csv,
    features_parquet=features_parquet,
    models_dir=models_dir,
    xgb_model_path=xgb_model_path,
    label_encoder_path=label_encoder_path,
    cold_start_path=cold_start_path,
    onnx_model_path=onnx_model_path,
    popularity_path=popularity_path,
    horaire_csv_path=horaire_csv_path,           # ← nouveau
    schedule_index_path=schedule_index_path,     # ← nouveau
)
```

- [ ] **Step 1.3: Vérifier que les tests existants passent encore**

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v --tb=short -q
```

Attendu : tous les tests passent (aucune modification de logique ici).

- [ ] **Step 1.4: Commit**

```powershell
git add src/rec_oncf/config.py
git commit -m "feat(config): add horaire_csv_path and schedule_index_path to Paths"
```

---

## Task 2 — Créer `src/rec_oncf/local_schedule.py` (TDD)

**Files:**
- Create: `src/rec_oncf/local_schedule.py`
- Create: `tests/test_local_schedule.py`

### Step 2.1 — Écrire le fichier de tests d'abord

- [ ] **Step 2.1: Créer `tests/test_local_schedule.py` complet**

```python
# tests/test_local_schedule.py
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rec_oncf.local_schedule import (
    build_od_index,
    get_local_schedule,
    load_schedule_index,
    parse_horaire_csv,
    save_schedule_index,
)

# ── fixture CSV minimal ────────────────────────────────────────────────────
# Train 1 : CASA PORT → MOHAMMEDIA → RABAT VILLE → KENITRA
# Train 10: KENITRA → RABAT VILLE → CASA PORT
SAMPLE_CSV = (
    "CASA PORT;00:00:00;06:15:00;1;1\n"
    "MOHAMMEDIA;06:36:00;06:37:00;5;1\n"
    "RABAT VILLE;07:31:00;07:33:00;12;1\n"
    "KENITRA;08:07:00;00:00:00;17;1\n"
    "KENITRA;00:00:00;08:07:00;1;10\n"
    "RABAT VILLE;08:40:00;08:42:00;5;10\n"
    "CASA PORT;09:20:00;00:00:00;12;10\n"
)


@pytest.fixture
def horaire_csv_path(tmp_path):
    p = tmp_path / "horaire.csv"
    p.write_text(SAMPLE_CSV, encoding="utf-8")
    return p


@pytest.fixture
def df(horaire_csv_path):
    return parse_horaire_csv(horaire_csv_path)


@pytest.fixture
def od_index(df):
    return build_od_index(df)


# ── parse_horaire_csv ──────────────────────────────────────────────────────

def test_parse_columns_exist(df):
    assert set(df.columns) == {"gare", "arrivee", "depart", "ordre", "num_commercial"}


def test_parse_row_count(df):
    assert len(df) == 7


def test_parse_normalizes_station_names(df):
    # strip + upper déjà fait par parse
    assert all(name == name.strip().upper() for name in df["gare"])


def test_parse_whitespace_padding(tmp_path):
    # Gare avec espaces de padding dans le CSV réel
    p = tmp_path / "h.csv"
    p.write_text("BENI NSAR VILLE               ;00:00:00;10:00:00;1;5\n"
                 "NADOR VILLE;11:00:00;00:00:00;3;5\n", encoding="utf-8")
    df2 = parse_horaire_csv(p)
    assert df2.iloc[0]["gare"] == "BENI NSAR VILLE"


# ── build_od_index ─────────────────────────────────────────────────────────

def test_od_index_is_dict(od_index):
    assert isinstance(od_index, dict)


def test_od_index_basic_pairs(od_index):
    # Train 1 doit générer (CASA PORT, KENITRA) et autres
    assert ("CASA PORT", "KENITRA") in od_index
    assert ("CASA PORT", "MOHAMMEDIA") in od_index
    assert ("MOHAMMEDIA", "RABAT VILLE") in od_index


def test_od_index_no_reverse_pair(od_index):
    # (A→B) existe mais pas (B→A) dans le même trajet
    assert ("KENITRA", "CASA PORT") in od_index      # Train 10
    assert ("CASA PORT", "KENITRA") in od_index      # Train 1
    # Vérifie que les deux sont indépendants (trains différents)
    assert od_index[("KENITRA", "CASA PORT")][0]["depart"] == "08:07"
    assert od_index[("CASA PORT", "KENITRA")][0]["depart"] == "06:15"


def test_od_index_terminus_not_origin(od_index):
    # KENITRA en train 1 a depart=00:00:00 → ne doit PAS être une origine dans ce trajet
    # La seule paire (KENITRA, x) vient du train 10
    kenitra_origins = [v for (o, _), v in od_index.items() if o == "KENITRA"]
    # Tous ces départs doivent être "08:07" (train 10), pas "00:00"
    for trips in kenitra_origins:
        for t in trips:
            assert t["depart"] != "00:00"


def test_od_index_origin_not_destination(od_index):
    # CASA PORT en train 1 a arrivee=00:00:00 → ne doit PAS être une destination dans ce trajet
    # La seule paire (x, CASA PORT) vient du train 10
    casa_dests = [v for (_, d), v in od_index.items() if d == "CASA PORT"]
    for trips in casa_dests:
        for t in trips:
            assert t["arrive"] != "00:00"


def test_od_index_times_are_hhmm(od_index):
    for trips in od_index.values():
        for t in trips:
            assert len(t["depart"]) == 5
            assert t["depart"][2] == ":"
            assert len(t["arrive"]) == 5
            assert t["arrive"][2] == ":"


def test_od_index_sorted_by_departure(od_index):
    # (CASA PORT, KENITRA) a un seul train, mais (KENITRA, ...) peut en avoir plusieurs
    for trips in od_index.values():
        deps = [t["depart"] for t in trips]
        assert deps == sorted(deps)


# ── get_local_schedule ─────────────────────────────────────────────────────

LIAISON_MAP = {
    "R1": ("CASA PORT", "KENITRA"),
    "R2": ("KENITRA", "CASA PORT"),
    "R3": ("CASA PORT", "NOWHERE"),   # gare absente de l'index
}


def test_get_local_schedule_hit(od_index):
    result = get_local_schedule("R1", LIAISON_MAP, od_index)
    assert isinstance(result, list)
    assert len(result) > 0
    assert result[0]["depart"] == "06:15"
    assert result[0]["arrive"] == "08:07"


def test_get_local_schedule_unknown_liaison(od_index):
    assert get_local_schedule("UNKNOWN", LIAISON_MAP, od_index) == []


def test_get_local_schedule_unmapped_station(od_index):
    # R3 → NOWHERE n'est pas dans l'index
    assert get_local_schedule("R3", LIAISON_MAP, od_index) == []


def test_get_local_schedule_time_filter_upcoming(od_index):
    # Train 1 part à 06:15. Si now=05:00, il doit être inclus.
    now = datetime(2026, 5, 19, 5, 0, tzinfo=ZoneInfo("Africa/Casablanca"))
    result = get_local_schedule("R1", LIAISON_MAP, od_index, now=now)
    assert len(result) > 0
    assert result[0]["depart"] == "06:15"


def test_get_local_schedule_time_filter_past(od_index):
    # Train 1 part à 06:15. Si now=09:00 (après tous les trains), on retourne le début de liste.
    now = datetime(2026, 5, 19, 9, 0, tzinfo=ZoneInfo("Africa/Casablanca"))
    result = get_local_schedule("R1", LIAISON_MAP, od_index, now=now)
    # Pas de train restant → retourne wrap (premiers de la liste)
    assert isinstance(result, list)
    assert len(result) > 0


def test_get_local_schedule_no_now_returns_all(od_index):
    result = get_local_schedule("R2", LIAISON_MAP, od_index)
    # Sans filtre horaire, retourne tous les trains
    assert len(result) >= 1


# ── save / load roundtrip ──────────────────────────────────────────────────

def test_save_load_roundtrip(od_index, tmp_path):
    path = tmp_path / "schedule_index.joblib"
    save_schedule_index(od_index, path)
    loaded = load_schedule_index(path)
    assert loaded == od_index


def test_load_missing_file_returns_empty(tmp_path):
    result = load_schedule_index(tmp_path / "nonexistent.joblib")
    assert result == {}
```

- [ ] **Step 2.2: Vérifier que les tests échouent (module pas encore créé)**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_local_schedule.py -v --tb=short 2>&1 | head -20
```

Attendu : `ModuleNotFoundError: No module named 'rec_oncf.local_schedule'`

### Step 2.3 — Implémenter `local_schedule.py`

- [ ] **Step 2.3: Créer `src/rec_oncf/local_schedule.py`**

```python
# src/rec_oncf/local_schedule.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd


def parse_horaire_csv(path: Path) -> pd.DataFrame:
    """Lit horaire.csv (sans en-tête, séparateur ;) et normalise les noms de gare."""
    df = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["gare", "arrivee", "depart", "ordre", "num_commercial"],
        dtype=str,
    )
    df["gare"] = df["gare"].str.strip().str.upper()
    df["ordre"] = df["ordre"].astype(int)
    return df


def build_od_index(
    df: pd.DataFrame,
) -> dict[tuple[str, str], list[dict[str, str]]]:
    """Génère toutes les paires O/D depuis les trajets de horaire.csv.

    Règle : 00:00:00 signifie terminus/gare de départ (pas minuit).
    - arrivee=00:00:00 → cette gare NE PEUT PAS être une destination
    - depart=00:00:00  → cette gare NE PEUT PAS être une origine
    """
    index: dict[tuple[str, str], list[dict[str, str]]] = {}

    for _num, group in df.groupby("num_commercial"):
        stops = group.sort_values("ordre").to_dict("records")

        for i, orig in enumerate(stops):
            if orig["depart"] == "00:00:00":
                continue  # terminus — pas de départ valide
            dep_hhmm = orig["depart"][:5]  # "HH:MM"

            for dest in stops[i + 1 :]:
                if dest["arrivee"] == "00:00:00":
                    continue  # gare de départ du trajet — pas d'arrivée valide
                arr_hhmm = dest["arrivee"][:5]  # "HH:MM"

                key = (orig["gare"], dest["gare"])
                index.setdefault(key, []).append(
                    {"depart": dep_hhmm, "arrive": arr_hhmm}
                )

    for key in index:
        index[key].sort(key=lambda x: x["depart"])

    return index


def get_local_schedule(
    liaison_id: str,
    liaison_map: dict[str, tuple[str, str]],
    od_index: dict[tuple[str, str], list[dict[str, str]]],
    now: datetime | None = None,
) -> list[dict[str, str]]:
    """Retourne les prochains départs pour un LiaisonId.

    Si now est fourni, filtre les départs déjà passés.
    Si tous les trains sont passés, retourne les 3 premiers (cycle du lendemain).
    """
    stations = liaison_map.get(str(liaison_id))
    if not stations:
        return []

    origin, dest = stations[0].strip().upper(), stations[1].strip().upper()
    trips = od_index.get((origin, dest), [])
    if not trips:
        return []

    if now is None:
        return trips

    current_hhmm = now.strftime("%H:%M")
    upcoming = [t for t in trips if t["depart"] >= current_hhmm]
    return upcoming if upcoming else trips[:3]


def save_schedule_index(
    od_index: dict[tuple[str, str], list[dict[str, str]]],
    path: Path,
) -> None:
    """Sérialise l'index O/D avec joblib."""
    joblib.dump(od_index, path)


def load_schedule_index(
    path: Path,
) -> dict[tuple[str, str], list[dict[str, str]]]:
    """Charge l'index O/D depuis le disque. Retourne {} si le fichier est absent."""
    if not path.exists():
        return {}
    return joblib.load(path)
```

- [ ] **Step 2.4: Vérifier que tous les tests passent**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_local_schedule.py -v
```

Attendu : **15/15 PASSED**

- [ ] **Step 2.5: Vérifier que les tests existants passent encore**

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v -q
```

Attendu : tous les tests précédents + les 15 nouveaux passent.

- [ ] **Step 2.6: Commit**

```powershell
git add src/rec_oncf/local_schedule.py tests/test_local_schedule.py
git commit -m "feat(schedule): add local_schedule module with O/D index from horaire.csv"
```

---

## Task 3 — Script `scripts/11_build_schedule_index.py`

**Files:**
- Create: `scripts/11_build_schedule_index.py`

- [ ] **Step 3.1: Créer le script**

```python
# scripts/11_build_schedule_index.py
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rec_oncf.config import default_paths
from rec_oncf.io import read_parquet
from rec_oncf.local_schedule import (
    build_od_index,
    parse_horaire_csv,
    save_schedule_index,
)
from rec_oncf.schedule import build_liaison_station_map


def main() -> None:
    paths = default_paths()

    if not paths.horaire_csv_path.exists():
        raise FileNotFoundError(
            f"horaire.csv introuvable : {paths.horaire_csv_path}\n"
            "Placez le fichier sur le Desktop."
        )

    print(f"Chargement de {paths.horaire_csv_path}...")
    df = parse_horaire_csv(paths.horaire_csv_path)
    n_trains = df["num_commercial"].nunique()
    n_gares = df["gare"].nunique()
    print(f"  {len(df)} arrêts  |  {n_trains} trajets  |  {n_gares} gares distinctes")

    print("Construction de l'index O/D...")
    od_index = build_od_index(df)
    print(f"  {len(od_index)} paires O/D générées")

    paths.models_dir.mkdir(exist_ok=True)
    save_schedule_index(od_index, paths.schedule_index_path)
    print(f"Index sauvegardé → {paths.schedule_index_path}")

    # Couverture : combien de LiaisonId sont servis par l'index ?
    if paths.processed_dataset_parquet.exists():
        clean = read_parquet(paths.processed_dataset_parquet)
        liaison_map = build_liaison_station_map(clean)
        covered = sum(
            1
            for (o, d) in liaison_map.values()
            if (o.strip().upper(), d.strip().upper()) in od_index
        )
        print(f"Couverture : {covered}/{len(liaison_map)} LiaisonId présents dans l'index")
    else:
        print("(oncf_clean.parquet absent — couverture non calculée)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.2: Exécuter le script**

```powershell
.venv\Scripts\python.exe scripts/11_build_schedule_index.py
```

Attendu (exemple) :
```
Chargement de C:\Users\omarc\Desktop\horaire.csv...
  2683 arrêts  |  395 trajets  |  NNN gares distinctes
Construction de l'index O/D...
  XXXX paires O/D générées
Index sauvegardé → ...\models\schedule_index.joblib
Couverture : NNN/1011 LiaisonId présents dans l'index
```

- [ ] **Step 3.3: Commit**

```powershell
git add scripts/11_build_schedule_index.py models/schedule_index.joblib
git commit -m "feat(schedule): add script 11 to build O/D index from horaire.csv"
```

---

## Task 4 — Modifier `apps/api/main.py`

**Files:**
- Modify: `apps/api/main.py`

Quatre changements dans ce fichier :
1. Import — ajouter `get_local_schedule`, `load_schedule_index` ; retirer `get_schedule`
2. Startup — charger `schedule_index` dans `app.state`
3. `/schedule/{id}` — remplacer `get_schedule` par `get_local_schedule`
4. `/recommend include_schedule` — même remplacement

- [ ] **Step 4.1: Modifier l'import ligne 28**

Remplacer :
```python
from rec_oncf.schedule import build_liaison_station_map, get_schedule
```
Par :
```python
from rec_oncf.local_schedule import get_local_schedule, load_schedule_index
from rec_oncf.schedule import build_liaison_station_map
```

- [ ] **Step 4.2: Ajouter le chargement de l'index dans le startup (lifespan)**

Dans la fonction lifespan, après la ligne `app.state.liaison_map = build_liaison_station_map(clean)`, ajouter :

```python
    app.state.schedule_index = load_schedule_index(paths.schedule_index_path)
    if app.state.schedule_index:
        logger.info(
            f"Schedule index loaded: {len(app.state.schedule_index)} O/D pairs"
        )
    else:
        logger.warning(
            "schedule_index.joblib not found — run scripts/11_build_schedule_index.py"
        )
```

- [ ] **Step 4.3: Modifier `schedule_endpoint`**

Remplacer :
```python
@app.get("/schedule/{liaison_id}", response_model=ScheduleResponse)
def schedule_endpoint(liaison_id: str):
    now = datetime.now(tz=ZoneInfo("Africa/Casablanca"))
    deps = get_schedule(liaison_id, app.state.liaison_map, now, redis_client=app.state.redis)
    return {"liaison_id": liaison_id, "schedule": deps}
```

Par :
```python
@app.get("/schedule/{liaison_id}", response_model=ScheduleResponse)
def schedule_endpoint(liaison_id: str):
    now = datetime.now(tz=ZoneInfo("Africa/Casablanca"))
    deps = get_local_schedule(
        liaison_id, app.state.liaison_map, app.state.schedule_index, now
    )
    return {"liaison_id": liaison_id, "schedule": deps}
```

- [ ] **Step 4.4: Modifier le bloc `include_schedule` dans `/recommend`**

Remplacer (environ ligne 146-149) :
```python
        def _fetch(lid: str) -> tuple[str, list]:
            return lid, get_schedule(lid, app.state.liaison_map, now, redis_client=app.state.redis)
```

Par :
```python
        def _fetch(lid: str) -> tuple[str, list]:
            return lid, get_local_schedule(
                lid, app.state.liaison_map, app.state.schedule_index, now
            )
```

- [ ] **Step 4.5: Vérifier que les tests existants passent**

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v -q
```

Note : certains tests api vont **échouer** car ils mockent encore `apps.api.main.get_schedule` — c'est attendu. On les corrige à l'étape suivante.

- [ ] **Step 4.6: Commit partiel (main.py seulement)**

```powershell
git add apps/api/main.py
git commit -m "feat(api): wire local schedule index into startup and endpoints"
```

---

## Task 5 — Mettre à jour `tests/test_api.py`

**Files:**
- Modify: `tests/test_api.py`

Trois changements :
1. Ajouter `app.state.schedule_index = {}` dans le fixture `client`
2. Remplacer `patch("apps.api.main.get_schedule", ...)` par `patch("apps.api.main.get_local_schedule", ...)`
3. Supprimer le champ `"train"` des mock data (notre index local ne produit que `depart`+`arrive`)

- [ ] **Step 5.1: Ajouter `schedule_index` dans le fixture `client` (environ ligne 106)**

Remplacer :
```python
    app.state.liaison_map = {}
    app.state.redis = None
    return TestClient(app)
```

Par :
```python
    app.state.liaison_map = {}
    app.state.schedule_index = {}
    app.state.redis = None
    return TestClient(app)
```

- [ ] **Step 5.2: Mettre à jour `test_recommend_include_schedule_adds_schedules_field`**

Remplacer :
```python
def test_recommend_include_schedule_adds_schedules_field(client):
    mock_sched = [{"depart": "07:00", "arrive": "09:30", "train": "1234"}]
    with patch("apps.api.main.get_schedule", return_value=mock_sched):
```

Par :
```python
def test_recommend_include_schedule_adds_schedules_field(client):
    mock_sched = [{"depart": "07:00", "arrive": "09:30"}]
    with patch("apps.api.main.get_local_schedule", return_value=mock_sched):
```

- [ ] **Step 5.3: Mettre à jour `test_schedule_endpoint_mocked_returns_data`**

Remplacer :
```python
def test_schedule_endpoint_mocked_returns_data(client):
    mock_deps = [{"depart": "08:00", "arrive": "10:30", "train": "705"}]
    with patch("apps.api.main.get_schedule", return_value=mock_deps):
        resp = client.get("/schedule/A")
    assert resp.status_code == 200
    body = resp.json()
    assert body["liaison_id"] == "A"
    assert body["schedule"] == mock_deps
```

Par :
```python
def test_schedule_endpoint_mocked_returns_data(client):
    mock_deps = [{"depart": "08:00", "arrive": "10:30"}]
    with patch("apps.api.main.get_local_schedule", return_value=mock_deps):
        resp = client.get("/schedule/A")
    assert resp.status_code == 200
    body = resp.json()
    assert body["liaison_id"] == "A"
    assert body["schedule"] == mock_deps
```

- [ ] **Step 5.4: Vérifier que tous les tests passent**

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

Attendu : **130/130 PASSED** (115 anciens + 15 nouveaux).

- [ ] **Step 5.5: Commit**

```powershell
git add tests/test_api.py
git commit -m "test(api): update schedule mocks from get_schedule to get_local_schedule"
```

---

## Task 6 — Mettre à jour CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 6.1: Mettre à jour la section `schedule.py` dans Repository Layout**

Dans le bloc `src/rec_oncf/`, ajouter après `schedule.py` :

```
│   ├── local_schedule.py   # index O/D depuis horaire.csv — parse_horaire_csv,
│   │                       #   build_od_index, get_local_schedule, save/load_schedule_index
```

- [ ] **Step 6.2: Mettre à jour le tableau des scripts**

Ajouter dans le tableau scripts :
```
│   ├── 11_build_schedule_index.py  # horaire.csv → models/schedule_index.joblib  ✅ done
```

- [ ] **Step 6.3: Mettre à jour Current Status**

Ajouter la ligne :
```
| Local schedule index (`local_schedule.py`) | ✅ Done | `models/schedule_index.joblib` — 395 trajets, offline, remplace scraper oncf.ma |
```

- [ ] **Step 6.4: Mettre à jour les tests dans CLAUDE.md**

Changer `115/115` → `130/130` et ajouter `test_local_schedule.py # 15 tests`.

- [ ] **Step 6.5: Mettre à jour l'Artifact Paths**

Ajouter :
```python
horaire_csv_path    = desktop / "horaire.csv"
schedule_index_path = models_dir / "schedule_index.joblib"  # ~50-200 KB
```

- [ ] **Step 6.6: Commit final**

```powershell
git add CLAUDE.md
git commit -m "docs(claude): update for local schedule integration (script 11, local_schedule.py)"
```

---

## Vérification finale

```powershell
# Tous les tests
.venv\Scripts\python.exe -m pytest tests/ -v

# Lint
.venv\Scripts\python.exe -m ruff check scripts/ src/

# Démarrer l'API et tester manuellement
.venv\Scripts\python.exe -m uvicorn apps.api.main:app --reload
# Puis : http://localhost:8000/schedule/18143
```

---

## Self-review

**Couverture spec :**
- ✅ `parse_horaire_csv` — Task 2
- ✅ `build_od_index` avec règle 00:00:00 — Task 2
- ✅ `get_local_schedule` avec filtre horaire — Task 2
- ✅ `save/load_schedule_index` — Task 2
- ✅ `scripts/11_build_schedule_index.py` — Task 3
- ✅ `config.py` paths — Task 1
- ✅ Startup API charge l'index — Task 4
- ✅ `schedule_endpoint` utilise local — Task 4
- ✅ `include_schedule` utilise local — Task 4
- ✅ Tests existants mis à jour — Task 5
- ✅ CLAUDE.md — Task 6

**Placeholders :** Aucun TBD/TODO dans le plan.

**Cohérence des types :**
- `od_index: dict[tuple[str, str], list[dict[str, str]]]` — cohérent Tasks 2→4
- `liaison_map: dict[str, tuple[str, str]]` — identique à `build_liaison_station_map` dans `schedule.py`
- `get_local_schedule(liaison_id, liaison_map, od_index, now=None)` — cohérent Tasks 2→4→5
