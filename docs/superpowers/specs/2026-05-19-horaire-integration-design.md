# Design Spec — Intégration horaire.csv (schedules locaux)

**Date :** 2026-05-19  
**Auteur :** Omar Chekroun  
**Statut :** Validé — en attente d'implémentation

---

## Contexte

L'API expose `GET /schedule/{liaison_id}` pour afficher les prochains départs sur la page de démo.  
Jusqu'ici, cet endpoint scrape **oncf.ma** (fragile, dépendance internet, parfois indisponible).  
On dispose désormais de **horaire.csv** (2 683 arrêts, 395 trajets) qui couvre l'intégralité du réseau.  
Objectif : remplacer le scraper par un **index local chargé au démarrage**, sans aucune dépendance externe.

---

## Périmètre

| Dans le scope | Hors scope |
|---|---|
| Parsing + indexation horaire.csv | Modification du modèle ML |
| Remplacement du scraper dans l'API | Pipeline retrain test1/test2 |
| Tests unitaires de la nouvelle logique | Interface graphique de l'horaire |
| Suppression de fetch_departures / dépendance requests/bs4 sur ce chemin | Mise à jour du rapport LaTeX |

---

## Schéma horaire.csv

```
gare ; heure_arrivée ; heure_départ ; ordre_arrêt ; num_commercial
```

- **Séparateur** : `;`
- **Pas d'en-tête** dans le fichier
- `heure_arrivée = 00:00:00` → gare de **départ du trajet** (le train commence là, pas de vrai horaire d'arrivée)
- `heure_départ = 00:00:00` → gare **terminus du trajet** (le train termine là, pas de vrai horaire de départ)
- Ces deux valeurs **ne représentent PAS minuit** — elles signifient "inexistant/non applicable"
- `num_commercial` groupe les arrêts d'un même trajet (395 valeurs distinctes)
- Certains noms de gare ont des espaces de padding → normalisation `strip().upper()` obligatoire
- Horaires **quotidiens récurrents** (pas de dates)

---

## Architecture — Approche : Index en mémoire

```
horaire.csv
    │
    ▼  scripts/11_build_schedule_index.py
    │  (optionnel — pré-calcul persisté)
    ▼
models/schedule_index.joblib          ← dict {(origin, dest): [{depart, arrive}, ...]}
    │
    ▼  apps/api/main.py (lifespan startup)
app.state.schedule_index              ← chargé une fois, servi en <1ms
    │
    ▼  GET /schedule/{liaison_id}
src/rec_oncf/local_schedule.py        ← lookup O/D → liste triée par heure de départ
```

---

## Composants à créer / modifier

### 1. `src/rec_oncf/local_schedule.py` (NOUVEAU)

```python
def parse_horaire_csv(path: Path) -> pd.DataFrame
    # lit horaire.csv, colonnes: gare, arr, dep, ordre, num_commercial
    # normalise: gare = strip().upper(), arr/dep = str HH:MM

def build_od_index(df: pd.DataFrame) -> dict[tuple[str, str], list[dict]]
    # pour chaque trajet (num_commercial):
    #   trie par ordre_arret
    #   filtre: origine = gares avec dep != "00:00:00"
    #           destination = gares avec arr != "00:00:00"
    #   génère toutes les paires (gare_i, gare_j) avec j > i
    #   ajoute {depart: dep_i, arrive: arr_j} dans l'index
    # retourne {(origin, dest): [{depart, arrive}, ...]} trié par heure de départ

def get_local_schedule(
    liaison_id: str,
    liaison_map: dict[str, tuple[str, str]],
    od_index: dict[tuple[str, str], list[dict]],
    now: datetime | None = None,           # ← NOUVEAU
) -> list[dict[str, str]]
    # liaison_map[liaison_id] → (origin, dest) (noms normalisés)
    # od_index.get((origin, dest), [])
    # si now fourni : filtre pour ne garder que les départs >= now.strftime("%H:%M")
    #   si aucun départ restant aujourd'hui → retourne quand même les N prochains (début de liste)
    # retourne [] si liaison inconnue ou gare hors couverture
```

### 2. `scripts/11_build_schedule_index.py` (NOUVEAU)

- Charge `horaire.csv` (chemin : `config.py` ou argument CLI)
- Appelle `build_od_index(parse_horaire_csv(...))`
- Sauvegarde dans `models/schedule_index.joblib`
- Affiche : nb de trajets, nb de paires O/D, gares non couvertes par `Liaison.csv`

### 3. `apps/api/main.py` (MODIFICATION)

**Startup (lifespan) :**
```python
app.state.schedule_index = load_schedule_index(paths)   # charge models/schedule_index.joblib
app.state.liaison_map    = build_liaison_station_map(clean_df)  # déjà existant
```

**Endpoint `/schedule/{liaison_id}` :**
```python
# Remplace : get_schedule(...) → scrape oncf.ma
# Par      : get_local_schedule(liaison_id, liaison_map, schedule_index)
```

Le scraper (`fetch_departures`) **n'est plus appelé**. Le code `schedule.py` peut rester mais `fetch_departures` est isolé (non importé depuis main.py).

### 4. `src/rec_oncf/config.py` (MODIFICATION MINEURE)

Ajouter :
```python
horaire_csv_path   = desktop_dir / "horaire.csv"   # ou project_root / "data/raw/horaire.csv"
schedule_index_path = models_dir / "schedule_index.joblib"
```

---

## Flux de données

```
/schedule/14332
    │
    ├── liaison_map[14332] → ("CASA PORT", "RABAT VILLE")
    │
    ├── od_index[("CASA PORT", "RABAT VILLE")]
    │       → [{depart: "06:15", arrive: "07:31"}, {depart: "07:00", arrive: "..."}]
    │
    └── retourne la liste triée (ou [] si non couvert)
```

---

## Gestion des cas limites

| Cas | Comportement |
|---|---|
| `liaison_id` inconnu dans `liaison_map` | retourne `[]` — identique à avant |
| Gare dans `liaison_map` absente de `horaire.csv` | retourne `[]` — silencieux |
| `schedule_index.joblib` absent au startup | warning log, `schedule_index = {}`, endpoint retourne `[]` |
| Noms avec espaces padding (ex: `"BENI NSAR VILLE               "`) | `strip().upper()` à la normalisation |

---

## Tests — `tests/test_local_schedule.py` (NOUVEAU, ~12 tests)

| Test | Ce qu'on teste et pourquoi |
|---|---|
| `test_parse_horaire_csv_columns` | S'assure que les 5 colonnes sont lues et nommées correctement |
| `test_normalize_strips_whitespace` | Gares avec padding ne brisent pas le lookup |
| `test_build_od_index_basic` | Un trajet A→B→C génère 3 paires : (A,B), (A,C), (B,C) |
| `test_build_od_index_direction` | Vérifie qu'on ne génère pas (B,A) depuis A→B (pas de retour) |
| `test_build_od_index_times` | L'heure de départ = heure depart de A, heure arrive = heure arrivée de B — et 00:00:00 exclut la gare comme origine ou comme destination |
| `test_build_od_index_sorted` | Les départs pour une paire O/D sont triés chronologiquement |
| `test_get_local_schedule_hit` | Liaison connue → retourne des départs non vides |
| `test_get_local_schedule_unknown_liaison` | Liaison absente de liaison_map → `[]` |
| `test_get_local_schedule_unmapped_station` | Gare dans liaison_map mais absente de horaire.csv → `[]` |
| `test_od_index_roundtrip_save_load` | `joblib.dump / load` préserve l'index |
| `test_multiple_trains_same_od` | Plusieurs trains sur même O/D → tous apparaissent, triés |
| `test_terminal_station_no_departure` | Gare terminus (dep=00:00:00) n'est jamais une origine |
| `test_time_filter_upcoming` | `now="08:00"` → seuls les départs ≥ 08:00 sont retournés |
| `test_time_filter_end_of_day` | Si `now="23:00"` et aucun train restant → retourne les premiers départs du jour (cycle suivant) |
| `test_ui_response_has_depart_arrive` | Le format `{depart, arrive}` est bien présent dans chaque entrée de schedule |

---

## Séquence d'implémentation (ordre recommandé)

1. `local_schedule.py` — logique pure, testable sans API
2. `test_local_schedule.py` — TDD : écrire les tests d'abord
3. `scripts/11_build_schedule_index.py` — valider l'index avec les vraies données
4. `config.py` — ajouter `schedule_index_path` et `horaire_csv_path`
5. `main.py` — connecter l'index au startup + modifier l'endpoint
6. `test_api.py` — mettre à jour les tests `/schedule/`
7. `CLAUDE.md` — mettre à jour

---

## Non-objectifs explicites

- On ne modifie pas le contrat de l'API (`GET /schedule/{id}` retourne toujours `{liaison_id, schedule: [...]}`)
- On ne supprime pas `schedule.py` du dépôt (il reste mais `fetch_departures` n'est plus appelé)
- On filtre par heure courante côté API : les départs passés ne sont pas retournés (sauf si aucun train ne reste, auquel cas on montre les premiers de la journée pour le lendemain)
- On ne gère pas les jours fériés ou horaires spéciaux
