# Pivot post-réunion ONCF — Design

> **Date :** 2026-05-15
> **Auteur :** Omar Chekroun
> **Deadline livraison :** lundi 2026-05-18 matin (~60h)
> **Contexte :** réunion ONCF du 2026-05-15. Six chantiers de livraison pour clôturer le projet.

---

## 1. Objet et contexte

À l'issue de la réunion de cadrage ONCF, six chantiers ont été décidés pour livrer le projet ZeroClickSearch dans sa version finale. Ce document fige le design retenu avant de passer à l'implémentation.

**Décisions de cadrage (validées avec l'utilisateur) :**

- **Deadline :** lundi 2026-05-18 matin (~60h disponibles à partir du vendredi soir).
- **Données :** l'ONCF fournira 2 CSV (historique users + horaires trains). Tant que les CSV ne sont pas reçus, on travaille sur les 5 autres chantiers ; la migration CSV passe en dernier (chantier 6).
- **IDA :** désigne **l'Ingénierie d'Attributs** (Feature Engineering), à fusionner dans la section *Exploration et Nettoyage des Données* du rapport.
- **Benchmark :** 1/3 ferroviaire (SNCF, DB, Trenitalia) + 2/3 patterns ML mobile (model serving, A/B, MLOps, latence).
- **Guide deploy :** Docker (voie nominale) + venv (fallback), PDF intégré comme pièce du rapport.
- **Rapport :** retirer **uniquement** les figures/mockups montrant la reco intégrée visuellement dans l'app ONCF Voyages (les diagrammes UML/séquence et le contexte métier restent).

---

## 2. Inventaire des 6 chantiers et ordre d'exécution

| Ordre | Chantier | Dépendances | Effort estimé |
|---|---|---|---|
| 1 | MAJ rapport (retrait mockups app + ajouts post-réunion) | aucune | 2-3 h |
| 2 | Partie IDA — restructure section *Exploration et Nettoyage* | aucune | 2-3 h |
| 3 | API deploy-ready (Dockerfile + compose + healthcheck) | aucune | 3-4 h |
| 4 | Guide de déploiement PDF — intégré au rapport | dépend du #3 | 4-5 h |
| 5 | Benchmark PDF — enrichit le chapitre *Étude de l'existant* | aucune | 6-8 h |
| 6 | Migration CSV (Option B — couche d'adaptation) | CSV ONCF reçus | 4-6 h |

**Total :** 21-29 h. Marge confortable pour relecture + bugs.

**Parallélisation :** série pour les chantiers 1-3 (qualité critique), parallélisation possible sur 4-5 (rédaction autonome).

**Non-régression :** les chantiers 1-5 ne touchent pas au code Python testé. Les 113 tests existants doivent rester verts à chaque commit. Le chantier 6 ajoute `tests/test_data_source.py`.

---

## 3. Architecture cible pour la migration CSV (chantier 6)

**Option retenue : couche d'adaptation des sources de données (Option B).**

```
┌──────────────────────────────────────────────────────────────────┐
│  CSV ONCF (à recevoir)                                           │
│   ├── users_history.csv  (historique réservations par CodeClient)│
│   └── trains_schedule.csv (horaires statiques par liaison)       │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  src/rec_oncf/data_source.py    ← NOUVEAU                        │
│   ├── UserHistorySource (Protocol)                               │
│   │    └── CsvUserHistorySource                                  │
│   └── ScheduleSource (Protocol)                                  │
│        ├── CsvScheduleSource (nominale)                          │
│        └── OncfMaScraperSource (legacy, kept for fallback)       │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  src/rec_oncf/recommender.py     ← MINIMAL CHANGES               │
│   Recommender.from_paths(paths, *, sources=...)                  │
│   history_lookup, popularity, ONNX, candidates → unchanged       │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  apps/api/main.py                ← LIFESPAN UPDATE               │
│   Lifespan: instancie les sources CSV → injecte dans Recommender │
│   Endpoints inchangés (/recommend, /schedule, /feedback, /)      │
└──────────────────────────────────────────────────────────────────┘
```

### Interfaces

```python
class UserHistorySource(Protocol):
    def get_history(self, code_client: str) -> pd.DataFrame | None: ...
    def get_all_users(self) -> list[str]: ...
    def get_clean_df(self) -> pd.DataFrame: ...

class ScheduleSource(Protocol):
    def get_schedule(self, liaison_id: str, date: str | None = None) -> list[dict]: ...
```

### Schémas CSV attendus (à ajuster à réception des fichiers ONCF)

**`users_history.csv` — colonnes minimales :**

| Colonne | Type | Notes |
|---|---|---|
| `CodeClient` | str | clé de recherche |
| `LiaisonId` | str | route O/D |
| `DateHeureDepartVoyageSegment` | datetime ISO | tri temporel |
| `PrixParLiaison` | float | feature |
| `NbrVoySegment` | int | feature |
| `DelaiAnticipation` | float | feature |
| `TypeParcoursId`, `ClassificationId`, `ClassePhysiqueId`, `NiveauPrixId`, `TrainAutocarId`, `CarteClientId` | str | features catégorielles |

**`trains_schedule.csv` — colonnes minimales :**

| Colonne | Type | Notes |
|---|---|---|
| `liaison_id` *ou* (`gare_depart`, `gare_arrivee`) | str | clé de jointure |
| `heure_depart` | "HH:MM" | |
| `heure_arrivee` | "HH:MM" | |
| `train_num` | str | optionnel |
| `jours_circulation` | str (ex. "LMMJVSD") | optionnel — défaut : tous jours |

### Script de validation

`scripts/09_load_oncf_csv.py` :
1. Lit chaque CSV avec pandas.
2. Affiche schéma détecté (colonnes, dtypes, nb lignes, nb users uniques).
3. Vérifie colonnes minimales ; warning + backfill safe si manquantes.
4. Affiche 5 lignes random pour inspection visuelle.
5. Sauvegarde en `data/processed/users_history.parquet` + `data/processed/trains_schedule.parquet` (parquet en interne pour perf, CSV en source).

### Fichiers impactés

- `+ src/rec_oncf/data_source.py` (~150 lignes)
- `~ src/rec_oncf/recommender.py` (constructor accepte sources injectables ; comportement identique sinon)
- `~ apps/api/main.py` (lifespan instancie les bonnes sources)
- `~ src/rec_oncf/config.py` (paths des CSV)
- `+ scripts/09_load_oncf_csv.py`
- `+ tests/test_data_source.py`

### Non-impactés

`candidates.py`, `popularity.py`, `features.py::compute_inference_row`, `training.py`, ONNX runtime, `FastPreprocessor`, et l'ensemble des 113 tests existants (les fixtures pointeront vers un mock `UserHistorySource` au lieu d'un parquet ; comportement identique).

---

## 4. Détail des livrables non-code (chantiers 1, 2, 4, 5)

### 4.1 Chantier 1 — MAJ `rapport_pfa_v2.tex`

**À retirer :**
- Toutes figures/mockups montrant la reco intégrée visuellement dans l'app ONCF Voyages. Avant tout retrait, je liste les `\includegraphics` et environnements `figure` qui concernent l'app pour validation utilisateur.

**À conserver :**
- Diagrammes UML, diagrammes de séquence, contexte métier ONCF, section *Présentation de l'app ONCF Voyages* (chap. 1).

**À ajouter :**
- Nouvelle section courte "Pivot post-réunion" insérée dans le chapitre 5 (*Phase 3 — Production*) à la fin, juste avant la conclusion. Contenu : 2 CSV ONCF comme source de données, architecture data source pluggable, guide de déploiement produit, benchmark enrichi. ~1 page.
- MAJ annexe "État actuel du projet" pour refléter le nouveau statut.

### 4.2 Chantier 2 — Partie IDA

Restructure la section *Exploration et Nettoyage des Données* du chapitre 3 (Sprint 1) avec ce plan :

1. Profilage du dataset brut (existant — gardé)
2. Règles métier de nettoyage (existant — gardé)
3. **NOUVEAU : Ingénierie d'Attributs (IDA)** — fusion de l'actuelle section *Ingénierie des Variables* avec une partie pédagogique sur **pourquoi** chaque feature a du sens métier (impact sur le signal prédictif). Regroupement en familles : comportementales, temporelles cycliques, contextuelles, dérivées (`user_top_liaison_share`).

### 4.3 Chantier 3 — API deploy-ready

**Fichiers produits :**

- `Dockerfile` (`python:3.12-slim`, multi-stage : builder pour deps, runtime minimaliste).
- `docker-compose.yml` (services : `api` + `redis` optionnel).
- `.env.example` : `LOG_LEVEL`, `MODEL_DIR`, `DATA_DIR`, `REDIS_URL`.
- `.dockerignore`.
- `healthcheck` enrichi : `/health` retourne `{status, model_loaded, popularity_loaded, n_users_history}` pour faciliter le monitoring ONCF.

### 4.4 Chantier 4 — Guide de déploiement PDF (~15 pages)

Plan :

1. Pré-requis hardware (CPU 4 cores+, 8 GB RAM, 5 GB disk pour modèles)
2. Pré-requis OS (Linux / Windows Server) + versions
3. **Voie nominale : Docker** (étape par étape, captures des commandes attendues)
4. **Voie alternative : venv + service Windows/systemd** (en annexe)
5. Configuration des variables d'environnement
6. Healthcheck et observabilité (logs JSON, `/health`)
7. Procédure de retraining (référence au script 07)
8. Troubleshooting (les 5-6 erreurs les plus probables)
9. Procédure de rollback

Compilé en PDF et intégré au rapport comme pièce annexe via `\includepdf`.

### 4.5 Chantier 5 — Benchmark PDF (~6-10 pages)

Plan :

1. Introduction (objectif, méthodologie)
2. **Section ferroviaire (~2-3 p)** : SNCF (assistant SNCF Connect), Deutsche Bahn (DB Navigator), Trenitalia. Pour chacun : approche reco/personnalisation, stack technique connue (sources publiques), enseignements.
3. **Section patterns ML mobile (~4-6 p)** :
   - Pattern 1 — Model serving : TensorFlow Serving / TorchServe / ONNX Runtime / vLLM, comparaison.
   - Pattern 2 — A/B testing prod : Booking.com, Netflix — validation d'un challenger.
   - Pattern 3 — MLOps cycles : Uber Michelangelo, Spotify — retraining cadence, drift monitoring, guardrails.
   - Pattern 4 — Latence p99 mobile : Amazon recommendations, Booking.
4. Synthèse : nos choix (ONNX Runtime, FastAPI, retraining quotidien, guardrail HR@1) positionnés vs l'industrie.
5. Bibliographie (papers + blogposts publics).

Intégré au rapport en **enrichissant le chapitre existant** *Étude de l'existant — Benchmark Industriel* (plus académique qu'une annexe séparée).

---

## 5. Tests et non-régression

- Chantiers 1-5 : aucune modification du code Python testé. Les 113 tests doivent rester verts à chaque commit.
- Chantier 6 : nouveau fichier `tests/test_data_source.py` couvrant :
  - `CsvUserHistorySource.get_history` (utilisateur connu / inconnu)
  - `CsvUserHistorySource.get_all_users`
  - `CsvScheduleSource.get_schedule` (liaison connue / inconnue / date filter)
  - Tolérance aux colonnes manquantes (warning + backfill safe)
  - Intégration avec `Recommender.from_paths(..., sources=...)`

---

## 6. Gestion d'erreurs et cas limites

- **CSV ONCF reçus tardivement :** travail séquentiel sur les 5 autres chantiers d'abord ; la migration CSV est isolée et bornée à 4-6h.
- **Colonnes CSV inattendues :** warning + backfill safe (NaN ou défaut documenté), pas de crash hard.
- **Recommender en l'absence de source CSV (rétro-compat) :** `Recommender.from_paths(paths)` sans argument `sources` fallback sur le parquet existant. L'archi double-mode pendant la migration.
- **Healthcheck en cas de modèle absent :** `/health` retourne `{status: "degraded", model_loaded: false, ...}` au lieu de planter ; permet à l'orchestrateur ONCF de redémarrer proprement.

---

## 7. Critères d'acceptation

- [ ] Les 113 tests existants restent verts à la fin de chaque chantier.
- [ ] Le rapport `rapport_pfa_v2.tex` compile sans erreur LaTeX après MAJ.
- [ ] Le `Dockerfile` build une image fonctionnelle qui démarre et répond à `/health` 200 OK.
- [ ] Le guide de déploiement PDF compile et est inclus dans le rapport.
- [ ] Le benchmark PDF est intégré au chapitre *Étude de l'existant* du rapport.
- [ ] Section IDA présente dans *Exploration et Nettoyage* avec rationale métier des 26 features.
- [ ] Aucun mockup d'intégration mobile ne subsiste dans le rapport.
- [ ] (Si chantier 6 atteint) — `CsvUserHistorySource` et `CsvScheduleSource` fonctionnels, tests verts.

---

## 8. Hors scope (explicitement reporté)

- Authentification / TLS / rate limiting (en attente d'arbitrage sécurité ONCF — cf. note de réunion §3).
- Vraie connexion à la base de production ONCF (les CSV statiques fournis sont le substitut convenu pour la durée du livrable).
- API horaires officielle ONCF (bloqué — cf. note de réunion §2.2).
- Intégration concrète côté app mobile (non démontrable, retiré du rapport).

---

*Document à valider avant passage en phase d'implémentation (writing-plans).*
