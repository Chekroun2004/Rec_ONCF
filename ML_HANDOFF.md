# Rec_ONCF — ML Handoff Document

> **Destinataire :** Agent IA chargé de la phase Machine Learning  
> **Statut :** Nettoyage terminé — dataset et features prêts à l'entraînement  
> **Date :** 2026-05-02

---

## 1. Objectif du projet

Système de **recommandation proactive "zero-click"** pour l'ONCF (Office National des Chemins de Fer, Maroc).

**But :** Prédire la liaison la plus probable pour un utilisateur et l'afficher dès l'écran d'accueil de l'application, **sans que l'utilisateur n'ait rien saisi**.

**Contraintes non négociables :**
- **Pas de GPS** — aucune donnée de localisation disponible ou autorisée
- **Cold-start strict** : si `nb_voyages_total < 3` → afficher la barre de recherche classique, **pas de recommandation du tout, pas de fallback**
- **Conformité Loi 09-08 / CNDP** (équivalent marocain du RGPD) : minimisation des données, pseudonymisation, rétention limitée

---

## 2. Architecture du pipeline

```
oncf_data.csv  ──┐
                 ├──► 01_make_dataset.py ──► oncf_clean.parquet
Liaison.csv    ──┘
                              │
                              ▼
                    02_build_features.py ──► features.parquet
                              │
                              ▼
                    03_train_ranker.py ──► models/xgb_ranker.json
                                          models/label_encoder.joblib
                                          reports/offline_metrics.json
                              │
                              ▼
                    apps/api/main.py  ──► POST /recommend
```

### Deux étapes de recommendation (à l'inférence)

1. **Candidate Generation** (`src/rec_oncf/candidates.py`)  
   - Génère ≤ 10 liaisons candidates basées sur l'historique de l'utilisateur  
   - Critère : most-frequent + most-recent dans une fenêtre de 50 derniers voyages  
   - Aucune popularité globale (cold-start géré en amont)

2. **Ranking XGBoost** (`src/rec_oncf/training.py`)  
   - Modèle global `multi:softprob` → probabilité sur chacune des 1 067 liaisons  
   - Retourne le top-k (k=1 par défaut, max 3)

---

## 3. Structure des fichiers

```
Rec_ONCF/
├── src/rec_oncf/
│   ├── cleaning.py      # Pipeline de nettoyage complet
│   ├── features.py      # Construction des features d'entraînement
│   ├── candidates.py    # Génération des candidats (inférence)
│   ├── training.py      # Train / save / load / predict XGBoost
│   ├── metrics.py       # hit_rate_at_k
│   ├── io.py            # read_csv / read_parquet / write_parquet
│   └── config.py        # Chemins centralisés (Paths dataclass)
├── scripts/
│   ├── 01_make_dataset.py   # Exécute le nettoyage
│   ├── 02_build_features.py # Exécute la feature engineering
│   └── 03_train_ranker.py   # Entraîne et évalue le modèle
├── apps/api/main.py         # API FastAPI
├── data/
│   ├── raw/                 # Vide — sources sur le Desktop (NE PAS MODIFIER)
│   └── processed/
│       ├── oncf_clean.parquet   # ✅ Dataset nettoyé
│       └── features.parquet     # ✅ Features prêtes à l'entraînement
├── models/                  # Vide — à remplir par 03_train_ranker.py
├── reports/
│   ├── cleaning_report.json
│   └── offline_metrics.json # À générer par 03_train_ranker.py
├── configs/privacy.md
├── requirements.txt
└── DATA_SOURCES_SCHEMA.json
```

---

## 4. Données sources (lecture seule)

| Fichier | Emplacement | Lignes brutes | Description |
|---|---|---|---|
| `oncf_data.csv` | `C:\Users\omarc\Desktop\` | 946 155 | Toutes les transactions ONCF |
| `Liaison.csv` | `C:\Users\omarc\Desktop\` | ~1 290 | Référentiel gares (départ/arrivée) |

**Ne jamais modifier ces fichiers.**

---

## 5. Dataset nettoyé — `data/processed/oncf_clean.parquet`

### Statistiques globales

| Indicateur | Valeur |
|---|---|
| Lignes brutes | 946 155 |
| Lignes après nettoyage | **491 680** |
| Lignes supprimées | 454 475 (−48%) |
| Clients actifs (≥ 3 voyages) | **69 449** |
| Liaisons distinctes | **1 067** |
| Médiane voyages / client | 4 |
| Moyenne voyages / client | 7.1 |
| P90 voyages / client | 12 |

### Pourquoi −48% ?

| Motif de suppression | Lignes | % du source |
|---|---:|---:|
| Clients cold-start (< 3 voyages) | 403 615 | 42.7% |
| Lignes d'annulation | 31 920 | 3.4% |
| Réservations annulées (cibles) | 18 939 | 2.0% |
| Jointure LiaisonId manquante | 1 | ~0% |
| **Total supprimé** | **454 475** | **48.0%** |

**Explication du cold-start (42.7%) :** 304 595 clients sur ~374 000 n'avaient que 1 ou 2 voyages dans la période. Ces clients ne recevront jamais de recommandation en production (règle métier stricte), donc les garder dans le training set n'aurait apporté que du bruit pur.

**Explication des annulations :** Les lignes avec `NbrVoySegment ≤ 0` ou `PrixParLiaison < 0` sont des lignes de reversal/annulation. Chaque ligne d'annulation supprime aussi la réservation précédente du même client pour le même `TrajetAllerRetour` (identifiant du voyage).

### Structure — une ligne = un segment de voyage

Chaque ligne représente **une réservation validée** d'un client pour une liaison donnée.

### Colonnes du dataset nettoyé

| Colonne | Type | Nulls | Description |
|---|---|---|---|
| `CodeClient` | str | 0 | Identifiant unique du client (pseudonymisé) |
| `LiaisonId` | str | 0 | **TARGET** — Identifiant de la liaison (ex: "345") |
| `DesignationFrGareDepart` | str | 0 | Nom gare de départ (ex: "Casablanca Voyageurs") |
| `DesignationFrGareArrive` | str | 0 | Nom gare d'arrivée (ex: "Fès") |
| `DateHeureDepartVoyageSegment` | datetime | 0 | Date/heure de départ du voyage |
| `DatePaiement` | datetime | 0 | Date/heure du paiement |
| `TrajetAllerRetour` | str | 0 | Identifiant du voyage (label, pas un nombre) |
| `AchteurId` | str/label | variable | ID de l'acheteur — peut différer du voyageur (famille, entreprise) |
| `TypeParcoursId` | float | 0 | Type de parcours (direct, correspondances…) — catégoriel |
| `ClassificationId` | float | 0 | Segment client — catégoriel |
| `ClassePhysiqueId` | float | 0 | Classe de transport (1ère, 2ème…) — catégoriel |
| `NiveauPrixId` | float | 0 | Niveau tarifaire (normal, réduit, premium…) — catégoriel |
| `TrainAutocarId` | float | 0 | ID train/autocar — catégoriel |
| `CarteClientId` | float | 0 | Type de carte client (jeune, étudiant, senior…) — catégoriel |
| `PrixParLiaison` | float | ~697 | Prix payé en MAD. **0 = réduction 100% (valide, pas une erreur)** |
| `NbrVoySegment` | float | 0 | Nombre de passagers dans ce segment |
| `DelaiAnticipation` | float | 0 | Jours entre achat et départ |
| `LiaisonVoyageurSegmentIdSTG` | str | 0 | ID brut liaison (source avant normalisation) |
| `depart_hour` | int | 0 | Heure de départ (0–23) |
| `depart_dow` | int | 0 | Jour de la semaine (0=Lundi … 6=Dimanche) |
| `depart_month` | int | 0 | Mois (1–12) |
| `depart_hour_sin` | float | 0 | Encodage cyclique heure — sin(2π·h/24) |
| `depart_hour_cos` | float | 0 | Encodage cyclique heure — cos(2π·h/24) |
| `depart_dow_sin` | float | 0 | Encodage cyclique jour semaine — sin(2π·d/7) |
| `depart_dow_cos` | float | 0 | Encodage cyclique jour semaine — cos(2π·d/7) |
| `depart_month_sin` | float | 0 | Encodage cyclique mois — sin(2π·m/12) |
| `depart_month_cos` | float | 0 | Encodage cyclique mois — cos(2π·m/12) |

> **Pourquoi l'encodage cyclique ?**  
> Un modèle qui reçoit `hour=23` et `hour=0` comme entiers les voit comme éloignés de 23 unités. Avec sin/cos, ils sont proches (discontinuité calendaire résolue). Idem pour décembre → janvier et dimanche → lundi.

---

## 6. Features d'entraînement — `data/processed/features.parquet`

Généré par `scripts/02_build_features.py` → `src/rec_oncf/features.py`.

**491 680 lignes × 24 colonnes.** Même granularité que le dataset nettoyé (une ligne = une réservation).

### Colonnes features

#### Identifiants & cible

| Colonne | Type | Description |
|---|---|---|
| `CodeClient` | str | Identifiant client (retiré de X avant training pour éviter la mémorisation) |
| `DateHeureDepartVoyageSegment` | datetime | Utilisé pour le tri temporel et le split train/test |
| `LiaisonId` | str | **TARGET** — 1 067 classes |

#### Features catégorielles (→ OneHotEncoding)

| Colonne | Type | Nulls | Description |
|---|---|---|---|
| `TypeParcoursId` | str | 0 | Type de parcours |
| `ClassificationId` | str | 0 | Segment client |
| `ClassePhysiqueId` | str | 0 | Classe de transport |
| `NiveauPrixId` | str | 0 | Niveau tarifaire |
| `TrainAutocarId` | str | 0 | ID train/autocar |
| `CarteClientId` | str | 0 | Type de carte client |
| `prev_liaison` | str | 69 449 | Liaison précédente du client (null = 1er voyage) |

> `prev_liaison` est la feature lag-1 la plus importante : elle encode l'habitude de déplacement.  
> Les 69 449 nulls correspondent exactement aux premiers voyages de chaque client (un null par client). Géré par `handle_unknown="ignore"` dans le OHE.

#### Features numériques

| Colonne | Type | Nulls | Description |
|---|---|---|---|
| `PrixParLiaison` | float64 | 697 | Prix en MAD (0 = réduction 100%) |
| `NbrVoySegment` | float64 | 0 | Nombre de passagers |
| `DelaiAnticipation` | float64 | 0 | Jours d'anticipation à l'achat |
| `user_trip_index` | int64 | 0 | Rang du voyage dans l'historique client (0 = 1er) |
| `days_since_prev` | float64 | 69 449 | Jours depuis le voyage précédent (null = 1er voyage) |
| `depart_hour` | int32 | 0 | Heure brute (0–23) |
| `depart_dow` | int32 | 0 | Jour semaine brut (0–6) |
| `depart_month` | int32 | 0 | Mois brut (1–12) |
| `depart_hour_sin` | float64 | 0 | sin cyclique heure |
| `depart_hour_cos` | float64 | 0 | cos cyclique heure |
| `depart_dow_sin` | float64 | 0 | sin cyclique jour semaine |
| `depart_dow_cos` | float64 | 0 | cos cyclique jour semaine |
| `depart_month_sin` | float64 | 0 | sin cyclique mois |
| `depart_month_cos` | float64 | 0 | cos cyclique mois |

---

## 7. Modèle actuel (baseline à améliorer)

### Architecture — `src/rec_oncf/training.py`

```
sklearn Pipeline
├── ColumnTransformer
│   ├── OneHotEncoder(handle_unknown="ignore")  → colonnes catégorielles
│   └── passthrough                              → colonnes numériques
└── XGBClassifier
    ├── objective      = "multi:softprob"
    ├── n_estimators   = 300
    ├── learning_rate  = 0.08
    ├── max_depth      = 8
    ├── subsample      = 0.9
    ├── colsample_bytree = 0.8
    ├── reg_lambda     = 1.0
    └── n_jobs         = 0  (tous les cœurs)
```

### Split temporel (pas aléatoire)

```python
# 80% des lignes triées par date → train
# 20% les plus récentes → test
df_sorted = df.sort_values("DateHeureDepartVoyageSegment")
cut = int(len(df) * 0.8)
train = df_sorted.iloc[:cut]   # ~393 344 lignes
test  = df_sorted.iloc[cut:]   # ~98 336 lignes
```

> Le split temporel évite le **data leakage** : le modèle ne voit jamais des voyages futurs pendant l'entraînement.

### Métriques offline

Calculées dans `src/rec_oncf/metrics.py` et sauvegardées dans `reports/offline_metrics.json`.

| Métrique | Description |
|---|---|
| `hit_rate@1` | La vraie liaison est-elle la #1 prédite ? |
| `hit_rate@3` | La vraie liaison est-elle dans le top-3 prédit ? |

> **hit_rate@1** est la métrique principale car l'app affiche une seule recommandation (k=1 par défaut).

### Ce qui est exclu de X à l'entraînement

- `CodeClient` : retiré pour éviter la mémorisation directe
- `DateHeureDepartVoyageSegment` : utilisé uniquement pour le tri
- `LiaisonId` : c'est la cible Y

### LabelEncoder

`LiaisonId` est encodé en entiers (0…1066) par `sklearn.LabelEncoder`. L'objet est sauvegardé dans `models/label_encoder.joblib` pour être réutilisé à l'inférence.

---

## 8. API — `apps/api/main.py`

```
POST /recommend
```

**Request body :**
```json
{
  "code_client": "CLI123456",
  "nb_voyages_total": 7,
  "k": 1
}
```

**Response :**
```json
{
  "mode": "model",
  "candidates": ["345", "678", "901"],
  "recommendations": ["345"]
}
```

**Logique :**
1. Si `nb_voyages_total < 3` → `mode: "cold_start"`, listes vides
2. Sinon :
   - Génère ≤ 10 candidats depuis l'historique client (most-frequent + most-recent)
   - Charge la dernière ligne de features du client
   - Prédit avec le modèle XGBoost → retourne top-k labels

**Lancer l'API :**
```bash
uvicorn apps.api.main:app --reload
GET /health  →  {"status": "ok", "ts": "..."}
```

---

## 9. Environnement

```bash
# Python 3.12, venv dans .venv/
.\.venv\Scripts\activate        # Windows PowerShell
pip install -r requirements.txt
```

**Dépendances principales :**

| Package | Version | Rôle |
|---|---|---|
| xgboost | 3.2.0 | Modèle de ranking |
| scikit-learn | 1.8.0 | Pipeline / OHE / LabelEncoder |
| pandas | 3.0.2 | Manipulation données |
| numpy | 2.4.4 | Calculs numériques |
| fastapi | 0.136.1 | API REST |
| uvicorn | 0.46.0 | Serveur ASGI |
| joblib | 1.5.3 | Sérialisation modèle |
| pyarrow | 24.0.0 | Lecture/écriture Parquet |

---

## 10. Ordre d'exécution du pipeline complet

```powershell
# 1. Nettoyage (déjà fait ✅)
python scripts/01_make_dataset.py

# 2. Feature engineering (déjà fait ✅)
python scripts/02_build_features.py

# 3. Entraînement + évaluation (à faire)
python scripts/03_train_ranker.py

# 4. API
uvicorn apps.api.main:app --reload
```

---

## 11. Points d'attention pour la phase ML

### Problèmes connus / axes d'amélioration

1. **Candidats non filtrés à l'inférence** — le modèle prédit sur toutes les 1 067 liaisons puis retourne le top-k, sans se restreindre aux candidats générés. L'idée prévue (commentaire dans `apps/api/main.py`) est de masquer les scores aux seules liaisons candidates avant le top-k.

2. **`prev_liaison` = null pour le 1er voyage** — 69 449 lignes de nuls (une par client). Le OHE avec `handle_unknown="ignore"` les encode comme vecteur de zéros, ce qui est correct.

3. **`PrixParLiaison` = 697 nulls** — à imputer (médiane par liaison) ou à laisser gérer par XGBoost (nativement supporté avec `tree_method="hist"`).

4. **Déséquilibre des classes** — 1 067 liaisons avec des fréquences très inégales. Envisager `scale_pos_weight` ou `sample_weight` basé sur la fréquence inverse.

5. **`TrajetAllerRetour`** — présent dans le dataset nettoyé mais pas encore dans les features. C'est un label catégoriel (type de trajet) qui pourrait améliorer le modèle.

6. **`AchteurId`** — présent dans le dataset nettoyé, non utilisé en feature. Pourrait servir à créer un feature booléen `is_self_purchase = (AchteurId == CodeClient)`.

7. **`DesignationFrGareDepart` / `DesignationFrGareArrive`** — noms de gare disponibles dans le parquet nettoyé mais non utilisés en feature (haute cardinalité). Envisager un embedding ou un regroupement par zone géographique.

8. **Métriques complémentaires à considérer** — MRR (Mean Reciprocal Rank), NDCG@k, couverture catalogue.

### Valeurs métier à respecter

| Règle | Valeur |
|---|---|
| Cold-start threshold | `nb_voyages_total < 3` → pas de reco |
| Max candidats | 10 |
| Max k (recommandations) | 3 |
| Target principal | `hit_rate@1` |

---

## 12. Fichiers de configuration centraux

- **`src/rec_oncf/config.py`** — tous les chemins (sources, processed, models)
- **`configs/privacy.md`** — règles de conformité CNDP
- **`DATA_SOURCES_SCHEMA.json`** — schéma complet des colonnes sources avec descriptions

---

*Document généré le 2026-05-02. Les fichiers parquet dans `data/processed/` sont à jour.*
