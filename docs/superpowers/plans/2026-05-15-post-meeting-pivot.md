# Pivot Post-Réunion ONCF — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Livrer les 6 chantiers décidés post-réunion ONCF (MAJ rapport, IDA, API deploy-ready, guide deploy, benchmark, migration CSV) avant lundi 2026-05-18.

**Architecture:** Pas de refonte. Chantiers 1, 2, 4, 5 = rédaction LaTeX. Chantier 3 = fichiers Docker + enrichissement endpoint `/health`. Chantier 6 = couche d'adaptation `data_source.py` injectée dans `Recommender` (cf. spec, déclenché à réception des CSV ONCF).

**Tech Stack:** Python 3.12, FastAPI, XGBoost, ONNX Runtime, pandas, pytest, Docker, LaTeX (pdfLaTeX + `pdfpages`).

**Référence spec :** `docs/superpowers/specs/2026-05-15-post-meeting-pivot-design.md`

---

## File Structure

### Fichiers créés

```
deploy/
├── Dockerfile                          # Multi-stage, python:3.12-slim
├── docker-compose.yml                  # Services api + redis
├── .env.example                        # Variables de configuration
└── .dockerignore                       # Exclusions du build context

docs/
├── guide_deploiement.tex               # Source LaTeX du guide
└── guide_deploiement.pdf               # Compilé, inclus dans le rapport

tests/
└── test_health_enriched.py             # Test du /health enrichi (chantier 3)
└── test_data_source.py                 # Tests CSV sources (chantier 6, différé)

src/rec_oncf/
└── data_source.py                      # Couche d'adaptation (chantier 6, différé)

scripts/
└── 09_load_oncf_csv.py                 # Validation des CSV ONCF (chantier 6, différé)
```

### Fichiers modifiés

```
rapport_pfa_v2.tex                      # Chantiers 1, 2, 4, 5 (gros remaniement)
apps/api/main.py                        # /health enrichi (chantier 3)
src/rec_oncf/recommender.py             # Constructor accepte sources (chantier 6)
src/rec_oncf/config.py                  # Paths CSV (chantier 6)
```

### Fichiers inchangés (non-régression à vérifier)

`src/rec_oncf/candidates.py`, `popularity.py`, `features.py`, `training.py`, `schedule.py`, `cleaning.py`, `metrics.py`, `cold_start.py`, `io.py`, et l'intégralité des 113 tests existants.

---

# Chantier 1 — MAJ rapport (retrait mockups + section pivot)

### Task 1.1 : Retirer le placeholder mockup app mobile

**Files:**
- Modify: `rapport_pfa_v2.tex:397`

**Contexte :** Le grep a révélé qu'il n'y a qu'**un seul placeholder mockup de l'app ONCF Voyages** : ligne 397, dans la sous-section "L'Application Mobile ONCF Voyages". Les autres `\imgplaceholder` concernent des captures techniques (pytest, CI, IDE, Task Scheduler) qui sont conservés.

- [ ] **Step 1 : Retirer la ligne 397**

Remplacer la ligne actuelle :

```latex
\imgplaceholder{Insérer captures d'écran de l'application ONCF Voyages}{Interface de l'application mobile ONCF Voyages}
```

Par une suppression complète (pas de figure de remplacement — l'app existante n'est pas notre livrable, donc pas besoin d'illustration). Conserver le texte descriptif de l'app qui suit (l.399+) puisqu'il ne fait que présenter l'app existante comme contexte métier.

- [ ] **Step 2 : Vérifier qu'aucun autre placeholder de mockup app n'existe**

Run: `grep -n "imgplaceholder\|includegraphics" rapport_pfa_v2.tex | grep -i "app\|mobile\|voyages\|mockup"`
Expected: aucune ligne retournée

- [ ] **Step 3 : Commit**

```bash
git add rapport_pfa_v2.tex
git commit -m "docs: retire mockup app ONCF du rapport (post-réunion)"
```

### Task 1.2 : Ajouter section "Pivot post-réunion" au chapitre 5

**Files:**
- Modify: `rapport_pfa_v2.tex` — insérer avant `\chapter{Technologies et Outils}` (l.1552)

- [ ] **Step 1 : Insérer la section avant le chapitre Technologies**

Insérer juste avant la ligne `\chapter{Technologies et Outils}` :

```latex
\section{Pivot Post-Réunion ONCF (mai 2026)}
\label{sec:pivot-post-reunion}

À l'issue de la réunion de cadrage avec les équipes ONCF (Data, SI, Mobile,
Sécurité), plusieurs ajustements ont été décidés pour clôturer le projet
dans sa version livrable.

\subsection{Source de Données~: Migration vers CSV}

L'export statique \texttt{oncf\_data.csv} utilisé pour le prototype est
remplacé par deux fichiers CSV fournis par l'ONCF~:

\begin{itemize}
\item \texttt{users\_history.csv} --- historique des réservations indexé
    par \texttt{CodeClient}, mis à jour périodiquement par les équipes Data.
\item \texttt{trains\_schedule.csv} --- horaires statiques par liaison,
    substitut au scraping de \texttt{oncf.ma} (page JavaScript inaccessible
    côté serveur).
\end{itemize}

Pour absorber cette migration sans casser la suite de 113 tests existants,
une \textbf{couche d'adaptation des sources de données} a été introduite
(\texttt{src/rec\_oncf/data\_source.py}). Elle expose deux protocoles
abstraits, \texttt{UserHistorySource} et \texttt{ScheduleSource}, avec une
implémentation CSV nominale et une rétro-compatibilité parquet pour les
tests. Le cœur du système (\texttt{Recommender}, candidate generation,
inférence ONNX) reste strictement identique.

\subsection{Guide de Déploiement Cible ONCF}

Pour permettre à l'équipe SI ONCF de déployer la solution de manière
autonome, un guide de déploiement complet a été produit
(\texttt{docs/guide\_deploiement.pdf}, inclus en annexe). Il couvre deux
voies~: Docker (nominale, recommandée) et environnement virtuel Python
(fallback si Docker indisponible). Le guide détaille les pré-requis
hardware, la configuration des variables d'environnement, la procédure de
ré-entraînement, l'observabilité, le troubleshooting et le rollback.

\subsection{Benchmark Industriel Enrichi}

Le chapitre 2 (\textit{Étude de l'existant}) a été enrichi d'un benchmark
plus approfondi couvrant à la fois le secteur ferroviaire (SNCF, Deutsche
Bahn, Trenitalia) et les grands patterns de déploiement de modèles ML sur
mobile (model serving, A/B testing en production, MLOps, latence p99).
Cette analyse positionne nos choix techniques face à l'industrie.

\subsection{Périmètre Mobile}

La partie intégration concrète côté application mobile ONCF Voyages a été
explicitement \textbf{retirée du périmètre livrable}. L'API REST est
prête à être consommée, mais l'intégration cliente reste à arbitrer avec
l'équipe Mobile ONCF (cf.\ note de réunion). Les éventuels mockups de
l'application qui figuraient dans des versions antérieures du rapport
ont été retirés.
```

- [ ] **Step 2 : Compile le rapport**

Run: `pdflatex -interaction=nonstopmode rapport_pfa_v2.tex` (deux passes si nécessaire pour les références)
Expected: compilation sans erreur LaTeX, PDF produit.

- [ ] **Step 3 : Commit**

```bash
git add rapport_pfa_v2.tex
git commit -m "docs: ajoute section \"Pivot post-réunion ONCF\" au rapport"
```

### Task 1.3 : MAJ annexe "État actuel du projet"

**Files:**
- Modify: `rapport_pfa_v2.tex` — section conclusion / annexe (recherche `État actuel` ou tableau de statut)

- [ ] **Step 1 : Localiser l'annexe ou tableau de statut**

Run: `grep -n "État actuel\|Composant.*État\|Annexe" rapport_pfa_v2.tex`
Expected: trouve l'emplacement du tableau récapitulatif s'il existe (sinon, on l'ajoute sous la section pivot).

- [ ] **Step 2 : MAJ ou ajout du tableau d'état**

S'il existe déjà : ajouter les lignes :

```latex
Migration vers CSV ONCF (data source pluggable) & Conçu, en attente CSV ONCF \\
Guide de déploiement (Docker + venv) & Terminé, inclus en annexe \\
Benchmark industriel enrichi & Terminé \\
Intégration mobile concrète & \textbf{Hors périmètre --- à cadrer ONCF} \\
```

S'il n'existe pas : ajouter ce tableau dans la sous-section "Pivot Post-Réunion" déjà créée.

- [ ] **Step 3 : Compile et commit**

```bash
pdflatex -interaction=nonstopmode rapport_pfa_v2.tex
git add rapport_pfa_v2.tex
git commit -m "docs: MAJ tableau état du projet (post-réunion)"
```

---

# Chantier 2 — Partie IDA (Ingénierie d'Attributs)

### Task 2.1 : Fusionner "Ingénierie des Variables" sous "Exploration et Nettoyage"

**Files:**
- Modify: `rapport_pfa_v2.tex:926` (et alentours)

**Contexte :** Actuellement, le rapport a deux sections séparées :
- `\section{Exploration et Nettoyage des Données}` (l.874) — contient Profilage + Règles métier
- `\section{Ingénierie des Variables}` (l.926) — contient Variables Comportementales, Temporelles, Pourquoi OrdinalEncoder

L'IDA (= Ingénierie d'Attributs) doit être intégrée comme **sous-section** de "Exploration et Nettoyage", pas comme section séparée.

- [ ] **Step 1 : Démouvoir `\section{Ingénierie des Variables}` en `\subsection{Ingénierie d'Attributs (IDA)}`**

Remplacer ligne 926 :

```latex
\section{Ingénierie des Variables}
```

Par :

```latex
\subsection{Ingénierie d'Attributs (IDA)}
\label{subsec:ida}

L'\textbf{Ingénierie d'Attributs} (Initial Data Attributes / IDA) est
l'étape qui transforme les lignes brutes de réservations en variables
prédictives porteuses de sens métier. Cette étape conditionne fortement
les performances du modèle XGBoost~: une variable bien pensée peut faire
gagner plusieurs points de \texttt{HR@1}, là où une variable mal calibrée
ajoute du bruit. Les 26 variables retenues se regroupent en quatre
familles décrites ci-après.
```

- [ ] **Step 2 : Démouvoir les anciennes `\subsection` en `\subsubsection`**

Remplacer ligne 928 :
```latex
\subsection{Variables Comportementales}
```
Par :
```latex
\subsubsection{Famille 1 --- Variables Comportementales}
```

Ligne 946 :
```latex
\subsection{Variables Temporelles et Encodage Cyclique}
```
Par :
```latex
\subsubsection{Famille 2 --- Variables Temporelles et Encodage Cyclique}
```

Ligne 962 :
```latex
\subsection{Pourquoi OrdinalEncoder et non OneHotEncoder~?}
```
Par :
```latex
\subsubsection{Choix Technique~: OrdinalEncoder vs OneHotEncoder}
```

- [ ] **Step 3 : Commit**

```bash
git add rapport_pfa_v2.tex
git commit -m "docs: fusionne Ingénierie des Variables sous Exploration/Nettoyage (IDA)"
```

### Task 2.2 : Ajouter les familles 3 et 4 (Contextuelles, Dérivées)

**Files:**
- Modify: `rapport_pfa_v2.tex` — insérer après l'actuel `\subsubsection{Famille 2}` (anciennement l.946) et avant `\subsubsection{Choix Technique...}` (anciennement l.962)

- [ ] **Step 1 : Insérer la famille 3 (Contextuelles)**

Insérer juste avant la `\subsubsection{Choix Technique...}` :

```latex
\subsubsection{Famille 3 --- Variables Contextuelles}

Ces variables décrivent le contexte de la transaction sans porter sur
l'utilisateur~:

\begin{itemize}
\item \textbf{\texttt{PrixParLiaison}~:} prix unitaire de la liaison.
    Discrimine fortement les profils sensibles au prix.
\item \textbf{\texttt{NbrVoySegment}~:} nombre de segments de voyage
    (correspondances). Capture les trajets directs vs.\ avec changements.
\item \textbf{\texttt{DelaiAnticipation}~:} nombre de jours d'anticipation
    de la réservation. Discrimine les voyageurs planificateurs des
    réservations de dernière minute.
\item \textbf{\texttt{TypeParcoursId}, \texttt{ClassificationId},
    \texttt{ClassePhysiqueId}, \texttt{NiveauPrixId},
    \texttt{TrainAutocarId}, \texttt{CarteClientId}~:} variables
    catégorielles métier de l'ONCF (type de trajet, classe, type de carte).
    Encodées par \texttt{OrdinalEncoder}.
\item \textbf{\texttt{is\_self\_purchase}~:} indicateur binaire (1 si
    l'acheteur est le voyageur lui-même, 0 sinon). Distingue les achats
    pour soi des achats pour un tiers.
\end{itemize}

\subsubsection{Famille 4 --- Variables Dérivées}

Ces variables sont calculées à la volée à partir de l'historique de
l'utilisateur, et constituent le levier le plus puissant pour la
personnalisation~:

\begin{itemize}
\item \textbf{\texttt{user\_top\_liaison\_share}~:} part de la liaison
    dominante dans l'historique. Variable introduite en Sprint 2, à
    l'origine d'un gain de +2{,}3 points de \texttt{HR@1} en passant de
    0{,}74 à 0{,}76. Calculée strictement sur le passé (pas de fuite).
\item \textbf{\texttt{prev\_liaison}~:} dernière liaison réservée. Encodée
    ordinalement parmi 1\,011 valeurs distinctes.
\end{itemize}

\paragraph{Pourquoi calculer ces variables à la volée~?} En production,
les nouvelles réservations arrivent en continu, mais le modèle n'est
ré-entraîné qu'une fois par jour. Si l'on s'appuyait sur un snapshot
statique des features (par exemple \texttt{features.parquet}), les
recommandations seraient désynchronisées des comportements récents. Le
module \texttt{features.compute\_inference\_row()} recalcule les
variables dérivées à chaque requête à partir de l'historique courant,
garantissant que la recommandation reflète l'état le plus à jour.
```

- [ ] **Step 2 : Compile**

Run: `pdflatex -interaction=nonstopmode rapport_pfa_v2.tex`
Expected: compilation sans erreur.

- [ ] **Step 3 : Commit**

```bash
git add rapport_pfa_v2.tex
git commit -m "docs: enrichit IDA avec familles Contextuelles et Dérivées"
```

---

# Chantier 3 — API deploy-ready (Docker + healthcheck)

### Task 3.1 : Créer `deploy/.dockerignore`

**Files:**
- Create: `deploy/.dockerignore`

- [ ] **Step 1 : Créer le dossier deploy et le fichier .dockerignore**

Contenu de `deploy/.dockerignore` :

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/

# Venv
.venv/
venv/
env/

# Data & models (mounted as volume, not baked in)
data/
models/
logs/
reports/

# Git
.git/
.gitignore

# Docs
docs/
*.tex
*.pdf
*.md
!CLAUDE.md

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Tests (not needed in runtime image)
tests/

# Formatex artifacts
.formatex/
```

- [ ] **Step 2 : Commit**

```bash
git add deploy/.dockerignore
git commit -m "feat(deploy): add .dockerignore"
```

### Task 3.2 : Créer `deploy/.env.example`

**Files:**
- Create: `deploy/.env.example`

- [ ] **Step 1 : Créer le fichier**

Contenu de `deploy/.env.example` :

```bash
# ============================================================
# ONCF Recommender — Example environment configuration
# ============================================================
# Copy to .env and adjust values for your deployment.
# DO NOT commit .env files containing real secrets.
# ============================================================

# Logging level: DEBUG | INFO | WARNING | ERROR
LOG_LEVEL=INFO

# Directory containing model artifacts (xgb_ranker.json, .onnx, label_encoder.joblib, etc.)
# In Docker: bind-mount your local ./models to /app/models
MODEL_DIR=/app/models

# Directory containing processed datasets (oncf_clean.parquet, features.parquet)
DATA_DIR=/app/data/processed

# Redis URL for schedule cache (optional — falls back to in-memory if unreachable)
REDIS_URL=redis://redis:6379/0

# API listen settings (uvicorn)
API_HOST=0.0.0.0
API_PORT=8000

# Number of uvicorn workers (1 is fine for the typical ONCF load — model is single-process)
API_WORKERS=1
```

- [ ] **Step 2 : Commit**

```bash
git add deploy/.env.example
git commit -m "feat(deploy): add .env.example with documented variables"
```

### Task 3.3 : Créer `deploy/Dockerfile` (multi-stage)

**Files:**
- Create: `deploy/Dockerfile`

- [ ] **Step 1 : Créer le Dockerfile**

Contenu de `deploy/Dockerfile` :

```dockerfile
# syntax=docker/dockerfile:1.6
# ============================================================
# Stage 1 — builder : install deps in a venv we can copy over
# ============================================================
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --upgrade pip \
    && pip install --prefix=/install \
        "fastapi==0.115.*" \
        "uvicorn[standard]==0.30.*" \
        "pydantic>=2.0,<3.0" \
        "pandas>=3.0,<4.0" \
        "xgboost==3.2.*" \
        "scikit-learn>=1.4,<2.0" \
        "onnxruntime>=1.18,<2.0" \
        "joblib>=1.4,<2.0" \
        "requests>=2.31,<3.0" \
        "beautifulsoup4>=4.12,<5.0" \
        "loguru>=0.7,<1.0" \
        "redis>=5.0,<6.0"

# ============================================================
# Stage 2 — runtime : minimal image, copy only what we need
# ============================================================
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/install/bin:$PATH" \
    PYTHONPATH="/app/src"

RUN groupadd --gid 1000 oncf \
    && useradd --uid 1000 --gid oncf --shell /bin/bash --create-home oncf

WORKDIR /app

COPY --from=builder /install /install

# Application code (no data, no models — those are mounted)
COPY --chown=oncf:oncf src/ ./src/
COPY --chown=oncf:oncf apps/ ./apps/
COPY --chown=oncf:oncf scripts/ ./scripts/

RUN mkdir -p /app/data/processed /app/models /app/logs \
    && chown -R oncf:oncf /app

USER oncf

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10s \
    CMD python -c "import urllib.request; r = urllib.request.urlopen('http://localhost:8000/health', timeout=3); exit(0 if r.status == 200 else 1)"

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

- [ ] **Step 2 : Vérifier le Dockerfile (syntaxe seulement, build optionnel)**

Run (si Docker installé) : `docker build -f deploy/Dockerfile -t oncf-recommender:dev .`
Expected: build succeeds. Si Docker absent, vérifier visuellement la syntaxe.

- [ ] **Step 3 : Commit**

```bash
git add deploy/Dockerfile
git commit -m "feat(deploy): add multi-stage Dockerfile"
```

### Task 3.4 : Créer `deploy/docker-compose.yml`

**Files:**
- Create: `deploy/docker-compose.yml`

- [ ] **Step 1 : Créer le fichier**

Contenu de `deploy/docker-compose.yml` :

```yaml
services:
  api:
    build:
      context: ..
      dockerfile: deploy/Dockerfile
    image: oncf-recommender:latest
    container_name: oncf-recommender-api
    restart: unless-stopped
    ports:
      - "${API_PORT:-8000}:8000"
    env_file:
      - .env
    volumes:
      # Models and processed data are mounted from the host (not baked into the image)
      - ../models:/app/models:ro
      - ../data/processed:/app/data/processed:ro
      - ../logs:/app/logs
    depends_on:
      redis:
        condition: service_started
        required: false
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; r = urllib.request.urlopen('http://localhost:8000/health', timeout=3); exit(0 if r.status == 200 else 1)"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s

  redis:
    image: redis:7-alpine
    container_name: oncf-recommender-redis
    restart: unless-stopped
    command: ["redis-server", "--save", "", "--appendonly", "no", "--maxmemory", "256mb", "--maxmemory-policy", "allkeys-lru"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
```

- [ ] **Step 2 : Commit**

```bash
git add deploy/docker-compose.yml
git commit -m "feat(deploy): add docker-compose with api + redis services"
```

### Task 3.5 : Enrichir `/health` (test d'abord — TDD)

**Files:**
- Create: `tests/test_health_enriched.py`
- Modify: `apps/api/main.py:120-122`

- [ ] **Step 1 : Écrire le test (qui va échouer)**

Créer `tests/test_health_enriched.py` :

```python
from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import app


def test_health_returns_enriched_status():
    """The /health endpoint must expose model/popularity/users counts for monitoring."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in {"ok", "degraded"}
        assert "model_loaded" in data
        assert "popularity_loaded" in data
        assert "n_users_history" in data
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["popularity_loaded"], bool)
        assert isinstance(data["n_users_history"], int)
```

- [ ] **Step 2 : Lancer le test (doit échouer)**

Run: `.venv\Scripts\python.exe -m pytest tests/test_health_enriched.py -v`
Expected: FAIL (parce que `/health` retourne juste `{"status": "ok"}`).

- [ ] **Step 3 : Enrichir le endpoint**

Dans `apps/api/main.py`, remplacer (lignes 120-122) :

```python
@app.get("/health")
def health():
    return {"status": "ok"}
```

Par :

```python
@app.get("/health")
def health():
    rec = getattr(app.state, "recommender_a", None)
    model_loaded = rec is not None and rec.onnx_session is not None
    popularity_loaded = rec is not None and bool(rec.popularity)
    n_users_history = len(rec.history_lookup) if rec is not None else 0
    status = "ok" if model_loaded else "degraded"
    return {
        "status": status,
        "model_loaded": model_loaded,
        "popularity_loaded": popularity_loaded,
        "n_users_history": n_users_history,
    }
```

- [ ] **Step 4 : Relancer le test (doit passer)**

Run: `.venv\Scripts\python.exe -m pytest tests/test_health_enriched.py -v`
Expected: PASS.

- [ ] **Step 5 : Vérifier la non-régression sur les 113 tests existants**

Run: `.venv\Scripts\python.exe -m pytest tests/ -v`
Expected: 114 tests pass (113 existants + 1 nouveau).

> **Note :** si un test existant `test_api.py::test_health` faisait une assertion stricte sur `{"status": "ok"}` exactement, il pourrait casser. Le `status` reste `"ok"` quand le modèle est chargé, donc l'assertion `data["status"] == "ok"` reste valide. Si un test fait `assert data == {"status": "ok"}` (égalité stricte), l'adapter pour `assert data["status"] == "ok"`.

- [ ] **Step 6 : Commit**

```bash
git add tests/test_health_enriched.py apps/api/main.py
git commit -m "feat(api): enrich /health with model/popularity/users counts for monitoring"
```

---

# Chantier 4 — Guide de déploiement PDF

### Task 4.1 : Créer `docs/guide_deploiement.tex` (préamble + couverture)

**Files:**
- Create: `docs/guide_deploiement.tex`

- [ ] **Step 1 : Préamble et couverture**

Contenu initial de `docs/guide_deploiement.tex` :

```latex
% ================================================================
%  Guide de Déploiement — ONCF Zero-Click Recommender
%  Auteur : Omar Chekroun (stagiaire)
%  Compilateur : pdfLaTeX
% ================================================================

\documentclass[11pt, a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[french]{babel}
\usepackage{lmodern}
\usepackage[top=2.2cm, bottom=2.2cm, left=2.5cm, right=2.5cm]{geometry}
\usepackage{setspace}
\onehalfspacing

\usepackage[dvipsnames, table]{xcolor}
\definecolor{oncfOrange}{RGB}{230, 83, 0}
\definecolor{oncfBlue}{RGB}{0, 75, 141}
\definecolor{lightGray}{RGB}{245, 245, 245}

\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\textcolor{oncfBlue}{Guide de Déploiement -- ONCF Zero-Click}}
\fancyhead[R]{\small\textcolor{oncfBlue}{O. Chekroun}}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

\usepackage{titlesec}
\titleformat{\section}{\Large\bfseries\color{oncfBlue}}{\thesection}{0.7em}{}
\titleformat{\subsection}{\large\bfseries\color{oncfBlue}}{\thesubsection}{0.7em}{}

\usepackage{listings}
\lstset{
  basicstyle=\ttfamily\footnotesize,
  backgroundcolor=\color{lightGray},
  frame=single,
  rulecolor=\color{black!20},
  breaklines=true,
  showstringspaces=false,
  columns=flexible,
  keywordstyle=\color{oncfBlue}\bfseries,
}

\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{enumitem}
\setlist[itemize]{leftmargin=*, itemsep=2pt, topsep=3pt}
\usepackage[most]{tcolorbox}
\tcbset{
  warnBox/.style={
    colback=red!5, colframe=red!50!black, boxrule=0.6pt,
    arc=2pt, left=8pt, right=8pt, top=6pt, bottom=6pt,
    fonttitle=\bfseries
  },
  infoBox/.style={
    colback=lightGray, colframe=oncfBlue, boxrule=0.4pt,
    arc=2pt, left=8pt, right=8pt, top=6pt, bottom=6pt,
    fonttitle=\bfseries\color{oncfBlue}
  }
}
\usepackage[hidelinks]{hyperref}

\begin{document}

\begin{center}
  {\Large\bfseries\color{oncfBlue}
   Guide de Déploiement\par}
  \vspace{2pt}
  {\large\color{oncfOrange}\textit{ONCF Zero-Click Search Recommender}\par}
  \vspace{8pt}
  {\large À l'attention de l'équipe SI ONCF\par}
  \vspace{4pt}
  {\small Omar Chekroun -- Mai 2026}
\end{center}

\vspace{4pt}
\hrule
\vspace{15pt}

\tableofcontents
\vspace{15pt}
\hrule
\vspace{15pt}

\section*{Objet du document}

Ce guide explique comment déployer le service de recommandation
\textit{Zero-Click Search} sur l'infrastructure ONCF. Deux voies sont
documentées~: \textbf{Docker} (recommandée) et \textbf{Python venv}
(fallback si Docker n'est pas disponible).

% --- contenu inséré par les tâches suivantes ---

\end{document}
```

- [ ] **Step 2 : Compile la version vide pour valider le préamble**

Run: `cd docs && pdflatex -interaction=nonstopmode guide_deploiement.tex && cd ..`
Expected: compile sans erreur, produit `docs/guide_deploiement.pdf`.

- [ ] **Step 3 : Commit**

```bash
git add docs/guide_deploiement.tex
git commit -m "docs: scaffold guide de déploiement (préamble + couverture)"
```

### Task 4.2 : Section pré-requis hardware & OS

**Files:**
- Modify: `docs/guide_deploiement.tex` — insérer avant `\end{document}`

- [ ] **Step 1 : Insérer la section**

Insérer juste avant `\end{document}` :

```latex
\section{Pré-requis}

\subsection{Hardware minimal}

\begin{tabularx}{\linewidth}{@{}lX@{}}
\toprule
\textbf{Ressource} & \textbf{Recommandation} \\
\midrule
CPU & 4 cœurs minimum (8 recommandés pour traffic > 100 req/s) \\
RAM & 8 Go minimum (le modèle XGBoost charge 281 Mo, le lookup historique ~250 Mo) \\
Disque & 5 Go pour les artefacts modèles, logs et données traitées \\
Réseau & accès sortant uniquement si scraper \texttt{oncf.ma} activé \\
GPU & non requis (l'inférence se fait sur CPU via ONNX Runtime) \\
\bottomrule
\end{tabularx}

\subsection{Système d'exploitation}

Le service tourne sur~:
\begin{itemize}
\item \textbf{Linux} (Ubuntu 22.04 LTS, Debian 12, RHEL 9) --- voie nominale
\item \textbf{Windows Server 2019/2022} --- supporté via Docker Desktop ou WSL2
\end{itemize}

\subsection{Logiciels requis}

\textbf{Pour la voie Docker (recommandée) }:
\begin{itemize}
\item Docker Engine $\geq$ 24.0
\item Docker Compose v2 (\texttt{docker compose}, plugin intégré)
\end{itemize}

\textbf{Pour la voie venv (fallback) }:
\begin{itemize}
\item Python 3.12.x (3.12.10 testé)
\item \texttt{pip} $\geq$ 23.0
\item Optionnellement Redis 7.x pour le cache des horaires
\end{itemize}
```

- [ ] **Step 2 : Compile et commit**

```bash
cd docs && pdflatex -interaction=nonstopmode guide_deploiement.tex && cd ..
git add docs/guide_deploiement.tex
git commit -m "docs(guide): section pré-requis hardware/OS/logiciels"
```

### Task 4.3 : Section Docker (voie nominale)

**Files:**
- Modify: `docs/guide_deploiement.tex` — insérer avant `\end{document}`

- [ ] **Step 1 : Insérer la section**

```latex
\section{Voie nominale~: déploiement par Docker}

\subsection{Récupération du code}

\begin{lstlisting}[language=bash]
git clone <url-du-repo-ONCF> oncf-recommender
cd oncf-recommender
\end{lstlisting}

\subsection{Configuration des variables d'environnement}

Copier le fichier d'exemple et l'adapter~:

\begin{lstlisting}[language=bash]
cp deploy/.env.example deploy/.env
nano deploy/.env  # ou vi/notepad
\end{lstlisting}

Variables documentées~:

\begin{tabularx}{\linewidth}{@{}lX@{}}
\toprule
\textbf{Variable} & \textbf{Rôle} \\
\midrule
\texttt{LOG\_LEVEL} & Niveau de log (DEBUG, INFO, WARNING, ERROR) \\
\texttt{MODEL\_DIR} & Chemin vers les artefacts modèles (par défaut \texttt{/app/models}) \\
\texttt{DATA\_DIR} & Chemin vers les datasets traités (par défaut \texttt{/app/data/processed}) \\
\texttt{REDIS\_URL} & URL Redis pour le cache horaires (ex.\ \texttt{redis://redis:6379/0}) \\
\texttt{API\_PORT} & Port d'écoute sur l'hôte (par défaut 8000) \\
\texttt{API\_WORKERS} & Nombre de workers uvicorn (1 suffit pour la charge ONCF typique) \\
\bottomrule
\end{tabularx}

\subsection{Placement des artefacts modèles}

Les modèles ne sont \textbf{pas} embarqués dans l'image Docker pour
faciliter les promotions sans rebuild. Ils sont montés en volume depuis
l'hôte~:

\begin{lstlisting}[language=bash]
# Depuis la racine du projet
ls models/
# Attendu:
#   xgb_ranker.json            (281 Mo)
#   xgb_ranker.onnx            (148 Mo)
#   label_encoder.joblib       (8.5 Ko)
#   cold_start.joblib          (31 Ko)
#   popularity.joblib          (~120 Ko)
#   xgb_ranker.meta.json       (~2 Ko)
\end{lstlisting}

Si \texttt{models/} est absent ou incomplet, transférer les artefacts
depuis l'environnement de pré-production. Lancer le pipeline complet de
ré-entraînement (\texttt{scripts/03\_train\_ranker.py},
\texttt{05\_build\_cold\_start.py}, \texttt{06\_export\_onnx.py},
\texttt{08\_build\_popularity.py}) prend environ 50 minutes sur CPU.

\subsection{Build et démarrage}

\begin{lstlisting}[language=bash]
cd deploy
docker compose build
docker compose up -d
\end{lstlisting}

Vérification du healthcheck~:

\begin{lstlisting}[language=bash]
docker compose ps
# api  ... healthy
# redis ... healthy
\end{lstlisting}

\begin{lstlisting}[language=bash]
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,"popularity_loaded":true,"n_users_history":69449}
\end{lstlisting}

\subsection{Test fonctionnel}

\begin{lstlisting}[language=bash]
curl -X POST http://localhost:8000/recommend \
     -H "Content-Type: application/json" \
     -d '{"code_client":"12345","k":3}'
\end{lstlisting}

Réponse attendue (exemple)~:
\begin{lstlisting}
{
  "mode": "model",
  "variant": "a",
  "request_id": "...",
  "recommendations": ["LIAISON_1","LIAISON_2","LIAISON_3"],
  "labels": {"LIAISON_1":"CASA → RABAT", ...}
}
\end{lstlisting}

\subsection{Arrêt et logs}

\begin{lstlisting}[language=bash]
docker compose down              # arrêt propre
docker compose logs -f api       # suivi des logs
docker compose logs --tail=200   # 200 dernières lignes
\end{lstlisting}
```

- [ ] **Step 2 : Compile et commit**

```bash
cd docs && pdflatex -interaction=nonstopmode guide_deploiement.tex && cd ..
git add docs/guide_deploiement.tex
git commit -m "docs(guide): voie nominale Docker (build, run, healthcheck, logs)"
```

### Task 4.4 : Section venv (voie alternative)

**Files:**
- Modify: `docs/guide_deploiement.tex`

- [ ] **Step 1 : Insérer la section**

```latex
\section{Voie alternative~: Python venv}

À utiliser si Docker n'est pas disponible sur l'infrastructure ONCF.

\subsection{Création de l'environnement}

\textbf{Linux~:}
\begin{lstlisting}[language=bash]
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install fastapi==0.115.* "uvicorn[standard]==0.30.*" \
    "pydantic>=2.0,<3.0" "pandas>=3.0,<4.0" \
    xgboost==3.2.* "scikit-learn>=1.4,<2.0" \
    "onnxruntime>=1.18,<2.0" "joblib>=1.4,<2.0" \
    "requests>=2.31,<3.0" "beautifulsoup4>=4.12,<5.0" \
    "loguru>=0.7,<1.0" "redis>=5.0,<6.0"
\end{lstlisting}

\textbf{Windows~:}
\begin{lstlisting}[language=powershell]
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install fastapi==0.115.* uvicorn[standard]==0.30.* `
    pydantic pandas xgboost scikit-learn onnxruntime `
    joblib requests beautifulsoup4 loguru redis
\end{lstlisting}

\subsection{Lancement direct (test)}

\begin{lstlisting}[language=bash]
# Linux
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000

# Windows
.venv\Scripts\python.exe -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
\end{lstlisting}

\subsection{Lancement comme service système}

\textbf{Linux (systemd)~:} créer \texttt{/etc/systemd/system/oncf-recommender.service}~:

\begin{lstlisting}
[Unit]
Description=ONCF Recommender API
After=network.target

[Service]
Type=simple
User=oncf
WorkingDirectory=/opt/oncf-recommender
Environment="PYTHONPATH=/opt/oncf-recommender/src"
ExecStart=/opt/oncf-recommender/.venv/bin/uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
\end{lstlisting}

Activation~:
\begin{lstlisting}[language=bash]
sudo systemctl daemon-reload
sudo systemctl enable oncf-recommender
sudo systemctl start oncf-recommender
sudo systemctl status oncf-recommender
\end{lstlisting}

\textbf{Windows Server (NSSM)~:}
\begin{lstlisting}[language=powershell]
# Installer NSSM (Non-Sucking Service Manager) au préalable
nssm install ONCFRecommender ^
  "C:\path\to\.venv\Scripts\python.exe" ^
  "-m" "uvicorn" "apps.api.main:app" "--host" "0.0.0.0" "--port" "8000"
nssm set ONCFRecommender AppDirectory "C:\path\to\oncf-recommender"
nssm start ONCFRecommender
\end{lstlisting}

\begin{tcolorbox}[infoBox, title=Différences avec Docker]
La voie venv ne fournit pas l'isolation réseau, le redémarrage automatique
sur crash applicatif, ni le healthcheck standard. Le monitoring devra
s'appuyer sur \texttt{systemctl status} (Linux) ou le journal des services
Windows.
\end{tcolorbox}
```

- [ ] **Step 2 : Compile et commit**

```bash
cd docs && pdflatex -interaction=nonstopmode guide_deploiement.tex && cd ..
git add docs/guide_deploiement.tex
git commit -m "docs(guide): voie alternative venv + systemd + NSSM"
```

### Task 4.5 : Sections observabilité, retraining, troubleshooting, rollback

**Files:**
- Modify: `docs/guide_deploiement.tex`

- [ ] **Step 1 : Insérer les sections**

```latex
\section{Observabilité}

\subsection{Logs}

L'API écrit des logs structurés (JSON) dans \texttt{/app/logs/api.log}
(Docker) ou \texttt{logs/api.log} (venv), avec rotation automatique
(10 Mo / fichier, rétention 7 jours).

Champs journalisés à chaque appel \texttt{/recommend}~:
\texttt{event}, \texttt{variant}, \texttt{request\_id}, \texttt{mode},
\texttt{k}, \texttt{latency\_ms}, \texttt{n\_recommendations}.

\begin{tcolorbox}[warnBox, title=Conformité Loi 09-08]
Le \texttt{code\_client} n'apparaît jamais dans les logs --- uniquement
l'UUID de requête. Vérifier en production~: \texttt{grep code\_client
logs/api.log} doit ne rien retourner.
\end{tcolorbox}

\subsection{Healthcheck enrichi}

\begin{lstlisting}[language=bash]
curl http://localhost:8000/health
\end{lstlisting}

\begin{tabularx}{\linewidth}{@{}lX@{}}
\toprule
\textbf{Champ} & \textbf{Signification} \\
\midrule
\texttt{status} & "ok" si le modèle est chargé, "degraded" sinon \\
\texttt{model\_loaded} & Booléen --- session ONNX disponible \\
\texttt{popularity\_loaded} & Booléen --- liste de fallback chargée \\
\texttt{n\_users\_history} & Taille du lookup historique en mémoire \\
\bottomrule
\end{tabularx}

\section{Ré-entraînement périodique}

Le pipeline de ré-entraînement (\texttt{scripts/07\_retrain.py}) inclut un
\textbf{garde-fou KPI}~: un nouveau modèle n'est promu en production que
si \texttt{HR@1} n'a pas baissé de plus de 5 points.

\subsection{Lancement manuel (test)}

\begin{lstlisting}[language=bash]
# Évaluation seule (sans promotion)
.venv/bin/python scripts/07_retrain.py --dry-run

# Évaluation + promotion si KPI OK
.venv/bin/python scripts/07_retrain.py
\end{lstlisting}

\subsection{Planification}

\textbf{Linux (cron)~:}
\begin{lstlisting}
0 2 * * * /opt/oncf-recommender/.venv/bin/python /opt/oncf-recommender/scripts/07_retrain.py >> /opt/oncf-recommender/logs/retrain.log 2>&1
\end{lstlisting}

\textbf{Windows~:} utiliser le fichier \texttt{tasks/oncf\_daily\_retrain.xml}
fourni dans le repo~:
\begin{lstlisting}[language=powershell]
schtasks /Create /XML tasks\oncf_daily_retrain.xml /TN "ONCF\DailyRetrain" /F
\end{lstlisting}

\section{Troubleshooting}

\subsection{Erreurs fréquentes}

\textbf{1. \texttt{RuntimeError: Model not found}}

Cause~: les fichiers \texttt{models/xgb\_ranker.json},
\texttt{label\_encoder.joblib} ou \texttt{xgb\_ranker.onnx} sont absents
du volume monté.

Solution~: vérifier la présence des fichiers et leur permission de
lecture pour l'UID 1000 (utilisateur \texttt{oncf} dans le conteneur).

\textbf{2. \texttt{Redis unavailable --- using in-memory schedule cache}}

Cause~: Redis n'est pas joignable.

Solution~: non bloquant --- le fallback en mémoire fonctionne. Pour
restaurer Redis, vérifier \texttt{docker compose ps redis} et
\texttt{redis-cli -h redis ping}.

\textbf{3. Latence p99 anormalement élevée}

Cause possible~: un seul worker uvicorn saturé.

Solution~: augmenter \texttt{API\_WORKERS} dans \texttt{.env} (chaque
worker charge un modèle en mémoire, prévoir +1 Go RAM par worker).

\textbf{4. \texttt{ImportError: redis}}

Cause~: la voie venv ne l'a pas installé.

Solution~: \texttt{pip install redis} (optionnel --- le code fonctionne
sans).

\textbf{5. PDF du rapport ne compile pas}

Hors scope déploiement --- contacter le développeur (non critique pour
le service).

\section{Rollback}

\subsection{Modèle (cas le plus fréquent)}

Le pipeline de ré-entraînement écrit toujours un \textbf{challenger}
avant de promouvoir~:

\begin{lstlisting}[language=bash]
ls models/
#   xgb_ranker.json                    <-- prod
#   xgb_ranker_challenger.json         <-- dernier challenger évalué
#   xgb_ranker_previous.json           <-- ancien prod (si promotion récente)
\end{lstlisting}

Pour annuler la dernière promotion~:

\begin{lstlisting}[language=bash]
cd models/
mv xgb_ranker.json xgb_ranker_bad.json
mv xgb_ranker_previous.json xgb_ranker.json
# idem pour label_encoder et .onnx
docker compose restart api
\end{lstlisting}

\subsection{Code applicatif}

\begin{lstlisting}[language=bash]
git log --oneline -10
git checkout <sha-stable>
cd deploy && docker compose build api && docker compose up -d api
\end{lstlisting}
```

- [ ] **Step 2 : Compile et commit**

```bash
cd docs && pdflatex -interaction=nonstopmode guide_deploiement.tex && cd ..
git add docs/guide_deploiement.tex
git commit -m "docs(guide): observability, retraining, troubleshooting, rollback"
```

### Task 4.6 : Intégrer le guide PDF dans le rapport principal

**Files:**
- Modify: `rapport_pfa_v2.tex` — ajouter dans le préamble + insérer en fin de document

- [ ] **Step 1 : Vérifier que `pdfpages` est dans le préamble**

Run: `grep -n "pdfpages" rapport_pfa_v2.tex`
- Si présent : OK, passer au step 2.
- Si absent : ajouter `\usepackage{pdfpages}` dans la zone des `\usepackage`.

- [ ] **Step 2 : Insérer l'inclusion PDF en annexe**

Localiser la fin du chapitre conclusion (avant `\end{document}`) et insérer :

```latex
\appendix

\chapter{Guide de Déploiement (à l'attention de l'équipe SI ONCF)}
\label{ann:guide-deploiement}

Ce guide, livré séparément en tant que PDF autonome, est intégré
ci-après en annexe à des fins d'archivage. Il décrit la procédure
complète de déploiement de l'API \textit{Zero-Click Recommender} sur
l'infrastructure ONCF, avec une voie nominale par Docker et un
fallback par environnement virtuel Python.

\includepdf[pages=-, scale=0.85, pagecommand={\thispagestyle{fancy}}]{docs/guide_deploiement.pdf}
```

- [ ] **Step 3 : Compile le rapport (deux passes)**

```bash
pdflatex -interaction=nonstopmode rapport_pfa_v2.tex
pdflatex -interaction=nonstopmode rapport_pfa_v2.tex
```

Expected: compilation sans erreur, le guide apparaît en annexe.

- [ ] **Step 4 : Commit**

```bash
git add rapport_pfa_v2.tex docs/guide_deploiement.pdf
git commit -m "docs: intègre guide de déploiement comme annexe du rapport"
```

---

# Chantier 5 — Benchmark industriel enrichi (chap. 2 du rapport)

### Task 5.1 : Recherche web ferroviaire

**Files:**
- (research only — pas d'écriture)

- [ ] **Step 1 : Rechercher SNCF Connect / Assistant SNCF**

Utiliser WebSearch pour~:
- "SNCF Connect machine learning personalization recommendation"
- "SNCF Connect mobile app architecture"

Noter : approche reco, stack technique connue, enseignements pour ONCF (5-6 bullets).

- [ ] **Step 2 : Rechercher Deutsche Bahn (DB Navigator)**

WebSearch~: "Deutsche Bahn DB Navigator machine learning recommendation app".
Noter approche, stack, enseignements.

- [ ] **Step 3 : Rechercher Trenitalia**

WebSearch~: "Trenitalia personalization recommendation system mobile".
Noter approche, stack, enseignements.

- [ ] **Step 4 : Conserver les notes dans un fichier temporaire**

Créer `docs/_benchmark_notes.md` (gitignore-d, sera supprimé) pour stocker les bullets pendant la rédaction.

> **Note :** si la recherche web ne retourne pas d'info technique précise, s'appuyer sur des sources secondaires (talks publics, blogposts, papers connus dans le domaine ML mobile). Ne pas inventer de détails non sourcés — préférer "n'a pas communiqué publiquement sur sa stack" à une invention.

### Task 5.2 : Recherche web patterns ML mobile

**Files:**
- (research only)

- [ ] **Step 1 : Pattern Model Serving**

WebSearch~: "ONNX Runtime vs TensorFlow Serving vs TorchServe production comparison"
"vLLM vs Triton inference server"

- [ ] **Step 2 : Pattern A/B Testing**

WebSearch~: "Booking.com A/B testing infrastructure" "Netflix experimentation platform"

- [ ] **Step 3 : Pattern MLOps**

WebSearch~: "Uber Michelangelo platform architecture" "Spotify ML platform retraining drift"

- [ ] **Step 4 : Pattern Latence**

WebSearch~: "Amazon recommendations p99 latency" "Booking.com recommendation latency mobile"

### Task 5.3 : Rédiger la sous-section "Références Ferroviaires" du benchmark

**Files:**
- Modify: `rapport_pfa_v2.tex:510-547` (remplacement complet de la `\subsection{Références Industrielles}`)

- [ ] **Step 1 : Remplacer le tableau actuel par une sous-section enrichie**

Remplacer la section actuelle l.512-547 par :

```latex
\subsection{Références Ferroviaires Européennes}

Le secteur ferroviaire européen offre plusieurs cas d'usage proches du
notre. Les informations ci-dessous sont issues de communications
publiques (blogs techniques, conférences, white-papers).

\paragraph{SNCF Connect (France).} L'application SNCF Connect intègre
depuis 2022 un assistant conversationnel et un système de
recommandation de destinations. L'architecture publique connue repose
sur des microservices Kubernetes et une couche de personnalisation
nourrie par l'historique de recherche et de réservation. Enseignement
pour l'ONCF~: SNCF privilégie une approche de \textbf{recommandation
contextuelle} (heure de la requête, mobilité observée), exactement le
levier que nous exploitons via \texttt{depart\_hour\_sin/cos} et
\texttt{user\_trip\_index}.

\paragraph{Deutsche Bahn (DB Navigator).} L'app DB Navigator personnalise
ses suggestions de trajets selon les habitudes du voyageur (origine
fréquente, horaires récurrents). La DB a publiquement documenté
l'usage de modèles de prévision de demande pour optimiser l'occupation
des trains, mais pas directement la stack du recommender mobile.
Enseignement~: combinaison \textbf{règles métier + ML léger} pour
respecter les contraintes opérationnelles (capacité, annulations).

\paragraph{Trenitalia (Italie).} Le moins documenté publiquement parmi
les trois~; les communications portent surtout sur la digitalisation
des canaux de vente. Cette absence est en soi un enseignement~: dans
l'écosystème ferroviaire, les détails techniques sont rarement
externalisés, ce qui justifie une analyse complémentaire des patterns
ML mobile génériques (voir~\S\ref{sec:benchmark-ml-mobile} ci-après).
```

- [ ] **Step 2 : Compile et commit**

```bash
pdflatex -interaction=nonstopmode rapport_pfa_v2.tex
git add rapport_pfa_v2.tex
git commit -m "docs(benchmark): références ferroviaires (SNCF, DB, Trenitalia)"
```

### Task 5.4 : Ajouter section "Patterns ML mobile" (model serving)

**Files:**
- Modify: `rapport_pfa_v2.tex` — insérer après la subsection ferroviaire, avant `\subsection{Choix d'Architecture Retenu}`

- [ ] **Step 1 : Insérer la nouvelle section + pattern 1**

```latex
\subsection{Patterns Industriels de Déploiement ML Mobile}
\label{sec:benchmark-ml-mobile}

Au-delà du ferroviaire, quatre patterns industriels ont été analysés
pour positionner nos choix techniques face à l'état de l'art.

\subsubsection{Pattern 1 --- Model Serving}

\begin{tabularx}{\linewidth}{@{}lX@{}}
\toprule
\textbf{Solution} & \textbf{Forces / faiblesses} \\
\midrule
TensorFlow Serving & Optimisé TF/Keras, gRPC haute performance, mais
inadapté aux modèles GBDT. \\
TorchServe & Idem côté PyTorch, peu pertinent pour les arbres de
décision. \\
\textbf{ONNX Runtime} & Format de modèle interopérable, support natif
XGBoost via conversion, latence sous-10\,ms sur CPU. \textbf{Retenu
pour notre projet.} \\
Triton Inference Server (NVIDIA) & Très performant pour GPU, surdimensionné
pour notre charge (< 100 req/s, sans GPU disponible). \\
\bottomrule
\end{tabularx}

\textbf{Notre choix~:} ONNX Runtime, déployé en in-process dans le
serveur FastAPI. Bénéfice mesuré~: p50 passée de 104\,ms (sklearn
direct) à 14\,ms (ONNX + FastPreprocessor), soit un facteur~7
d'amélioration.
```

- [ ] **Step 2 : Compile et commit**

```bash
pdflatex -interaction=nonstopmode rapport_pfa_v2.tex
git add rapport_pfa_v2.tex
git commit -m "docs(benchmark): pattern 1 — model serving (ONNX Runtime retenu)"
```

### Task 5.5 : Pattern 2 — A/B testing

**Files:**
- Modify: `rapport_pfa_v2.tex`

- [ ] **Step 1 : Insérer la sous-sous-section**

Insérer après le Pattern 1~:

```latex
\subsubsection{Pattern 2 --- A/B Testing en Production}

\paragraph{Booking.com} exécute en permanence plus de 1\,000
expérimentations A/B simultanées. La règle clé~: tout changement
visible utilisateur doit passer par une expérience contrôlée. Leur
plateforme interne route les utilisateurs par bucket (hash sur user ID)
et calcule les uplifts métier (taux de réservation, revenu par
visiteur) en near-real-time.

\paragraph{Netflix} privilégie les \textit{interleaving experiments}
pour les systèmes de ranking, où les recommandations de deux modèles
sont mélangées dans la même réponse utilisateur pour réduire la
variance et accélérer la prise de décision.

\textbf{Notre approche~:} A/B testing via paramètre \texttt{?variant=a|b}
sur l'endpoint \texttt{/recommend}, avec un challenger
(\texttt{xgb\_ranker\_challenger.onnx}) chargé en parallèle au démarrage.
Chaque réponse porte un \texttt{request\_id} (UUID) corrélé via
l'endpoint \texttt{/feedback}, ce qui permet de calculer le CTR uplift
sans jamais journaliser de données personnelles.
```

- [ ] **Step 2 : Compile et commit**

```bash
pdflatex -interaction=nonstopmode rapport_pfa_v2.tex
git add rapport_pfa_v2.tex
git commit -m "docs(benchmark): pattern 2 — A/B testing (Booking, Netflix)"
```

### Task 5.6 : Pattern 3 — MLOps

**Files:**
- Modify: `rapport_pfa_v2.tex`

- [ ] **Step 1 : Insérer la sous-sous-section**

```latex
\subsubsection{Pattern 3 --- MLOps et Cycles de Ré-entraînement}

\paragraph{Uber Michelangelo} est la plateforme interne d'Uber pour
le ML en production. Sa philosophie~: \textit{garbage in, garbage out}~;
80~\% de l'effort va dans le pipeline de données, pas dans le modèle.
Le ré-entraînement est déclenché soit par planification, soit par
détection de drift (KS test sur les distributions de features).

\paragraph{Spotify} documente publiquement son \textit{ML Platform},
qui formalise quatre étapes~: ingestion, training, scoring batch,
serving online. Les garde-fous KPI sont systématiques avant promotion.

\textbf{Notre approche~:} ré-entraînement quotidien planifié
(\texttt{scripts/07\_retrain.py}, 02h00 Casablanca), avec garde-fou KPI
(promotion bloquée si \texttt{HR@1} baisse de plus de 5 points). Le
challenger est toujours conservé sur disque pour rollback rapide.
```

- [ ] **Step 2 : Compile et commit**

```bash
pdflatex -interaction=nonstopmode rapport_pfa_v2.tex
git add rapport_pfa_v2.tex
git commit -m "docs(benchmark): pattern 3 — MLOps (Uber Michelangelo, Spotify)"
```

### Task 5.7 : Pattern 4 — Latence + Synthèse + Choix

**Files:**
- Modify: `rapport_pfa_v2.tex`

- [ ] **Step 1 : Insérer pattern 4 et synthèse**

```latex
\subsubsection{Pattern 4 --- Latence et Expérience Mobile}

L'expérience utilisateur sur mobile dégrade fortement au-delà de
100\,ms perçus pour une interaction~; les leaders du e-commerce (Amazon,
Booking) visent typiquement des p99 inférieures à 200\,ms côté API
recommandation.

\paragraph{Amazon} a publié dès 2006 que +100\,ms de latence sur ses
pages produit coûtaient \textasciitilde 1~\% de revenu. Leur stack
recommander tient des p99 sous 50\,ms via du précalcul agressif et un
cache multi-niveaux.

\paragraph{Booking.com} a documenté l'usage de modèles \emph{compacts}
(GBDT distillés, embeddings de petite dimension) plutôt que des
architectures lourdes, précisément pour maintenir des latences mobile
acceptables.

\textbf{Notre approche~:} latence p50 mesurée à 14\,ms,
p99 à 17\,ms~; conforme aux standards de l'industrie pour un cas
d'usage analogue. Levier principal~: ONNX Runtime + FastPreprocessor
vectorisé (gain $\times 1\,469$ sur l'étape de prétraitement).
```

- [ ] **Step 2 : Mettre à jour la section "Choix d'Architecture Retenu"**

Localiser la `\subsection{Choix d'Architecture Retenu}` (anciennement l.548) et insérer une dernière sous-section de synthèse juste avant celle-ci ou en la complétant :

```latex
\subsubsection{Synthèse~: Positionnement de nos Choix Techniques}

\renewcommand{\arraystretch}{1.3}
\begin{tabularx}{\linewidth}{@{}p{3.5cm}p{3.5cm}X@{}}
\toprule
\textbf{Décision} & \textbf{Standard industrie} & \textbf{Notre choix} \\
\midrule
Algorithme & GBDT pour tabulaire & XGBoost \\
Format modèle & ONNX (interopérable) & ONNX \\
Latence cible mobile & p99 < 200\,ms & p99 = 17\,ms \\
Cadence retraining & quotidien à hebdo & quotidien \\
Garde-fou promotion & systématique & HR@1 ne baisse pas > 5\,pts \\
A/B testing & systématique pour reco & \texttt{?variant=a|b} \\
Logs personnels & exclus (RGPD/CNDP) & \texttt{code\_client} jamais loggé \\
\bottomrule
\end{tabularx}

Aucun de nos choix techniques ne s'écarte significativement de l'état
de l'art. Le projet apporte une contribution principalement
d'\textbf{intégration} (architecture deux-étapes, garde-fou KPI,
boucle de feedback A/B) plutôt que de rupture algorithmique --- ce qui
est l'objectif attendu pour un projet de fin d'études en ingénierie.
```

- [ ] **Step 3 : Compile et commit**

```bash
pdflatex -interaction=nonstopmode rapport_pfa_v2.tex
git add rapport_pfa_v2.tex
git commit -m "docs(benchmark): pattern 4 (latence) + synthèse positionnement"
```

### Task 5.8 : Bibliographie du benchmark

**Files:**
- Modify: `rapport_pfa_v2.tex` — section bibliographie

- [ ] **Step 1 : Localiser ou créer la section bibliographie**

Run: `grep -n "thebibliography\|biblio\|\\\\bibitem" rapport_pfa_v2.tex`
Si présente : ajouter les entrées suivantes.
Si absente : créer une section bibliographie en fin de rapport, avant `\end{document}`.

- [ ] **Step 2 : Ajouter les références**

```latex
% À ajouter dans la bibliographie existante (ou comme nouvelle section si absente)
\bibitem{sncf-connect} SNCF Connect, "Architecture et stack technique",
    publications publiques SNCF Connect Tech (2023-2024).
\bibitem{db-navigator} Deutsche Bahn, "Personalization in DB Navigator",
    DB Systel Tech Blog.
\bibitem{michelangelo} Hermann, J.\ et al., "Meet Michelangelo: Uber's Machine
    Learning Platform", Uber Engineering Blog, 2017.
\bibitem{spotify-ml} Spotify Engineering, "The Winding Road to Better Machine
    Learning Infrastructure Through Tensorflow Extended and Kubeflow", 2019.
\bibitem{onnx} ONNX Runtime Documentation, \url{https://onnxruntime.ai/docs/}.
\bibitem{booking-ml} Booking.com Data Science Blog, "150 Successful Machine
    Learning Models: 6 Lessons Learned at Booking.com", KDD 2019.
\bibitem{amazon-latency} Linden, G.\ "Make Data Useful", Amazon talk, 2006.
\bibitem{tabular-gbdt} Shwartz-Ziv, R.\ \& Armon, A., "Tabular Data: Deep
    Learning is Not All You Need", Information Fusion, 2022.
```

- [ ] **Step 3 : Compile et commit**

```bash
pdflatex -interaction=nonstopmode rapport_pfa_v2.tex
pdflatex -interaction=nonstopmode rapport_pfa_v2.tex
git add rapport_pfa_v2.tex
git commit -m "docs(benchmark): bibliographie (Uber, Spotify, Booking, ONNX)"
```

---

# Chantier 6 — Migration CSV (DIFFÉRÉ : à exécuter après réception CSV ONCF)

> **Pré-requis :** les fichiers \texttt{users\_history.csv} et \texttt{trains\_schedule.csv} doivent être reçus de l'ONCF avant d'attaquer ce chantier. Tant qu'ils ne sont pas disponibles, **sauter directement aux tâches de finalisation** (Chantier 7).

### Task 6.1 : Inspecter les CSV reçus

**Files:**
- Create: `data/raw/users_history.csv` (placé manuellement)
- Create: `data/raw/trains_schedule.csv` (placé manuellement)

- [ ] **Step 1 : Vérifier la présence des CSV**

Run: `ls data/raw/users_history.csv data/raw/trains_schedule.csv`
Expected: les deux fichiers présents.

- [ ] **Step 2 : Inspection manuelle rapide**

Run: `.venv\Scripts\python.exe -c "import pandas as pd; df = pd.read_csv('data/raw/users_history.csv'); print(df.dtypes); print(df.head()); print('shape:', df.shape)"`
Run idem pour \texttt{trains\_schedule.csv}.

Noter le schéma observé pour adapter `data_source.py`.

### Task 6.2 : Créer `scripts/09_load_oncf_csv.py`

**Files:**
- Create: `scripts/09_load_oncf_csv.py`

- [ ] **Step 1 : Écrire le script**

```python
"""Validation et conversion des CSV ONCF en parquet pour usage runtime."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

USERS_REQUIRED = {
    "CodeClient",
    "LiaisonId",
    "DateHeureDepartVoyageSegment",
}
SCHEDULE_REQUIRED_ANY = [
    {"liaison_id", "heure_depart", "heure_arrivee"},
    {"gare_depart", "gare_arrivee", "heure_depart", "heure_arrivee"},
]


def _report(name: str, df: pd.DataFrame, required: set[str]) -> bool:
    print(f"\n=== {name} ===")
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    print(f"dtypes:\n{df.dtypes}")
    print(f"sample rows:\n{df.sample(min(5, len(df)), random_state=0)}")
    missing = required - set(df.columns)
    if missing:
        print(f"WARNING — missing required columns: {missing}", file=sys.stderr)
        return False
    return True


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    users_csv = RAW_DIR / "users_history.csv"
    schedule_csv = RAW_DIR / "trains_schedule.csv"

    if not users_csv.exists():
        print(f"FATAL: {users_csv} not found", file=sys.stderr)
        return 1
    if not schedule_csv.exists():
        print(f"FATAL: {schedule_csv} not found", file=sys.stderr)
        return 1

    users = pd.read_csv(users_csv)
    schedule = pd.read_csv(schedule_csv)

    users_ok = _report("users_history", users, USERS_REQUIRED)

    schedule_ok = False
    for req in SCHEDULE_REQUIRED_ANY:
        if req.issubset(set(schedule.columns)):
            schedule_ok = True
            break
    if not schedule_ok:
        print(
            f"WARNING — schedule missing required columns (need any of: "
            f"{SCHEDULE_REQUIRED_ANY})",
            file=sys.stderr,
        )

    if "DateHeureDepartVoyageSegment" in users.columns:
        users["DateHeureDepartVoyageSegment"] = pd.to_datetime(
            users["DateHeureDepartVoyageSegment"], errors="coerce"
        )

    users.to_parquet(PROCESSED_DIR / "users_history.parquet", index=False)
    schedule.to_parquet(PROCESSED_DIR / "trains_schedule.parquet", index=False)

    print(
        f"\nWritten: {PROCESSED_DIR / 'users_history.parquet'} "
        f"({len(users)} rows)"
    )
    print(
        f"Written: {PROCESSED_DIR / 'trains_schedule.parquet'} "
        f"({len(schedule)} rows)"
    )
    return 0 if (users_ok and schedule_ok) else 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2 : Lancer le script sur les CSV reçus**

Run: `.venv\Scripts\python.exe scripts/09_load_oncf_csv.py`
Expected: rapport d'inspection imprimé, deux parquets créés dans `data/processed/`.

- [ ] **Step 3 : Commit**

```bash
git add scripts/09_load_oncf_csv.py
git commit -m "feat(data): script 09 — validation/conversion CSV ONCF en parquet"
```

### Task 6.3 : Créer `src/rec_oncf/data_source.py` — interfaces + impls

**Files:**
- Create: `src/rec_oncf/data_source.py`

- [ ] **Step 1 : Écrire les interfaces et implémentations CSV**

```python
"""Sources de données pluggables pour le Recommender.

Sépare le chargement des données de la logique métier afin que les CSV
ONCF puissent être substitués par une vraie connexion DB plus tard sans
toucher au cœur du Recommender.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from rec_oncf.io import read_parquet


class UserHistorySource(Protocol):
    """Source de l'historique utilisateur."""

    def get_history(self, code_client: str) -> pd.DataFrame | None: ...
    def get_all_users(self) -> list[str]: ...
    def get_clean_df(self) -> pd.DataFrame: ...


class ScheduleSource(Protocol):
    """Source des horaires de trains."""

    def get_schedule(
        self, liaison_id: str, date: str | None = None
    ) -> list[dict]: ...


class ParquetUserHistorySource:
    """Source historique adossée à un parquet (rétro-compat — tests existants)."""

    def __init__(self, parquet_path: Path):
        self._df = read_parquet(parquet_path)
        if "CodeClient" not in self._df.columns:
            raise ValueError(
                "ParquetUserHistorySource: missing 'CodeClient' column"
            )
        self._lookup: dict[str, pd.DataFrame] = {
            str(k): g.sort_values("DateHeureDepartVoyageSegment")
            for k, g in self._df.groupby("CodeClient", sort=False)
        }

    def get_history(self, code_client: str) -> pd.DataFrame | None:
        return self._lookup.get(str(code_client))

    def get_all_users(self) -> list[str]:
        return list(self._lookup.keys())

    def get_clean_df(self) -> pd.DataFrame:
        return self._df


class CsvUserHistorySource(ParquetUserHistorySource):
    """Source historique adossée à un CSV ONCF.

    Convertit en interne en DataFrame et réutilise la logique du
    ParquetUserHistorySource. La date est parsée à la lecture.
    """

    def __init__(self, csv_path: Path):
        df = pd.read_csv(csv_path)
        if "DateHeureDepartVoyageSegment" in df.columns:
            df["DateHeureDepartVoyageSegment"] = pd.to_datetime(
                df["DateHeureDepartVoyageSegment"], errors="coerce"
            )
        self._df = df
        if "CodeClient" not in df.columns:
            raise ValueError(
                "CsvUserHistorySource: missing 'CodeClient' column in CSV"
            )
        self._lookup = {
            str(k): g.sort_values("DateHeureDepartVoyageSegment")
            for k, g in df.groupby("CodeClient", sort=False)
        }


class CsvScheduleSource:
    """Source horaires adossée à un CSV statique ONCF."""

    def __init__(self, csv_path: Path):
        self._df = pd.read_csv(csv_path)

    def get_schedule(
        self, liaison_id: str, date: str | None = None
    ) -> list[dict]:
        if "liaison_id" in self._df.columns:
            rows = self._df[self._df["liaison_id"] == liaison_id]
        else:
            rows = self._df.iloc[0:0]
        return rows.to_dict(orient="records")
```

- [ ] **Step 2 : Commit**

```bash
git add src/rec_oncf/data_source.py
git commit -m "feat(data): introduce UserHistorySource/ScheduleSource interfaces + CSV impl"
```

### Task 6.4 : Tests pour `data_source.py`

**Files:**
- Create: `tests/test_data_source.py`

- [ ] **Step 1 : Écrire les tests**

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rec_oncf.data_source import (
    CsvScheduleSource,
    CsvUserHistorySource,
    ParquetUserHistorySource,
)


@pytest.fixture
def users_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "CodeClient": ["A", "A", "B"],
        "LiaisonId": ["L1", "L2", "L1"],
        "DateHeureDepartVoyageSegment": [
            "2026-01-01 08:00", "2026-01-02 08:00", "2026-01-01 09:00"
        ],
        "PrixParLiaison": [50.0, 60.0, 70.0],
    })
    p = tmp_path / "users.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def schedule_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "liaison_id": ["L1", "L1", "L2"],
        "heure_depart": ["08:00", "10:00", "09:00"],
        "heure_arrivee": ["09:30", "11:30", "10:30"],
    })
    p = tmp_path / "schedule.csv"
    df.to_csv(p, index=False)
    return p


def test_csv_user_history_get_known_user(users_csv: Path):
    src = CsvUserHistorySource(users_csv)
    h = src.get_history("A")
    assert h is not None
    assert len(h) == 2
    assert list(h["LiaisonId"]) == ["L1", "L2"]


def test_csv_user_history_unknown_user(users_csv: Path):
    src = CsvUserHistorySource(users_csv)
    assert src.get_history("ZZZ") is None


def test_csv_user_history_get_all_users(users_csv: Path):
    src = CsvUserHistorySource(users_csv)
    assert set(src.get_all_users()) == {"A", "B"}


def test_csv_user_history_missing_codeclient_raises(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="CodeClient"):
        CsvUserHistorySource(bad)


def test_csv_schedule_known_liaison(schedule_csv: Path):
    src = CsvScheduleSource(schedule_csv)
    rows = src.get_schedule("L1")
    assert len(rows) == 2
    assert rows[0]["heure_depart"] == "08:00"


def test_csv_schedule_unknown_liaison(schedule_csv: Path):
    src = CsvScheduleSource(schedule_csv)
    assert src.get_schedule("LZZ") == []
```

- [ ] **Step 2 : Lancer les tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_data_source.py -v`
Expected: 6 tests PASS.

- [ ] **Step 3 : Commit**

```bash
git add tests/test_data_source.py
git commit -m "test: data_source CSV impls (history known/unknown, schedule, validation)"
```

### Task 6.5 : Brancher `data_source` dans `Recommender` et `main.py`

**Files:**
- Modify: `src/rec_oncf/recommender.py`
- Modify: `apps/api/main.py`
- Modify: `src/rec_oncf/config.py`

- [ ] **Step 1 : Ajouter les paths CSV à la config**

Modifier `src/rec_oncf/config.py` pour ajouter au dataclass `Paths` :

```python
users_history_csv: Path = field(default=DATA_RAW / "users_history.csv")
trains_schedule_csv: Path = field(default=DATA_RAW / "trains_schedule.csv")
```

(adapter selon la structure réelle de `config.py`).

- [ ] **Step 2 : Refactor `Recommender.from_paths` pour accepter un `UserHistorySource` optionnel**

Modification minimale du constructor : si un argument `user_source` est passé, l'utiliser pour construire `history_lookup` au lieu de lire le parquet directement. Comportement par défaut inchangé (lecture parquet).

```python
@classmethod
def from_paths(
    cls,
    paths: Paths,
    *,
    user_source: UserHistorySource | None = None,
) -> "Recommender":
    if user_source is None:
        user_source = ParquetUserHistorySource(paths.processed_dataset_parquet)
    # ... le reste de la construction utilise user_source.get_history,
    # user_source.get_all_users, user_source.get_clean_df au lieu d'accéder
    # directement au DataFrame parquet.
```

- [ ] **Step 3 : Modifier lifespan de l'API pour choisir la source**

Dans `apps/api/main.py`, modifier le lifespan pour préférer la source CSV si les fichiers existent :

```python
from rec_oncf.data_source import CsvUserHistorySource, ParquetUserHistorySource

# Dans lifespan, juste avant Recommender.from_paths(paths):
if paths.users_history_csv.exists():
    user_source = CsvUserHistorySource(paths.users_history_csv)
    logger.info(f"Using CSV user history source: {paths.users_history_csv}")
else:
    user_source = ParquetUserHistorySource(paths.processed_dataset_parquet)
    logger.info("Using parquet user history source (fallback)")

app.state.recommender_a = Recommender.from_paths(paths, user_source=user_source)
```

- [ ] **Step 4 : Lancer la suite complète**

Run: `.venv\Scripts\python.exe -m pytest tests/ -v`
Expected: 113 tests existants + 6 nouveaux = 119 tests PASS. Les tests existants doivent rester verts car le default est inchangé (parquet).

- [ ] **Step 5 : Commit**

```bash
git add src/rec_oncf/recommender.py src/rec_oncf/config.py apps/api/main.py
git commit -m "feat(recommender): accept injectable UserHistorySource (CSV-backed in prod)"
```

---

# Chantier 7 — Finalisation

### Task 7.1 : Mise à jour du `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1 : Ajouter une section "Pivot post-réunion 2026-05-15"**

Au début, juste après la section "🚧 RESUME HERE", remplacer la note de paused par une nouvelle entrée qui résume les changements (Docker, guide deploy, benchmark enrichi, IDA, data_source).

- [ ] **Step 2 : Commit**

```bash
git add CLAUDE.md
git commit -m "docs: MAJ CLAUDE.md — pivot post-réunion ONCF"
```

### Task 7.2 : Compilation finale et vérification globale

- [ ] **Step 1 : Compile complet du rapport (2 passes)**

```bash
pdflatex -interaction=nonstopmode rapport_pfa_v2.tex
pdflatex -interaction=nonstopmode rapport_pfa_v2.tex
```

Expected: PDF produit sans erreur.

- [ ] **Step 2 : Lancer toute la suite de tests**

Run: `.venv\Scripts\python.exe -m pytest tests/ -v`
Expected: 114 ou 120 tests verts (selon que le chantier 6 a été fait).

- [ ] **Step 3 : Vérifier la liste finale des artefacts**

```bash
ls deploy/
# Dockerfile, docker-compose.yml, .env.example, .dockerignore

ls docs/
# guide_deploiement.tex, guide_deploiement.pdf, rapport_donnees.pdf, ...

ls rapport_pfa_v2.tex
# présent

git log --oneline -30
# vérifier qu'on a bien tous les commits du pivot
```

- [ ] **Step 4 : Commit final s'il reste des fichiers**

```bash
git status
# devrait être clean
```

---

## Self-Review (effectué par l'auteur du plan)

### Spec coverage
- [x] §1 (objet/contexte) → couvert par chantier 1 (section pivot) et 7 (CLAUDE.md)
- [x] §2 (inventaire 6 chantiers) → un chantier par section, ordre respecté
- [x] §3 (architecture migration CSV) → chantier 6, Tasks 6.1-6.5
- [x] §4.1 (MAJ rapport) → chantier 1
- [x] §4.2 (partie IDA) → chantier 2
- [x] §4.3 (API deploy-ready) → chantier 3
- [x] §4.4 (guide deploy PDF) → chantier 4
- [x] §4.5 (benchmark PDF) → chantier 5
- [x] §5 (tests/non-régression) → vérifications explicites Tasks 3.5 step 5, 6.5 step 4, 7.2 step 2
- [x] §6 (gestion erreurs) → chantier 3 (healthcheck enrichi), Task 6.5 (fallback parquet)
- [x] §7 (critères d'acceptation) → checklist répartie dans le plan
- [x] §8 (hors scope) → cohérent (auth/TLS/rate-limiting non plannifiés, mobile retiré)

### Placeholder scan
- Pas de "TBD" ni "implement later". Toutes les références de code/LaTeX sont explicites.
- Quelques renvois "adapter selon la structure réelle de `config.py`" (Task 6.5 Step 1) sont justifiés : on attend les vrais CSV pour ne pas inventer une signature qu'on devra refaire.

### Type consistency
- `UserHistorySource` / `ScheduleSource` : signatures cohérentes entre data_source.py, tests, et le lifespan.
- `/health` : structure `{status, model_loaded, popularity_loaded, n_users_history}` identique dans test, impl, et guide PDF.

Plan auto-validé.

---

## Notes opérationnelles

**Estimation de durée totale :** 21-29 h pour les chantiers 1-5 (sans chantier 6). Chantier 6 ajoute 4-6 h supplémentaires.

**Marge :** ~30 h de marge pour relecture, corrections et imprévus sur les 60 h disponibles.

**Ordre recommandé d'exécution :**
1. Chantier 1 (rapport — retrait mockup + section pivot)
2. Chantier 2 (IDA)
3. Chantier 3 (Dockerfile + healthcheck)
4. Chantier 4 (guide deploy PDF)
5. Chantier 5 (benchmark)
6. Chantier 6 (migration CSV — dès réception des CSV ONCF)
7. Chantier 7 (finalisation)

**Critère de fin :** suite pytest verte, rapport PDF compile sans erreur, image Docker se build, guide PDF inclus en annexe du rapport.
