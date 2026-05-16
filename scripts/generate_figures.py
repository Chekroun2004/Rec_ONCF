"""
generate_figures.py - genere toutes les figures du rapport PFA ONCF.
Usage : .venv/Scripts/python.exe scripts/generate_figures.py
Output: pic/<nom_figure>.png
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyPatch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, FancyArrow
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

ROOT = Path(__file__).parent.parent
PIC  = ROOT / "pic"
PIC.mkdir(exist_ok=True)

# ── Palette ONCF ──────────────────────────────────────────────────────────────
ORANGE = "#E65300"
BLUE   = "#004B8D"
LGRAY  = "#F8F8F8"
MGRAY  = "#CCCCCC"
GREEN  = "#2E7D32"
RED    = "#C62828"
TEAL   = "#00695C"

plt.rcParams.update({
    "font.family"   : "DejaVu Sans",
    "font.size"     : 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi"    : 150,
    "savefig.dpi"   : 200,
    "savefig.bbox"  : "tight",
    "savefig.facecolor": "white",
})

def save(name):
    path = PIC / name
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  OK  {name}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. ZERO-CLICK SEARCH — Schéma conceptuel
# ══════════════════════════════════════════════════════════════════════════════
def fig_zero_click():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")

    def box(x, y, w, h, color, text, fontsize=9, text_color="white"):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor="white", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold",
                wrap=True, multialignment="center")

    def arrow(x1, x2, y=2.0):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE,
                                   lw=2, mutation_scale=20))

    # WITHOUT system (top)
    boxes_before = [
        (0.2, 2.5, 1.8, 1.0, MGRAY, "Utilisateur\nouvre l'app", 9, "#333"),
        (2.3, 2.5, 1.8, 1.0, MGRAY, "Saisit gare\ndépart", 9, "#333"),
        (4.4, 2.5, 1.8, 1.0, MGRAY, "Saisit gare\narrivée", 9, "#333"),
        (6.5, 2.5, 1.8, 1.0, MGRAY, "Sélectionne\ndate", 9, "#333"),
        (8.6, 2.5, 1.8, 1.0, MGRAY, "Résultats\n(5 étapes)", 9, "#333"),
    ]
    for b in boxes_before:
        box(*b)
    for x in [2.1, 4.2, 6.3, 8.4]:
        arrow(x, x + 0.2, y=3.0)
    ax.text(5.5, 3.8, "SANS Zero-Click Search (friction de recherche)", ha="center",
            fontsize=10, color=RED, fontweight="bold")
    ax.annotate("", xy=(10.7, 3.0), xytext=(10.4, 3.0),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2, mutation_scale=20))

    # WITH system (bottom)
    boxes_after = [
        (0.2, 0.4, 2.2, 1.0, BLUE,   "Utilisateur\nouvre l'app", 9, "white"),
        (3.0, 0.4, 3.5, 1.0, ORANGE, "Système prédit\nautomatiquement\nle trajet", 9, "white"),
        (7.5, 0.4, 2.5, 1.0, GREEN,  "Suggestion\nimmédiate ✓", 9, "white"),
        (10.2, 0.4, 1.5, 1.0, TEAL,  "1 clic !", 10, "white"),
    ]
    for b in boxes_after:
        box(*b)
    for x1, x2 in [(2.4, 3.0), (6.5, 7.5), (10.0, 10.2)]:
        arrow(x1, x2, y=0.9)
    ax.text(5.5, 0.0, "AVEC Zero-Click Search (zéro friction)", ha="center",
            fontsize=10, color=GREEN, fontweight="bold")

    ax.text(5.5, 4.15, "Zero-Click Search — Principe de recommandation proactive",
            ha="center", fontsize=12, fontweight="bold", color=BLUE)

    save("zero_click_concept.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. GANTT — Planning du projet
# ══════════════════════════════════════════════════════════════════════════════
def fig_gantt():
    phases = [
        ("Étude & Benchmark",           1, 2,  MGRAY),
        ("Sprint 1 : Données & Modèle", 2, 5,  BLUE),
        ("Sprint 2 : Architecture & API",5, 7,  ORANGE),
        ("Phase 3 : Production",         7, 8,  TEAL),
        ("Consolidation & Rapport",      8, 9,  GREEN),
    ]
    livrables = [
        (2,  "Cahier des charges"),
        (5,  "Modèle v1 + métriques"),
        (7,  "API REST + tests"),
        (8,  "ONNX + A/B + retrain"),
        (9,  "Rapport final\n115 tests"),
    ]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(-0.5, len(phases))
    ax.set_xlabel("Semaine du stage", fontsize=10)
    ax.set_xticks(range(1, 10))
    ax.set_xticklabels([f"S{i}" for i in range(1, 10)])
    ax.set_yticks(range(len(phases)))
    ax.set_yticklabels([p[0] for p in phases], fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_title("Planning du projet — Phases et sprints", fontweight="bold",
                 color=BLUE, fontsize=12)

    for i, (label, start, end, color) in enumerate(phases):
        ax.barh(i, end - start, left=start, height=0.5, color=color,
                edgecolor="white", linewidth=1)

    for week, text in livrables:
        ax.axvline(week, color=ORANGE, linestyle=":", linewidth=1, alpha=0.7)
        ax.text(week + 0.05, len(phases) - 0.2, text, fontsize=7,
                color=ORANGE, va="top", rotation=0)

    save("gantt.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. ARCHITECTURE GLOBALE — 4 couches
# ══════════════════════════════════════════════════════════════════════════════
def fig_archi_globale():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11); ax.set_ylim(0, 8); ax.axis("off")

    layers = [
        (0.5, 6.5, 10, 1.2, BLUE,   "4 — Couche API",
         ["FastAPI  /recommend", "FastAPI  /feedback", "FastAPI  /health", "Demo Web (/)"], ORANGE),
        (0.5, 4.8, 10, 1.4, TEAL,   "3 — Couche Logique (rec_oncf)",
         ["Recommender", "CandidateGen", "Features on-the-fly", "ColdStart CF", "Schedule scraping"], LGRAY),
        (0.5, 3.1, 10, 1.4, "#5C6BC0", "2 — Couche Modèle (models/)",
         ["xgb_ranker.onnx  148 Mo", "label_encoder.joblib", "cold_start.joblib", "popularity.joblib"], LGRAY),
        (0.5, 1.2, 10, 1.6, "#546E7A", "1 — Couche Données (data/processed/)",
         ["oncf_clean.parquet  491 680 lignes", "features.parquet  26 colonnes", "(Statique entre deux réentraînements)"], LGRAY),
    ]

    for x, y, w, h, color, title, items, ic in layers:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                              facecolor=color, edgecolor="white", linewidth=2, alpha=0.92)
        ax.add_patch(rect)
        ax.text(x + 0.3, y + h - 0.25, title, fontsize=10.5,
                fontweight="bold", color="white", va="top")
        item_text = "   |   ".join(items)
        ax.text(x + 0.3, y + 0.25, item_text, fontsize=8.2, color=ic, va="bottom")

    # arrows between layers
    for y_from, y_to in [(4.8, 5.7), (3.1+1.4, 4.8), (3.1, 2.5), (1.2+1.6, 3.1)]:
        ax.annotate("", xy=(5.5, y_to), xytext=(5.5, y_from - 0.01),
                    arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.5,
                                   mutation_scale=14))

    # External actors
    ax.text(0.2, 7.8, "📱 App Mobile ONCF", fontsize=9, color=BLUE)
    ax.annotate("", xy=(5.5, 7.7), xytext=(5.5, 7.65),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))

    ax.set_title("Architecture Globale du Système de Recommandation ONCF",
                 fontsize=12, fontweight="bold", color=BLUE, pad=10)
    save("archi_globale.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. UML COMPOSANTS
# ══════════════════════════════════════════════════════════════════════════════
def fig_uml_composants():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 9); ax.axis("off")
    ax.set_title("Diagramme de Composants UML", fontsize=12,
                 fontweight="bold", color=BLUE)

    def comp(x, y, w, h, title, items=None, color=BLUE):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              facecolor=LGRAY, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        # Title bar
        title_rect = FancyBboxPatch((x, y+h-0.4), w, 0.4,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor=color)
        ax.add_patch(title_rect)
        ax.text(x + w/2, y + h - 0.2, f"«component»\n{title}",
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white")
        if items:
            for i, item in enumerate(items):
                ax.text(x + 0.2, y + h - 0.65 - i*0.35, f"• {item}",
                        fontsize=7.5, va="top", color="#333")

    def interface(x, y, label):
        circle = plt.Circle((x, y), 0.13, color=ORANGE, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y-0.28, label, ha="center", fontsize=7, color=ORANGE)

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.3))

    # Components
    comp(0.3, 6.5, 3.5, 2.2, "apps.api.main",
         ["FastAPI app", "lifespan startup", "/recommend, /feedback, /health", "Logging structuré"], BLUE)
    comp(4.5, 6.5, 3.5, 2.2, "rec_oncf.recommender",
         ["Recommender.from_paths()", "recommend(code_client, k)", "history_lookup [dict]",
          "_fallback(k)"], TEAL)
    comp(8.5, 6.5, 3.7, 2.2, "rec_oncf.training",
         ["train_xgb_multiclass()", "FastPreprocessor", "predict_proba_onnx()", "TrainArtifacts"], "#5C6BC0")

    comp(0.3, 3.5, 3.5, 2.2, "rec_oncf.candidates",
         ["generate_candidates(history)", "Règles heuristiques", "Top fréquence + récence"], ORANGE)
    comp(4.5, 3.5, 3.5, 2.2, "rec_oncf.features",
         ["build_training_rows()", "compute_inference_row()", "Encodage cyclique",
          "user_top_liaison_share"], GREEN)
    comp(8.5, 3.5, 3.7, 2.2, "rec_oncf.cold_start",
         ["ColdStartRecommender", "Matrice co-occurrence", "recommend(history, k)"], RED)

    comp(1.5, 0.5, 2.5, 2.2, "models/ (artefacts)",
         ["xgb_ranker.onnx", "label_encoder.joblib", "cold_start.joblib",
          "popularity.joblib"], "#546E7A")
    comp(5.0, 0.5, 2.5, 2.2, "data/processed/",
         ["oncf_clean.parquet", "features.parquet"], "#546E7A")
    comp(8.5, 0.5, 3.7, 2.2, "rec_oncf.schedule",
         ["STATION_CODES (24 gares)", "fetch_departures()", "Cache Redis/mémoire TTL 1h"], "#795548")

    # Arrows
    arr(3.8, 7.6, 4.5, 7.6)
    arr(8.0, 7.6, 8.5, 7.6)
    arr(4.5, 7.0, 3.8, 6.5)   # recommender → candidates
    arr(4.5+1.75, 6.5, 4.5+1.75, 3.5+2.2)  # recommender → features
    arr(8.5+1.85, 6.5, 8.5+1.85, 3.5+2.2)  # training → cold_start
    arr(4.5+1.75, 3.5, 2.75, 0.5+2.2)  # features → models
    arr(5.0+1.25, 3.5, 5.0+1.25, 0.5+2.2)
    arr(8.5+1.85, 3.5, 8.5+1.85, 0.5+2.2)  # cold_start → schedule

    # External actor
    ax.text(6.0, 8.9, "📱 Application Mobile ONCF", ha="center",
            fontsize=9.5, color=BLUE, fontweight="bold")
    arr(6.0, 8.7, 2.0, 8.7)
    ax.annotate("", xy=(2.0, 8.7), xytext=(2.0, 6.5+2.2),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.5))

    save("uml_composants.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. ARCHITECTURE DEUX ÉTAPES
# ══════════════════════════════════════════════════════════════════════════════
def fig_archi_deux_etapes():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("Architecture Deux Étapes : Candidate Generation + Ranking",
                 fontsize=12, fontweight="bold", color=BLUE)

    def block(x, y, w, h, title, body, color):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                           facecolor=color, edgecolor="white", linewidth=2, alpha=0.92)
        ax.add_patch(r)
        ax.text(x+w/2, y+h-0.3, title, ha="center", va="top",
                fontsize=9.5, fontweight="bold", color="white")
        ax.text(x+w/2, y+0.2, body, ha="center", va="bottom",
                fontsize=8, color="white", multialignment="center")

    def arr(x1, x2, y=2.5):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE,
                                   lw=2.5, mutation_scale=18))

    block(0.2, 1.5, 2.2, 2.0, "Historique", "CodeClient\n→ history_lookup\n(DataFrame)", "#546E7A")
    arr(2.4, 3.0)
    block(3.0, 1.5, 3.0, 2.0, "Étape 1\nCandidate Generation",
          "Heuristiques :\n• 3 derniers voyages\n• Top fréquence\n• Co-occurrence CF\n≤ 10 candidats",
          TEAL)
    arr(6.0, 6.8)
    block(6.8, 1.5, 3.0, 2.0, "Étape 2\nRanking XGBoost",
          "• Features on-the-fly (26)\n• FastPreprocessor\n• ONNX Runtime\n• Filtre candidats",
          BLUE)
    arr(9.8, 10.6)
    block(10.6, 1.5, 2.2, 2.0, "Top-k\nRéponse",
          "mode: \"model\"\nk ∈ {1, 2, 3}\n< 14 ms (p50)", GREEN)

    # Metrics labels
    ax.text(4.5, 1.3, "< 3 ms", ha="center", fontsize=8, color=TEAL, style="italic")
    ax.text(8.3, 1.3, "~11 ms", ha="center", fontsize=8, color=BLUE, style="italic")

    # cold start bypass
    ax.text(1.3, 1.3, "< 3 voyages ?", ha="center", fontsize=7.5, color=RED)
    ax.annotate("", xy=(11.7, 1.5), xytext=(1.3, 1.3),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2,
                                connectionstyle="arc3,rad=-0.4",
                                mutation_scale=12))
    ax.text(6.5, 0.4, "Cold Start CF / Popularité (bypass)", ha="center",
            fontsize=7.5, color=RED, style="italic")

    save("archi_deux_etapes.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. UML USE CASE
# ══════════════════════════════════════════════════════════════════════════════
def fig_uml_usecase():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")
    ax.set_title("Diagramme de Cas d'Utilisation — Système Zero-Click Search",
                 fontsize=12, fontweight="bold", color=BLUE)

    # System boundary
    rect = FancyBboxPatch((2.0, 0.5), 8.5, 8.0, boxstyle="square,pad=0.1",
                          facecolor="#F0F4FF", edgecolor=BLUE, linewidth=2)
    ax.add_patch(rect)
    ax.text(6.25, 8.35, "Système de Recommandation ONCF", ha="center",
            fontsize=10, fontweight="bold", color=BLUE)

    def actor(x, y, name, color=BLUE):
        # Stick figure
        ax.plot([x, x], [y+0.5, y+0.9], color=color, lw=2)
        ax.plot([x-0.3, x+0.3], [y+0.7, y+0.7], color=color, lw=2)
        ax.plot([x, x-0.25], [y+0.5, y+0.2], color=color, lw=2)
        ax.plot([x, x+0.25], [y+0.5, y+0.2], color=color, lw=2)
        circle = plt.Circle((x, y+1.0), 0.18, color=color, fill=False, lw=2)
        ax.add_patch(circle)
        ax.text(x, y-0.05, name, ha="center", fontsize=8,
                fontweight="bold", color=color)

    def usecase(x, y, w, h, text, color=BLUE):
        e = mpatches.Ellipse((x, y), w, h, facecolor="white",
                             edgecolor=color, linewidth=1.5)
        ax.add_patch(e)
        ax.text(x, y, text, ha="center", va="center", fontsize=7.5,
                multialignment="center", color="#222")

    def assoc(ax1, ay1, ux, uy):
        ax.plot([ax1, ux], [ay1, uy], color="#777", lw=1.2)

    # Actors
    actor(0.5, 5.5, "Utilisateur\nmobile", BLUE)
    actor(0.5, 2.5, "Application\nmobile", TEAL)
    actor(11.2, 5.5, "Système\nRéentraînement", GREEN)
    actor(11.2, 2.5, "Data Scientist\n/ Ops", ORANGE)

    # Use cases
    usecase(5.5, 7.5, 3.5, 0.8, "Consulter recommandation\nà l'ouverture")
    usecase(5.5, 6.2, 3.2, 0.8, "Cliquer sur une\nliaison suggérée")
    usecase(5.5, 4.9, 3.0, 0.8, "Effectuer une\nréservation")
    usecase(5.5, 3.6, 3.2, 0.8, "Obtenir horaires\n(include_schedule)")
    usecase(5.5, 2.3, 3.2, 0.8, "Envoyer feedback\n(POST /feedback)")
    usecase(5.5, 1.0, 3.2, 0.8, "Vérifier l'état\n(GET /health)")

    usecase(8.8, 7.5, 2.8, 0.7, "Réentraîner\nle modèle")
    usecase(8.8, 6.3, 2.8, 0.7, "Évaluer avec\nguardrail KPI")
    usecase(8.8, 5.1, 2.8, 0.7, "Promouvoir\nle challenger")
    usecase(8.8, 3.0, 2.8, 0.7, "Consulter\nles métriques CTR")
    usecase(8.8, 1.8, 2.8, 0.7, "Configurer le\nguardrail")

    # Associations
    for uy in [7.5, 6.2, 4.9, 3.6]:
        assoc(0.9, 6.2, 3.85, uy)
    for uy in [7.5, 6.2, 4.9, 3.6, 2.3, 1.0]:
        assoc(0.9, 3.2, 3.85, uy)
    for uy in [7.5, 6.3, 5.1]:
        assoc(11.0, 6.2, 10.2, uy)
    for uy in [3.0, 1.8]:
        assoc(11.0, 3.2, 10.2, uy)

    save("uml_usecase.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. DIAGRAMME DE SÉQUENCE — POST /recommend
# ══════════════════════════════════════════════════════════════════════════════
def fig_sequence():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis("off")
    ax.set_title("Diagramme de Séquence — POST /recommend",
                 fontsize=12, fontweight="bold", color=BLUE)

    participants = [
        (1.0,  "App Mobile",       BLUE),
        (3.5,  "FastAPI\n/recommend", ORANGE),
        (6.0,  "Recommender",      TEAL),
        (8.5,  "CandidateGen\n& Features", GREEN),
        (11.0, "ONNX Runtime",     "#5C6BC0"),
    ]

    TOP = 8.5
    for x, name, color in participants:
        rect = FancyBboxPatch((x-0.6, TOP-0.25), 1.2, 0.5,
                              boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor="white", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, TOP, name, ha="center", va="center", fontsize=8,
                fontweight="bold", color="white", multialignment="center")
        ax.plot([x, x], [0.3, TOP-0.25], color=color, lw=1,
                linestyle="--", alpha=0.5)

    msgs = [
        (1.0, 3.5, 8.0, "POST /recommend {code_client, k, variant}",    BLUE),
        (3.5, 6.0, 7.4, "history_lookup.get(code_client)",               ORANGE),
        (6.0, 3.5, 7.0, "[None] → _fallback(k) → 'popularity'",          RED),
        (6.0, 8.5, 6.5, "[<3 voyages] → cold_start_rec.recommend()",     TEAL),
        (8.5, 6.0, 6.5, "candidates : List[str]",                        GREEN),
        (6.0, 8.5, 6.0, "generate_candidates(history)",                   TEAL),
        (8.5, 6.0, 5.6, "feat_row (26 cols)",                            GREEN),
        (6.0, 8.5, 5.2, "compute_inference_row(history)",                 TEAL),
        (6.0, 11.0,4.8, "fast_preprocessor.encode(row_dict)",             TEAL),
        (11.0, 6.0,4.4, "X float32 (1 × 23)",                           "#5C6BC0"),
        (6.0, 11.0,4.0, "session.run(['probabilities'], X)",              TEAL),
        (11.0, 6.0,3.6, "proba[1011]",                                   "#5C6BC0"),
        (6.0, 8.5, 3.2, "argsort(-proba[cand_idx])[:k]",                 TEAL),
        (8.5, 6.0, 2.8, "top-k candidates",                             GREEN),
        (6.0, 3.5, 2.4, "{mode, recs, labels, variant, request_id}",    TEAL),
        (3.5, 1.0, 2.0, "HTTP 200 JSON",                                 ORANGE),
    ]

    for i, (x1, x2, y, label, color) in enumerate(msgs):
        dx = x2 - x1
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=1.3, mutation_scale=12))
        mx = (x1 + x2) / 2
        ax.text(mx, y + 0.1, label, ha="center", va="bottom",
                fontsize=7, color=color)

    # Lifeline boxes
    for x, _, color in participants:
        for y in [4.8, 6.5]:
            rect = FancyBboxPatch((x-0.2, y-0.15), 0.4, 1.2,
                                  boxstyle="square,pad=0.02",
                                  facecolor=color, edgecolor="white",
                                  linewidth=1, alpha=0.3)
            ax.add_patch(rect)

    ax.text(0.1, 0.7, "t ↑", fontsize=9, color="#888", style="italic")
    ax.text(7.0, 1.5, "⏱ ~14 ms (p50)", ha="center", fontsize=9,
            color=GREEN, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor=LGRAY, edgecolor=GREEN))

    save("uml_sequence.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. DIAGRAMME DE CLASSES
# ══════════════════════════════════════════════════════════════════════════════
def fig_classes():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis("off")
    ax.set_title("Diagramme de Classes Simplifié", fontsize=12,
                 fontweight="bold", color=BLUE)

    def uml_class(x, y, w, title, attributes, methods, color=BLUE):
        h_title = 0.5
        h_attrs = len(attributes) * 0.32 + 0.1
        h_mths  = len(methods)    * 0.32 + 0.1
        h = h_title + h_attrs + h_mths

        rect = FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0",
                              facecolor="white", edgecolor=color, linewidth=2)
        ax.add_patch(rect)

        # Title
        r_title = FancyBboxPatch((x, y+h-h_title), w, h_title,
                                 boxstyle="square,pad=0",
                                 facecolor=color, edgecolor=color)
        ax.add_patch(r_title)
        ax.text(x+w/2, y+h-h_title/2, title, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")

        # Separator
        ax.plot([x, x+w], [y+h_mths, y+h_mths], color=color, lw=1)

        # Attributes
        for i, attr in enumerate(attributes):
            ax.text(x+0.1, y+h_mths + h_attrs - 0.1 - i*0.32, f"  {attr}",
                    fontsize=7.5, va="top", color="#333", family="monospace")

        # Methods
        for i, mth in enumerate(methods):
            ax.text(x+0.1, y+h_mths - 0.1 - i*0.32, f"  {mth}",
                    fontsize=7.5, va="top", color="#222", family="monospace")

        return x+w/2, y+h, x+w/2, y  # cx, top_y, cx, bot_y

    def assoc(x1, y1, x2, y2, label="", style="->"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color="#555", lw=1.3))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.05, my+0.05, label, fontsize=7, color="#555", style="italic")

    cx1, ty1, _, by1 = uml_class(0.2, 4.5, 3.8, "Recommender",
        ["artifacts : TrainArtifacts",
         "history_lookup : dict[str, DataFrame]",
         "cold_start_rec : ColdStartRecommender",
         "onnx_session : InferenceSession | None",
         "fast_preprocessor : FastPreprocessor",
         "popularity : list[str]",
         "liaison_label_lookup : dict"],
        ["+ from_paths(paths) : Recommender",
         "+ from_data(arts, df) : Recommender",
         "+ recommend(code_client, k) : dict",
         "- _fallback(k) : dict"],
        BLUE)

    cx2, ty2, _, by2 = uml_class(4.5, 6.2, 3.3, "TrainArtifacts",
        ["pipeline : Pipeline",
         "label_encoder : LabelEncoder"],
        ["+ load(path) : TrainArtifacts"],
        TEAL)

    cx3, ty3, _, by3 = uml_class(4.5, 3.0, 3.3, "FastPreprocessor",
        ["_cat_maps : dict[str, dict]",
         "_num_cols : list[str]",
         "_col_order : list[str]"],
        ["+ encode(row: dict) → ndarray",
         "+ from_column_transformer(ct)"],
        "#5C6BC0")

    cx4, ty4, _, by4 = uml_class(4.5, 0.3, 3.3, "ColdStartRecommender",
        ["co_matrix : dict[str, Counter]"],
        ["+ recommend(history, k) : list",
         "+ from_path(path)",
         "+ save(path)"],
        ORANGE)

    cx5, ty5, _, by5 = uml_class(8.5, 5.5, 3.2, "CandidateGen",
        ["(module-level functions)"],
        ["+ generate_candidates(history,\n   user_id) : list[str]"],
        GREEN)

    cx6, ty6, _, by6 = uml_class(8.5, 3.5, 3.2, "features",
        ["(module-level functions)"],
        ["+ compute_inference_row(\n   history) : DataFrame",
         "+ build_training_rows(df)"],
        GREEN)

    cx7, ty7, _, by7 = uml_class(8.5, 1.0, 3.2, "popularity",
        ["(module-level functions)"],
        ["+ build_popularity_list(df)",
         "+ save_popularity(lst, path)",
         "+ load_popularity(path)"],
        "#795548")

    # Associations
    assoc(cx1, ty1, cx2, by2, "1", "->")
    assoc(cx1, 6.2, cx3, ty3, "1", "->")
    assoc(cx1, 5.0, cx4, ty4, "1", "->")
    assoc(4.0, 7.0, 8.5, 6.2, "uses", "->")
    assoc(4.0, 6.5, 8.5, 5.2, "uses", "->")
    assoc(cx1, 5.2, 8.5, 2.5, "uses", "->")

    save("uml_classes.png")


# ══════════════════════════════════════════════════════════════════════════════
# 9. HISTOGRAMME — Distribution des liaisons
# ══════════════════════════════════════════════════════════════════════════════
def fig_hist_liaison():
    parquet = ROOT / "data" / "processed" / "oncf_clean.parquet"
    print("  → Chargement oncf_clean.parquet …", end=" ", flush=True)
    df = pd.read_parquet(parquet, columns=["LiaisonId"])
    print("OK")

    counts = df["LiaisonId"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: top-30 liaisons
    top30 = counts.head(30)
    ax = axes[0]
    bars = ax.barh(range(len(top30)), top30.values, color=BLUE, alpha=0.85)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels([f"Liaison {lid}" for lid in top30.index],
                       fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Nombre de réservations", fontsize=9)
    ax.set_title("Top 30 liaisons par fréquence", fontsize=10,
                 fontweight="bold", color=BLUE)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=7)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Right: distribution histogram (log scale)
    ax2 = axes[1]
    ax2.hist(counts.values, bins=50, color=ORANGE, edgecolor="white",
             alpha=0.85, log=True)
    ax2.set_xlabel("Fréquence d'une liaison (nb réservations)", fontsize=9)
    ax2.set_ylabel("Nombre de liaisons (échelle log)", fontsize=9)
    ax2.set_title(f"Distribution des fréquences ({len(counts)} liaisons)", fontsize=10,
                  fontweight="bold", color=BLUE)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    # Stats annotation
    ax2.axvline(counts.median(), color=GREEN, lw=2, linestyle="--",
                label=f"Médiane = {int(counts.median())}")
    ax2.axvline(counts.mean(), color=RED, lw=2, linestyle=":",
                label=f"Moyenne = {int(counts.mean())}")
    ax2.legend(fontsize=8)

    plt.suptitle("Distribution des liaisons par fréquence de réservation",
                 fontsize=12, fontweight="bold", color=BLUE, y=1.01)
    plt.tight_layout()
    save("hist_liaison_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10. RAPPORT DE NETTOYAGE — Funnel / Sankey simplifié
# ══════════════════════════════════════════════════════════════════════════════
def fig_rapport_nettoyage():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: funnel
    ax = axes[0]
    steps = [
        ("Dataset brut\n(oncf_data.csv)",      946155, MGRAY),
        ("Après propagation\ndes annulations",   640000, "#64B5F6"),
        ("Après déduplication",                  510000, TEAL),
        ("Dataset propre\n(oncf_clean.parquet)", 491680, GREEN),
    ]
    heights = [s[1] for s in steps]
    max_h = max(heights)

    for i, (label, val, color) in enumerate(steps):
        width = val / max_h
        left  = (1 - width) / 2
        bar = ax.barh(i, width, left=left, height=0.6, color=color,
                      edgecolor="white", linewidth=1.5)
        ax.text(0.5, i, f"{val:,}\n({val/946155*100:.1f}%)",
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white")
        ax.text(left - 0.02, i, label, ha="right", va="center",
                fontsize=8.5, color="#333")

    ax.set_xlim(0, 1); ax.set_ylim(-0.5, len(steps))
    ax.axis("off"); ax.invert_yaxis()
    ax.set_title("Funnel de nettoyage", fontsize=10,
                 fontweight="bold", color=BLUE)

    # Right: pie règles
    ax2 = axes[1]
    categories = {
        "Réservations annulées\n(propagation statut)": 946155-640000,
        "Doublons consolidés":                          640000-510000,
        "Utilisateurs cold start\n(< 3 voyages)":       510000-491680,
        "Dataset final propre":                         491680,
    }
    colors = [RED, ORANGE, "#64B5F6", GREEN]
    wedges, texts, autotexts = ax2.pie(
        list(categories.values()),
        labels=list(categories.keys()),
        colors=colors,
        autopct="%1.1f%%",
        pctdistance=0.78,
        startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for t in texts:
        t.set_fontsize(8)
    for at in autotexts:
        at.set_fontsize(8); at.set_fontweight("bold")
    ax2.set_title(f"Répartition des lignes retirées\n(946 155 → 491 680 lignes)",
                  fontsize=10, fontweight="bold", color=BLUE)

    plt.suptitle("Bilan du nettoyage — Entrées et Sorties",
                 fontsize=12, fontweight="bold", color=BLUE)
    plt.tight_layout()
    save("rapport_nettoyage.png")


# ══════════════════════════════════════════════════════════════════════════════
# 11. ENCODAGE CYCLIQUE — Heure, DOW, Mois
# ══════════════════════════════════════════════════════════════════════════════
def fig_encodage_cyclique():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    configs = [
        ("Heure de départ", 24, "depart_hour"),
        ("Jour de la semaine", 7, "depart_dow"),
        ("Mois", 12, "depart_month"),
    ]

    for ax, (title, period, col) in zip(axes, configs):
        vals = np.arange(period)
        sin_v = np.sin(2 * np.pi * vals / period)
        cos_v = np.cos(2 * np.pi * vals / period)

        ax.plot(vals, sin_v, "o-", color=ORANGE, label="sin", lw=2, markersize=5)
        ax.plot(vals, cos_v, "s--", color=BLUE, label="cos", lw=2, markersize=5)

        # Unit circle inset
        ax_ins = ax.inset_axes([0.68, 0.55, 0.30, 0.40])
        theta = np.linspace(0, 2*np.pi, 200)
        ax_ins.plot(np.cos(theta), np.sin(theta), color=MGRAY, lw=1)
        for v in vals:
            angle = 2 * np.pi * v / period
            ax_ins.plot(np.cos(angle), np.sin(angle), ".", color=TEAL, ms=4)
        ax_ins.set_aspect("equal"); ax_ins.axis("off")
        ax_ins.set_title("Cercle\nunité", fontsize=6)

        ax.set_xlabel(col.replace("_", " "), fontsize=9)
        ax.set_ylabel("Valeur encodée", fontsize=9)
        ax.set_title(f"Encodage cyclique\n{title} (T={period})", fontsize=9.5,
                     fontweight="bold", color=BLUE)
        ax.legend(fontsize=8)
        ax.grid(linestyle="--", alpha=0.3)
        ax.axhline(0, color="#ccc", lw=0.8)

        # Annotation: no discontinuity
        if period == 24:
            ax.annotate("Pas de discontinuité\n23h → 0h",
                        xy=(0, sin_v[0]), xytext=(5, -0.7),
                        fontsize=7, color=GREEN,
                        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))

    plt.suptitle("Encodage Cyclique des Variables Temporelles (sin/cos)",
                 fontsize=12, fontweight="bold", color=BLUE, y=1.01)
    plt.tight_layout()
    save("encodage_cyclique.png")


# ══════════════════════════════════════════════════════════════════════════════
# 12. DISTRIBUTION user_top_liaison_share
# ══════════════════════════════════════════════════════════════════════════════
def fig_user_top_liaison_share():
    parquet = ROOT / "data" / "processed" / "features.parquet"
    print("  → Chargement features.parquet …", end=" ", flush=True)
    df = pd.read_parquet(parquet, columns=["user_top_liaison_share"])
    s  = df["user_top_liaison_share"].dropna()
    print(f"OK  ({len(s):,} valeurs)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: histogram
    ax = axes[0]
    ax.hist(s, bins=40, color=BLUE, edgecolor="white", alpha=0.85)
    ax.axvline(s.mean(), color=ORANGE, lw=2, linestyle="--",
               label=f"Moyenne = {s.mean():.3f}")
    ax.axvline(s.median(), color=GREEN, lw=2, linestyle=":",
               label=f"Médiane = {s.median():.3f}")
    ax.set_xlabel("user_top_liaison_share", fontsize=9)
    ax.set_ylabel("Nombre de lignes", fontsize=9)
    ax.set_title("Distribution de user_top_liaison_share", fontsize=10,
                 fontweight="bold", color=BLUE)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Right: boxplot by segment
    ax2 = axes[1]
    # Compute user-level stats
    feat = pd.read_parquet(parquet, columns=["user_top_liaison_share",
                                              "user_trip_index"])
    feat = feat.dropna()
    feat["segment"] = pd.cut(feat["user_trip_index"],
                             bins=[0, 2, 5, 20, 10000],
                             labels=["0-2 voyages", "3-5", "6-20", "21+"])
    groups = [feat.loc[feat["segment"] == seg, "user_top_liaison_share"].values
              for seg in ["0-2 voyages", "3-5", "6-20", "21+"]]
    bp = ax2.boxplot(groups, patch_artist=True, notch=False,
                     labels=["0-2 voyages", "3-5", "6-20", "21+"],
                     widths=0.5)
    colors_bp = ["#90CAF9", "#64B5F6", TEAL, BLUE]
    for patch, c in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
    ax2.set_xlabel("Segment d'utilisateur (taille historique)", fontsize=9)
    ax2.set_ylabel("user_top_liaison_share", fontsize=9)
    ax2.set_title("user_top_liaison_share par segment", fontsize=10,
                  fontweight="bold", color=BLUE)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    plt.suptitle("Variable user_top_liaison_share — Loyauté utilisateur",
                 fontsize=12, fontweight="bold", color=BLUE)
    plt.tight_layout()
    save("dist_user_top_liaison_share.png")


# ══════════════════════════════════════════════════════════════════════════════
# 13. COLD START CF — Schéma matrice co-occurrence
# ══════════════════════════════════════════════════════════════════════════════
def fig_cold_start():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("Module Cold Start — Filtrage Collaboratif par Co-occurrence",
                 fontsize=12, fontweight="bold", color=BLUE)

    def box(x, y, w, h, text, color, fsize=9, tcol="white"):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                           facecolor=color, edgecolor="white", linewidth=1.5)
        ax.add_patch(r)
        ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                fontsize=fsize, color=tcol, fontweight="bold",
                multialignment="center")

    def arr(x1, x2, y, label=""):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE,
                                   lw=2, mutation_scale=16))
        if label:
            ax.text((x1+x2)/2, y+0.12, label, ha="center", fontsize=8,
                    color=ORANGE)

    box(0.2, 2.5, 2.0, 1.0, "Utilisateur\nfroid\n1-2 voyages", ORANGE)
    arr(2.2, 3.0, 3.0)
    box(3.0, 2.5, 2.0, 1.0, "Liaison(s)\nconnue(s)\nex: L42", TEAL)
    arr(5.0, 6.0, 3.0, "cherche co-réservations")

    # Co-occurrence matrix
    liaisons = ["L42", "L77", "L13", "L99", "L05"]
    co_matrix = np.array([
        [0, 312, 89, 45, 201],
        [312, 0, 67, 23, 145],
        [89, 67, 0, 188, 34],
        [45, 23, 188, 0, 12],
        [201, 145, 34, 12, 0],
    ])
    im = ax.imshow(co_matrix, cmap="Blues", aspect="auto",
                   extent=[6.0, 10.0, 1.5, 5.5])
    plt.colorbar(im, ax=ax, shrink=0.7, label="Co-occurrences")
    for i in range(5):
        for j in range(5):
            val = co_matrix[i, j]
            ax.text(6.4 + j*0.8, 5.1 - i*0.8, str(val),
                    ha="center", va="center", fontsize=7.5,
                    color="white" if val > 150 else "#333",
                    fontweight="bold")
    ax.set_xticks([6.4 + j*0.8 for j in range(5)])
    ax.set_xticklabels(liaisons, fontsize=8)
    ax.set_yticks([5.1 - i*0.8 for i in range(5)])
    ax.set_yticklabels(liaisons, fontsize=8)
    ax.text(8.0, 5.7, "Matrice de co-occurrence", ha="center",
            fontsize=9, fontweight="bold", color=BLUE)

    arr(10.0, 10.8, 3.0, "Top-k")
    box(10.8, 2.5, 1.0, 1.0, "Recs\n[L77, L05]", GREEN)

    # Decision table
    rows = [
        ("0 voyage",    "Popularité",    ORANGE),
        ("1-2 voyages", "Cold Start CF", TEAL),
        ("≥ 3 voyages", "XGBoost",       BLUE),
    ]
    ax.text(0.3, 1.9, "Stratégie selon historique :", fontsize=8.5,
            fontweight="bold", color="#333")
    for i, (hist, mode, color) in enumerate(rows):
        box(0.3, 1.3 - i*0.45, 1.5, 0.38, hist, MGRAY, 7.5, "#333")
        box(1.85, 1.3 - i*0.45, 1.5, 0.38, mode, color, 7.5, "white")

    save("cold_start_cf.png")


# ══════════════════════════════════════════════════════════════════════════════
# 14. ARCHITECTURE API FastAPI
# ══════════════════════════════════════════════════════════════════════════════
def fig_archi_api():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Architecture de l'API FastAPI — ONCF Recommender",
                 fontsize=12, fontweight="bold", color=BLUE)

    def box(x, y, w, h, text, color, fsize=8.5, tcol="white"):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                           facecolor=color, edgecolor="white", linewidth=1.5)
        ax.add_patch(r)
        ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                fontsize=fsize, color=tcol, fontweight="bold",
                multialignment="center")
        return x+w/2, y, x+w/2, y+h

    def arr(x1, y1, x2, y2, label="", bidirect=False):
        style = "<->" if bidirect else "-|>"
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=ORANGE,
                                   lw=1.5, mutation_scale=14))
        if label:
            ax.text((x1+x2)/2+0.05, (y1+y2)/2+0.05, label,
                    fontsize=7.5, color=ORANGE)

    # External
    box(0.2, 5.5, 1.8, 1.0, "📱\nApp Mobile", "#546E7A")
    box(0.2, 3.5, 1.8, 1.0, "🌐\nBrowser\n(Demo UI)", "#795548")

    # FastAPI layer
    box(2.5, 3.0, 3.5, 4.5, "", LGRAY, tcol="#333")
    ax.text(4.25, 7.3, "FastAPI (apps/api/main.py)", ha="center",
            fontsize=9, fontweight="bold", color=BLUE)

    box(2.7, 6.5, 3.0, 0.7, "GET /  (Demo page)", BLUE, 8)
    box(2.7, 5.7, 3.0, 0.7, "GET /health", TEAL, 8)
    box(2.7, 4.8, 3.0, 0.75, "POST /recommend\n?variant=a|b", ORANGE, 8)
    box(2.7, 3.95, 3.0, 0.75, "GET /schedule/{id}", GREEN, 8)
    box(2.7, 3.1, 3.0, 0.75, "POST /feedback", "#5C6BC0", 8)

    # Recommender
    box(7.0, 4.5, 2.8, 1.5, "Recommender A\n(prod)", BLUE)
    box(7.0, 2.5, 2.8, 1.5, "Recommender B\n(challenger)", "#5C6BC0")

    # ONNX
    box(10.5, 4.5, 2.2, 1.5, "ONNX Runtime\n~3ms", TEAL)
    box(10.5, 2.5, 2.2, 1.5, "ONNX Runtime\n(challenger)", "#546E7A")

    # Models
    box(10.5, 0.5, 2.2, 1.5, "models/\n.onnx .joblib", "#37474F", 8)

    # Redis
    box(7.0, 0.5, 2.8, 1.5, "Redis Cache\nTTL 1h\n(horaires)", RED)

    # Arrows
    arr(2.0, 6.0, 2.5, 6.85)
    arr(2.0, 4.0, 2.5, 5.2)
    arr(5.7, 5.2, 7.0, 5.2, "variant=a")
    arr(5.7, 4.0, 7.0, 3.2, "variant=b")
    arr(9.8, 5.2, 10.5, 5.2)
    arr(9.8, 3.2, 10.5, 3.2)
    arr(11.6, 4.5, 11.6, 2.0, "")
    arr(8.4, 4.5, 8.4, 2.0, "schedule")
    arr(10.5, 1.25, 8.4, 4.5)

    # Lifespan note
    ax.text(4.25, 3.0, "lifespan: charge les modèles\nune seule fois au démarrage",
            ha="center", fontsize=7.5, color=BLUE, style="italic",
            bbox=dict(boxstyle="round", facecolor=LGRAY, edgecolor=BLUE))

    save("archi_api.png")


# ══════════════════════════════════════════════════════════════════════════════
# 15. PROFILING LATENCE par étape
# ══════════════════════════════════════════════════════════════════════════════
def fig_latence_profiling():
    steps = [
        "Candidate\nGeneration",
        "compute_\ninference_row",
        "ColumnTransformer\n.transform (BOTTLENECK)",
        "ONNX Inference\n(session.run)",
        "Scoring &\ntri candidats",
    ]
    durations_before = [2.7, 1.4, 11.3, 3.2, 0.002]
    durations_after  = [2.7, 1.4, 0.01, 3.2, 0.002]
    colors_b = [TEAL, GREEN, RED, BLUE, "#546E7A"]
    colors_a = [TEAL, GREEN, GREEN, BLUE, "#546E7A"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, durations, colors, title in [
        (axes[0], durations_before, colors_b, "Avant optimisation (~23ms total)"),
        (axes[1], durations_after,  colors_a, "Après FastPreprocessor (~7ms total)"),
    ]:
        bars = ax.barh(steps, durations, color=colors, edgecolor="white",
                       linewidth=1.5, height=0.55)
        ax.bar_label(bars, fmt="%.2f ms", padding=5, fontsize=8.5)
        ax.set_xlabel("Durée médiane (ms)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", color=BLUE)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_xlim(0, max(durations_before) * 1.5)
        ax.invert_yaxis()

    axes[0].axvline(11.3, color=RED, lw=1.5, linestyle=":",
                    alpha=0.5, label="Bottleneck 49%")
    axes[0].legend(fontsize=8)

    plt.suptitle("Décomposition de la latence par étape du pipeline d'inférence",
                 fontsize=12, fontweight="bold", color=BLUE)
    plt.tight_layout()
    save("latence_profiling.png")


# ══════════════════════════════════════════════════════════════════════════════
# 16. FASTPREPROCESSOR vs SKLEARN comparaison
# ══════════════════════════════════════════════════════════════════════════════
def fig_fastpreprocessor():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: latency comparison bar chart
    ax = axes[0]
    labels_comp = ["sklearn\nColumnTransformer\n.transform",
                   "FastPreprocessor\n(dict lookup)"]
    values_comp = [11.31, 0.0077]
    cols = [RED, GREEN]
    bars = ax.bar(labels_comp, values_comp, color=cols, width=0.5,
                  edgecolor="white", linewidth=1.5)
    ax.bar_label(bars, labels=[f"{v:.3f} ms" for v in values_comp],
                 padding=5, fontsize=10, fontweight="bold")
    ax.set_ylabel("Durée médiane (ms)", fontsize=9)
    ax.set_title("Latence de prétraitement", fontsize=10,
                 fontweight="bold", color=BLUE)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(0, 15)
    ax.text(0.5, 7.5, "×1469\nplus rapide !", ha="center",
            fontsize=14, color=GREEN, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor=LGRAY, edgecolor=GREEN))

    # Right: end-to-end pipeline comparison
    ax2 = axes[1]
    configs_e2e = ["Baseline\n(sklearn predict)", "+ ONNX\nRuntime",
                   "+ FastPreprocessor\n(final)"]
    p50_vals = [852, 56, 13.74]
    cols2 = [RED, ORANGE, GREEN]
    bars2 = ax2.bar(configs_e2e, p50_vals, color=cols2, width=0.55,
                    edgecolor="white", linewidth=1.5)
    ax2.bar_label(bars2, labels=[f"{v:.1f} ms" for v in p50_vals],
                  padding=5, fontsize=9, fontweight="bold")
    ax2.set_ylabel("Latence p50 (ms)", fontsize=9)
    ax2.set_title("Évolution de la latence API p50\n(speedup total ×62)", fontsize=10,
                  fontweight="bold", color=BLUE)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.axhline(100, color=ORANGE, lw=2, linestyle="--",
                label="Seuil p99 < 100ms")
    ax2.axhline(50, color=GREEN, lw=2, linestyle=":",
                label="Seuil p50 < 50ms")
    ax2.legend(fontsize=8)

    plt.suptitle("FastPreprocessor — Optimisation du Prétraitement sur le Chemin Chaud",
                 fontsize=12, fontweight="bold", color=BLUE)
    plt.tight_layout()
    save("fastpreprocessor_vs_sklearn.png")


# ══════════════════════════════════════════════════════════════════════════════
# 17. PIPELINE DE RÉENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════
def fig_retrain_pipeline():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Pipeline de Réentraînement Automatique avec Guardrail KPI",
                 fontsize=12, fontweight="bold", color=BLUE)

    def step(x, y, w, h, num, text, color):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                           facecolor=color, edgecolor="white", linewidth=2)
        ax.add_patch(r)
        ax.text(x+0.2, y+h-0.2, f"({num})", fontsize=8, color="white",
                va="top", fontweight="bold")
        ax.text(x+w/2, y+h/2-0.1, text, ha="center", va="center",
                fontsize=8.5, color="white", multialignment="center")

    def diamond(x, y, w, h, text, color=ORANGE):
        diamond_pts = np.array([
            [x+w/2, y+h],
            [x+w,   y+h/2],
            [x+w/2, y],
            [x,     y+h/2],
        ])
        poly = plt.Polygon(diamond_pts, facecolor=color, edgecolor="white",
                           linewidth=2)
        ax.add_patch(poly)
        ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                fontsize=8, color="white", fontweight="bold",
                multialignment="center")

    def arr(x1, y1, x2, y2, label="", color=ORANGE):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=2, mutation_scale=16))
        if label:
            ax.text((x1+x2)/2+0.05, (y1+y2)/2+0.05, label,
                    fontsize=7.5, color=color)

    # Trigger
    ax.text(0.5, 7.5, "⏰ 02h00 (Task Scheduler)", fontsize=9,
            color=BLUE, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor=LGRAY, edgecolor=BLUE))
    arr(1.5, 7.5, 2.0, 7.5)

    # Steps
    step(2.0, 6.5, 2.5, 1.0, "1", "Chargement\ndonnées\n(--window-months)", TEAL)
    arr(3.25, 6.5, 3.25, 5.0)
    step(2.0, 3.9, 2.5, 1.0, "2", "Entraînement\nXGBoost\n(~43 min CPU)", BLUE)
    arr(3.25, 3.9, 3.25, 2.8)
    step(2.0, 1.7, 2.5, 1.0, "3", "Évaluation\nHR@1, HR@3\nMRR@3", GREEN)
    arr(3.25, 1.7, 5.0, 1.7)

    diamond(5.0, 0.9, 3.0, 1.5, "Guardrail\nHR@1 chute\n> 5 points ?", ORANGE)
    arr(6.5, 0.9, 6.5, 0.0, "NON → PASS", GREEN)

    # Write challenger (always)
    arr(6.5, 0.9, 6.5, 0.0)
    step(4.5, 2.5, 2.5, 1.0, "4", "Écriture\nchallenger\n(toujours)", "#5C6BC0")
    arr(5.75, 3.5, 5.75, 2.5)

    arr(8.0, 1.6, 9.0, 1.6, "FAIL → BLOQUE", RED)
    step(9.0, 1.1, 3.0, 1.0, "!", "Promotion\nBLOQUÉE\n(alerte log)", RED)
    ax.text(9.5, 0.8, "Challenger conservé\npour analyse", fontsize=7.5,
            color=RED, ha="center")

    step(6.0, 4.0, 3.0, 1.0, "5", "Promotion\nchallenger → prod\n(si guardrail OK)", GREEN)
    arr(5.75, 2.5, 7.5, 4.0)
    arr(5.75, 4.0, 5.75, 4.2, "→ prod")

    # Rolling window note
    ax.text(0.3, 5.0, "Rolling Window\noption :", fontsize=8.5,
            color=TEAL, fontweight="bold")
    ax.text(0.3, 4.2, "--window-months 24\n→ 160 K lignes\n→ ~18 min", fontsize=7.5,
            color=TEAL,
            bbox=dict(boxstyle="round", facecolor=LGRAY, edgecolor=TEAL))

    # Metrics note
    ax.text(9.5, 5.5, "Métriques cibles :", fontsize=8.5,
            fontweight="bold", color=BLUE)
    for i, txt in enumerate(["HR@1 > 50%", "HR@3 > 60%", "MRR@3 > 60%"]):
        ax.text(9.5, 5.0 - i*0.45, f"✓ {txt}", fontsize=8, color=GREEN)

    save("retrain_pipeline.png")


# ══════════════════════════════════════════════════════════════════════════════
# 18. FRAMEWORK A/B TESTING
# ══════════════════════════════════════════════════════════════════════════════
def fig_ab_testing():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Framework A/B Testing — Architecture et Flux",
                 fontsize=12, fontweight="bold", color=BLUE)

    def box(x, y, w, h, text, color, fsize=8.5, tcol="white"):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                           facecolor=color, edgecolor="white", linewidth=2)
        ax.add_patch(r)
        ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                fontsize=fsize, color=tcol, fontweight="bold",
                multialignment="center")

    def arr(x1, y1, x2, y2, label="", color=ORANGE, lw=1.8):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=lw, mutation_scale=14))
        if label:
            ax.text((x1+x2)/2+0.08, (y1+y2)/2+0.08, label,
                    fontsize=7.5, color=color)

    # App mobile
    box(0.2, 4.5, 2.0, 1.0, "📱\nApp Mobile", "#546E7A")

    # Router
    box(3.0, 5.5, 2.5, 0.8, "variant=a\n50% traffic", BLUE)
    box(3.0, 4.2, 2.5, 0.8, "variant=b\n50% traffic", "#5C6BC0")

    arr(2.2, 5.0, 3.0, 5.9, "?variant=a")
    arr(2.2, 5.0, 3.0, 4.6, "?variant=b")

    # Recommenders
    box(6.2, 5.3, 2.8, 1.2, "Recommender A\n(prod)\nxgb_ranker.onnx", BLUE)
    box(6.2, 3.9, 2.8, 1.2, "Recommender B\n(challenger)\nxgb_ranker_challenger.onnx", "#5C6BC0")

    arr(5.5, 5.9, 6.2, 5.9)
    arr(5.5, 4.6, 6.2, 4.6)

    # Response with request_id
    box(9.5, 4.5, 2.8, 1.2, "Réponse JSON\n+ request_id\n(UUID4)", TEAL)
    arr(9.0, 5.9, 9.5, 5.1)
    arr(9.0, 4.6, 9.5, 4.8)

    # Feedback loop
    box(9.5, 2.5, 2.8, 1.2, "POST /feedback\n{request_id,\nliaison_id, clicked}", ORANGE)
    arr(9.5+1.4, 4.5, 9.5+1.4, 3.7, "", ORANGE)

    # Logs
    box(6.2, 1.5, 2.8, 1.2, "api.log\n(JSON structuré)\ncode_client absent ✓", "#37474F")
    arr(9.5+1.4, 2.5, 7.6, 2.7, "", "#37474F")
    arr(9.0, 5.9, 7.6, 2.7, "", "#37474F", lw=1)
    arr(9.0, 4.6, 7.6, 2.7, "", "#37474F", lw=1)

    # Analysis
    box(2.5, 0.8, 3.0, 1.0, "Analyse CTR offline\nJointure request_id\nVariant A vs B", GREEN)
    arr(6.2, 2.1, 5.5, 1.3, "logs → analyse", GREEN)

    # Promotion
    box(0.3, 0.8, 2.0, 1.0, "Promotion\nchallenger\n→ prod", ORANGE)
    arr(2.5, 1.3, 2.3, 1.3, "si CTR ↑", ORANGE)

    # Privacy note
    ax.text(0.2, 3.5, "🔒 Loi 09-08", fontsize=9, color=RED, fontweight="bold")
    ax.text(0.2, 3.1, "code_client jamais\ndans les logs.\nrequest_id = proxy\nanonyme.", fontsize=8, color=RED)

    # Workflow steps
    steps_txt = [
        "① Retrain produit challenger.*",
        "② Redémarrage API (lifespan)",
        "③ 50% trafic → variant=b",
        "④ 7 jours → analyse CTR",
        "⑤ Si meilleur → promotion",
    ]
    for i, t in enumerate(steps_txt):
        ax.text(0.2, 8.0 - i*0.5, t, fontsize=8, color=BLUE)

    save("ab_testing.png")


# ══════════════════════════════════════════════════════════════════════════════
# 19. BONUS — Métriques par segment (bar chart)
# ══════════════════════════════════════════════════════════════════════════════
def fig_metrics_segment():
    segments = ["0-2 voyages\n(n=44 397)", "3-5 voyages\n(n=16 780)",
                "6-20 voyages\n(n=23 737)", "21+ voyages\n(n=13 347)"]
    hr1  = [73.9, 75.2, 77.9, 82.7]
    hr3  = [89.3, 90.4, 91.6, 93.1]
    mrr3 = [80.9, 82.1, 84.1, 87.4]

    x = np.arange(len(segments))
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 5.5))
    b1 = ax.bar(x - width, hr1,  width, label="HR@1",  color=BLUE,   edgecolor="white", lw=1.2)
    b2 = ax.bar(x,          hr3,  width, label="HR@3",  color=ORANGE, edgecolor="white", lw=1.2)
    b3 = ax.bar(x + width,  mrr3, width, label="MRR@3", color=GREEN,  edgecolor="white", lw=1.2)

    ax.bar_label(b1, fmt="%.1f%%", fontsize=7.5, padding=2)
    ax.bar_label(b2, fmt="%.1f%%", fontsize=7.5, padding=2)
    ax.bar_label(b3, fmt="%.1f%%", fontsize=7.5, padding=2)

    ax.set_xticks(x); ax.set_xticklabels(segments, fontsize=9)
    ax.set_ylabel("Métrique (%)", fontsize=10)
    ax.set_ylim(50, 100)
    ax.set_title("Métriques par Segment d'Utilisateurs (Sprint 2)",
                 fontsize=12, fontweight="bold", color=BLUE)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.axhline(76.28, color=BLUE, lw=1.5, linestyle=":",
               alpha=0.5, label="HR@1 moy.")
    ax.text(3.7, 77.5, "HR@1 moy=76.3%", fontsize=8, color=BLUE, style="italic")

    plt.tight_layout()
    save("metrics_segment.png")


# ══════════════════════════════════════════════════════════════════════════════
# 20. BONUS — Comparaison Sprint 1 vs Sprint 2 vs Baselines
# ══════════════════════════════════════════════════════════════════════════════
def fig_metrics_comparison():
    models = ["global_top\n(popularité)", "prev_liaison\n(dernier trajet)",
              "most_frequent\n(freq+récence)", "XGBoost\nSprint 1",
              "XGBoost\nSprint 2"]
    hr1  = [3.99,  26.20, 27.51, 73.95, 76.28]
    hr3  = [11.25, 32.04, 51.28, 88.77, 90.55]
    mrr3 = [7.07,  28.81, 38.65, 80.64, 82.77]

    x = np.arange(len(models)); width = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))

    b1 = ax.bar(x - width, hr1,  width, label="HR@1",  color=BLUE,   edgecolor="white", lw=1.2)
    b2 = ax.bar(x,          hr3,  width, label="HR@3",  color=ORANGE, edgecolor="white", lw=1.2)
    b3 = ax.bar(x + width,  mrr3, width, label="MRR@3", color=GREEN,  edgecolor="white", lw=1.2)
    ax.bar_label(b1, fmt="%.1f%%", fontsize=7, padding=2, rotation=0)
    ax.bar_label(b2, fmt="%.1f%%", fontsize=7, padding=2, rotation=0)
    ax.bar_label(b3, fmt="%.1f%%", fontsize=7, padding=2, rotation=0)

    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Métrique (%)", fontsize=10); ax.set_ylim(0, 105)
    ax.set_title("Comparaison XGBoost vs Baselines (Sprint 1 & Sprint 2)",
                 fontsize=12, fontweight="bold", color=BLUE)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Threshold lines
    for thresh, label, color in [(50, "Seuil HR@1>50%", BLUE),
                                  (60, "Seuil HR@3>60%", ORANGE),
                                  (60, "Seuil MRR>60%",  GREEN)]:
        ax.axhline(thresh, color=color, lw=1, linestyle=":", alpha=0.5)

    # Separator between baselines and XGBoost
    ax.axvline(2.5, color=RED, lw=2, linestyle="--", alpha=0.4)
    ax.text(2.6, 95, "XGBoost", fontsize=9, color=RED, style="italic")

    plt.tight_layout()
    save("metrics_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{'='*60}")
    print("  Génération des figures — Rapport PFA ONCF")
    print(f"  Output : {PIC}")
    print(f"{'='*60}\n")

    # Figures without data dependencies
    tasks = [
        ("zero_click_concept.png",        fig_zero_click),
        ("gantt.png",                      fig_gantt),
        ("archi_globale.png",              fig_archi_globale),
        ("uml_composants.png",             fig_uml_composants),
        ("archi_deux_etapes.png",          fig_archi_deux_etapes),
        ("uml_usecase.png",                fig_uml_usecase),
        ("uml_sequence.png",               fig_sequence),
        ("uml_classes.png",                fig_classes),
        ("rapport_nettoyage.png",          fig_rapport_nettoyage),
        ("encodage_cyclique.png",          fig_encodage_cyclique),
        ("cold_start_cf.png",             fig_cold_start),
        ("archi_api.png",                  fig_archi_api),
        ("latence_profiling.png",          fig_latence_profiling),
        ("fastpreprocessor_vs_sklearn.png",fig_fastpreprocessor),
        ("retrain_pipeline.png",           fig_retrain_pipeline),
        ("ab_testing.png",                 fig_ab_testing),
        ("metrics_segment.png",            fig_metrics_segment),
        ("metrics_comparison.png",         fig_metrics_comparison),
    ]

    data_tasks = [
        ("hist_liaison_distribution.png",       fig_hist_liaison),
        ("dist_user_top_liaison_share.png",     fig_user_top_liaison_share),
    ]

    errors = []
    for name, fn in tasks:
        try:
            print(f"  Génération : {name}")
            fn()
        except Exception as e:
            print(f"  ✗ ERREUR : {e}")
            errors.append((name, str(e)))

    print("\n  --- Figures avec données réelles ---")
    for name, fn in data_tasks:
        try:
            print(f"  Génération : {name}")
            fn()
        except Exception as e:
            print(f"  ✗ ERREUR : {e}")
            errors.append((name, str(e)))

    print(f"\n{'='*60}")
    total = len(tasks) + len(data_tasks)
    ok = total - len(errors)
    print(f"  {ok}/{total} figures générées avec succès.")
    if errors:
        print(f"\n  Erreurs :")
        for name, err in errors:
            print(f"    • {name} : {err}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
