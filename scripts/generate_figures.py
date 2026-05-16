"""
generate_figures.py — ONCF Recommender PFA — Figures (v2)
Layouts refactorises : aucun chevauchement, flux directionnels clairs,
typographie formelle, aucun emoji dans les schemas.
Usage : .venv/Scripts/python.exe scripts/generate_figures.py
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
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Polygon as MPoly

ROOT = Path(__file__).parent.parent
PIC  = ROOT / "pic"
PIC.mkdir(exist_ok=True)

# ── Palette ONCF ──────────────────────────────────────────────────────────────
BLUE   = "#004B8D"
ORANGE = "#E65300"
GREEN  = "#2E7D32"
TEAL   = "#00695C"
RED    = "#C62828"
PURPLE = "#5C6BC0"
GRAY_D = "#546E7A"
GRAY_M = "#CCCCCC"
LGRAY  = "#F5F5F5"
WHITE  = "#FFFFFF"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    13,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})


# ── Helpers communs ────────────────────────────────────────────────────────────
def save(name):
    plt.savefig(PIC / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  OK  {name}")


def rbox(ax, x, y, w, h, fc, ec=WHITE, lw=1.8, pad=0.08):
    """Rectangle arrondi rempli."""
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(p)


def txt(ax, x, y, s, fs=9, color=WHITE, ha="center", va="center",
        bold=False, style="normal"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", fontstyle=style,
            multialignment="center", zorder=5)


def varrow(ax, x, y1, y2, color=ORANGE, lw=1.8):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=14), zorder=2)


def harrow(ax, x1, x2, y, color=ORANGE, lw=1.8):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=14), zorder=2)


def darrow(ax, x1, y1, x2, y2, color=ORANGE, lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=14), zorder=2)


def dmnd(ax, cx, cy, hw, hh, fc, ec=WHITE, lw=2):
    pts = [[cx, cy + hh], [cx + hw, cy], [cx, cy - hh], [cx - hw, cy]]
    ax.add_patch(MPoly(pts, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))


# ══════════════════════════════════════════════════════════════════════════════
# 1. ZERO-CLICK SEARCH — Schema conceptuel
# ══════════════════════════════════════════════════════════════════════════════
def fig_zero_click():
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title("Zero-Click Search — Principe de recommandation proactive",
                 fontsize=13, fontweight="bold", color=BLUE, pad=10)

    BW, BH = 2.1, 1.0

    # ── Ligne du haut : SANS le systeme ──────────────────────────────────────
    y1 = 3.6
    labels_b = ["Utilisateur\nouvre l'app", "Saisit gare\ndepart",
                 "Saisit gare\narrivee", "Selectionne\ndate", "Resultats\n(5 etapes)"]
    xs = [0.4, 2.9, 5.4, 7.9, 10.4]
    for lbl, px in zip(labels_b, xs):
        rbox(ax, px, y1, BW, BH, GRAY_M, ec="#999", lw=1.3, pad=0.06)
        txt(ax, px + BW / 2, y1 + BH / 2, lbl, fs=8.5, color="#333")
    for i in range(len(xs) - 1):
        harrow(ax, xs[i] + BW, xs[i + 1], y1 + BH / 2, color="#888", lw=1.5)
    ax.text(7.0, 4.85, "SANS Zero-Click Search  —  Friction de recherche (5 etapes)",
            ha="center", fontsize=9.5, color=RED, fontweight="bold")

    # ── Ligne du bas : AVEC le systeme ───────────────────────────────────────
    y2 = 1.3
    boxes_a = [
        (0.4,  2.2, GRAY_D, "Utilisateur\nouvre l'app"),
        (3.0,  4.0, ORANGE, "Systeme predit\nautomatiquement\nle trajet"),
        (7.4,  2.2, GREEN,  "Suggestion\naffichee"),
        (10.0, 1.6, TEAL,   "1 clic"),
    ]
    for px, bw, fc, lbl in boxes_a:
        rbox(ax, px, y2, bw, BH, fc)
        txt(ax, px + bw / 2, y2 + BH / 2, lbl, fs=8.5)
    pairs_a = [(0.4 + 2.2, 3.0), (3.0 + 4.0, 7.4), (7.4 + 2.2, 10.0)]
    for x1, x2 in pairs_a:
        harrow(ax, x1, x2, y2 + BH / 2, color=GREEN, lw=2)
    ax.text(7.0, 0.7, "AVEC Zero-Click Search  —  Zero friction, experience immediate",
            ha="center", fontsize=9.5, color=GREEN, fontweight="bold")

    # Separateur central
    ax.axhline(2.85, color=GRAY_M, lw=1.2, linestyle="--", xmin=0.02, xmax=0.98)

    save("zero_click_concept.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. GANTT — Planning du projet
# ══════════════════════════════════════════════════════════════════════════════
def fig_gantt():
    phases = [
        ("Etude & Benchmark",           1, 2,  GRAY_M),
        ("Sprint 1 : Donnees & Modele", 2, 5,  BLUE),
        ("Sprint 2 : Architecture & API", 5, 7, ORANGE),
        ("Phase 3 : Production",         7, 8,  TEAL),
        ("Consolidation & Rapport",      8, 9,  GREEN),
    ]
    livrables = [
        (2,  "Cahier des charges"),
        (5,  "Modele v1 + metriques"),
        (7,  "API REST + tests"),
        (8,  "ONNX + A/B + retrain"),
        (9,  "Rapport final\n(115 tests)"),
    ]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.subplots_adjust(top=0.82)
    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(-1.5, len(phases))
    ax.set_xlabel("Semaine du stage", fontsize=10)
    ax.set_xticks(range(1, 10))
    ax.set_xticklabels([f"S{i}" for i in range(1, 10)], fontsize=9)
    ax.set_yticks(range(len(phases)))
    ax.set_yticklabels([p[0] for p in phases], fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_title("Planning du projet — Phases et sprints",
                 fontweight="bold", color=BLUE, fontsize=12, pad=12)

    for i, (_, start, end, color) in enumerate(phases):
        ax.barh(i, end - start, left=start, height=0.55, color=color,
                edgecolor="white", linewidth=1.2)

    for week, label in livrables:
        ax.axvline(week, color=ORANGE, linestyle=":", linewidth=1.3, alpha=0.8, ymin=0.05)
        ax.text(week, -0.7, label, fontsize=7.5, color=ORANGE,
                ha="center", va="top", rotation=30,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=ORANGE, lw=0.8))

    save("gantt.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. ARCHITECTURE GLOBALE — 4 couches
# ══════════════════════════════════════════════════════════════════════════════
def fig_archi_globale():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Architecture Globale du Systeme de Recommandation ONCF",
                 fontsize=13, fontweight="bold", color=BLUE, pad=12)

    # Couche externe : App Mobile (haut, centre)
    rbox(ax, 4.5, 7.0, 3.0, 0.7, GRAY_D)
    txt(ax, 6.0, 7.35, "Application Mobile ONCF  (client)", fs=9, bold=True)
    varrow(ax, 6.0, 7.0, 6.65, color=BLUE, lw=1.5)

    # Couches empilees
    layers = [
        (0.4, 5.8, 11.2, 1.0, BLUE,    "4 — Couche API (apps/api/main.py)",
         "GET /  |  GET /health  |  POST /recommend  |  GET /schedule/{id}  |  POST /feedback", ORANGE),
        (0.4, 4.1, 11.2, 1.2, TEAL,    "3 — Couche Logique (rec_oncf.*)",
         "Recommender  |  generate_candidates  |  compute_inference_row  |  ColdStartRecommender  |  schedule", LGRAY),
        (0.4, 2.4, 11.2, 1.2, PURPLE,  "2 — Couche Modele (models/)",
         "xgb_ranker.onnx  148 Mo  |  label_encoder.joblib  |  cold_start.joblib  |  popularity.joblib", LGRAY),
        (0.4, 0.5, 11.2, 1.4, GRAY_D,  "1 — Couche Donnees (data/processed/)",
         "oncf_clean.parquet  491 680 lignes  |  features.parquet  26 colonnes", LGRAY),
    ]

    for x, y, w, h, fc, title, items, ic in layers:
        rbox(ax, x, y, w, h, fc, lw=2)
        txt(ax, x + 0.4, y + h - 0.27, title, fs=10, bold=True, ha="left")
        txt(ax, x + 0.4, y + 0.25, items, fs=8.2, color=ic, ha="left", va="bottom")

    # Fleches entre couches (a droite pour eviter de traverser le texte)
    arrow_x = 10.5
    for y_from, y_to in [(5.8, 5.3), (4.1 + 1.2, 4.1), (2.4 + 1.2, 3.6), (0.5 + 1.4, 2.4)]:
        ax.annotate("", xy=(arrow_x, y_to), xytext=(arrow_x, y_from),
                    arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.5,
                                   mutation_scale=12), zorder=2)

    save("archi_globale.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. UML COMPOSANTS
# ══════════════════════════════════════════════════════════════════════════════
def fig_uml_composants():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("Diagramme de Composants UML  —  Application Mobile ONCF",
                 fontsize=13, fontweight="bold", color=BLUE, pad=10)

    def comp(x, y, w, h, title, items=None, color=BLUE):
        rbox(ax, x, y, w, h, LGRAY, ec=color, lw=2, pad=0.05)
        # Barre de titre
        rbox(ax, x, y + h - 0.45, w, 0.45, color, ec=color, lw=0, pad=0.02)
        txt(ax, x + w / 2, y + h - 0.22, f"<<component>>\n{title}",
            fs=8, bold=True, color=WHITE)
        if items:
            for i, item in enumerate(items):
                ax.text(x + 0.18, y + h - 0.62 - i * 0.34, f"• {item}",
                        fontsize=7.5, va="top", color="#333")

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.3,
                                   mutation_scale=12), zorder=2)

    # Rangee 1
    comp(0.3, 6.3, 4.0, 2.3, "apps.api.main",
         ["FastAPI app, lifespan startup",
          "GET /, GET /health, POST /recommend",
          "GET /schedule, POST /feedback",
          "Logging structure JSON"], BLUE)
    comp(4.8, 6.3, 4.2, 2.3, "rec_oncf.recommender",
         ["Recommender.from_paths(paths)",
          "recommend(code_client, k) : dict",
          "history_lookup [dict[str, DataFrame]]",
          "_fallback(k) : dict"], TEAL)
    comp(9.5, 6.3, 4.2, 2.3, "rec_oncf.training",
         ["train_xgb_multiclass()",
          "FastPreprocessor (dict lookup)",
          "predict_proba_onnx(session, X)",
          "TrainArtifacts (pipeline + encoder)"], PURPLE)

    # Rangee 2
    comp(0.3, 3.3, 4.0, 2.3, "rec_oncf.candidates",
         ["generate_candidates(history)",
          "Regles heuristiques",
          "Top frequence + recence",
          "Co-occurrence CF"], ORANGE)
    comp(4.8, 3.3, 4.2, 2.3, "rec_oncf.features",
         ["build_training_rows(df)",
          "compute_inference_row(history)",
          "Encodage cyclique (sin/cos)",
          "user_top_liaison_share"], GREEN)
    comp(9.5, 3.3, 4.2, 2.3, "rec_oncf.cold_start",
         ["ColdStartRecommender",
          "Matrice co-occurrence",
          "recommend(history, k)"], RED)

    # Rangee 3
    comp(1.5, 0.4, 3.3, 2.3, "models/  (artefacts)",
         ["xgb_ranker.onnx  148 Mo",
          "label_encoder.joblib",
          "cold_start.joblib",
          "popularity.joblib"], GRAY_D)
    comp(5.5, 0.4, 3.0, 2.3, "data/processed/",
         ["oncf_clean.parquet\n491 680 lignes",
          "features.parquet\n26 colonnes"], GRAY_D)
    comp(9.5, 0.4, 4.2, 2.3, "rec_oncf.schedule",
         ["STATION_CODES (24 gares)",
          "fetch_departures(origin, dest, date)",
          "Cache Redis/memoire  TTL 1h"], "#795548")

    # Fleches rangees 1→2
    arr(4.3, 7.45, 4.8, 7.45)
    arr(9.0, 7.45, 9.5, 7.45)
    arr(4.8 + 2.1, 6.3, 4.8 + 2.1, 3.3 + 2.3)
    arr(9.5 + 2.1, 6.3, 9.5 + 2.1, 3.3 + 2.3)
    arr(0.3 + 2.0, 6.3, 0.3 + 2.0, 3.3 + 2.3)

    # Fleches rangees 2→3
    arr(4.8 + 2.1, 3.3, 3.15, 0.4 + 2.3)
    arr(4.8 + 2.1, 3.3, 7.0, 0.4 + 2.3)
    arr(9.5 + 2.1, 3.3, 9.5 + 2.1, 0.4 + 2.3)

    # Acteur externe
    ax.text(7.0, 8.85, "Application Mobile ONCF  (acteur externe)",
            ha="center", fontsize=9.5, color=BLUE, fontweight="bold")
    arr(7.0, 8.72, 4.3, 8.5)
    ax.annotate("", xy=(4.3, 8.5), xytext=(4.3, 6.3 + 2.3),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.5,
                                mutation_scale=12), zorder=2)

    save("uml_composants.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. ARCHITECTURE DEUX ETAPES  (Candidate Generation + Ranking)
# ══════════════════════════════════════════════════════════════════════════════
def fig_archi_deux_etapes():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Architecture Deux Etapes : Candidate Generation + Ranking",
                 fontsize=13, fontweight="bold", color=BLUE, pad=10)

    # ── Flux principal (haut) ─────────────────────────────────────────────────
    blocks = [
        (0.3,  4.0, 2.5, 2.0, GRAY_D, "Historique",
         "CodeClient\n→ history_lookup\n(DataFrame)"),
        (3.2,  3.7, 3.2, 2.5, TEAL,   "Etape 1\nCandidate Generation",
         "Heuristiques :\n  3 derniers voyages\n  Top frequence\n  Co-occurrence CF\n  <= 10 candidats"),
        (6.9,  3.7, 3.2, 2.5, BLUE,   "Etape 2\nRanking XGBoost",
         "Features on-the-fly (26)\nFastPreprocessor\nONNX Runtime (~3 ms)\nFiltre sur candidats"),
        (10.6, 3.7, 3.0, 2.5, GREEN,  "Reponse Top-k",
         "mode: \"model\"\nk in {1, 2, 3}\nLatence < 14 ms (p50)"),
    ]
    for x, y, w, h, fc, title, body in blocks:
        rbox(ax, x, y, w, h, fc, lw=2)
        txt(ax, x + w / 2, y + h - 0.35, title, fs=9.5, bold=True)
        txt(ax, x + w / 2, y + 0.5, body, fs=8)

    # Fleches horizontales entre blocs (haut)
    harrow(ax, 0.3 + 2.5, 3.2, 5.0, color=ORANGE, lw=2)
    harrow(ax, 3.2 + 3.2, 6.9, 5.0, color=ORANGE, lw=2)
    harrow(ax, 6.9 + 3.2, 10.6, 5.0, color=ORANGE, lw=2)

    # Labels de latence sous les fleches
    ax.text(4.8, 3.5, "< 3 ms", ha="center", fontsize=8.5, color=TEAL, style="italic")
    ax.text(8.5, 3.5, "~ 11 ms", ha="center", fontsize=8.5, color=BLUE, style="italic")

    # ── Branche Cold Start (bas) ──────────────────────────────────────────────
    # Indicateur sous le bloc Historique
    ax.text(1.55, 3.75, "Si < 3 voyages :", ha="center", fontsize=8, color=RED)

    # Fleche vers le bas (depuis Historique)
    varrow(ax, 1.55, 3.75, 2.8, color=RED, lw=1.5)

    # Boite Cold Start
    rbox(ax, 0.3, 1.3, 5.5, 1.2, "#FFF3E0", ec=RED, lw=1.5, pad=0.06)
    txt(ax, 3.05, 1.9, "Cold Start CF  /  Popularite  (bypass)",
        fs=9, color=RED, bold=True)
    ax.text(3.05, 1.4, "Si 1-2 voyages : Filtrage Collaboratif (co-occurrence)   "
            "|   Si 0 voyage : Liste popularite globale",
            ha="center", va="center", fontsize=7.5, color="#444",
            multialignment="center")

    # Fleche depuis Cold Start vers Top-k (en bas a droite)
    ax.annotate("", xy=(12.1, 3.7), xytext=(5.8, 1.9),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5,
                                mutation_scale=12,
                                connectionstyle="arc3,rad=-0.2"), zorder=2)
    ax.text(9.5, 2.3, "bypass direct", fontsize=8, color=RED,
            style="italic", ha="center")

    save("archi_deux_etapes.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. UML CAS D'UTILISATION
# ══════════════════════════════════════════════════════════════════════════════
def fig_uml_usecase():
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("Diagramme de Cas d'Utilisation — Systeme Zero-Click Search",
                 fontsize=13, fontweight="bold", color=BLUE, pad=10)

    # Frontiere systeme
    ax.add_patch(FancyBboxPatch((2.0, 0.3), 9.0, 8.0,
                                boxstyle="square,pad=0.1",
                                facecolor="#F0F4FF", edgecolor=BLUE, linewidth=2))
    ax.text(6.5, 8.22, "Systeme de Recommandation ONCF",
            ha="center", fontsize=10, fontweight="bold", color=BLUE)

    def actor(x, y, name, color=BLUE):
        ax.plot([x, x], [y + 0.45, y + 0.85], color=color, lw=2)
        ax.plot([x - 0.28, x + 0.28], [y + 0.65, y + 0.65], color=color, lw=2)
        ax.plot([x, x - 0.22], [y + 0.45, y + 0.18], color=color, lw=2)
        ax.plot([x, x + 0.22], [y + 0.45, y + 0.18], color=color, lw=2)
        ax.add_patch(plt.Circle((x, y + 0.96), 0.17, color=color,
                                fill=False, lw=2))
        ax.text(x, y, name, ha="center", fontsize=8,
                fontweight="bold", color=color, multialignment="center")

    def usecase(x, y, w, h, text, color=BLUE):
        ax.add_patch(mpatches.Ellipse((x, y), w, h, facecolor="white",
                                      edgecolor=color, linewidth=1.5))
        ax.text(x, y, text, ha="center", va="center",
                fontsize=7.5, multialignment="center", color="#222")

    def assoc(ax1, ay1, ux, uy):
        ax.plot([ax1, ux], [ay1, uy], color="#888", lw=1.2)

    actor(0.8, 5.5, "Utilisateur\nmobile", BLUE)
    actor(0.8, 2.5, "Application\nmobile", TEAL)
    actor(12.2, 5.5, "Systeme\nReentrainement", GREEN)
    actor(12.2, 2.5, "Data Scientist\n/ Ops", ORANGE)

    usecase(5.8, 7.5, 3.5, 0.8, "Consulter recommandation\na l'ouverture")
    usecase(5.8, 6.3, 3.2, 0.8, "Cliquer sur une\nliaison suggeree")
    usecase(5.8, 5.1, 3.0, 0.8, "Effectuer une\nreservation")
    usecase(5.8, 3.9, 3.2, 0.8, "Obtenir horaires\n(include_schedule)")
    usecase(5.8, 2.7, 3.2, 0.8, "Envoyer feedback\n(POST /feedback)")
    usecase(5.8, 1.5, 3.2, 0.8, "Verifier l'etat\n(GET /health)")

    usecase(9.5, 7.5, 2.8, 0.7, "Reentrainer\nle modele")
    usecase(9.5, 6.3, 2.8, 0.7, "Evaluer avec\nguardrail KPI")
    usecase(9.5, 5.1, 2.8, 0.7, "Promouvoir\nle challenger")
    usecase(9.5, 3.0, 2.8, 0.7, "Consulter\nles metriques CTR")
    usecase(9.5, 1.8, 2.8, 0.7, "Configurer\nle guardrail")

    for uy in [7.5, 6.3, 5.1, 3.9, 2.7, 1.5]:
        assoc(1.2, 6.2, 4.2, uy)
    for uy in [7.5, 6.3, 5.1, 3.9, 2.7, 1.5]:
        assoc(1.2, 3.2, 4.2, uy)
    for uy in [7.5, 6.3, 5.1]:
        assoc(11.9, 6.2, 10.9, uy)
    for uy in [3.0, 1.8]:
        assoc(11.9, 3.2, 10.9, uy)

    save("uml_usecase.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. DIAGRAMME DE SEQUENCE — POST /recommend
# ══════════════════════════════════════════════════════════════════════════════
def fig_sequence():
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.set_title("Diagramme de Sequence — POST /recommend",
                 fontsize=13, fontweight="bold", color=BLUE, pad=10)

    participants = [
        (1.3,  "App Mobile",           BLUE),
        (4.0,  "FastAPI\n/recommend",  ORANGE),
        (7.0,  "Recommender",          TEAL),
        (10.5, "CandidateGen\n& Features", GREEN),
        (14.0, "ONNX Runtime",         PURPLE),
    ]
    TOP = 10.3

    for x, name, color in participants:
        rbox(ax, x - 0.75, TOP - 0.35, 1.5, 0.7, color, lw=1.5, pad=0.05)
        txt(ax, x, TOP, name, fs=9, bold=True)
        ax.plot([x, x], [0.5, TOP - 0.35], color=color, lw=1.0,
                linestyle="--", alpha=0.5, zorder=1)

    msgs = [
        (1.3,  4.0,  9.6, "POST /recommend  {code_client, k, variant}",        BLUE,   "→"),
        (4.0,  7.0,  9.0, "history_lookup.get(code_client)",                   ORANGE, "→"),
        (7.0,  4.0,  8.5, "[None]  →  _fallback(k)  →  mode: 'popularity'",   RED,    "←"),
        (7.0,  10.5, 8.0, "generate_candidates(history)",                       TEAL,   "→"),
        (10.5, 7.0,  7.5, "candidates : List[str]  (<=10)",                    GREEN,  "←"),
        (7.0,  10.5, 7.0, "compute_inference_row(history)",                     TEAL,   "→"),
        (10.5, 7.0,  6.5, "feat_row  (26 colonnes)",                           GREEN,  "←"),
        (7.0,  10.5, 6.0, "fast_preprocessor.encode(row_dict)",                TEAL,   "→"),
        (10.5, 14.0, 5.5, "session.run(['probabilities'],  X float32 [1x23])", GREEN,  "→"),
        (14.0, 10.5, 5.0, "proba[1011]",                                       PURPLE, "←"),
        (10.5, 7.0,  4.5, "argsort(-proba[cand_idx])[:k]  →  top-k",          GREEN,  "←"),
        (7.0,  4.0,  3.8, "{mode, recs, labels, variant, request_id}",         TEAL,   "←"),
        (4.0,  1.3,  3.1, "HTTP 200  JSON",                                    ORANGE, "←"),
    ]

    for x1, x2, y, label, color, direction in msgs:
        is_right = x2 > x1
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(
                        arrowstyle="-|>" if is_right else "<|-",
                        color=color, lw=1.5, mutation_scale=13), zorder=2)
        mx = (x1 + x2) / 2
        ax.text(mx, y + 0.14, label, ha="center", va="bottom",
                fontsize=7.8, color=color)

    # Boite de resultat total
    rbox(ax, 3.0, 0.5, 6.0, 0.65, "#E8F5E9", ec=GREEN, lw=1.5, pad=0.05)
    txt(ax, 6.0, 0.82, "Latence totale :  ~ 14 ms  (p50)  |  ~ 17 ms  (p99)",
        fs=9.5, color=GREEN, bold=True)

    # Axe temps
    ax.text(0.2, 9.5, "t", fontsize=11, color="#888")
    ax.annotate("", xy=(0.3, 0.8), xytext=(0.3, 9.4),
                arrowprops=dict(arrowstyle="-|>", color="#aaa", lw=1.2,
                                mutation_scale=10))

    save("uml_sequence.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. DIAGRAMME DE CLASSES
# ══════════════════════════════════════════════════════════════════════════════
def fig_classes():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("Diagramme de Classes Simplifie", fontsize=13,
                 fontweight="bold", color=BLUE, pad=10)

    def uml_class(x, y, w, title, attributes, methods, color=BLUE):
        h_title = 0.48
        h_attrs = len(attributes) * 0.33 + 0.12
        h_mths  = len(methods)    * 0.33 + 0.12
        h = h_title + h_attrs + h_mths

        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0",
                                   facecolor="white", edgecolor=color, linewidth=2))
        ax.add_patch(FancyBboxPatch((x, y + h - h_title), w, h_title,
                                   boxstyle="square,pad=0",
                                   facecolor=color, edgecolor=color))
        ax.text(x + w / 2, y + h - h_title / 2, title,
                ha="center", va="center", fontsize=9, fontweight="bold",
                color="white")
        ax.plot([x, x + w], [y + h_mths, y + h_mths], color=color, lw=1)
        for i, attr in enumerate(attributes):
            ax.text(x + 0.1, y + h_mths + h_attrs - 0.08 - i * 0.33,
                    f"  {attr}", fontsize=7.2, va="top", color="#333",
                    family="monospace")
        for i, mth in enumerate(methods):
            ax.text(x + 0.1, y + h_mths - 0.08 - i * 0.33,
                    f"  {mth}", fontsize=7.2, va="top", color="#222",
                    family="monospace")
        return x + w / 2, y + h, x + w / 2, y

    def assoc(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.2,
                                   mutation_scale=12))
        if label:
            ax.text((x1 + x2) / 2 + 0.05, (y1 + y2) / 2 + 0.05,
                    label, fontsize=7, color="#555", style="italic")

    uml_class(0.2, 4.2, 4.2, "Recommender",
        ["artifacts : TrainArtifacts",
         "history_lookup : dict[str, DataFrame]",
         "cold_start_rec : ColdStartRecommender",
         "onnx_session : InferenceSession | None",
         "fast_preprocessor : FastPreprocessor",
         "popularity : list[str]",
         "liaison_label_lookup : dict[str, str]"],
        ["+ from_paths(paths) : Recommender",
         "+ from_data(arts, df) : Recommender",
         "+ recommend(code_client, k) : dict",
         "- _fallback(k) : dict"],
        BLUE)

    uml_class(5.0, 6.5, 3.8, "TrainArtifacts",
        ["pipeline : Pipeline",
         "label_encoder : LabelEncoder"],
        ["+ load(path) : TrainArtifacts"],
        TEAL)

    uml_class(5.0, 3.6, 3.8, "FastPreprocessor",
        ["_cat_maps : dict[str, dict]",
         "_num_cols : list[str]",
         "_col_order : list[str]"],
        ["+ encode(row: dict) : ndarray",
         "+ from_column_transformer(ct)"],
        PURPLE)

    uml_class(5.0, 0.5, 3.8, "ColdStartRecommender",
        ["co_matrix : dict[str, Counter]"],
        ["+ recommend(history, k) : list",
         "+ from_path(path)",
         "+ save(path)"],
        ORANGE)

    uml_class(9.5, 6.2, 4.8, "CandidateGen",
        ["(fonctions module-level)"],
        ["+ generate_candidates(history,",
         "    user_id) : list[str]"],
        GREEN)

    uml_class(9.5, 3.5, 4.8, "features",
        ["(fonctions module-level)"],
        ["+ compute_inference_row(history)",
         "    : DataFrame",
         "+ build_training_rows(df)"],
        GREEN)

    uml_class(9.5, 0.5, 4.8, "popularity",
        ["(fonctions module-level)"],
        ["+ build_popularity_list(df)",
         "+ save_popularity(lst, path)",
         "+ load_popularity(path)"],
        "#795548")

    # Associations
    assoc(2.3, 8.28, 6.9, 7.75, "uses")
    assoc(2.3, 7.5, 6.9, 6.2, "uses")
    assoc(2.3, 6.0, 6.9, 4.1, "uses")
    assoc(4.4, 7.0, 9.5, 7.0, "calls")
    assoc(4.4, 6.5, 9.5, 5.0, "calls")
    assoc(4.4, 5.5, 9.5, 2.0, "calls")

    save("uml_classes.png")


# ══════════════════════════════════════════════════════════════════════════════
# 9. HISTOGRAMME — Distribution des liaisons
# ══════════════════════════════════════════════════════════════════════════════
def fig_hist_liaison():
    parquet = ROOT / "data" / "processed" / "oncf_clean.parquet"
    print("  → Chargement oncf_clean.parquet ...", end=" ", flush=True)
    df = pd.read_parquet(parquet, columns=["LiaisonId"])
    print("OK")

    counts = df["LiaisonId"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    top30 = counts.head(30)
    ax = axes[0]
    bars = ax.barh(range(len(top30)), top30.values, color=BLUE, alpha=0.85)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels([f"Liaison {lid}" for lid in top30.index], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Nombre de reservations", fontsize=9)
    ax.set_title("Top 30 liaisons par frequence", fontsize=10,
                 fontweight="bold", color=BLUE)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=7)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    ax2 = axes[1]
    ax2.hist(counts.values, bins=50, color=ORANGE, edgecolor="white",
             alpha=0.85, log=True)
    ax2.set_xlabel("Frequence d'une liaison (nb reservations)", fontsize=9)
    ax2.set_ylabel("Nombre de liaisons (echelle log)", fontsize=9)
    ax2.set_title(f"Distribution des frequences ({len(counts)} liaisons)",
                  fontsize=10, fontweight="bold", color=BLUE)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.axvline(counts.median(), color=GREEN, lw=2, linestyle="--",
                label=f"Mediane = {int(counts.median())}")
    ax2.axvline(counts.mean(), color=RED, lw=2, linestyle=":",
                label=f"Moyenne = {int(counts.mean())}")
    ax2.legend(fontsize=8)

    plt.suptitle("Distribution des liaisons par frequence de reservation",
                 fontsize=12, fontweight="bold", color=BLUE, y=1.01)
    plt.tight_layout()
    save("hist_liaison_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10. RAPPORT DE NETTOYAGE
# ══════════════════════════════════════════════════════════════════════════════
def fig_rapport_nettoyage():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    steps = [
        ("Dataset brut\n(oncf_data.csv)",      946155, GRAY_M),
        ("Apres propagation\ndes annulations",   640000, "#64B5F6"),
        ("Apres deduplication",                  510000, TEAL),
        ("Dataset propre\n(oncf_clean.parquet)", 491680, GREEN),
    ]
    max_h = max(s[1] for s in steps)
    for i, (label, val, color) in enumerate(steps):
        width = val / max_h
        left  = (1 - width) / 2
        ax.barh(i, width, left=left, height=0.6, color=color,
                edgecolor="white", linewidth=1.5)
        ax.text(0.5, i, f"{val:,}\n({val/946155*100:.1f}%)",
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white")
        ax.text(left - 0.02, i, label, ha="right", va="center", fontsize=8.5,
                color="#333")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(steps))
    ax.axis("off")
    ax.invert_yaxis()
    ax.set_title("Funnel de nettoyage", fontsize=10,
                 fontweight="bold", color=BLUE)

    ax2 = axes[1]
    categories = {
        "Reservations annulees\n(propagation statut)": 946155 - 640000,
        "Doublons consolides":                          640000 - 510000,
        "Utilisateurs cold start\n(< 3 voyages)":       510000 - 491680,
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
        at.set_fontsize(8)
        at.set_fontweight("bold")
    ax2.set_title("Repartition des lignes retirees\n(946 155 → 491 680 lignes)",
                  fontsize=10, fontweight="bold", color=BLUE)

    plt.suptitle("Bilan du nettoyage — Entrees et Sorties",
                 fontsize=12, fontweight="bold", color=BLUE)
    plt.tight_layout()
    save("rapport_nettoyage.png")


# ══════════════════════════════════════════════════════════════════════════════
# 11. ENCODAGE CYCLIQUE
# ══════════════════════════════════════════════════════════════════════════════
def fig_encodage_cyclique():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    configs = [
        ("Heure de depart", 24, "depart_hour"),
        ("Jour de la semaine", 7, "depart_dow"),
        ("Mois", 12, "depart_month"),
    ]
    for ax, (title, period, col) in zip(axes, configs):
        vals = np.arange(period)
        sin_v = np.sin(2 * np.pi * vals / period)
        cos_v = np.cos(2 * np.pi * vals / period)
        ax.plot(vals, sin_v, "o-", color=ORANGE, label="sin", lw=2, markersize=5)
        ax.plot(vals, cos_v, "s--", color=BLUE, label="cos", lw=2, markersize=5)
        ax_ins = ax.inset_axes([0.68, 0.55, 0.30, 0.40])
        theta = np.linspace(0, 2 * np.pi, 200)
        ax_ins.plot(np.cos(theta), np.sin(theta), color=GRAY_M, lw=1)
        for v in vals:
            angle = 2 * np.pi * v / period
            ax_ins.plot(np.cos(angle), np.sin(angle), ".", color=TEAL, ms=4)
        ax_ins.set_aspect("equal")
        ax_ins.axis("off")
        ax_ins.set_title("Cercle\nunite", fontsize=6)
        ax.set_xlabel(col.replace("_", " "), fontsize=9)
        ax.set_ylabel("Valeur encodee", fontsize=9)
        ax.set_title(f"Encodage cyclique\n{title} (T={period})",
                     fontsize=9.5, fontweight="bold", color=BLUE)
        ax.legend(fontsize=8)
        ax.grid(linestyle="--", alpha=0.3)
        ax.axhline(0, color="#ccc", lw=0.8)
        if period == 24:
            ax.annotate("Pas de discontinuite\n23h → 0h",
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
    print("  → Chargement features.parquet ...", end=" ", flush=True)
    df = pd.read_parquet(parquet, columns=["user_top_liaison_share"])
    s  = df["user_top_liaison_share"].dropna()
    print(f"OK  ({len(s):,} valeurs)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.hist(s, bins=40, color=BLUE, edgecolor="white", alpha=0.85)
    ax.axvline(s.mean(), color=ORANGE, lw=2, linestyle="--",
               label=f"Moyenne = {s.mean():.3f}")
    ax.axvline(s.median(), color=GREEN, lw=2, linestyle=":",
               label=f"Mediane = {s.median():.3f}")
    ax.set_xlabel("user_top_liaison_share", fontsize=9)
    ax.set_ylabel("Nombre de lignes", fontsize=9)
    ax.set_title("Distribution de user_top_liaison_share",
                 fontsize=10, fontweight="bold", color=BLUE)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax2 = axes[1]
    feat = pd.read_parquet(parquet,
                           columns=["user_top_liaison_share", "user_trip_index"])
    feat = feat.dropna()
    feat["segment"] = pd.cut(feat["user_trip_index"],
                             bins=[0, 2, 5, 20, 10000],
                             labels=["0-2 voyages", "3-5", "6-20", "21+"])
    groups = [feat.loc[feat["segment"] == seg, "user_top_liaison_share"].values
              for seg in ["0-2 voyages", "3-5", "6-20", "21+"]]
    bp = ax2.boxplot(groups, patch_artist=True, notch=False,
                     labels=["0-2 voyages", "3-5", "6-20", "21+"], widths=0.5)
    for patch, c in zip(bp["boxes"], ["#90CAF9", "#64B5F6", TEAL, BLUE]):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
    ax2.set_xlabel("Segment d'utilisateur (taille historique)", fontsize=9)
    ax2.set_ylabel("user_top_liaison_share", fontsize=9)
    ax2.set_title("user_top_liaison_share par segment",
                  fontsize=10, fontweight="bold", color=BLUE)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    plt.suptitle("Variable user_top_liaison_share — Loyaute utilisateur",
                 fontsize=12, fontweight="bold", color=BLUE)
    plt.tight_layout()
    save("dist_user_top_liaison_share.png")


# ══════════════════════════════════════════════════════════════════════════════
# 13. COLD START CF
# ══════════════════════════════════════════════════════════════════════════════
def fig_cold_start():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Module Cold Start — Filtrage Collaboratif par Co-occurrence",
                 fontsize=13, fontweight="bold", color=BLUE, pad=10)

    # Flux principal
    rbox(ax, 0.3, 3.8, 2.2, 1.2, ORANGE)
    txt(ax, 1.4, 4.4, "Utilisateur froid\n1-2 voyages", fs=9, bold=True)

    rbox(ax, 3.0, 3.8, 2.2, 1.2, TEAL)
    txt(ax, 4.1, 4.4, "Liaison(s) connue(s)\nexemple : L42", fs=9, bold=True)

    harrow(ax, 2.5, 3.0, 4.4, color=ORANGE, lw=2)
    ax.text(4.4, 5.25, "Recherche co-reservations",
            ha="center", fontsize=8.5, color=ORANGE)
    harrow(ax, 5.2, 6.2, 4.4, color=ORANGE, lw=2)

    # Matrice co-occurrence (centree)
    liaisons = ["L42", "L77", "L13", "L99", "L05"]
    co_matrix = np.array([
        [0,   312, 89,  45,  201],
        [312, 0,   67,  23,  145],
        [89,  67,  0,   188, 34 ],
        [45,  23,  188, 0,   12 ],
        [201, 145, 34,  12,  0  ],
    ])
    im = ax.imshow(co_matrix, cmap="Blues", aspect="auto",
                   extent=[6.2, 10.2, 2.3, 6.3])
    plt.colorbar(im, ax=ax, shrink=0.6, label="Co-occurrences", pad=0.01)
    for i in range(5):
        for j in range(5):
            val = co_matrix[i, j]
            ax.text(6.6 + j * 0.8, 5.9 - i * 0.8, str(val),
                    ha="center", va="center", fontsize=8,
                    color="white" if val > 150 else "#333", fontweight="bold")
    ax.set_xticks([6.6 + j * 0.8 for j in range(5)])
    ax.set_xticklabels(liaisons, fontsize=8.5)
    ax.set_yticks([5.9 - i * 0.8 for i in range(5)])
    ax.set_yticklabels(liaisons, fontsize=8.5)
    ax.text(8.2, 6.55, "Matrice de co-occurrence",
            ha="center", fontsize=9.5, fontweight="bold", color=BLUE)

    harrow(ax, 10.5, 11.2, 4.4, color=ORANGE, lw=2)
    ax.text(10.85, 4.85, "Top-k", ha="center", fontsize=8.5, color=ORANGE)
    rbox(ax, 11.2, 3.8, 1.5, 1.2, GREEN)
    txt(ax, 11.95, 4.4, "Recs\n[L77, L05]", fs=9, bold=True)

    # Table de strategie (bas gauche)
    ax.text(0.3, 3.2, "Strategie selon historique :", fontsize=9,
            fontweight="bold", color="#333")
    rows = [
        ("0 voyage",    "Popularite globale", ORANGE),
        ("1-2 voyages", "Cold Start CF",      TEAL),
        (">= 3 voyages","XGBoost ONNX",       BLUE),
    ]
    for i, (hist, mode, color) in enumerate(rows):
        rbox(ax, 0.3, 2.6 - i * 0.55, 1.8, 0.45, GRAY_M, ec="#999", lw=1)
        txt(ax, 1.2, 2.83 - i * 0.55, hist, fs=8, color="#333")
        rbox(ax, 2.15, 2.6 - i * 0.55, 2.0, 0.45, color)
        txt(ax, 3.15, 2.83 - i * 0.55, mode, fs=8, bold=True)

    save("cold_start_cf.png")


# ══════════════════════════════════════════════════════════════════════════════
# 14. ARCHITECTURE API FastAPI  — redesign en 3 colonnes
# ══════════════════════════════════════════════════════════════════════════════
def fig_archi_api():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("Architecture de l'API FastAPI — ONCF Recommender",
                 fontsize=13, fontweight="bold", color=BLUE, pad=10)

    # ── Colonne 1 : Clients ──────────────────────────────────────────────────
    rbox(ax, 0.3, 5.9, 2.0, 1.0, GRAY_D)
    txt(ax, 1.3, 6.4, "Application\nMobile ONCF", fs=9, bold=True)

    rbox(ax, 0.3, 4.2, 2.0, 1.0, "#795548")
    txt(ax, 1.3, 4.7, "Navigateur Web\n(Demo UI)", fs=9, bold=True)

    # ── Colonne 2 : FastAPI ──────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((3.0, 1.8), 4.5, 6.8,
                                boxstyle="round,pad=0.1",
                                facecolor=LGRAY, edgecolor=BLUE, linewidth=1.8, zorder=1))
    ax.text(5.25, 8.82, "FastAPI  —  apps/api/main.py",
            ha="center", fontsize=9.5, fontweight="bold", color=BLUE)

    endpoints = [
        (BLUE,    "GET /            (Demo page)"),
        (TEAL,    "GET /health"),
        (ORANGE,  "POST /recommend  ?variant=a|b"),
        (GREEN,   "GET /schedule/{id}"),
        (PURPLE,  "POST /feedback"),
    ]
    for i, (fc, label) in enumerate(endpoints):
        ey = 7.8 - i * 1.1
        rbox(ax, 3.2, ey - 0.35, 4.1, 0.7, fc, lw=1.5)
        txt(ax, 5.25, ey, label, fs=8.5, bold=True)

    ax.text(5.25, 2.2, "lifespan : charge les modeles\nune seule fois au demarrage",
            ha="center", fontsize=8, color=BLUE, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=BLUE, lw=1))

    # ── Colonne 3a : Recommenders ─────────────────────────────────────────────
    rbox(ax, 8.2, 6.5, 2.8, 1.2, BLUE)
    txt(ax, 9.6, 7.1, "Recommender A\n(prod)", fs=9, bold=True)

    rbox(ax, 8.2, 4.7, 2.8, 1.2, PURPLE)
    txt(ax, 9.6, 5.3, "Recommender B\n(challenger)", fs=9, bold=True)

    rbox(ax, 8.2, 2.5, 2.8, 1.2, RED)
    txt(ax, 9.6, 3.1, "Redis Cache\nTTL 1h  (horaires)", fs=9, bold=True)

    # ── Colonne 3b : ONNX / models ────────────────────────────────────────────
    rbox(ax, 11.5, 6.5, 2.3, 1.2, TEAL)
    txt(ax, 12.65, 7.1, "ONNX Runtime\n~3 ms  (prod)", fs=9, bold=True)

    rbox(ax, 11.5, 4.7, 2.3, 1.2, GRAY_D)
    txt(ax, 12.65, 5.3, "ONNX Runtime\n(challenger)", fs=9, bold=True)

    rbox(ax, 11.5, 2.5, 2.3, 1.2, "#37474F")
    txt(ax, 12.65, 3.1, "models/\n.onnx  .joblib", fs=9, bold=True)

    # ── Fleches ──────────────────────────────────────────────────────────────
    # Clients → FastAPI
    harrow(ax, 2.3, 3.0, 6.4, color=BLUE, lw=1.5)
    harrow(ax, 2.3, 3.0, 4.7, color="#795548", lw=1.5)

    # POST /recommend → Rec A / Rec B
    ax.annotate("", xy=(8.2, 7.1), xytext=(7.3, 7.45),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.5,
                                mutation_scale=12), zorder=2)
    ax.text(7.7, 7.5, "variant=a", fontsize=7.5, color=ORANGE, ha="center")

    ax.annotate("", xy=(8.2, 5.3), xytext=(7.3, 7.1),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.5,
                                mutation_scale=12), zorder=2)
    ax.text(7.4, 6.0, "variant=b", fontsize=7.5, color=ORANGE, ha="right")

    # GET /schedule → Redis
    ax.annotate("", xy=(8.2, 3.1), xytext=(7.3, 3.75),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5,
                                mutation_scale=12), zorder=2)

    # Rec A / Rec B → ONNX
    harrow(ax, 11.0, 11.5, 7.1, color=TEAL, lw=1.5)
    harrow(ax, 11.0, 11.5, 5.3, color=GRAY_D, lw=1.5)

    # ONNX → models (pointille)
    ax.plot([12.65, 12.65], [6.5, 3.7], color="#888", lw=1.2,
            linestyle="--", zorder=2)
    ax.annotate("", xy=(12.65, 3.7), xytext=(12.65, 4.7),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.2,
                                mutation_scale=10), zorder=2)

    save("archi_api.png")


# ══════════════════════════════════════════════════════════════════════════════
# 15. PROFILING LATENCE
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
    colors_b = [TEAL, GREEN, RED, BLUE, GRAY_D]
    colors_a = [TEAL, GREEN, GREEN, BLUE, GRAY_D]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, durations, colors, title in [
        (axes[0], durations_before, colors_b, "Avant optimisation (~23 ms total)"),
        (axes[1], durations_after,  colors_a, "Apres FastPreprocessor (~7 ms total)"),
    ]:
        bars = ax.barh(steps, durations, color=colors,
                       edgecolor="white", linewidth=1.5, height=0.55)
        ax.bar_label(bars, fmt="%.2f ms", padding=5, fontsize=8.5)
        ax.set_xlabel("Duree mediane (ms)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", color=BLUE)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_xlim(0, max(durations_before) * 1.5)
        ax.invert_yaxis()
    axes[0].axvline(11.3, color=RED, lw=1.5, linestyle=":", alpha=0.5,
                    label="Bottleneck 49%")
    axes[0].legend(fontsize=8)
    plt.suptitle("Decomposition de la latence par etape du pipeline d'inference",
                 fontsize=12, fontweight="bold", color=BLUE)
    plt.tight_layout()
    save("latence_profiling.png")


# ══════════════════════════════════════════════════════════════════════════════
# 16. FASTPREPROCESSOR vs SKLEARN
# ══════════════════════════════════════════════════════════════════════════════
def fig_fastpreprocessor():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    labels_comp = ["sklearn\nColumnTransformer\n.transform",
                   "FastPreprocessor\n(dict lookup)"]
    values_comp = [11.31, 0.0077]
    bars = ax.bar(labels_comp, values_comp, color=[RED, GREEN], width=0.5,
                  edgecolor="white", linewidth=1.5)
    ax.bar_label(bars, labels=[f"{v:.3f} ms" for v in values_comp],
                 padding=5, fontsize=10, fontweight="bold")
    ax.set_ylabel("Duree mediane (ms)", fontsize=9)
    ax.set_title("Latence de pretraitement", fontsize=10,
                 fontweight="bold", color=BLUE)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(0, 15)
    ax.text(0.5, 7.5, "x1469\nplus rapide !", ha="center",
            fontsize=14, color=GREEN, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor=LGRAY, edgecolor=GREEN))

    ax2 = axes[1]
    configs_e2e = ["Baseline\n(sklearn predict)", "+ ONNX\nRuntime",
                   "+ FastPreprocessor\n(final)"]
    p50_vals = [852, 56, 13.74]
    bars2 = ax2.bar(configs_e2e, p50_vals, color=[RED, ORANGE, GREEN], width=0.55,
                    edgecolor="white", linewidth=1.5)
    ax2.bar_label(bars2, labels=[f"{v:.1f} ms" for v in p50_vals],
                  padding=5, fontsize=9, fontweight="bold")
    ax2.set_ylabel("Latence p50 (ms)", fontsize=9)
    ax2.set_title("Evolution de la latence API p50\n(speedup total x62)",
                  fontsize=10, fontweight="bold", color=BLUE)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.axhline(100, color=ORANGE, lw=2, linestyle="--", label="Seuil p99 < 100ms")
    ax2.axhline(50, color=GREEN, lw=2, linestyle=":", label="Seuil p50 < 50ms")
    ax2.legend(fontsize=8)

    plt.suptitle("FastPreprocessor — Optimisation du Pretraitement sur le Chemin Chaud",
                 fontsize=12, fontweight="bold", color=BLUE)
    plt.tight_layout()
    save("fastpreprocessor_vs_sklearn.png")


# ══════════════════════════════════════════════════════════════════════════════
# 17. PIPELINE DE REENTRAINEMENT  — redesign vertical propre
# ══════════════════════════════════════════════════════════════════════════════
def fig_retrain_pipeline():
    fig, ax = plt.subplots(figsize=(12, 11))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.set_title("Pipeline de Reentrainement Automatique avec Guardrail KPI",
                 fontsize=13, fontweight="bold", color=BLUE, pad=12)

    CX   = 5.5    # centre colonne principale
    BW   = 3.6    # largeur boite
    BH   = 0.85   # hauteur boite
    X0   = CX - BW / 2

    # Positions y (bas de chaque boite)
    Y_TRIG  = 9.8
    Y_S1    = 8.4
    Y_S2    = 7.0
    Y_S3    = 5.6
    D_CY    = 4.3   # centre du diamant
    D_HW    = 2.1   # demi-largeur diamant
    D_HH    = 0.65  # demi-hauteur diamant
    Y_S4    = 2.9
    Y_S5    = 1.5

    # ── Boites principales ────────────────────────────────────────────────────
    # Trigger
    rbox(ax, X0, Y_TRIG, BW, BH, GRAY_D)
    txt(ax, CX, Y_TRIG + BH / 2,
        "Declenchement : 02h00 quotidien\n(Windows Task Scheduler)", fs=9, bold=True)

    # Step 1
    rbox(ax, X0, Y_S1, BW, BH, TEAL)
    txt(ax, CX, Y_S1 + BH / 2,
        "(1)  Chargement des donnees\n(option : --window-months N)", fs=9, bold=True)

    # Step 2
    rbox(ax, X0, Y_S2, BW, BH, BLUE)
    txt(ax, CX, Y_S2 + BH / 2,
        "(2)  Entrainement XGBoost\n(~43 min CPU,  n_estimators=200)", fs=9, bold=True)

    # Step 3
    rbox(ax, X0, Y_S3, BW, BH, PURPLE)
    txt(ax, CX, Y_S3 + BH / 2,
        "(3)  Evaluation :  HR@1,  HR@3,  MRR@3\n(jeu de test temporel)", fs=9, bold=True)

    # Diamant Guardrail
    dmnd(ax, CX, D_CY, D_HW, D_HH, ORANGE, lw=2)
    txt(ax, CX, D_CY, "Guardrail\nHR@1 chute > 5 pts ?", fs=8.5, bold=True)

    # Step 4  (branche PASS, vers le bas)
    rbox(ax, X0, Y_S4, BW, BH, TEAL)
    txt(ax, CX, Y_S4 + BH / 2,
        "(4)  Ecriture modele challenger\n(models/xgb_ranker_challenger.*)", fs=9, bold=True)

    # Step 5
    rbox(ax, X0, Y_S5, BW, BH, GREEN)
    txt(ax, CX, Y_S5 + BH / 2,
        "(5)  Promotion challenger → prod\n(remplacement modele de production)", fs=9, bold=True)

    # Boite FAIL (a droite du diamant)
    FAIL_CX = 9.8
    FAIL_BW = 2.3
    FAIL_BH = 1.0
    rbox(ax, FAIL_CX - FAIL_BW / 2, D_CY - FAIL_BH / 2, FAIL_BW, FAIL_BH, RED)
    txt(ax, FAIL_CX, D_CY,
        "Promotion BLOQUEE\nAlerte dans api.log\nChallenger conserve", fs=8)

    # ── Fleches principales ───────────────────────────────────────────────────
    varrow(ax, CX, Y_TRIG, Y_S1 + BH, color=GRAY_D, lw=2)
    varrow(ax, CX, Y_S1,   Y_S2 + BH, color=TEAL,   lw=2)
    varrow(ax, CX, Y_S2,   Y_S3 + BH, color=BLUE,   lw=2)
    varrow(ax, CX, Y_S3,   D_CY + D_HH, color=PURPLE, lw=2)

    # PASS : bas du diamant → S4
    varrow(ax, CX, D_CY - D_HH, Y_S4 + BH, color=GREEN, lw=2)
    ax.text(CX + 0.15, (D_CY - D_HH + Y_S4 + BH) / 2, "NON  (PASS)",
            fontsize=8.5, color=GREEN, va="center")

    varrow(ax, CX, Y_S4, Y_S5 + BH, color=GREEN, lw=2)

    # FAIL : droite du diamant → boite FAIL
    harrow(ax, CX + D_HW, FAIL_CX - FAIL_BW / 2, D_CY, color=RED, lw=2)
    ax.text((CX + D_HW + FAIL_CX - FAIL_BW / 2) / 2, D_CY + 0.18,
            "OUI (FAIL)", fontsize=8.5, color=RED, ha="center")

    # ── Panneaux lateraux ──────────────────────────────────────────────────────
    # Rolling Window (gauche, aligne sur S1)
    ax.text(0.3, Y_S1 + BH / 2,
            "Option Rolling Window :\n  --window-months 24\n  → 160 K lignes\n  → ~18 min",
            fontsize=8, color=TEAL, va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc=LGRAY, ec=TEAL, lw=1.2))
    ax.annotate("", xy=(X0, Y_S1 + BH / 2), xytext=(2.4, Y_S1 + BH / 2),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.0,
                                linestyle="dashed"), zorder=1)

    # Seuils cibles (droite, aligne sur S3)
    ax.text(9.6, Y_S3 + BH / 2,
            "Seuils cibles :\n  HR@1  >  50 %\n  HR@3  >  60 %\n  MRR@3 >  60 %",
            fontsize=8, color=GREEN, va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="#E8F5E9", ec=GREEN, lw=1.2))
    ax.annotate("", xy=(X0 + BW, Y_S3 + BH / 2), xytext=(9.35, Y_S3 + BH / 2),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0,
                                linestyle="dashed"), zorder=1)

    save("retrain_pipeline.png")


# ══════════════════════════════════════════════════════════════════════════════
# 18. FRAMEWORK A/B TESTING  — redesign flux organise
# ══════════════════════════════════════════════════════════════════════════════
def fig_ab_testing():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Framework A/B Testing — Architecture et Flux",
                 fontsize=13, fontweight="bold", color=BLUE, pad=10)

    CX = 6.5   # centre colonne principale

    # ── Boites du flux principal ───────────────────────────────────────────────
    # App Mobile
    rbox(ax, CX - 1.4, 9.0, 2.8, 0.75, GRAY_D)
    txt(ax, CX, 9.38, "Application Mobile ONCF", fs=9.5, bold=True)

    # FastAPI Router
    varrow(ax, CX, 9.0, 8.5, color=ORANGE, lw=2)
    rbox(ax, CX - 2.2, 7.6, 4.4, 0.75, BLUE)
    txt(ax, CX, 7.975, "FastAPI  —  POST /recommend?variant=a|b", fs=9, bold=True)

    # Split labels
    ax.text(3.8, 7.4, "variant=a", fontsize=8.5, color=BLUE, ha="center",
            style="italic")
    ax.text(9.2, 7.4, "variant=b", fontsize=8.5, color=PURPLE, ha="center",
            style="italic")

    # Recommender A (gauche)
    darrow(ax, CX - 1.5, 7.6, 3.0 + 1.7, 6.75, color=BLUE, lw=1.8)
    rbox(ax, 1.5, 5.4, 3.3, 1.2, BLUE)
    txt(ax, 3.15, 6.0,
        "Recommender A  (prod)\nxgb_ranker.onnx\n148 Mo", fs=8.5, bold=True)

    # Recommender B (droite)
    darrow(ax, CX + 1.5, 7.6, 9.0 - 1.7, 6.75, color=PURPLE, lw=1.8)
    rbox(ax, 8.2, 5.4, 3.3, 1.2, PURPLE)
    txt(ax, 9.85, 6.0,
        "Recommender B  (challenger)\nxgb_ranker_challenger.onnx", fs=8.5, bold=True)

    # Reponse (converge)
    darrow(ax, 3.15, 5.4, CX - 2.2, 4.75, color=BLUE, lw=1.8)
    darrow(ax, 9.85, 5.4, CX + 2.2, 4.75, color=PURPLE, lw=1.8)
    rbox(ax, CX - 2.5, 3.9, 5.0, 0.75, TEAL)
    txt(ax, CX, 4.275,
        "Reponse JSON  {mode, recs, labels, variant, request_id  (UUID4)}",
        fs=8.5, bold=True)

    # Feedback
    varrow(ax, CX, 3.9, 3.4, color=ORANGE, lw=2)
    rbox(ax, CX - 2.2, 2.6, 4.4, 0.65, ORANGE)
    txt(ax, CX, 2.925,
        "POST /feedback  {request_id,  liaison_id,  clicked}", fs=8.5, bold=True)

    # api.log
    varrow(ax, CX, 2.6, 2.1, color=GRAY_D, lw=2)
    rbox(ax, CX - 2.0, 1.35, 4.0, 0.65, GRAY_D)
    txt(ax, CX, 1.68,
        "api.log  (JSON structure)  —  code_client absent des logs", fs=8.5, bold=True)

    # CTR Analysis + Promotion
    darrow(ax, CX - 1.0, 1.35, 5.0, 0.9, color=GREEN, lw=1.8)
    rbox(ax, 2.8, 0.2, 3.8, 0.65, GREEN)
    txt(ax, 4.7, 0.525,
        "Analyse CTR offline  —  jointure request_id\nVariant A vs B", fs=8, bold=True)

    harrow(ax, 2.8, 1.6, 0.525, color=GREEN, lw=1.8)
    rbox(ax, 0.2, 0.2, 1.35, 0.65, ORANGE)
    txt(ax, 0.875, 0.525, "Promotion\nsi CTR(B) > CTR(A)", fs=7.5, bold=True)

    # ── Panneau lateral droit : protocole + loi ───────────────────────────────
    rbox(ax, 10.8, 0.2, 2.9, 7.2, "#1A237E", ec=WHITE, lw=1.5, pad=0.1)
    ax.text(12.25, 7.15, "Protocole A/B", ha="center",
            fontsize=9, fontweight="bold", color=WHITE)
    steps_txt = [
        "1. Retrain → challenger.*",
        "2. Redemarrage API",
        "   (lifespan recharge)",
        "3. 50 % traffic → variant=a",
        "   50 % traffic → variant=b",
        "4. 7 jours → analyse CTR",
        "5. Si CTR(B) > CTR(A)",
        "   → promotion challenger",
    ]
    for i, line in enumerate(steps_txt):
        ax.text(11.0, 6.7 - i * 0.5, line, fontsize=7.8, color=WHITE, va="top")

    ax.plot([10.8, 13.7], [3.5, 3.5], color=WHITE, lw=0.8, linestyle="--")
    ax.text(12.25, 3.3, "Conformite Loi 09-08", ha="center",
            fontsize=8, fontweight="bold", color="#FFCC02")
    ax.text(11.0, 2.9,
            "code_client : jamais dans\nles logs ni la reponse.\n"
            "request_id  = correlateur\nanonymise (UUID4).",
            fontsize=7.8, color=WHITE, va="top")

    save("ab_testing.png")


# ══════════════════════════════════════════════════════════════════════════════
# 19. METRIQUES PAR SEGMENT
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
    ax.set_xticks(x)
    ax.set_xticklabels(segments, fontsize=9)
    ax.set_ylabel("Metrique (%)", fontsize=10)
    ax.set_ylim(50, 100)
    ax.set_title("Metriques par Segment d'Utilisateurs (Sprint 2)",
                 fontsize=12, fontweight="bold", color=BLUE)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.axhline(76.28, color=BLUE, lw=1.5, linestyle=":", alpha=0.5)
    ax.text(3.7, 77.5, "HR@1 moy=76.3%", fontsize=8, color=BLUE, style="italic")
    plt.tight_layout()
    save("metrics_segment.png")


# ══════════════════════════════════════════════════════════════════════════════
# 20. COMPARAISON SPRINT 1 vs SPRINT 2 vs BASELINES
# ══════════════════════════════════════════════════════════════════════════════
def fig_metrics_comparison():
    models = ["global_top\n(popularite)", "prev_liaison\n(dernier trajet)",
              "most_frequent\n(freq+recence)", "XGBoost\nSprint 1",
              "XGBoost\nSprint 2"]
    hr1  = [3.99,  26.20, 27.51, 73.95, 76.28]
    hr3  = [11.25, 32.04, 51.28, 88.77, 90.55]
    mrr3 = [7.07,  28.81, 38.65, 80.64, 82.77]

    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - width, hr1,  width, label="HR@1",  color=BLUE,   edgecolor="white", lw=1.2)
    b2 = ax.bar(x,          hr3,  width, label="HR@3",  color=ORANGE, edgecolor="white", lw=1.2)
    b3 = ax.bar(x + width,  mrr3, width, label="MRR@3", color=GREEN,  edgecolor="white", lw=1.2)
    ax.bar_label(b1, fmt="%.1f%%", fontsize=7, padding=2)
    ax.bar_label(b2, fmt="%.1f%%", fontsize=7, padding=2)
    ax.bar_label(b3, fmt="%.1f%%", fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Metrique (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title("Comparaison XGBoost vs Baselines (Sprint 1 & Sprint 2)",
                 fontsize=12, fontweight="bold", color=BLUE)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.axvline(2.5, color=RED, lw=2, linestyle="--", alpha=0.4)
    ax.text(2.6, 95, "XGBoost", fontsize=9, color=RED, style="italic")
    plt.tight_layout()
    save("metrics_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{'=' * 60}")
    print("  Generation des figures — Rapport PFA ONCF (v2)")
    print(f"  Output : {PIC}")
    print(f"{'=' * 60}\n")

    tasks = [
        ("zero_click_concept.png",         fig_zero_click),
        ("gantt.png",                       fig_gantt),
        ("archi_globale.png",               fig_archi_globale),
        ("uml_composants.png",              fig_uml_composants),
        ("archi_deux_etapes.png",           fig_archi_deux_etapes),
        ("uml_usecase.png",                 fig_uml_usecase),
        ("uml_sequence.png",                fig_sequence),
        ("uml_classes.png",                 fig_classes),
        ("rapport_nettoyage.png",           fig_rapport_nettoyage),
        ("encodage_cyclique.png",           fig_encodage_cyclique),
        ("cold_start_cf.png",               fig_cold_start),
        ("archi_api.png",                   fig_archi_api),
        ("latence_profiling.png",           fig_latence_profiling),
        ("fastpreprocessor_vs_sklearn.png", fig_fastpreprocessor),
        ("retrain_pipeline.png",            fig_retrain_pipeline),
        ("ab_testing.png",                  fig_ab_testing),
        ("metrics_segment.png",             fig_metrics_segment),
        ("metrics_comparison.png",          fig_metrics_comparison),
    ]

    data_tasks = [
        ("hist_liaison_distribution.png",   fig_hist_liaison),
        ("dist_user_top_liaison_share.png", fig_user_top_liaison_share),
    ]

    errors = []
    for name, fn in tasks:
        try:
            print(f"  Generation : {name}")
            fn()
        except Exception as e:
            print(f"  ERREUR : {e}")
            errors.append((name, str(e)))

    print("\n  --- Figures avec donnees reelles ---")
    for name, fn in data_tasks:
        try:
            print(f"  Generation : {name}")
            fn()
        except Exception as e:
            print(f"  ERREUR : {e}")
            errors.append((name, str(e)))

    print(f"\n{'=' * 60}")
    if errors:
        print(f"  {len(errors)} erreur(s) :")
        for name, msg in errors:
            print(f"    - {name} : {msg}")
    else:
        print("  Toutes les figures generees avec succes.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
