r"""Génère toutes les figures scriptables du rapport PFA ONCF dans pic/.

Les noms de fichiers sont IDENTIQUES à ceux référencés dans rapport_pfa_v2.tex
(\graphicspath{{pic/}}) pour un remplacement direct.

Couvre 20 figures :
  Graphes (données)      : gantt, hist_liaison_distribution, rapport_nettoyage,
                           encodage_cyclique, metrics_comparison,
                           dist_user_top_liaison_share, metrics_segment,
                           latence_profiling, fastpreprocessor_vs_sklearn
  Schémas (architecture) : zero_click_concept, archi_globale, archi_deux_etapes,
                           archi_api, cold_start_cf, retrain_pipeline, ab_testing
  UML                    : uml_composants, uml_usecase, uml_sequence, uml_classes

NON générées (à fournir à la main) : oncf.png, LogoFsr.png (logos),
  github_actions_ci.png, pytest_output.png, task_scheduler.png (captures écran).

Lancer :  .venv\\Scripts\\python.exe scripts/generate_report_figures.py
"""
from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
)

ROOT = Path(__file__).resolve().parents[1]
PIC = ROOT / "pic"
PIC.mkdir(exist_ok=True)
CLEAN_DIR = ROOT / "data" / "clean" / "parquet"
FEAT_DIR  = ROOT / "data" / "features" / "parquet"

# ── Palette ONCF ───────────────────────────────────────────────────
ORANGE = "#E65300"
BLUE = "#004B8D"
LBLUE = "#E8F1F8"
LOR = "#FBE3D5"
GREEN = "#2E7D32"
LGREEN = "#E4F1E4"
RED = "#C62828"
LRED = "#FBE3E3"
GREY = "#ECEEF1"
DGREY = "#6B7280"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "savefig.dpi": 170,
        "axes.edgecolor": "#444",
    }
)


# ── Boîte à outils diagrammes ──────────────────────────────────────
def canvas(xmax=100, ymax=70, figw=11):
    figh = figw * ymax / xmax
    fig, ax = plt.subplots(figsize=(figw, figh))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _wrap(t, w):
    return "\n".join(textwrap.wrap(t, w)) if w else t


def box(ax, cx, cy, w, h, text, fc=LBLUE, ec=BLUE, fs=11, weight="normal",
        tc="#111", wrapw=0, round=True):
    bs = "round,pad=0,rounding_size=1.4" if round else "square,pad=0"
    ax.add_patch(
        FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=bs,
                       lw=1.5, ec=ec, fc=fc, zorder=3, mutation_aspect=1)
    )
    ax.text(cx, cy, _wrap(text, wrapw), ha="center", va="center",
            fontsize=fs, weight=weight, color=tc, zorder=4)


def arrow(ax, p1, p2, color=ORANGE, lw=2.0, style="-|>", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16, lw=lw,
                        color=color, connectionstyle=f"arc3,rad={rad}",
                        zorder=2, shrinkA=0, shrinkB=0)
    )


def title(ax, t, ymax=70):
    ax.text(50, ymax - 2, t, ha="center", va="top", fontsize=14,
            weight="bold", color=BLUE)


def save(fig, name):
    fig.savefig(PIC / name, bbox_inches="tight", facecolor="white", pad_inches=0.2)
    plt.close(fig)
    print(f"  [ok] {name}")


# ===================================================================
# 1. GRAPHES DE DONNÉES
# ===================================================================
def fig_gantt():
    base = date(2026, 3, 16)
    phases = [
        ("Cadrage & étude de l'existant", date(2026, 3, 16), date(2026, 3, 29), DGREY),
        ("Sprint 1 : Données & Modèle v1", date(2026, 3, 30), date(2026, 4, 26), BLUE),
        ("Sprint 2 : Architecture & API", date(2026, 4, 27), date(2026, 5, 24), BLUE),
        ("Phase 3 : Production & A/B Test", date(2026, 5, 25), date(2026, 6, 8), ORANGE),
        ("Consolidation & rapport", date(2026, 6, 9), date(2026, 6, 16), GREEN),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, (name, s, e, c) in enumerate(phases):
        y = len(phases) - 1 - i
        start = (s - base).days
        dur = (e - s).days + 1
        ax.barh(y, dur, left=start, height=0.55, color=c, edgecolor="white", zorder=3)
        ax.text(start + dur / 2, y, f"{dur} j", ha="center", va="center",
                color="white", fontsize=9, weight="bold", zorder=4)
        ax.text(-2, y, name, ha="right", va="center", fontsize=10)
    # axes mensuels
    ticks, labels = [], []
    for m, lbl in [(date(2026, 3, 16), "16 mars"), (date(2026, 3, 30), "30 mars"),
                   (date(2026, 4, 13), "13 avr"), (date(2026, 4, 27), "27 avr"),
                   (date(2026, 5, 11), "11 mai"), (date(2026, 5, 25), "25 mai"),
                   (date(2026, 6, 8), "8 juin"), (date(2026, 6, 16), "16 juin")]:
        ticks.append((m - base).days)
        labels.append(lbl)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([])
    ax.set_xlim(-30, 95)
    ax.set_ylim(-0.6, len(phases) - 0.3)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", ls=":", color="#bbb", zorder=0)
    ax.set_title("Planning du stage — 13 semaines (16 mars → 16 juin 2026)",
                 color=BLUE, weight="bold", fontsize=13, pad=12)
    save(fig, "gantt.png")


def fig_hist_liaison_distribution():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    try:
        s = pd.read_parquet(CLEAN_DIR / "oncf_clean.parquet", columns=["LiaisonId"])["LiaisonId"]
        counts = s.astype(str).value_counts().head(25)
        src = "Données réelles (oncf_clean.parquet)"
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] oncf_clean.parquet illisible ({exc}) — données synthétiques")
        rng = np.random.default_rng(42)
        vals = np.sort(rng.zipf(1.6, 25))[::-1] * 800
        counts = pd.Series(vals, index=[f"L{i}" for i in range(25)])
        src = "Données synthétiques (parquet absent)"
    ax.bar(range(len(counts)), counts.values, color=BLUE, edgecolor="white", zorder=3)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Nombre de réservations")
    ax.set_xlabel("LiaisonId (top 25)")
    ax.set_title("Distribution des liaisons par fréquence de réservation\n"
                 "(longue traîne : quelques navettes dominent)",
                 color=BLUE, weight="bold", fontsize=12)
    ax.grid(axis="y", ls=":", color="#bbb", zorder=0)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.text(0.99, 0.95, src, transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=DGREY, style="italic")
    save(fig, "hist_liaison_distribution.png")


def fig_rapport_nettoyage():
    # Waterfall : 946 155 -> 491 680
    steps = [
        ("Lignes\nbrutes", 946_155, BLUE),
        ("Annulations\n(-38 %)", -172_701, RED),
        ("Cold start\n(-31 %)", -140_887, ORANGE),
        ("Doublons\n(-19 %)", -86_350, "#8E5BB5"),
        ("Incomplètes\n(-12 %)", -54_537, DGREY),
        ("Dataset\npropre", 491_680, GREEN),
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    running = 0
    for i, (label, val, c) in enumerate(steps):
        if i == 0 or i == len(steps) - 1:
            ax.bar(i, val, color=c, edgecolor="white", zorder=3)
            ax.text(i, val + 15000, f"{val:,}".replace(",", " "), ha="center",
                    fontsize=9, weight="bold")
            running = val if i == 0 else running
        else:
            ax.bar(i, val, bottom=running + val, color=c, edgecolor="white", zorder=3)
            ax.text(i, running + val / 2, f"{val:,}".replace(",", " "), ha="center",
                    va="center", fontsize=8, color="white", weight="bold")
            running += val
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], fontsize=9)
    ax.set_ylabel("Nombre de lignes")
    ax.set_title("Bilan du nettoyage : 946 155 → 491 680 lignes "
                 "(taux de conservation 51,9 %)", color=BLUE, weight="bold", fontsize=12)
    ax.grid(axis="y", ls=":", color="#bbb", zorder=0)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    save(fig, "rapport_nettoyage.png")


def fig_encodage_cyclique():
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    ax.set_aspect("equal")
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="#cfd8e3", lw=1.5, zorder=1)
    for h in range(24):
        a = 2 * np.pi * h / 24
        x, y = np.cos(a), np.sin(a)
        big = h in (0, 6, 12, 18, 23)
        ax.scatter([x], [y], s=90 if big else 45,
                   color=ORANGE if big else BLUE, zorder=3)
        if big:
            ax.annotate(f"{h}h", (x, y), (x * 1.18, y * 1.18),
                        ha="center", va="center", fontsize=11, weight="bold")
    # met en évidence l'adjacence 23h <-> 0h
    for h in (23, 0):
        a = 2 * np.pi * h / 24
        ax.plot([0, np.cos(a)], [0, np.sin(a)], color=ORANGE, ls="--", lw=1.2, zorder=2)
    ax.text(0, -1.42, "23h et 0h sont adjacents sur le cercle\n"
            "→ pas de discontinuité artificielle", ha="center", fontsize=10,
            color=DGREY)
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.65, 1.55)
    ax.axis("off")
    ax.set_title("Encodage cyclique de l'heure  (sin, cos)", color=BLUE,
                 weight="bold", fontsize=13)
    save(fig, "encodage_cyclique.png")


def _grouped_metrics(ax, models, hr1, hr3, mrr, colors):
    x = np.arange(len(models))
    w = 0.26
    ax.bar(x - w, hr1, w, label="HR@1", color=colors[0], zorder=3)
    ax.bar(x, hr3, w, label="HR@3", color=colors[1], zorder=3)
    ax.bar(x + w, mrr, w, label="MRR@3", color=colors[2], zorder=3)
    for xi, vals in zip(x, zip(hr1, hr3, mrr)):
        for dx, v in zip((-w, 0, w), vals):
            ax.text(xi + dx, v + 1.3, f"{v:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", ls=":", color="#bbb", zorder=0)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=9)


def fig_metrics_comparison():
    fig, ax = plt.subplots(figsize=(10, 5))
    models = ["global_top", "prev_liaison", "most_frequent", "XGBoost\n(Sprint 1)"]
    _grouped_metrics(
        ax, models,
        [3.99, 26.20, 27.51, 73.95],
        [11.25, 32.04, 51.28, 88.77],
        [7.07, 28.81, 38.65, 80.64],
        [BLUE, ORANGE, GREEN],
    )
    ax.set_title("XGBoost vs baselines — supériorité sur les trois métriques\n"
                 "(×2,69 vs la meilleure baseline sur HR@1)", color=BLUE,
                 weight="bold", fontsize=12)
    save(fig, "metrics_comparison.png")


def fig_metrics_segment():
    fig, ax = plt.subplots(figsize=(10, 5))
    segs = ["0–2\nvoyages", "3–5\nvoyages", "6–20\nvoyages", "21+\n(fidèles)"]
    _grouped_metrics(
        ax, segs,
        [73.9, 75.2, 77.9, 82.7],
        [89.3, 90.4, 91.6, 93.1],
        [80.9, 82.1, 84.1, 87.4],
        [BLUE, ORANGE, GREEN],
    )
    ax.set_ylim(60, 100)
    ax.set_title("Métriques par profondeur d'historique\n"
                 "amélioration monotone → le modèle exploite l'historique individuel",
                 color=BLUE, weight="bold", fontsize=12)
    save(fig, "metrics_segment.png")


def fig_dist_user_top_liaison_share():
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    try:
        s = pd.read_parquet(FEAT_DIR / "oncf_features.parquet",
                            columns=["user_top_liaison_share"])["user_top_liaison_share"]
        s = pd.to_numeric(s, errors="coerce").dropna()
        src = "Données réelles (features.parquet)"
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] features.parquet illisible ({exc}) — données synthétiques")
        rng = np.random.default_rng(1)
        s = pd.Series(np.clip(rng.beta(2, 1.2, 50000), 0, 1))
        src = "Données synthétiques (parquet absent)"
    ax.hist(s, bins=24, color=ORANGE, edgecolor="white", zorder=3)
    ax.set_xlabel("user_top_liaison_share (part de la liaison dominante)")
    ax.set_ylabel("Nombre de lignes")
    ax.set_title("Distribution de user_top_liaison_share\n"
                 "pic à droite = voyageurs fidèles à un seul trajet",
                 color=BLUE, weight="bold", fontsize=12)
    ax.grid(axis="y", ls=":", color="#bbb", zorder=0)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.text(0.02, 0.95, src, transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color=DGREY, style="italic")
    save(fig, "dist_user_top_liaison_share.png")


def fig_latence_profiling():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    steps = ["Scoring & tri\ndes candidats", "compute_\ninference_row",
             "Candidate\ngeneration", "ONNX inference\n(session.run)",
             "ColumnTransformer\n.transform"]
    vals = [0.002, 1.4, 2.7, 3.2, 11.3]
    cols = [DGREY, BLUE, BLUE, GREEN, RED]
    ax.barh(range(len(steps)), vals, color=cols, edgecolor="white", zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + 0.15, i, f"{v} ms", va="center", fontsize=9, weight="bold")
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels(steps, fontsize=9)
    ax.set_xlabel("Latence (ms) — médiane sur 100 appels")
    ax.set_xlim(0, 13.5)
    ax.set_title("Décomposition du chemin d'inférence (ONNX actif)\n"
                 "→ ColumnTransformer = 49 % du temps : le goulot à éliminer",
                 color=BLUE, weight="bold", fontsize=12)
    ax.grid(axis="x", ls=":", color="#bbb", zorder=0)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    save(fig, "latence_profiling.png")


def fig_fastpreprocessor_vs_sklearn():
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    labels = ["ColumnTransformer\n.transform", "FastPreprocessor\n(dict lookup)"]
    vals = [11.3, 0.01]
    ax.bar(labels, vals, color=[RED, GREEN], edgecolor="white", width=0.55, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("Latence de l'encodage (ms, échelle log)")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.4, f"{v} ms", ha="center", fontsize=11, weight="bold")
    ax.text(0.5, 0.5, "× ~1 130\nplus rapide", transform=ax.transAxes,
            ha="center", va="center", fontsize=13, weight="bold", color=BLUE,
            bbox=dict(boxstyle="round", fc=LBLUE, ec=BLUE))
    ax.set_title("FastPreprocessor vs ColumnTransformer.transform",
                 color=BLUE, weight="bold", fontsize=12)
    ax.grid(axis="y", ls=":", color="#bbb", zorder=0)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    save(fig, "fastpreprocessor_vs_sklearn.png")


# ===================================================================
# 2. SCHÉMAS D'ARCHITECTURE
# ===================================================================
def fig_zero_click_concept():
    fig, ax = canvas(100, 62, figw=12)
    title(ax, "Concept Zero-Click Search : 5 étapes → 0 saisie", ymax=62)
    # Avant (téléphone gauche)
    box(ax, 24, 30, 30, 44, "", fc="white", ec="#888")
    ax.text(24, 49, "AVANT", ha="center", fontsize=12, weight="bold", color=RED)
    etapes = ["1. Saisir gare de départ", "2. Saisir gare d'arrivée",
              "3. Sélectionner une date", "4. Lancer la recherche",
              "5. Choisir un train"]
    for i, e in enumerate(etapes):
        box(ax, 24, 44 - i * 5.2, 26, 4, e, fc=LRED, ec=RED, fs=8.5)
    # flèche
    arrow(ax, (40, 30), (60, 30), color=ORANGE, lw=3)
    ax.text(50, 34, "modèle\nprédictif", ha="center", fontsize=9, color=ORANGE, weight="bold")
    # Après (téléphone droit)
    box(ax, 76, 30, 30, 44, "", fc="white", ec="#888")
    ax.text(76, 49, "APRÈS", ha="center", fontsize=12, weight="bold", color=GREEN)
    box(ax, 76, 40, 26, 6, "Bonjour,\nVotre trajet habituel ?", fc=LBLUE, ec=BLUE, fs=8.5)
    box(ax, 76, 30, 26, 8, "Casa Voyageurs → Rabat Agdal\n07:00  •  Confirmer",
        fc=LGREEN, ec=GREEN, fs=8.5, weight="bold")
    box(ax, 76, 20, 26, 5, "Suggestion proactive à l'ouverture", fc=GREY, ec="#888", fs=8)
    save(fig, "zero_click_concept.png")


def fig_archi_globale():
    fig, ax = canvas(100, 78, figw=9)
    title(ax, "Architecture globale — 4 couches", ymax=78)
    layers = [
        ("Couche API  —  FastAPI", "/recommend   /feedback   /schedule   /health", LBLUE, BLUE, 60),
        ("Couche Logique  —  rec_oncf", "recommender · candidates · features · cold_start · training", LOR, ORANGE, 45),
        ("Couche Modèle  —  models/", "xgb_ranker.onnx · label_encoder · cold_start · popularity", LGREEN, GREEN, 30),
        ("Couche Données  —  data/processed/", "oncf_clean.parquet · features.parquet (lecture seule)", GREY, DGREY, 15),
    ]
    for name, sub, fc, ec, y in layers:
        box(ax, 50, y, 80, 11, "", fc=fc, ec=ec)
        ax.text(50, y + 2.4, name, ha="center", fontsize=12, weight="bold", color=ec)
        ax.text(50, y - 2.6, sub, ha="center", fontsize=9, color="#333")
    for y in (54.5, 39.5, 24.5):
        arrow(ax, (50, y), (50, y - 4), color="#555", lw=2, style="-|>")
        arrow(ax, (50, y - 4), (50, y), color="#999", lw=1.2, style="-|>", rad=0.0)
    save(fig, "archi_globale.png")


def fig_archi_deux_etapes():
    fig, ax = canvas(100, 44, figw=12)
    title(ax, "Architecture deux étapes : Candidate Generation → Ranking", ymax=44)
    box(ax, 13, 22, 20, 14, "Historique\nutilisateur", fc=GREY, ec=DGREY, fs=10, weight="bold")
    box(ax, 40, 22, 24, 16, "Étape 1\nCandidate Generation\n(heuristique, < 3 ms)\n→ ≤ 10 candidats",
        fc=LOR, ec=ORANGE, fs=9.5)
    box(ax, 70, 22, 24, 16, "Étape 2\nRanking XGBoost\n(1 011 classes)\nrestreint aux candidats",
        fc=LBLUE, ec=BLUE, fs=9.5)
    box(ax, 92, 22, 13, 12, "Top-k\n(k ≤ 3)", fc=LGREEN, ec=GREEN, fs=10, weight="bold")
    arrow(ax, (23, 22), (28, 22))
    arrow(ax, (52, 22), (58, 22))
    arrow(ax, (82, 22), (85.5, 22))
    save(fig, "archi_deux_etapes.png")


def fig_archi_api():
    fig, ax = canvas(100, 60, figw=10)
    title(ax, "Architecture de l'API FastAPI", ymax=60)
    box(ax, 50, 50, 30, 8, "Application mobile\nONCF Voyages", fc=GREY, ec=DGREY, fs=10, weight="bold")
    box(ax, 50, 36, 46, 10, "FastAPI (main.py)\nPOST /recommend · /feedback · GET /schedule · /health",
        fc=LBLUE, ec=BLUE, fs=9)
    box(ax, 50, 22, 50, 9, "Recommender  (lifespan : chargé 1× au démarrage)\nhistory_lookup · onnx_session · fast_preprocessor",
        fc=LOR, ec=ORANGE, fs=9)
    box(ax, 25, 8, 26, 8, "models/\nonnx · label_encoder", fc=LGREEN, ec=GREEN, fs=9)
    box(ax, 60, 8, 30, 8, "schedule_index.joblib\n(horaires offline)", fc=LGREEN, ec=GREEN, fs=9)
    arrow(ax, (50, 46), (50, 41), color=ORANGE)
    arrow(ax, (50, 41.5), (50, 46.5), color="#999", lw=1.2)
    arrow(ax, (50, 31), (50, 26.5), color=ORANGE)
    arrow(ax, (40, 17.5), (28, 12))
    arrow(ax, (58, 17.5), (60, 12))
    save(fig, "archi_api.png")


def fig_cold_start_cf():
    fig, ax = canvas(100, 58, figw=10)
    title(ax, "Cold start : filtrage collaboratif par co-occurrence", ymax=58)
    # noeuds liaisons
    nodes = {"A": (25, 40), "B": (25, 18), "C": (55, 45), "D": (55, 13), "E": (78, 30)}
    edges = [("A", "C"), ("A", "D"), ("B", "C"), ("B", "E"), ("C", "E"), ("D", "E")]
    for a, b in edges:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]],
                color="#cfd8e3", lw=1.6, zorder=1)
    known = {"A", "B"}
    for n, (x, y) in nodes.items():
        c = BLUE if n in known else ORANGE
        ax.add_patch(Circle((x, y), 4.6, fc=c, ec="white", lw=1.5, zorder=3))
        ax.text(x, y, n, ha="center", va="center", color="white",
                fontsize=12, weight="bold", zorder=4)
    ax.text(25, 50, "Liaisons connues\nde l'utilisateur", ha="center", fontsize=9, color=BLUE, weight="bold")
    ax.text(72, 48, "Liaisons recommandées\n(les plus co-réservées)", ha="center",
            fontsize=9, color=ORANGE, weight="bold")
    box(ax, 50, 4, 70, 5,
        "Deux liaisons sont co-associées si un même client les a réservées toutes les deux",
        fc=GREY, ec="#888", fs=8.5)
    save(fig, "cold_start_cf.png")


def fig_retrain_pipeline():
    fig, ax = canvas(100, 84, figw=8.5)
    title(ax, "Pipeline de réentraînement avec guardrail", ymax=84)
    box(ax, 50, 74, 50, 8, "1. Chargement des données\n(oncf_clean + features.parquet)", fc=GREY, ec=DGREY, fs=9)
    box(ax, 50, 62, 50, 8, "2. Entraînement XGBoost\n(mêmes hyperparamètres)", fc=LBLUE, ec=BLUE, fs=9)
    box(ax, 50, 50, 50, 8, "3. Évaluation\nHR@1 · HR@3 · MRR@3 (split 20 %)", fc=LBLUE, ec=BLUE, fs=9)
    # losange guardrail
    d = Polygon([(50, 42), (74, 32), (50, 22), (26, 32)], closed=True,
                fc=LOR, ec=ORANGE, lw=1.6, zorder=3)
    ax.add_patch(d)
    ax.text(50, 32, "Guardrail\nHR@1 chute\n> 5 pp ?", ha="center", va="center", fontsize=9, weight="bold")
    box(ax, 16, 32, 22, 9, "Challenger\ntoujours\nsauvegardé", fc=LGREEN, ec=GREEN, fs=8.5)
    box(ax, 50, 9, 30, 9, "Promotion\n(remplace prod)", fc=LGREEN, ec=GREEN, fs=10, weight="bold")
    box(ax, 84, 32, 22, 9, "Promotion\nbloquée", fc=LRED, ec=RED, fs=9)
    arrow(ax, (50, 70), (50, 66))
    arrow(ax, (50, 58), (50, 54))
    arrow(ax, (50, 46), (50, 42))
    arrow(ax, (50, 22), (50, 13.5), color=GREEN)
    ax.text(53, 17, "OK", fontsize=9, color=GREEN, weight="bold")
    arrow(ax, (74, 32), (95, 32), color=RED)
    ax.text(84, 38, "BLOQUÉ", fontsize=9, color=RED, weight="bold")
    arrow(ax, (26, 32), (27, 32), color=GREEN)  # marqueur
    arrow(ax, (50, 54), (27, 36.5), color="#999", lw=1.2)
    save(fig, "retrain_pipeline.png")


def fig_ab_testing():
    fig, ax = canvas(100, 58, figw=11)
    title(ax, "Framework A/B testing (routing piloté par le client)", ymax=58)
    box(ax, 14, 30, 20, 10, "Application\nmobile", fc=GREY, ec=DGREY, fs=10, weight="bold")
    box(ax, 40, 30, 18, 9, "/recommend\n?variant=a|b", fc=LBLUE, ec=BLUE, fs=9)
    box(ax, 70, 44, 26, 9, "Variant A (prod)\nxgb_ranker.onnx", fc=LGREEN, ec=GREEN, fs=9)
    box(ax, 70, 16, 30, 9, "Variant B (challenger)\nxgb_ranker_challenger.onnx", fc=LOR, ec=ORANGE, fs=9)
    box(ax, 50, 5, 80, 5, "Réponse : request_id (UUID4) → /feedback → analyse CTR par jointure (sans CodeClient)",
        fc=GREY, ec="#888", fs=8.5)
    arrow(ax, (24, 30), (31, 30))
    arrow(ax, (49, 32), (57, 42))
    arrow(ax, (49, 28), (55, 18))
    save(fig, "ab_testing.png")


# ===================================================================
# 3. DIAGRAMMES UML
# ===================================================================
def fig_uml_composants():
    """Graphe de dépendances de modules rec_oncf."""
    fig, ax = canvas(100, 88, figw=10)
    title(ax, "Diagramme de composants : dépendances de modules", ymax=88)

    BW, BH = 17, 6
    pos = {
        "API\n(main.py)":  (18, 80),
        "simulation":      (82, 80),
        "recommender":     (28, 62),
        "retrain":         (72, 62),
        "candidates":      ( 8, 42),
        "cold_start":      (25, 42),
        "features":        (46, 42),
        "training":        (64, 42),
        "metrics":         (83, 42),
        "config":          (52, 22),
    }
    edges = [
        ("API\n(main.py)", "recommender",   0.0),
        ("simulation",     "recommender",  -0.2),
        ("simulation",     "retrain",       0.0),
        ("recommender",    "candidates",    0.1),
        ("recommender",    "cold_start",    0.05),
        ("recommender",    "features",      0.0),
        ("recommender",    "training",     -0.15),
        ("retrain",        "features",      0.12),
        ("retrain",        "training",      0.0),
        ("retrain",        "metrics",      -0.05),
        ("features",       "config",        0.05),
        ("training",       "config",       -0.05),
    ]
    OR_KEYS = {"simulation", "config"}
    for src, dst, rad in edges:
        sx, sy = pos[src]
        dx, dy = pos[dst]
        arrow(ax, (sx, sy - BH / 2), (dx, dy + BH / 2),
              color="#aaa", lw=1.0, style="-|>", rad=rad)
    for label, (cx, cy) in pos.items():
        fc = LOR if label in OR_KEYS else LBLUE
        ec = ORANGE if label in OR_KEYS else BLUE
        box(ax, cx, cy, BW, BH, label, fc=fc, ec=ec, fs=8.5)
    ax.text(50, 10, "→   dépend de", ha="center", fontsize=9, color="#666", style="italic")
    save(fig, "uml_composants.png")


def fig_uml_usecase():
    fig, ax = canvas(100, 76, figw=11)
    title(ax, "Diagramme de cas d'utilisation", ymax=76)
    # frontière système
    ax.add_patch(FancyBboxPatch((30, 4), 40, 62, boxstyle="round,pad=0,rounding_size=1.5",
                                fc="#FBFCFE", ec=BLUE, lw=1.6, zorder=1))
    ax.text(50, 63.5, "Système de recommandation ONCF", ha="center", fontsize=10,
            weight="bold", color=BLUE, zorder=2)
    ucs = [
        ("Consulter\nrecommandations",     50, 56),
        ("Envoyer feedback\n(clic)",        50, 47),
        ("Obtenir horaires\nde la liaison", 50, 38),
        ("Réentraîner\nle modèle",          50, 28),
        ("Évaluer &\npromouvoir",           50, 19),
        ("Configurer le\nguardrail",        50,  9),
    ]
    pos = {}
    for name, x, y in ucs:
        ax.add_patch(Ellipse((x, y), 24, 6.6, fc=LBLUE, ec=BLUE, lw=1.3, zorder=2))
        ax.text(x, y, name, ha="center", va="center", fontsize=8.5, zorder=3)
        pos[name] = (x, y)

    def actor(ax, x, y, label):
        ax.add_patch(Circle((x, y + 4), 1.6, fc="white", ec="#222", lw=1.5, zorder=3))
        ax.plot([x, x], [y + 2.4, y - 1], color="#222", lw=1.5, zorder=3)
        ax.plot([x - 2.4, x + 2.4], [y + 1, y + 1], color="#222", lw=1.5, zorder=3)
        ax.plot([x, x - 2], [y - 1, y - 4], color="#222", lw=1.5, zorder=3)
        ax.plot([x, x + 2], [y - 1, y - 4], color="#222", lw=1.5, zorder=3)
        ax.text(x, y - 6.5, label, ha="center", fontsize=8.5, weight="bold")

    actor(ax, 10, 50, "Utilisateur\nmobile")
    actor(ax, 10, 29, "Application\nmobile")
    actor(ax, 90, 39, "Système de\nréentraînement")
    actor(ax, 90, 13, "Data Scientist\n/ Ops")
    # (actor_x, actor_y, uc_name) — edge connects to nearest ellipse border
    links = [
        (13, 50, "Consulter\nrecommandations"),
        (13, 50, "Envoyer feedback\n(clic)"),
        (13, 29, "Envoyer feedback\n(clic)"),
        (13, 29, "Obtenir horaires\nde la liaison"),
        (87, 39, "Réentraîner\nle modèle"),
        (87, 39, "Évaluer &\npromouvoir"),
        (87, 13, "Configurer le\nguardrail"),
    ]
    for actor_x, actor_y, name in links:
        bx, by = pos[name]
        ex = bx - 12 if actor_x < 50 else bx + 12
        ax.plot([actor_x, ex], [actor_y, by], color="#888", lw=1.1, zorder=1)
    save(fig, "uml_usecase.png")


def fig_uml_sequence():
    fig, ax = canvas(100, 86, figw=11)
    title(ax, "Diagramme de séquence : POST /recommend", ymax=86)
    lanes = [("Application\nmobile", 12), ("API\nFastAPI", 38),
             ("Recommender", 64), ("ONNX\nRuntime", 88)]
    top, bot = 70, 8
    for name, x in lanes:
        box(ax, x, top + 3.5, 18, 6, name, fc=LBLUE, ec=BLUE, fs=9, weight="bold")
        ax.plot([x, x], [top, bot], color="#9aa6b2", ls="--", lw=1.2, zorder=1)
    X = {n.split("\n")[0]: x for n, x in lanes}

    def msg(y, a, b, text, ret=False):
        x1, x2 = X[a], X[b]
        arrow(ax, (x1, y), (x2, y), color=(DGREY if ret else ORANGE),
              lw=1.6, style="-|>" if not ret else "->")
        mid = (x1 + x2) / 2
        ax.text(mid, y + 1.2, text, ha="center", fontsize=7.8,
                color="#333", style="italic" if ret else "normal")

    def selfmsg(y, a, text):
        x = X[a]
        ax.add_patch(FancyArrowPatch((x, y), (x + 7, y - 2.2),
                     connectionstyle="arc3,rad=-0.6", arrowstyle="-|>",
                     mutation_scale=12, color=ORANGE, lw=1.4, zorder=2))
        ax.text(x + 9, y - 1, text, ha="left", fontsize=7.6, color="#333")

    y = 65
    msg(y, "Application", "API", "POST /recommend {code_client, k, variant}")
    y -= 7
    msg(y, "API", "Recommender", "recommend(code_client, k)")
    y -= 7
    selfmsg(y, "Recommender", "history_lookup.get() — si <3 → cold start CF")
    y -= 8
    selfmsg(y, "Recommender", "generate_candidates() + compute_inference_row()")
    y -= 8
    msg(y, "Recommender", "ONNX", "session.run(features encodées)")
    y -= 7
    msg(y, "ONNX", "Recommender", "probabilités (1 011 classes)", ret=True)
    y -= 7
    selfmsg(y, "Recommender", "filtre top-k aux candidats valides")
    y -= 8
    msg(y, "Recommender", "API", "{mode, recommendations, labels, request_id}", ret=True)
    y -= 7
    msg(y, "API", "Application", "200 OK  (~14 ms)", ret=True)
    save(fig, "uml_sequence.png")


def fig_uml_classes():
    fig, ax = canvas(100, 70, figw=11)
    title(ax, "Diagramme de classes simplifié", ymax=70)

    def uclass(cx, cy, w, name, attrs, meths):
        nh, lh = 4.5, 3.0
        h = nh + lh * (len(attrs) + len(meths)) + 2
        top = cy + h / 2
        # header
        ax.add_patch(FancyBboxPatch((cx - w / 2, top - nh), w, nh, boxstyle="square,pad=0",
                                    fc=BLUE, ec=BLUE, zorder=3))
        ax.text(cx, top - nh / 2, name, ha="center", va="center", color="white",
                fontsize=10.5, weight="bold", zorder=4)
        body_top = top - nh
        ah = lh * len(attrs) + 1
        ax.add_patch(FancyBboxPatch((cx - w / 2, body_top - ah), w, ah, boxstyle="square,pad=0",
                                    fc="white", ec=BLUE, zorder=3))
        for i, a in enumerate(attrs):
            ax.text(cx - w / 2 + 1.5, body_top - 2.4 - i * lh, a, ha="left", va="center",
                    fontsize=8.2, zorder=4)
        mh = lh * len(meths) + 1
        ax.add_patch(FancyBboxPatch((cx - w / 2, body_top - ah - mh), w, mh, boxstyle="square,pad=0",
                                    fc=LBLUE, ec=BLUE, zorder=3))
        for i, m in enumerate(meths):
            ax.text(cx - w / 2 + 1.5, body_top - ah - 2.4 - i * lh, m, ha="left", va="center",
                    fontsize=8.2, zorder=4)
        return (cx, cy - h / 2, cx, cy + h / 2, w)

    uclass(28, 40, 34, "Recommender",
                 ["+ artifacts", "+ history_lookup", "+ cold_start_rec",
                  "+ onnx_session", "+ fast_preprocessor", "+ popularity"],
                 ["+ from_paths()", "+ from_data()", "+ recommend()"])
    uclass(78, 56, 30, "TrainArtifacts", ["+ pipeline", "+ label_encoder"], ["+ predict_proba()"])
    uclass(78, 33, 30, "FastPreprocessor", ["+ cat_maps", "+ num_cols"], ["+ encode()"])
    uclass(78, 12, 30, "ColdStartRecommender", ["+ cooccurrence"], ["+ recommend()"])
    # relations (composition losange côté Recommender)
    for ty in (56, 33, 12):
        arrow(ax, (45, 40), (63, ty), color="#555", style="-|>", rad=0.05)
    ax.text(54, 50, "utilise", fontsize=8, color="#555")
    save(fig, "uml_classes.png")


# ===================================================================
def main():
    print("Génération des figures dans pic/ …")
    funcs = [
        fig_gantt, fig_hist_liaison_distribution, fig_rapport_nettoyage,
        fig_encodage_cyclique, fig_metrics_comparison, fig_metrics_segment,
        fig_dist_user_top_liaison_share, fig_latence_profiling,
        fig_fastpreprocessor_vs_sklearn,
        fig_zero_click_concept, fig_archi_globale, fig_archi_deux_etapes,
        fig_archi_api, fig_cold_start_cf, fig_retrain_pipeline, fig_ab_testing,
        fig_uml_composants, fig_uml_usecase, fig_uml_sequence, fig_uml_classes,
    ]
    for f in funcs:
        f()
    print(f"\nTerminé — {len(funcs)} figures dans {PIC}")
    print("À fournir à la main : oncf.png, LogoFsr.png (logos), "
          "github_actions_ci.png, pytest_output.png, task_scheduler.png (captures).")


if __name__ == "__main__":
    main()
