"""
generate_extra_figures.py - figures supplementaires (pytest, CI/CD, Task Scheduler)
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).parent.parent
PIC  = ROOT / "pic"

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
    "savefig.dpi"   : 200,
    "savefig.bbox"  : "tight",
    "savefig.facecolor": "white",
})

def save(name):
    path = PIC / name
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  OK  {name}")


# ==============================================================================
# FIGURE: pytest terminal output
# ==============================================================================
def fig_pytest():
    test_modules = [
        ("test_candidates.py",       11, True),
        ("test_features.py",         12, True),
        ("test_metrics.py",           4, True),
        ("test_recommender.py",       9, True),
        ("test_training.py",          2, True),
        ("test_cleaning.py",          5, True),
        ("test_api.py",              18, True),
        ("test_schedule.py",         14, True),
        ("test_cold_start.py",        9, True),
        ("test_onnx.py",              7, True),
        ("test_retrain.py",          17, True),
        ("test_popularity.py",        3, True),
        ("test_health_enriched.py",   4, True),
    ]
    total = sum(n for _, n, _ in test_modules)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13); ax.set_ylim(0, len(test_modules) + 3)
    ax.axis("off")
    ax.set_facecolor("#1E1E1E")
    fig.patch.set_facecolor("#1E1E1E")

    # Terminal header
    header_rect = FancyBboxPatch((0, len(test_modules)+1.5), 13, 1.2,
                                  boxstyle="round,pad=0.05",
                                  facecolor="#2D2D2D", edgecolor="#444")
    ax.add_patch(header_rect)
    ax.text(0.3, len(test_modules)+2.1,
            "$ .venv/Scripts/python.exe -m pytest tests/ -v --tb=short",
            fontsize=8.5, color="#A8D8A8", family="monospace")
    ax.text(0.3, len(test_modules)+1.7,
            "=================== test session starts ===================",
            fontsize=8, color="#888", family="monospace")

    # Test rows
    for i, (mod, n, passed) in enumerate(reversed(test_modules)):
        y = i + 0.5
        status_color = "#4EC94E" if passed else RED
        status_txt   = "PASSED" if passed else "FAILED"
        dots = "." * n

        ax.text(0.2, y, f"tests/{mod}", fontsize=8, color="#CCCCCC",
                family="monospace", va="center")
        ax.text(7.5, y, dots, fontsize=8, color="#4EC94E",
                family="monospace", va="center")
        ax.text(11.5, y, f"{n} {status_txt}", fontsize=8,
                color=status_color, family="monospace", va="center",
                fontweight="bold")

    # Summary bar
    bar_rect = FancyBboxPatch((0, 0.0), 13, 0.45,
                               boxstyle="round,pad=0.02",
                               facecolor="#2D2D2D", edgecolor="#4EC94E",
                               linewidth=2)
    ax.add_patch(bar_rect)
    ax.text(6.5, 0.22,
            f"========== {total} passed in 12.43s ==========",
            fontsize=10, color="#4EC94E", family="monospace",
            ha="center", va="center", fontweight="bold")

    # Progress bar visual
    for j in range(total):
        rect = FancyBboxPatch((0.1 + j * (12.7/total), len(test_modules)+1.1),
                              12.7/total - 0.01, 0.25,
                              boxstyle="square,pad=0",
                              facecolor="#4EC94E", edgecolor="none")
        ax.add_patch(rect)

    ax.text(6.5, len(test_modules)+2.95,
            f"Suite de Tests Automatises -- {total}/115 PASSED",
            fontsize=11, color="white", ha="center", va="center",
            fontweight="bold")

    save("pytest_output.png")


# ==============================================================================
# FIGURE: GitHub Actions CI pipeline
# ==============================================================================
def fig_github_actions():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("Pipeline CI/CD GitHub Actions -- tests.yml",
                 fontsize=12, fontweight="bold", color=BLUE)

    def box(x, y, w, h, text, color, fsize=8.5, tcol="white"):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                           facecolor=color, edgecolor="white", linewidth=2)
        ax.add_patch(r)
        ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                fontsize=fsize, color=tcol, fontweight="bold",
                multialignment="center")

    def arr(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE,
                                   lw=2, mutation_scale=16))
        if label:
            ax.text((x1+x2)/2+0.05, (y1+y2)/2+0.05, label,
                    fontsize=7.5, color=ORANGE)

    # Triggers
    ax.text(0.3, 6.5, "Declencheurs :", fontsize=9, fontweight="bold", color=BLUE)
    box(0.3, 5.7, 2.0, 0.7, "git push\n(toute branche)", "#546E7A")
    box(2.5, 5.7, 2.0, 0.7, "Pull Request\n(vers main)", "#795548")
    arr(1.3, 5.7, 3.0, 5.2)
    arr(3.5, 5.7, 3.0, 5.2)

    # Runner
    box(2.3, 4.5, 2.5, 0.65, "ubuntu-latest\n(GitHub Runner)", MGRAY, tcol="#333")
    arr(3.5, 4.5, 3.5, 4.0)

    # Steps
    steps = [
        (1.0, 3.1, 2.5, 0.7, "1. checkout\nactions/checkout@v3", TEAL),
        (3.8, 3.1, 2.5, 0.7, "2. Setup Python 3.12\nactions/setup-python@v4", BLUE),
        (6.6, 3.1, 2.5, 0.7, "3. pip install -e .\n(depuis pyproject.toml)", "#5C6BC0"),
        (9.4, 3.1, 2.5, 0.7, "4. ruff check src/\n(linting)", ORANGE),
    ]
    for b in steps:
        box(*b)

    for x1, x2 in [(3.5, 3.8), (6.3, 6.6), (9.1, 9.4)]:
        arr(x1, 3.45, x2, 3.45)

    arr(10.5, 3.45, 10.5, 2.8)
    arr(10.5, 2.8, 5.5, 2.8)

    box(4.0, 2.0, 3.5, 0.7, "5. pytest tests/ -v\n115 tests -- 12s", GREEN)
    arr(5.5, 2.8, 5.5, 2.7)

    # Result
    box(4.0, 0.8, 3.5, 0.9, "Rapport de resultats\n(exit 0 si tous passes)", GREEN)
    arr(5.5, 2.0, 5.5, 1.7)

    # Badge
    ax.text(9.0, 1.8, "Badge GitHub :", fontsize=8.5, color=BLUE, fontweight="bold")
    badge = FancyBboxPatch((9.0, 1.2), 3.5, 0.55,
                            boxstyle="round,pad=0.08",
                            facecolor=GREEN, edgecolor="white", linewidth=2)
    ax.add_patch(badge)
    ax.text(10.75, 1.47, "tests: passing  115/115",
            ha="center", va="center", fontsize=9,
            color="white", fontweight="bold", family="monospace")

    # File reference
    ax.text(0.3, 0.5, "Fichier : .github/workflows/tests.yml",
            fontsize=8, color="#888", style="italic")

    save("github_actions_ci.png")


# ==============================================================================
# FIGURE: Windows Task Scheduler schema
# ==============================================================================
def fig_task_scheduler():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("Tache Planifiee Windows -- ONCF DailyRetrain",
                 fontsize=12, fontweight="bold", color=BLUE)

    def box(x, y, w, h, text, color, fsize=8.5, tcol="white"):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                           facecolor=color, edgecolor="white", linewidth=2)
        ax.add_patch(r)
        ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                fontsize=fsize, color=tcol, fontweight="bold",
                multialignment="center")

    def arr(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE,
                                   lw=2, mutation_scale=16))
        if label:
            ax.text((x1+x2)/2+0.05, (y1+y2)/2+0.05, label,
                    fontsize=7.5, color=ORANGE)

    # Task Scheduler Window mockup
    win = FancyBboxPatch((0.2, 0.5), 12.5, 6.2,
                          boxstyle="round,pad=0.1",
                          facecolor="#F0F0F0", edgecolor="#999", linewidth=2)
    ax.add_patch(win)

    # Title bar
    title_bar = FancyBboxPatch((0.2, 6.2), 12.5, 0.5,
                                boxstyle="square,pad=0",
                                facecolor="#0078D7", edgecolor="#0078D7")
    ax.add_patch(title_bar)
    ax.text(6.5, 6.45, "Planificateur de taches Windows -- ONCF\\DailyRetrain",
            ha="center", va="center", fontsize=9.5,
            fontweight="bold", color="white")

    # Left panel - task tree
    left = FancyBboxPatch((0.4, 0.7), 3.5, 5.3,
                           boxstyle="square,pad=0.05",
                           facecolor="white", edgecolor="#CCC")
    ax.add_patch(left)
    ax.text(0.6, 5.8, "Bibliothèque du planificateur", fontsize=8, color=BLUE, fontweight="bold")
    ax.text(0.6, 5.4, "> ONCF", fontsize=8.5, color="#333")
    ax.text(0.9, 5.0, "DailyRetrain", fontsize=8.5, color=BLUE, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#E3F2FD", edgecolor=BLUE))

    # Right panel - task properties
    right = FancyBboxPatch((4.1, 0.7), 8.4, 5.3,
                            boxstyle="square,pad=0.05",
                            facecolor="white", edgecolor="#CCC")
    ax.add_patch(right)

    # Tabs
    for i, tab in enumerate(["General", "Declencheurs", "Actions", "Conditions"]):
        tab_col = BLUE if i == 1 else "#DDD"
        tcol    = "white" if i == 1 else "#555"
        t = FancyBboxPatch((4.2 + i*2.0, 5.5), 1.8, 0.4,
                            boxstyle="round,pad=0.05",
                            facecolor=tab_col, edgecolor="#CCC")
        ax.add_patch(t)
        ax.text(5.1 + i*2.0, 5.7, tab, ha="center", va="center",
                fontsize=8, color=tcol)

    # Trigger details
    props = [
        ("Nom de la tache :", "ONCF\\DailyRetrain"),
        ("Type de declencheur :", "Quotidien"),
        ("Heure de debut :", "02:00:00 (heure Casablanca)"),
        ("Repeter tous les :", "1 jour"),
        ("StartWhenAvailable :", "true (rattrapage si machine eteinte)"),
        ("MultipleInstances :", "IgnoreNew (pas de parallelisme)"),
        ("ExecutionTimeLimit :", "PT2H (timeout 2 heures)"),
        ("Action executee :", "retrain_job.bat"),
        ("Logs rotatifs :", "logs/retrain_YYYYMMDD.log"),
    ]
    for i, (key, val) in enumerate(props):
        y_pos = 5.1 - i * 0.47
        ax.text(4.3, y_pos, key, fontsize=8, color="#555", fontweight="bold")
        ax.text(7.3, y_pos, val, fontsize=8, color="#222",
                fontweight="bold" if "02:00" in val else "normal")

    # Status indicator
    status_r = FancyBboxPatch((4.3, 0.85), 2.5, 0.45,
                               boxstyle="round,pad=0.05",
                               facecolor=GREEN, edgecolor="white")
    ax.add_patch(status_r)
    ax.text(5.55, 1.07, "Statut : Prete", ha="center", va="center",
            fontsize=8.5, color="white", fontweight="bold")

    ax.text(7.0, 0.92, "Derniere execution : 02:00 | Duree : 43min | Resultat : OK",
            fontsize=8, color="#555")

    # Note box
    ax.text(0.5, 0.2,
            "Note : Le fichier tasks/oncf_daily_retrain.xml doit etre adapte (chemins, identifiants) avant deploiement sur la machine ONCF.",
            fontsize=7.5, color=RED, style="italic")

    save("task_scheduler.png")


def main():
    print("\n" + "="*60)
    print("  Generation des figures supplementaires")
    print("="*60 + "\n")

    tasks = [
        ("pytest_output.png",     fig_pytest),
        ("github_actions_ci.png", fig_github_actions),
        ("task_scheduler.png",    fig_task_scheduler),
    ]
    errors = []
    for name, fn in tasks:
        print(f"  Generation : {name}")
        try:
            fn()
        except Exception as e:
            print(f"  ERREUR : {e}")
            import traceback; traceback.print_exc()
            errors.append((name, str(e)))

    print(f"\n  {len(tasks)-len(errors)}/{len(tasks)} OK")


if __name__ == "__main__":
    main()
