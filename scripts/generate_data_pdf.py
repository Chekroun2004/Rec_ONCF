"""
Generate a detailed PDF explaining the data pipeline of the ONCF recommender project.
Output: docs/rapport_donnees.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fpdf import FPDF

DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)
OUT_PATH = DOCS_DIR / "rapport_donnees.pdf"

# -- Palette ----------------------------------------------------------------
ONCF_GREEN  = (0, 130, 80)
ONCF_DARK   = (20, 50, 40)
LIGHT_GREEN = (220, 245, 230)
LIGHT_GRAY  = (245, 245, 245)
DARK_GRAY   = (80, 80, 80)
WHITE       = (255, 255, 255)
RED_SOFT    = (255, 235, 235)
BLUE_SOFT   = (235, 240, 255)
YELLOW_SOFT = (255, 252, 220)
CODE_BG     = (30, 35, 45)
CODE_FG     = (180, 220, 180)


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(20, 20, 20)

    # -- Header / Footer ----------------------------------------------------
    def header(self):
        if self.page_no() == 1:
            return
        self.set_fill_color(*ONCF_GREEN)
        self.rect(0, 0, 210, 12, "F")
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*WHITE)
        self.set_xy(10, 3)
        self.cell(0, 6, "ONCF Zero-Click Recommender  |  Pipeline de Donnees", align="L")
        self.set_xy(-30, 3)
        self.cell(20, 6, f"Page {self.page_no()}", align="R")
        self.set_text_color(0, 0, 0)
        self.ln(10)

    def footer(self):
        self.set_y(-12)
        self.set_draw_color(*ONCF_GREEN)
        self.set_line_width(0.5)
        self.line(20, self.get_y(), 190, self.get_y())
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*DARK_GRAY)
        self.cell(0, 6, "Omar Chekroun  |  PFA 2026  |  Confidentiel", align="C")

    # -- Helpers ------------------------------------------------------------
    def chapter_title(self, num: str, title: str):
        self.ln(4)
        self.set_fill_color(*ONCF_GREEN)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 13)
        self.set_x(20)
        self.cell(0, 10, f"  {num}  {title}", fill=True, ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def section_title(self, title: str):
        self.ln(3)
        self.set_fill_color(*LIGHT_GREEN)
        self.set_text_color(*ONCF_DARK)
        self.set_font("Helvetica", "B", 11)
        self.set_x(20)
        self.cell(0, 8, f"  {title}", fill=True, ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def subsection_title(self, title: str):
        self.ln(2)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*ONCF_GREEN)
        self.set_x(20)
        self.cell(0, 7, title, ln=True)
        self.set_text_color(0, 0, 0)

    def body(self, text: str, indent: int = 0):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.set_x(20 + indent)
        self.multi_cell(170 - indent, 6, text)
        self.ln(1)

    def bullet(self, text: str, level: int = 1):
        indent = 5 * level
        bullet_char = "-" if level == 1 else "o"
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.set_x(20 + indent)
        self.multi_cell(170 - indent, 6, f"{bullet_char}  {text}")

    def code_block(self, lines: list[str]):
        self.ln(2)
        self.set_fill_color(*CODE_BG)
        total_h = len(lines) * 5.5 + 6
        self.set_x(22)
        x0 = self.get_x()
        y0 = self.get_y()
        self.rect(x0, y0, 166, total_h, "F")
        self.set_font("Courier", "", 8.5)
        self.set_text_color(*CODE_FG)
        self.set_xy(x0 + 4, y0 + 3)
        for line in lines:
            self.set_x(x0 + 4)
            self.cell(162, 5.5, line, ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def info_box(self, label: str, text: str, color=BLUE_SOFT):
        self.ln(2)
        self.set_fill_color(*color)
        self.set_x(20)
        y0 = self.get_y()
        self.rect(20, y0, 170, 1, "F")
        self.set_x(22)
        self.set_y(y0 + 2)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*ONCF_DARK)
        self.cell(0, 5, label, ln=True)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        self.set_x(22)
        self.multi_cell(166, 5.5, text, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def stat_row(self, label: str, value: str, pct: str = "", color=WHITE):
        self.set_fill_color(*color)
        self.set_font("Helvetica", "", 9)
        self.set_x(20)
        self.cell(110, 7, "  " + label, fill=True, border=1)
        self.cell(35, 7, value, fill=True, border=1, align="R")
        self.cell(25, 7, pct, fill=True, border=1, align="C")
        self.ln()

    def table_header(self, cols: list[tuple[str, int]], color=ONCF_GREEN):
        self.set_fill_color(*color)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 9)
        self.set_x(20)
        for text, w in cols:
            self.cell(w, 7, "  " + text, fill=True, border=1)
        self.ln()
        self.set_text_color(0, 0, 0)

    def table_row(self, vals: list[tuple[str, int]], color=WHITE):
        self.set_fill_color(*color)
        self.set_font("Helvetica", "", 9)
        self.set_x(20)
        for text, w in vals:
            self.multi_cell(w, 6, "  " + text, fill=True, border=1, max_line_height=6, new_x="RIGHT", new_y="TOP")
        self.ln()

    def divider(self):
        self.ln(3)
        self.set_draw_color(*ONCF_GREEN)
        self.set_line_width(0.3)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(4)

    def pipeline_step(self, num: str, title: str, subtitle: str,
                      input_: str, output: str, color=LIGHT_GREEN):
        self.set_fill_color(*color)
        y0 = self.get_y()
        self.set_x(20)
        self.rect(20, y0, 170, 22, "F")
        self.set_draw_color(*ONCF_GREEN)
        self.rect(20, y0, 170, 22, "D")
        # number badge
        self.set_fill_color(*ONCF_GREEN)
        self.rect(20, y0, 14, 22, "F")
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*WHITE)
        self.set_xy(20, y0 + 6)
        self.cell(14, 10, num, align="C")
        # title
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*ONCF_DARK)
        self.set_xy(36, y0 + 2)
        self.cell(134, 7, title)
        self.ln(0)
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*DARK_GRAY)
        self.set_xy(36, y0 + 10)
        self.cell(134, 5, subtitle)
        self.ln(0)
        self.set_font("Helvetica", "I", 7.5)
        self.set_xy(36, y0 + 16)
        self.set_text_color(*ONCF_GREEN)
        self.cell(60, 5, f"IN: {input_}")
        self.set_x(100)
        self.cell(70, 5, f"OUT: {output}")
        self.set_text_color(0, 0, 0)
        self.set_y(y0 + 25)


# ???????????????????????????????????????????????????????????????????????????
def build_pdf():
    pdf = PDF()

    # ??????????????????????????????????????????????????????????????????????
    # PAGE DE TITRE
    # ??????????????????????????????????????????????????????????????????????
    pdf.add_page()
    # Bande verte haute
    pdf.set_fill_color(*ONCF_GREEN)
    pdf.rect(0, 0, 210, 60, "F")
    # Logo text
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(20, 12)
    pdf.cell(0, 14, "ONCF", align="L")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_xy(20, 28)
    pdf.cell(0, 7, "Office National des Chemins de Fer", align="L")

    # Titre principal
    pdf.set_fill_color(*ONCF_DARK)
    pdf.rect(0, 60, 210, 45, "F")
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(20, 66)
    pdf.cell(0, 12, "Pipeline de Donnees", align="L")
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_xy(20, 80)
    pdf.cell(0, 8, "Systeme de Recommandation Zero-Click", align="L")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(20, 91)
    pdf.cell(0, 7, "Nettoyage  |  Transformation  |  Feature Engineering", align="L")

    # Encadre infos
    pdf.set_fill_color(*WHITE)
    pdf.rect(20, 118, 170, 52, "F")
    pdf.set_draw_color(*ONCF_GREEN)
    pdf.rect(20, 118, 170, 52, "D")

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*ONCF_DARK)
    pdf.set_xy(28, 124)
    pdf.cell(70, 7, "Auteur :")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(70, 124)
    pdf.cell(0, 7, "Omar Chekroun")

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_xy(28, 133)
    pdf.cell(70, 7, "Projet :")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(70, 133)
    pdf.cell(0, 7, "PFA 2026  -  Recommandation ferroviaire")

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_xy(28, 142)
    pdf.cell(70, 7, "Dataset initial :")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(70, 142)
    pdf.cell(0, 7, "946 155 reservations ONCF")

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_xy(28, 151)
    pdf.cell(70, 7, "Dataset final :")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(70, 151)
    pdf.cell(0, 7, "491 680 lignes  |  69 449 clients  |  1 067 liaisons")

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_xy(28, 160)
    pdf.cell(70, 7, "Features construites :")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(70, 160)
    pdf.cell(0, 7, "26 colonnes  (7 categoriel + 15 numerique + 4 meta)")

    # Table des matières
    pdf.set_text_color(0, 0, 0)
    pdf.set_fill_color(*LIGHT_GREEN)
    pdf.rect(20, 182, 170, 8, "F")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*ONCF_DARK)
    pdf.set_xy(22, 182)
    pdf.cell(0, 8, "  Table des matieres")
    pdf.set_text_color(0, 0, 0)

    toc = [
        ("1.", "Sources de donnees brutes", "3"),
        ("2.", "Vue d'ensemble du pipeline", "4"),
        ("3.", "Etape 1 - Nettoyage des donnees (cleaning.py)", "5"),
        ("   3.1", "Validation des colonnes obligatoires", "5"),
        ("   3.2", "Conversion des types", "5"),
        ("   3.3", "Colonnes supprimees (constantes)", "6"),
        ("   3.4", "Jointure avec Liaison.csv", "7"),
        ("   3.5", "Suppression des valeurs manquantes", "7"),
        ("   3.6", "Detection et suppression des annulations", "8"),
        ("   3.7", "Suppression des clients cold-start", "10"),
        ("   3.8", "Encodage cyclique du temps", "11"),
        ("4.", "Etape 2 - Feature Engineering (features.py)", "12"),
        ("   4.1", "Les 7 features categorielles", "13"),
        ("   4.2", "Les features numeriques", "14"),
        ("   4.3", "La feature cle : user_top_liaison_share", "15"),
        ("   4.4", "Inference en production : compute_inference_row", "16"),
        ("5.", "Schema final des colonnes", "17"),
        ("6.", "Bilan et chiffres cles", "18"),
    ]
    for num, title, page in toc:
        pdf.set_font("Helvetica", "B" if not num.startswith(" ") else "", 9)
        pdf.set_x(24)
        pdf.cell(10, 6, num)
        pdf.cell(140, 6, title)
        pdf.cell(0, 6, page, align="R")
        pdf.ln()

    # ??????????????????????????????????????????????????????????????????????
    # PAGE 2  -  SOURCES DE DONNEES
    # ??????????????????????????????????????????????????????????????????????
    pdf.add_page()
    pdf.chapter_title("1.", "Sources de donnees brutes")

    pdf.body(
        "Le systeme de recommandation part de deux fichiers CSV bruts fournis par l'ONCF. "
        "Ces fichiers representent les reservations historiques des clients et la table "
        "de reference des liaisons ferroviaires."
    )

    pdf.section_title("Fichier 1  -  oncf_data.csv (reservations)")
    pdf.body("946 155 lignes. Chaque ligne represente une reservation de billet par un client.")
    pdf.ln(2)

    cols1 = [("Colonne", 60), ("Type brut", 30), ("Description", 80)]
    pdf.table_header(cols1)
    rows1 = [
        ("CodeClient", "str/int", "Identifiant unique du client (cle de lookup  -  jamais feature)"),
        ("LiaisonVoyageurSegmentIdSTG", "str/float", "Code de la liaison (devient LiaisonId apres normalisation)"),
        ("DateHeureDepartVoyageSegment", "str/datetime", "Date et heure de depart du voyage  -  colonne temporelle principale"),
        ("DatePaiement", "str/datetime", "Date de paiement de la reservation"),
        ("TypeParcoursId", "str/int", "Type de trajet : aller simple, aller-retour, etc."),
        ("ClassificationId", "str/int", "Classification de la reservation"),
        ("ClassePhysiqueId", "str/int", "Classe physique du train (1ere, 2eme...)"),
        ("NiveauPrixId", "str/int", "Niveau de prix / tarif applique"),
        ("TrainAutocarId", "str/int", "Indicateur train ou autocar"),
        ("CarteClientId", "str/int", "Type de carte de fidelite du client"),
        ("PrixParLiaison", "str/float", "Prix paye pour cette liaison (peut etre 0 = 100% reduction)"),
        ("NbrVoySegment", "str/float", "Nombre de segments  -  clef pour detecter les annulations"),
        ("DelaiAnticipation", "str/float", "Nombre de jours entre achat et depart"),
        ("TrajetAllerRetour", "str", "Identifiant du trajet aller-retour (optionnel)"),
        ("AchteurId", "str", "Identifiant de l'acheteur (? CodeClient si achat pour autrui)"),
        ("PosteVenteId", "float", "ID du poste de vente  -  SUPPRIME (constante)"),
        ("TypeOperationVenteApresVenteId", "float", "Type operation apres-vente  -  SUPPRIME (constante)"),
        ("PrixServices", "float", "Prix des services  -  SUPPRIME (constante)"),
    ]
    for i, (a, b, c) in enumerate(rows1):
        clr = LIGHT_GRAY if i % 2 == 0 else WHITE
        pdf.table_row([(a, 60), (b, 30), (c, 80)], color=clr)

    pdf.ln(4)
    pdf.section_title("Fichier 2  -  Liaison.csv (referentiel routes)")
    pdf.body("Table de correspondance entre les identifiants de liaison et les noms des gares.")
    pdf.ln(2)

    cols2 = [("Colonne", 60), ("Description", 110)]
    pdf.table_header(cols2)
    pdf.table_row([("LiaisonId", 60), ("Identifiant unique de la liaison ferroviaire", 110)], WHITE)
    pdf.table_row([("DesignationFrGareDepart", 60), ("Nom complet de la gare de depart en francais", 110)], LIGHT_GRAY)
    pdf.table_row([("DesignationFrGareArrive", 60), ("Nom complet de la gare d'arrivee en francais", 110)], WHITE)

    pdf.ln(5)
    pdf.info_box(
        "Jointure entre les deux fichiers",
        "oncf_data.csv contient LiaisonVoyageurSegmentIdSTG (code brut). "
        "Apres normalisation (suppression du '.0' final), ce code est joint avec LiaisonId de Liaison.csv "
        "pour recuperer les noms des gares. Taux de correspondance : 99.92% (1 seule ligne sans match).",
        color=BLUE_SOFT
    )

    # ??????????????????????????????????????????????????????????????????????
    # PAGE 3  -  VUE D'ENSEMBLE DU PIPELINE
    # ??????????????????????????????????????????????????????????????????????
    pdf.add_page()
    pdf.chapter_title("2.", "Vue d'ensemble du pipeline de donnees")

    pdf.body(
        "Le pipeline transforme les donnees brutes en features pret pour l'entrainement "
        "en deux grandes etapes executees par deux scripts independants."
    )
    pdf.ln(4)

    pdf.pipeline_step("1", "Nettoyage (script 01_make_dataset.py)",
                      "cleaning.py  ->  make_clean_dataset()",
                      "oncf_data.csv + Liaison.csv",
                      "oncf_clean.parquet (491 680 lignes)")
    pdf.ln(2)
    pdf.set_fill_color(*ONCF_GREEN)
    pdf.set_x(100)
    pdf.cell(10, 6, "v", align="C")
    pdf.ln(2)

    pdf.pipeline_step("2", "Feature Engineering (script 02_build_features.py)",
                      "features.py  ->  build_training_rows()",
                      "oncf_clean.parquet (491 680 lignes)",
                      "features.parquet (491 680 x 26 colonnes)")
    pdf.ln(2)
    pdf.set_x(100)
    pdf.cell(10, 6, "v", align="C")
    pdf.ln(2)

    pdf.pipeline_step("3", "Entrainement XGBoost (script 03_train_ranker.py)",
                      "training.py  ->  train_xgb_multiclass()",
                      "features.parquet",
                      "xgb_ranker.json + label_encoder.joblib",
                      color=YELLOW_SOFT)

    pdf.ln(6)
    pdf.section_title("Bilan numerique du nettoyage")
    pdf.ln(2)

    pdf.set_fill_color(*ONCF_GREEN)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_x(20)
    pdf.cell(110, 7, "  Etape", fill=True, border=1)
    pdf.cell(35, 7, "Lignes supprimees", fill=True, border=1, align="R")
    pdf.cell(25, 7, "% du total", fill=True, border=1, align="C")
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    pdf.stat_row("Lignes initiales (oncf_data.csv)",       "946 155", "100.0%",  WHITE)
    pdf.stat_row("- Pas de correspondance Liaison.csv",    "1",       "< 0.01%", LIGHT_GRAY)
    pdf.stat_row("- Valeurs trop manquantes",              "0",       "0.00%",   WHITE)
    pdf.stat_row("- Annulations + reservations annulees",  "50 859",  "5.37%",   LIGHT_GRAY)
    pdf.stat_row("- Champs essentiels manquants",          "0",       "0.00%",   WHITE)
    pdf.stat_row("- Doublons exacts",                      "0",       "0.00%",   LIGHT_GRAY)
    pdf.stat_row("- Clients cold-start (< 3 voyages)",     "403 615", "42.66%",  RED_SOFT)

    pdf.set_fill_color(*ONCF_DARK)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_x(20)
    pdf.cell(110, 8, "  = TOTAL SUPPRIME", fill=True, border=1)
    pdf.cell(35, 8, "454 475", fill=True, border=1, align="R")
    pdf.cell(25, 8, "48.03%", fill=True, border=1, align="C")
    pdf.ln()

    pdf.set_fill_color(*ONCF_GREEN)
    pdf.set_x(20)
    pdf.cell(110, 8, "  = DATASET FINAL (oncf_clean.parquet)", fill=True, border=1)
    pdf.cell(35, 8, "491 680", fill=True, border=1, align="R")
    pdf.cell(25, 8, "51.97%", fill=True, border=1, align="C")
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    # ??????????????????????????????????????????????????????????????????????
    # PAGE 4  -  NETTOYAGE ETAPE PAR ETAPE
    # ??????????????????????????????????????????????????????????????????????
    pdf.add_page()
    pdf.chapter_title("3.", "Etape 1  -  Nettoyage des donnees (cleaning.py)")

    pdf.body(
        "La fonction make_clean_dataset(oncf_df, liaison_df) est le coeur du pipeline de nettoyage. "
        "Elle prend les deux dataframes bruts en entree et retourne : "
        "(1) le dataframe nettoye, (2) un rapport de nettoyage JSON, (3) une table de provenance "
        "qui enregistre la decision prise pour chaque ligne originale."
    )

    # 3.1
    pdf.section_title("3.1  Validation des colonnes obligatoires")
    pdf.body(
        "Avant toute transformation, le code verifie que les colonnes requises sont presentes. "
        "Si une seule colonne obligatoire est absente, une exception est levee immediatement."
    )
    pdf.code_block([
        "REQUIRED_ONCF_COLS = [",
        "    'TrajetAllerRetour', 'TypeParcoursId', 'CodeClient', 'ClassificationId',",
        "    'ClassePhysiqueId', 'NiveauPrixId', 'TrainAutocarId', 'LiaisonVoyageurSegmentIdSTG',",
        "    'CarteClientId', 'PrixParLiaison', 'NbrVoySegment', 'DatePaiement',",
        "    'DateHeureDepartVoyageSegment', 'DelaiAnticipation'",
        "]",
        "missing = [c for c in REQUIRED_ONCF_COLS if c not in oncf_df.columns]",
        "if missing:",
        "    raise ValueError(f'Missing columns: {missing}')",
    ])
    pdf.info_box("Pourquoi ?",
                 "Le systeme sera deploye en production et recevra de nouveaux exports CSV periodiquement. "
                 "La validation stricte a l'entree garantit qu'une erreur de schema (colonne renommee, "
                 "manquante) est detectee immediatement plutot que de produire des donnees silencieusement corrompues.",
                 color=YELLOW_SOFT)

    # 3.2
    pdf.section_title("3.2  Conversion des types")
    pdf.body(
        "Les CSV sont charges avec des types generiques (str/object). "
        "Le code convertit explicitement chaque colonne vers son type correct."
    )
    pdf.code_block([
        "# Dates : deux formats tentes (ISO puis DD/MM/YYYY)",
        "df['DatePaiement'] = pd.to_datetime(series, errors='coerce')",
        "# Si > 20% echouent, retente avec dayfirst=True",
        "",
        "# IDs et prix : conversion numerique, erreurs -> NaN",
        "for col in numeric_cols:",
        "    df[col] = pd.to_numeric(df[col], errors='coerce')",
        "",
        "# AchteurId : intentionnellement EXCLU de la conversion numerique",
        "# C'est un identifiant de personne, pas une quantite.",
    ])

    pdf.bullet("DateHeureDepartVoyageSegment : datetime64  -  colonne temporelle principale pour le tri et le split")
    pdf.bullet("TypeParcoursId, ClassePhysiqueId, etc. : convertis en float64  -  indispensable pour la detection des valeurs negatives (annulations)")
    pdf.bullet("AchteurId : reste str  -  compare avec CodeClient pour is_self_purchase")
    pdf.bullet("PrixServices, PosteVenteId, TypeOperationId : convertis mais supprimes ensuite (constantes)")

    # 3.3
    pdf.add_page()
    pdf.section_title("3.3  Colonnes supprimees  -  Colonnes constantes")

    pdf.body(
        "Une colonne 'constante' est une colonne dont tous les valeurs non-nulles sont identiques. "
        "Elle n'apporte aucun signal discriminant pour le modele (variance = 0)."
    )
    pdf.code_block([
        "def _drop_constant_columns(df, *, ignore):",
        "    constant_cols = []",
        "    for column in df.columns:",
        "        if column in ignore: continue",
        "        non_null = df[column].dropna()",
        "        if non_null.empty or non_null.nunique() <= 1:",
        "            constant_cols.append(column)",
        "    return df.drop(columns=constant_cols), constant_cols",
    ])

    pdf.ln(3)
    pdf.body("3 colonnes ont ete supprimees dans notre dataset :")
    pdf.ln(2)

    cols_supp = [("Colonne supprimee", 60), ("Valeur unique observee", 40), ("Raison de suppression", 70)]
    pdf.table_header(cols_supp)
    pdf.table_row([("PosteVenteId", 60), ("Constante", 40), ("Une seule valeur  -  zero information pour le modele", 70)], LIGHT_GRAY)
    pdf.table_row([("TypeOperationVenteApresVenteId", 60), ("Constante", 40), ("Idem  -  aucune variabilite", 70)], WHITE)
    pdf.table_row([("PrixServices", 60), ("Constante", 40), ("Idem  -  aucune variabilite", 70)], LIGHT_GRAY)

    pdf.ln(3)
    pdf.info_box("Colonnes protegees (non testees pour la constance)",
                 "CodeClient, LiaisonVoyageurSegmentIdSTG, TrajetAllerRetour, AchteurId, "
                 "DatePaiement, DateHeureDepartVoyageSegment. Ces colonnes sont des cles "
                 "ou des identifiants qui pourraient techniquement etre constantes sans que cela "
                 "soit une erreur  -  elles sont donc ignorees dans le test de constance.",
                 color=YELLOW_SOFT)

    # 3.4
    pdf.section_title("3.4  Normalisation de LiaisonId et jointure")
    pdf.body(
        "La colonne LiaisonVoyageurSegmentIdSTG contient des codes comme '123.0', '456.0'. "
        "Ces suffixes '.0' apparaissent quand pandas lit une colonne d'entiers qui contient "
        "des NaN (converti en float). La normalisation supprime ce suffixe avant la jointure."
    )
    pdf.code_block([
        "# Normalisation : '123.0' -> '123'",
        "df['LiaisonId'] = series.astype(str).str.replace(r'\\.0$', '', regex=True).str.strip()",
        "",
        "# Jointure LEFT pour conserver toutes les reservations",
        "# et ajouter DesignationFrGareDepart + DesignationFrGareArrive",
        "out = df.merge(liaison, on='LiaisonId', how='left')",
        "",
        "# Lignes sans correspondance -> supprimees (1 ligne dans notre dataset)",
        "join_missing = out[['DesignationFrGareDepart', 'DesignationFrGareArrive']].isna().any(axis=1)",
        "out = out.loc[~join_missing]",
    ])
    pdf.bullet("Taux de correspondance : 99.92%  -  1 290 LiaisonId distincts sur 1 291 trouvent leur match")
    pdf.bullet("1 seule ligne supprimee faute de correspondance dans notre dataset")
    pdf.bullet("DesignationFrGareDepart / DesignationFrGareArrive : utilisees uniquement pour l'affichage UI, pas comme features")

    # 3.5
    pdf.section_title("3.5  Suppression des lignes avec trop de valeurs manquantes")
    pdf.body(
        "Une ligne est supprimee si elle a PLUS DE 1 valeur manquante parmi toutes ses colonnes "
        "(en excluant les colonnes de noms de gares qui peuvent legitimement etre absentes). "
        "Dans notre dataset : 0 ligne supprimee par cette regle."
    )
    pdf.info_box("Champs essentiels",
                 "CodeClient, LiaisonId, DateHeureDepartVoyageSegment sont verifies separement : "
                 "si l'UN de ces trois champs est manquant, la ligne est supprimee independamment "
                 "du nombre total de valeurs manquantes. 0 cas dans notre dataset.",
                 color=BLUE_SOFT)

    # ??????????????????????????????????????????????????????????????????????
    # PAGE 5  -  ANNULATIONS
    # ??????????????????????????????????????????????????????????????????????
    pdf.add_page()
    pdf.section_title("3.6  Detection et suppression des annulations")

    pdf.body(
        "C'est l'etape la plus complexe du nettoyage. 50 859 lignes sont supprimees. "
        "Une annulation dans le systeme ONCF est encodee comme une nouvelle ligne dans le CSV, "
        "pas comme la modification d'une ligne existante. Il faut donc la detecter et supprimer "
        "AUSSI la reservation d'origine qu'elle annule."
    )

    pdf.subsection_title("Comment reconnaitre une ligne d'annulation ?")
    pdf.ln(2)

    cols_ann = [("Indicateur", 55), ("Condition", 50), ("Explication metier", 65)]
    pdf.table_header(cols_ann)
    pdf.table_row([("NbrVoySegment <= 0", 55), ("nb_segments = 0", 50), ("Zero passager = voyage annule. Note : 0 est une annulation, pas une reduction.", 65)], LIGHT_GRAY)
    pdf.table_row([("PrixParLiaison < 0", 55), ("prix < 0 (negatif)", 50), ("Montant negatif = remboursement / reversal. ATTENTION : 0 est valide = billet 100% reduit.", 65)], WHITE)
    pdf.table_row([("Autres IDs < 0", 55), ("TypeParcoursId < 0, etc.", 50), ("Identifiants negatifs = lignes d'inversion comptable generees automatiquement.", 65)], LIGHT_GRAY)

    pdf.ln(3)
    pdf.code_block([
        "is_cancellation = pd.Series(False, index=ordered.index)",
        "",
        "# Critere 1 : zero segment = annulation",
        "is_cancellation |= ordered['NbrVoySegment'].le(0)",
        "",
        "# Critere 2 : prix negatif = remboursement",
        "# (IMPORTANT : .lt(0) et non .le(0) car PrixParLiaison=0 est valide !)",
        "is_cancellation |= ordered['PrixParLiaison'].lt(0)",
        "",
        "# Critere 3 : IDs categoriel negatifs = reversal comptable",
        "is_cancellation |= ordered[cancel_strict_neg_cols].lt(0).any(axis=1)",
    ])

    pdf.subsection_title("Comment identifier la reservation annulee ?")
    pdf.body(
        "Une annulation void la reservation qui la PRECEDE dans l'historique du meme client. "
        "Le code trie d'abord par (CodeClient, DateHeureDepartVoyageSegment), puis pour chaque "
        "ligne d'annulation, recupere la ligne precedente via un groupby + shift(1)."
    )
    pdf.code_block([
        "# Tri par client et date",
        "ordered = out.sort_values(['CodeClient', 'DateHeureDepartVoyageSegment'])",
        "",
        "# Pour chaque ligne, l'_orig_row_id de la ligne precedente du meme client",
        "group_prev_orig = ordered.groupby(['CodeClient', 'TrajetAllerRetour'])['_orig_row_id'].shift(1)",
        "",
        "# IDs des lignes d'annulation",
        "neg_orig_ids   = set(ordered.loc[is_cancellation, '_orig_row_id'])",
        "",
        "# IDs des reservations qui precedaient ces annulations",
        "prev_orig_ids  = set(group_prev_orig.loc[is_cancellation].dropna().astype(int))",
        "",
        "# Supprimer les deux : l'annulation ET la reservation d'origine",
        "indices_to_drop = neg_orig_ids | prev_orig_ids",
    ])

    pdf.ln(3)
    pdf.body("Exemple concret pour un client :")
    pdf.ln(2)
    pdf.set_fill_color(*LIGHT_GREEN)
    pdf.set_draw_color(*ONCF_GREEN)
    pdf.set_x(20)
    pdf.rect(20, pdf.get_y(), 170, 30, "FD")
    y0 = pdf.get_y() + 3
    pdf.set_font("Courier", "", 9)
    pdf.set_text_color(30, 30, 30)
    pdf.set_xy(25, y0)
    pdf.cell(0, 5.5, "Client 12345 - historique trie par date :")
    y0 += 6
    pdf.set_xy(25, y0)
    pdf.cell(0, 5.5, "  Ligne A (2025-01-10) : Casablanca -> Rabat    | NbrVoy=1, Prix=120  <- RESERVATION ORIGINALE")
    y0 += 5.5
    pdf.set_xy(25, y0)
    pdf.cell(0, 5.5, "  Ligne B (2025-01-12) : Casablanca -> Rabat    | NbrVoy=0, Prix=-120 <- ANNULATION (void Ligne A)")
    y0 += 5.5
    pdf.set_xy(25, y0)
    pdf.cell(0, 5.5, "  => Lignes A et B supprimees toutes les deux")
    pdf.set_y(y0 + 10)
    pdf.set_text_color(0, 0, 0)

    pdf.ln(3)
    pdf.info_box(
        "Chiffres cles  -  annulations",
        "32 010 lignes d'annulation detectees\n"
        "+ 18 849 reservations d'origine supprimees (la reservation qui precedait chaque annulation)\n"
        "= 50 859 lignes supprimees au total (5.37% du dataset initial)",
        color=RED_SOFT
    )

    # ??????????????????????????????????????????????????????????????????????
    # PAGE 6  -  COLD START
    # ??????????????????????????????????????????????????????????????????????
    pdf.add_page()
    pdf.section_title("3.7  Suppression des clients cold-start")

    pdf.body(
        "C'est la plus grande suppression : 403 615 lignes, 304 595 clients. "
        "Un client 'cold-start' est un client avec moins de 3 reservations au total dans le dataset."
    )

    pdf.subsection_title("La regle : MIN_TRIPS_FOR_TRAINING = 3")
    pdf.code_block([
        "MIN_TRIPS_FOR_TRAINING = 3",
        "",
        "# Compte le nombre de voyages par client",
        "trip_counts = out.groupby('CodeClient').size()",
        "",
        "# Identifie les clients sous le seuil",
        "cold_start_clients = trip_counts[trip_counts < MIN_TRIPS_FOR_TRAINING].index",
        "",
        "# Supprime TOUTES les lignes de ces clients",
        "cold_start_mask = out['CodeClient'].isin(cold_start_clients)",
        "out = out[~cold_start_mask]",
    ])

    pdf.subsection_title("Pourquoi supprimer ces clients du training ?")
    pdf.body(
        "Trois raisons complementaires justifient cette decision :"
    )
    pdf.ln(2)

    reasons = [
        ("1. Pas de signal apprennable",
         "Un client avec 1 ou 2 voyages n'a pas d'historique suffisant pour que le modele "
         "apprenne son comportement. Inclure ces lignes ajoute du bruit sans signal utile. "
         "Le modele ne peut pas generaliser depuis 1 exemple."),
        ("2. Coherence train / production",
         "En production, ces memes clients (< 3 voyages) ne passent PAS par le modele XGBoost. "
         "Ils recoivent une recommandation par popularite globale (cold-start path). "
         "Les inclure dans le training creerait une inconsistance : entraineent sur des profils "
         "qu'on ne modelise pas en prod."),
        ("3. Alignement metier",
         "L'objectif est de predire le comportement des voyageurs reguliers de l'ONCF. "
         "Un client avec 1 voyage pourrait etre occasionnel ou un touriste  -  pas la cible principale."),
    ]
    for title, desc in reasons:
        pdf.set_fill_color(*LIGHT_GREEN)
        pdf.set_x(20)
        y0 = pdf.get_y()
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*ONCF_DARK)
        pdf.cell(0, 7, "  " + title, fill=True, ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(30, 30, 30)
        pdf.set_x(22)
        pdf.multi_cell(166, 5.5, desc)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    pdf.subsection_title("Que se passe-t-il pour ces clients en production ?")
    pdf.body(
        "En production, si un client a < 3 voyages dans l'historique, le Recommender "
        "applique un fallback en cascade :"
    )
    pdf.code_block([
        "if len(history) < 3:",
        "    # Essai 1 : co-occurrence (cold_start.joblib)",
        "    result = cold_start_rec.recommend()  # mode: 'cold_start_cf'",
        "    if not result:",
        "        # Essai 2 : popularite globale (popularity.joblib)",
        "        return _fallback(k)  # mode: 'popularity'",
    ])
    pdf.bullet("mode: 'cold_start_cf'  -  recommandation par co-occurrence (qui prend quelles routes en meme temps)")
    pdf.bullet("mode: 'popularity'  -  top-k global des liaisons les plus reservees")
    pdf.bullet("mode: 'cold_start'  -  liste vide (uniquement si popularity.joblib est absent)")

    pdf.ln(4)
    pdf.info_box(
        "Impact sur la distribution des clients",
        "Avant nettoyage : ~374 044 clients distincts\n"
        "Apres suppression cold-start : 69 449 clients actifs (avec >= 3 voyages)\n\n"
        "Cela signifie que 304 595 clients (81.4% des clients) ont moins de 3 voyages. "
        "Ce chiffre est typique des systemes de reservation ferroviaire : "
        "la majorite des clients voyagent rarement, et seule une minorite est reguliere.",
        color=YELLOW_SOFT
    )

    # ??????????????????????????????????????????????????????????????????????
    # PAGE 7  -  ENCODAGE CYCLIQUE
    # ??????????????????????????????????????????????????????????????????????
    pdf.add_page()
    pdf.section_title("3.8  Encodage cyclique du temps")

    pdf.body(
        "Apres le nettoyage, le code extrait trois features temporelles brutes depuis "
        "DateHeureDepartVoyageSegment, puis les encode de maniere cyclique."
    )

    pdf.subsection_title("Les trois features temporelles brutes")
    cols_tmp = [("Feature", 50), ("Calcul", 60), ("Plage de valeurs", 60)]
    pdf.table_header(cols_tmp)
    pdf.table_row([("depart_hour", 50), ("dt.hour", 60), ("0 a 23", 60)], LIGHT_GRAY)
    pdf.table_row([("depart_dow", 50), ("dt.dayofweek", 60), ("0 (lundi) a 6 (dimanche)", 60)], WHITE)
    pdf.table_row([("depart_month", 50), ("dt.month", 60), ("1 a 12", 60)], LIGHT_GRAY)

    pdf.ln(4)
    pdf.subsection_title("Pourquoi encoder cycliquement ?")
    pdf.body(
        "Si on donne les valeurs brutes au modele, il voit que l'heure 23 et l'heure 0 "
        "sont tres eloignees (|23 - 0| = 23). Mais metrologiquement, 23h et 0h sont adjacentes "
        "(5 minutes d'ecart possible). L'encodage cyclique resout ce probleme en projetant "
        "chaque valeur sur un cercle via sinus et cosinus."
    )
    pdf.code_block([
        "# Formule generale pour une periode P :",
        "sin_feature = sin(2 * pi * valeur / P)",
        "cos_feature = cos(2 * pi * valeur / P)",
        "",
        "# Heure (P=24) :",
        "depart_hour_sin = sin(2 * pi * depart_hour / 24)",
        "depart_hour_cos = cos(2 * pi * depart_hour / 24)",
        "",
        "# Jour semaine (P=7) :",
        "depart_dow_sin = sin(2 * pi * depart_dow / 7)",
        "depart_dow_cos = cos(2 * pi * depart_dow / 7)",
        "",
        "# Mois (P=12) :",
        "depart_month_sin = sin(2 * pi * depart_month / 12)",
        "depart_month_cos = cos(2 * pi * depart_month / 12)",
    ])

    pdf.subsection_title("Propriete cle : continuite aux extremes")
    pdf.body(
        "Avec l'encodage cyclique (sin, cos), la distance entre heure 23 et heure 0 est "
        "petite  -  exactement comme la distance entre heure 1 et heure 2. "
        "Le modele peut donc detecter que 'voyage de nuit' (22h-1h) est une categorie coherente "
        "sans qu'une frontiere artificielle brise le signal. "
        "On utilise TOUJOURS les deux (sin ET cos) car sin seul est ambigu : "
        "sin(30 degres) = sin(150 degres), donc deux heures differentes auraient le meme sin. "
        "Avec sin + cos ensemble, chaque heure a une paire (sin, cos) unique."
    )

    pdf.ln(3)
    pdf.info_box(
        "6 colonnes ajoutees au total par l'encodage cyclique",
        "depart_hour_sin, depart_hour_cos   (heure de depart)\n"
        "depart_dow_sin,  depart_dow_cos    (jour de la semaine)\n"
        "depart_month_sin, depart_month_cos (mois de l'annee)\n\n"
        "Ces 6 colonnes s'ajoutent aux 3 colonnes brutes (depart_hour, depart_dow, depart_month) "
        "qui sont egalement conservees car XGBoost peut exploiter les deux representations.",
        color=LIGHT_GREEN
    )

    # ??????????????????????????????????????????????????????????????????????
    # PAGE 8  -  FEATURE ENGINEERING
    # ??????????????????????????????????????????????????????????????????????
    pdf.add_page()
    pdf.chapter_title("4.", "Etape 2  -  Feature Engineering (features.py)")

    pdf.body(
        "La fonction build_training_rows(clean_df) transforme le dataset nettoye en table "
        "d'entrainement. Elle produit une ligne par voyage historique, ou chaque ligne "
        "represente 'ce que le modele savait au moment de ce voyage' pour predire la liaison."
    )
    pdf.body(
        "Entree : oncf_clean.parquet (491 680 lignes)\n"
        "Sortie : features.parquet (491 680 lignes x 26 colonnes)"
    )

    pdf.code_block([
        "def build_training_rows(clean_df):",
        "    df = clean_df.copy()",
        "    df = df.sort_values(['CodeClient', 'DateHeureDepartVoyageSegment'])",
        "",
        "    # Features calculees par groupe (historique de chaque client)",
        "    df['user_trip_index']       = df.groupby('CodeClient').cumcount()",
        "    df['prev_liaison']          = df.groupby('CodeClient')['LiaisonId'].shift(1)",
        "    df['days_since_prev']       = ...  # difference de dates",
        "    df['user_top_liaison_share']= ...  # rolling fraction (voir section 4.3)",
        "    df['is_self_purchase']      = (df['AchteurId'] == df['CodeClient']).astype(int)",
        "",
        "    # Conversion des categories en str pour OrdinalEncoder",
        "    for c in cat_cols:",
        "        df[c] = df[c].astype('Int64').astype(str)",
        "",
        "    return df[out_cols]",
    ])

    # 4.1
    pdf.section_title("4.1  Les 7 features categorielles")
    pdf.body(
        "Ces features sont passees dans un OrdinalEncoder (pas OneHotEncoder) pour deux raisons : "
        "(1) XGBoost gere bien les entiers ordinaux, "
        "(2) prev_liaison a 1 011 valeurs uniques  -  OHE creerait 1 011 colonnes supplementaires."
    )
    pdf.ln(2)

    cols_cat = [("Feature", 55), ("Source", 30), ("Description et signal", 85)]
    pdf.table_header(cols_cat)
    rows_cat = [
        ("TypeParcoursId", "brut", "Type de trajet : aller simple, aller-retour, etc. Influence la liaison choisie."),
        ("ClassificationId", "brut", "Classification de la reservation (abonnement, billet ponctuel...)."),
        ("ClassePhysiqueId", "brut", "Classe physique : 1ere ou 2eme classe. Corre le au budget."),
        ("NiveauPrixId", "brut", "Niveau de tarif applique. Capte la sensibilite au prix."),
        ("TrainAutocarId", "brut", "Train ou autocar. Certaines liaisons n'existent qu'en autocar."),
        ("CarteClientId", "brut", "Type de carte fidelite. Proxy du profil voyageur."),
        ("prev_liaison", "calcule", "Liaison precedente du client. Feature la plus predictive : les voyageurs repetent souvent leurs routes."),
    ]
    for i, (a, b, c) in enumerate(rows_cat):
        clr = LIGHT_GRAY if i % 2 == 0 else WHITE
        pdf.table_row([(a, 55), (b, 30), (c, 85)], color=clr)

    pdf.ln(3)
    pdf.info_box(
        "prev_liaison  -  la feature categorielle la plus importante",
        "prev_liaison = la LiaisonId du voyage precedent du client (calculee par groupby + shift(1)).\n"
        "NaN pour le premier voyage d'un client -> converti en la chaine 'nan' pour OrdinalEncoder.\n\n"
        "Signal : un voyageur qui a pris Casablanca->Rabat la derniere fois a une forte probabilite "
        "de reprendre la meme route. La baseline 'prev_liaison' (toujours recommander la derniere route) "
        "atteint HR@1 = 0.26  -  le modele monte a 0.76 en integrant le contexte complet.",
        color=LIGHT_GREEN
    )

    # 4.2
    pdf.add_page()
    pdf.section_title("4.2  Les features numeriques")
    pdf.ln(2)

    cols_num = [("Feature", 55), ("Calcul", 45), ("Description", 70)]
    pdf.table_header(cols_num)
    rows_num = [
        ("CodeClient", "brut (int64)", "ID client  -  CLE DE LOOKUP UNIQUEMENT. Jamais passe au modele XGBoost (Loi 09-08 / vie privee)."),
        ("PrixParLiaison", "brut (float64)", "Prix paye. Capte les habitudes de budget. Nullable."),
        ("NbrVoySegment", "brut (float64)", "Nombre de segments du voyage."),
        ("DelaiAnticipation", "brut (float64)", "Jours entre achat et depart. Planifie=grand, spontane=petit."),
        ("user_trip_index", "cumcount()", "Numero du voyage dans l'historique client. 0=1er voyage, 1=2eme... Proxy de la fidelite et de l'experience."),
        ("days_since_prev", "diff. de dates", "Jours depuis le voyage precedent. Capte le rythme: voyageur hebdo, mensuel, occasionnel."),
        ("user_top_liaison_share", "rolling frac.", "Part de la liaison la plus frequente dans l'historique passe. Voir section 4.3."),
        ("depart_hour", "dt.hour", "Heure brute (0-23). Conservee en plus du cyclique."),
        ("depart_dow", "dt.dayofweek", "Jour semaine brut (0=Lundi). Conserve en plus du cyclique."),
        ("depart_month", "dt.month", "Mois brut (1-12). Conserve en plus du cyclique."),
        ("depart_hour_sin/cos", "sin/cos cyclique", "Encodage cyclique de l'heure (periode=24). Voir section 3.8."),
        ("depart_dow_sin/cos", "sin/cos cyclique", "Encodage cyclique du jour (periode=7)."),
        ("depart_month_sin/cos", "sin/cos cyclique", "Encodage cyclique du mois (periode=12)."),
        ("is_self_purchase", "AchteurId==CodeClient", "1 si achat pour soi-meme, 0 si achat pour quelqu'un d'autre. Proxy du type de deplacement."),
    ]
    for i, (a, b, c) in enumerate(rows_num):
        clr = LIGHT_GRAY if i % 2 == 0 else WHITE
        pdf.table_row([(a, 55), (b, 45), (c, 70)], color=clr)

    # 4.3
    pdf.add_page()
    pdf.section_title("4.3  La feature cle : user_top_liaison_share")

    pdf.body(
        "Cette feature a ete ajoutee en Sprint 2 et a produit la plus grande amelioration "
        "des metriques (+2.33pp sur HR@1, de 0.7395 a 0.7628). "
        "Elle mesure le degre de 'fidelite' d'un utilisateur a une route dominante."
    )

    pdf.subsection_title("Definition")
    pdf.body(
        "Pour un utilisateur a la position i dans son historique trie par date : "
        "user_top_liaison_share = (nombre de fois ou la liaison la plus frequente apparait "
        "dans les i-1 voyages precedents) / (i-1)."
    )
    pdf.body("Exemples :")
    pdf.bullet("0.9 = 90% de mes voyages passes sont sur la meme route (tres fidele)")
    pdf.bullet("0.3 = je prends plusieurs routes differentes (voyageur varie)")
    pdf.bullet("NaN = premier voyage (aucun historique)")

    pdf.subsection_title("Implementation : calcul strictement sur le passe (sans data leakage)")
    pdf.code_block([
        "def _rolling_top_share(series):",
        "    counts = Counter()",
        "    out = np.empty(len(series), dtype=float)",
        "    out[0] = np.nan  # NaN pour le 1er voyage : pas d'historique",
        "",
        "    for i, value in enumerate(series.tolist()):",
        "        if i > 0:",
        "            # Fraction de la liaison la plus frequente",
        "            # dans les i observations PRECEDENTES",
        "            top_n = counts.most_common(1)[0][1]",
        "            out[i] = top_n / float(i)",
        "        counts[str(value)] += 1  # mise a jour APRES le calcul",
        "    return pd.Series(out, index=series.index)",
        "",
        "# Appliquee par groupe client",
        "df['user_top_liaison_share'] = (",
        "    df.groupby('CodeClient')['LiaisonId']",
        "    .apply(_rolling_top_share)",
        ")",
    ])

    pdf.subsection_title("Pourquoi 'rolling' et pas global ?")
    pdf.body(
        "Si on calculait la part globale (incluant les voyages futurs), la ligne du voyage N "
        "contiendrait des informations sur les voyages N+1, N+2... qui n'existaient pas encore "
        "au moment du voyage N. C'est du DATA LEAKAGE  -  le modele verrait le futur pendant "
        "l'entrainement et serait artificiellement bon. "
        "La version rolling calcule uniquement depuis les observations passees : "
        "la ligne du voyage i n'utilise que les voyages 0 a i-1."
    )

    pdf.info_box(
        "Impact sur les metriques (Sprint 2)",
        "Avant user_top_liaison_share : HR@1 = 0.7395  (Sprint 1)\n"
        "Apres user_top_liaison_share  : HR@1 = 0.7628  (Sprint 2)  -> +2.33 points de pourcentage\n\n"
        "Le modele capture maintenant le profil de fidelite de chaque utilisateur, "
        "ce qui lui permet de distinguer les voyageurs 'monotones' (recommander toujours la meme route) "
        "des voyageurs 'varies' (explorer plus de candidats).",
        color=LIGHT_GREEN
    )

    # 4.4
    pdf.add_page()
    pdf.section_title("4.4  Inference en production : compute_inference_row()")

    pdf.body(
        "build_training_rows() produit les donnees d'entrainement : une ligne par voyage historique. "
        "Mais en production, quand un client ouvre l'application, on ne connait pas encore son "
        "prochain voyage. On doit construire UNE seule ligne de features qui represente "
        "'a quoi ressemble ce client maintenant', pour predire ce qu'il va reserver."
    )
    pdf.body(
        "C'est le role de compute_inference_row(history_df, asof=now)."
    )

    pdf.subsection_title("Differences cles avec build_training_rows")
    pdf.ln(2)

    cols_diff = [("Aspect", 55), ("build_training_rows (training)", 55), ("compute_inference_row (prod)", 60)]
    pdf.table_header(cols_diff)
    pdf.table_row([
        ("Nombre de lignes",
         "1 ligne par voyage historique",
         "1 seule ligne (la prediction)"),
        55, 55, 60
    ] if False else [("Nombre de lignes", 55), ("1 ligne par voyage historique", 55), ("1 seule ligne (la prediction)", 60)], LIGHT_GRAY)
    pdf.table_row([("Date de reference", 55), ("DateHeureDepartVoyageSegment de chaque ligne", 55), ("asof = maintenant (quand l'app s'ouvre)", 60)], WHITE)
    pdf.table_row([("prev_liaison", 55), ("shift(1) = liaison du voyage precedent", 55), ("last['LiaisonId'] = derniere liaison connue", 60)], LIGHT_GRAY)
    pdf.table_row([("user_trip_index", 55), ("cumcount() = index cumule", 55), ("len(history) = total voyages connus", 60)], WHITE)
    pdf.table_row([("days_since_prev", 55), ("diff entre voyage N et N-1", 55), ("(maintenant - derniere date) en jours", 60)], LIGHT_GRAY)
    pdf.table_row([("user_top_liaison_share", 55), ("fraction sur i-1 obs. passees", 55), ("fraction sur TOUT l'historique connu", 60)], WHITE)
    pdf.table_row([("Colonnes prix/classe", 55), ("valeurs de la ligne courante", 55), ("proxy = valeurs du DERNIER voyage connu", 60)], LIGHT_GRAY)
    pdf.table_row([("LiaisonId (cible)", 55), ("vraie valeur (etiquette d'entrainement)", 55), ("'__unknown__' - supprimee avant predict", 60)], WHITE)

    pdf.ln(3)
    pdf.code_block([
        "def compute_inference_row(history_df, asof=None):",
        "    if asof is None:",
        "        asof = pd.Timestamp(datetime.now())",
        "",
        "    sorted_hist = history_df.sort_values('DateHeureDepartVoyageSegment')",
        "    last = sorted_hist.iloc[-1]  # dernier voyage connu",
        "",
        "    user_trip_index = float(len(sorted_hist))",
        "    days_since_prev = (asof - last['DateHeureDepartVoyageSegment']).total_seconds() / 86400",
        "",
        "    # user_top_liaison_share sur tout l'historique",
        "    liaison_counts = sorted_hist['LiaisonId'].value_counts()",
        "    top_share = liaison_counts.iloc[0] / len(sorted_hist)",
        "",
        "    # Features temporelles basees sur 'maintenant'",
        "    row['depart_hour'] = asof.hour",
        "    row['depart_hour_sin'] = sin(2*pi*asof.hour/24)",
        "    # ... etc",
        "",
        "    return pd.DataFrame([row])  # 1 ligne",
    ])

    pdf.info_box(
        "Pourquoi proxy les features prix/classe avec le dernier voyage ?",
        "En production, on ne connait pas encore les details du PROCHAIN voyage (prix, classe...). "
        "On utilise le dernier voyage connu comme approximation : l'hypothese est que le prochain "
        "voyage ressemblera au dernier en termes de classe et de prix. C'est un proxy imparfait "
        "mais c'est la meilleure approximation disponible sans information sur le futur.",
        color=YELLOW_SOFT
    )

    # ??????????????????????????????????????????????????????????????????????
    # PAGE  -  SCHEMA FINAL
    # ??????????????????????????????????????????????????????????????????????
    pdf.add_page()
    pdf.chapter_title("5.", "Schema final des colonnes (features.parquet  -  26 colonnes)")

    pdf.body(
        "Voici la liste complete des 26 colonnes du fichier features.parquet, "
        "avec leur role dans le pipeline."
    )
    pdf.ln(2)

    cols_schema = [("Colonne", 62), ("Type", 28), ("Role", 20), ("Description", 60)]
    pdf.table_header(cols_schema)
    schema = [
        ("DateHeureDepartVoyageSegment", "datetime64", "META",    "Date de depart  -  pour temporal split, jamais feature"),
        ("LiaisonId",                   "str",        "CIBLE",   "La liaison a predire (1 011 classes)"),
        ("CodeClient",                  "float64",    "META",    "ID client  -  cle de lookup, jamais feature modele"),
        ("TypeParcoursId",              "str",        "CAT",     "Type de trajet (OrdinalEncoder)"),
        ("ClassificationId",            "str",        "CAT",     "Classification reservation (OrdinalEncoder)"),
        ("ClassePhysiqueId",            "str",        "CAT",     "Classe physique (OrdinalEncoder)"),
        ("NiveauPrixId",                "str",        "CAT",     "Niveau de prix (OrdinalEncoder)"),
        ("TrainAutocarId",              "str",        "CAT",     "Train ou autocar (OrdinalEncoder)"),
        ("CarteClientId",               "str",        "CAT",     "Carte de fidelite (OrdinalEncoder)"),
        ("prev_liaison",                "str",        "CAT",     "Liaison precedente, 'nan' si 1er voyage (OrdinalEncoder)"),
        ("PrixParLiaison",              "float64",    "NUM",     "Prix paye (nullable)"),
        ("NbrVoySegment",               "float64",    "NUM",     "Nombre de segments"),
        ("DelaiAnticipation",           "float64",    "NUM",     "Jours anticipes avant depart"),
        ("user_trip_index",             "int64",      "NUM",     "Index cumule du voyage du client"),
        ("days_since_prev",             "float64",    "NUM",     "Jours depuis le voyage precedent"),
        ("user_top_liaison_share",      "float64",    "NUM",     "Part liaison dominante (rolling, pas de leakage)"),
        ("depart_hour",                 "int32",      "NUM",     "Heure de depart brute (0-23)"),
        ("depart_dow",                  "int32",      "NUM",     "Jour semaine brut (0=Lundi)"),
        ("depart_month",                "int32",      "NUM",     "Mois brut (1-12)"),
        ("depart_hour_sin",             "float64",    "NUM",     "sin(2*pi*heure/24)"),
        ("depart_hour_cos",             "float64",    "NUM",     "cos(2*pi*heure/24)"),
        ("depart_dow_sin",              "float64",    "NUM",     "sin(2*pi*dow/7)"),
        ("depart_dow_cos",              "float64",    "NUM",     "cos(2*pi*dow/7)"),
        ("depart_month_sin",            "float64",    "NUM",     "sin(2*pi*mois/12)"),
        ("depart_month_cos",            "float64",    "NUM",     "cos(2*pi*mois/12)"),
        ("is_self_purchase",            "int64",      "NUM",     "1 si acheteur = client, 0 sinon"),
    ]

    role_colors = {"META": YELLOW_SOFT, "CIBLE": RED_SOFT, "CAT": BLUE_SOFT, "NUM": LIGHT_GREEN}
    for i, (col, typ, role, desc) in enumerate(schema):
        clr = role_colors.get(role, WHITE)
        pdf.table_row([(col, 62), (typ, 28), (role, 20), (desc, 60)], color=clr)

    pdf.ln(3)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*DARK_GRAY)
    pdf.set_x(20)
    pdf.cell(12, 5, "")
    pdf.set_fill_color(*YELLOW_SOFT); pdf.rect(32, pdf.get_y(), 8, 5, "F"); pdf.set_x(42); pdf.cell(20, 5, "META")
    pdf.set_fill_color(*RED_SOFT);    pdf.rect(62, pdf.get_y()-5, 8, 5, "F"); pdf.set_x(72); pdf.cell(20, 5, "CIBLE")
    pdf.set_fill_color(*BLUE_SOFT);   pdf.rect(92, pdf.get_y()-5, 8, 5, "F"); pdf.set_x(102); pdf.cell(25, 5, "CAT = categorielle")
    pdf.set_fill_color(*LIGHT_GREEN); pdf.rect(127, pdf.get_y()-5, 8, 5, "F"); pdf.set_x(137); pdf.cell(30, 5, "NUM = numerique")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(8)

    # ??????????????????????????????????????????????????????????????????????
    # PAGE  -  BILAN
    # ??????????????????????????????????????????????????????????????????????
    pdf.add_page()
    pdf.chapter_title("6.", "Bilan et chiffres cles")

    pdf.section_title("Chiffres cles du pipeline")
    pdf.ln(2)

    stats = [
        ("Lignes initiales", "946 155", "", WHITE),
        ("Lignes apres annulations", "895 295", "-50 859 (-5.4%)", LIGHT_GRAY),
        ("Lignes apres cold-start", "491 680", "-403 615 (-45.1%)", WHITE),
        ("Clients actifs (>= 3 voyages)", "69 449", "18.6% des clients initiaux", LIGHT_GRAY),
        ("Liaisons distinctes", "1 067", "routes ferroviaires differentes", WHITE),
        ("Colonnes features finales", "26", "7 cat + 15 num + 4 meta", LIGHT_GRAY),
        ("Colonnes supprimees (constantes)", "3", "PosteVenteId, TypeOpVAV, PrixServices", WHITE),
        ("Taux de jointure Liaison.csv", "99.92%", "1 290 / 1 291 LiaisonIds matchent", LIGHT_GRAY),
    ]
    pdf.set_fill_color(*ONCF_GREEN)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_x(20)
    pdf.cell(75, 7, "  Metrique", fill=True, border=1)
    pdf.cell(45, 7, "Valeur", fill=True, border=1, align="R")
    pdf.cell(50, 7, "Contexte", fill=True, border=1, align="C")
    pdf.ln()
    pdf.set_text_color(0, 0, 0)
    for label, val, ctx, clr in stats:
        pdf.set_fill_color(*clr)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_x(20)
        pdf.cell(75, 7, "  " + label, fill=True, border=1)
        pdf.cell(45, 7, val, fill=True, border=1, align="R")
        pdf.cell(50, 7, ctx, fill=True, border=1, align="C")
        pdf.ln()

    pdf.ln(6)
    pdf.section_title("Decisions d'architecture importantes")

    decisions = [
        ("OrdinalEncoder plutot que OneHotEncoder",
         "prev_liaison a 1 011 valeurs uniques. OneHotEncoder creerait 1 011 colonnes = 5 000+ "
         "colonnes au total. Impossible a gerer par XGBoost dans ce contexte. OrdinalEncoder "
         "garde tout en 1 colonne avec des entiers (0, 1, 2...)."),
        ("Encodage cyclique plutot que brut pour le temps",
         "Les valeurs brutes (0-23 pour l'heure) creent une discontinuite artificielle aux extremes. "
         "L'encodage sin/cos preserve la continuite : heure 23 et heure 0 sont proches."),
        ("Rolling share plutot que global share pour user_top_liaison_share",
         "Calculer la fraction sur l'historique global inclurait les voyages futurs (data leakage). "
         "La version rolling n'utilise que les observations strictement anterieures a chaque ligne."),
        ("Suppression des clients cold-start (< 3 voyages)",
         "Ces clients representent 81.4% des clients mais n'ont pas assez de signal pour etre "
         "appris par le modele. Ils sont geres separement en production par un fallback cascade."),
        ("CodeClient jamais feature du modele",
         "Loi 09-08 (equivalant RGPD marocain) / CNDP. CodeClient est utilise uniquement comme "
         "cle de lookup pour recuperer l'historique. Il n'est jamais passe au modele XGBoost, "
         "jamais logue dans les requetes API."),
        ("Proxy des features de voyage futures par le dernier voyage connu",
         "En production, les details du prochain voyage (prix, classe...) sont inconnus. "
         "On utilise le dernier voyage comme approximation, ce qui est justifie par la regularite "
         "des comportements des voyageurs ferroviaires."),
    ]
    for i, (title, desc) in enumerate(decisions):
        clr = LIGHT_GREEN if i % 2 == 0 else BLUE_SOFT
        pdf.set_fill_color(*clr)
        y0 = pdf.get_y()
        pdf.set_x(20)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*ONCF_DARK)
        pdf.cell(0, 7, "  " + title, fill=True, ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(30, 30, 30)
        pdf.set_x(22)
        pdf.multi_cell(166, 5.5, desc, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    # ??????????????????????????????????????????????????????????????????????
    # DERNIERE PAGE  -  RESUME VISUEL
    # ??????????????????????????????????????????????????????????????????????
    pdf.add_page()
    pdf.chapter_title("", "Recapitulatif visuel du pipeline complet")

    pdf.set_fill_color(*ONCF_DARK)
    pdf.rect(20, pdf.get_y(), 170, 11, "F")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*WHITE)
    pdf.set_x(22)
    pdf.cell(0, 11, "  DONNEES BRUTES")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(11)

    pdf.set_x(20)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(80, 6, "  oncf_data.csv")
    pdf.cell(90, 6, "Liaison.csv")
    pdf.ln()
    pdf.set_x(20)
    pdf.cell(80, 6, "  946 155 reservations")
    pdf.cell(90, 6, "1 290 liaisons + noms de gares")
    pdf.ln(8)

    arrow_steps = [
        ("SCRIPT 01  -  make_clean_dataset()", [
            "1. Validation colonnes obligatoires",
            "2. Conversion types (dates, numeriques)",
            "3. Suppression 3 colonnes constantes (PosteVenteId, TypeOpVAV, PrixServices)",
            "4. Normalisation LiaisonId + LEFT JOIN Liaison.csv",
            "5. Suppression lignes sans match (-1 ligne)",
            "6. Detection annulations (NbrVoySegment<=0, Prix<0, IDs<0)",
            "   + Suppression annulation ET reservation d'origine (-50 859 lignes)",
            "7. Suppression champs essentiels manquants (-0 lignes)",
            "8. Suppression doublons exacts (-0 lignes)",
            "9. Ajout features temporelles (depart_hour, dow, month + sin/cos x3)",
            "10. Suppression clients < 3 voyages (cold-start) (-403 615 lignes, -304 595 clients)",
        ], "oncf_clean.parquet  :  491 680 lignes  |  69 449 clients  |  1 067 liaisons"),
        ("SCRIPT 02  -  build_training_rows()", [
            "1. Tri par (CodeClient, DateHeureDepartVoyageSegment)",
            "2. Calcul user_trip_index : groupby cumcount",
            "3. Calcul prev_liaison : groupby shift(1)",
            "4. Calcul days_since_prev : difference de dates",
            "5. Calcul user_top_liaison_share : rolling fraction (sans data leakage)",
            "6. Calcul is_self_purchase : AchteurId == CodeClient",
            "7. Conversion categories -> str pour OrdinalEncoder",
            "8. Selection des 26 colonnes finales",
        ], "features.parquet  :  491 680 lignes  x  26 colonnes"),
    ]

    for step_title, ops, result in arrow_steps:
        pdf.set_fill_color(*ONCF_GREEN)
        pdf.set_x(100)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*ONCF_GREEN)
        pdf.cell(0, 8, "  |")
        pdf.ln(0)
        pdf.set_x(100)
        pdf.cell(0, 5, "  v")
        pdf.ln(6)
        pdf.set_text_color(0, 0, 0)

        pdf.set_fill_color(*LIGHT_GREEN)
        pdf.set_draw_color(*ONCF_GREEN)
        y0 = pdf.get_y()
        n_lines = len(ops)
        box_h = 9 + n_lines * 5.5 + 6
        pdf.rect(20, y0, 170, box_h, "FD")
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*ONCF_DARK)
        pdf.set_xy(24, y0 + 2)
        pdf.cell(0, 6, step_title)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(30, 30, 30)
        for op in ops:
            pdf.set_xy(28, pdf.get_y())
            pdf.cell(0, 5.5, op)
            pdf.ln(0)
            pdf.set_xy(28, pdf.get_y() + 5.5)
        pdf.set_text_color(0, 0, 0)
        pdf.set_y(y0 + box_h + 3)

        pdf.set_fill_color(*ONCF_DARK)
        pdf.rect(20, pdf.get_y(), 170, 9, "F")
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*WHITE)
        pdf.set_x(22)
        pdf.cell(0, 9, "  RESULTAT  :  " + result)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(9)

    # Arrow to model
    pdf.set_x(100)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*ONCF_GREEN)
    pdf.cell(0, 8, "  |")
    pdf.ln(0)
    pdf.set_x(100)
    pdf.cell(0, 5, "  v")
    pdf.ln(6)
    pdf.set_text_color(0, 0, 0)

    pdf.set_fill_color(*ONCF_GREEN)
    pdf.rect(20, pdf.get_y(), 170, 12, "F")
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*WHITE)
    pdf.set_x(22)
    pdf.cell(0, 12, "  SCRIPT 03  -  XGBoost multiclass  :  1 011 classes (liaisons)")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(12)

    return pdf


if __name__ == "__main__":
    pdf = build_pdf()
    pdf.output(str(OUT_PATH))
    print(f"PDF genere : {OUT_PATH}")
    print(f"Pages : {pdf.page}")
