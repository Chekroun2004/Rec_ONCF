"""Génère les stats exactes pour le fichier de documentation ML."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

clean   = pd.read_parquet(PROJECT_ROOT / "data/processed/oncf_clean.parquet")
feats   = pd.read_parquet(PROJECT_ROOT / "data/processed/features.parquet")
with open(PROJECT_ROOT / "reports/cleaning_report.json", encoding="utf-8") as f:
    report = json.load(f)

print("=== CLEAN DATASET ===")
print(f"Shape: {clean.shape}")
print(f"Rows per client (median): {clean.groupby('CodeClient').size().median():.0f}")
print(f"Rows per client (mean):   {clean.groupby('CodeClient').size().mean():.1f}")
print(f"Date range: {clean['DateHeureDepartVoyageSegment'].min()} -> {clean['DateHeureDepartVoyageSegment'].max()}")
print(f"Prix=0 count: {(clean['PrixParLiaison']==0).sum()}")
print(f"Prix>0 count: {(clean['PrixParLiaison']>0).sum()}")
print(f"Prix range (excl 0): min={clean.loc[clean['PrixParLiaison']>0,'PrixParLiaison'].min()}, max={clean.loc[clean['PrixParLiaison']>0,'PrixParLiaison'].max():.0f}")

print("\n=== CLEAN COLUMNS ===")
for col in clean.columns:
    dtype = str(clean[col].dtype)
    n_null = clean[col].isna().sum()
    n_unique = clean[col].nunique()
    sample = clean[col].dropna().iloc[0] if clean[col].notna().any() else "NULL"
    print(f"  {col:<40} {dtype:<20} unique={n_unique:<8} nulls={n_null:<8} ex={sample}")

print("\n=== FEATURES COLUMNS ===")
for col in feats.columns:
    dtype = str(feats[col].dtype)
    n_null = feats[col].isna().sum()
    sample = feats[col].dropna().iloc[0] if feats[col].notna().any() else "NULL"
    print(f"  {col:<40} {dtype:<20} nulls={n_null:<8} ex={sample}")

print("\n=== LIAISON DISTRIBUTION (top 15) ===")
top = clean["LiaisonId"].value_counts().head(15)
for lid, cnt in top.items():
    dep = clean.loc[clean["LiaisonId"]==lid, "DesignationFrGareDepart"].iloc[0]
    arr = clean.loc[clean["LiaisonId"]==lid, "DesignationFrGareArrive"].iloc[0]
    print(f"  {lid:<8} {dep:<30} -> {arr:<30} : {cnt:,}")

print("\n=== TRAJET ALLER RETOUR ===")
if "TrajetAllerRetour" in clean.columns:
    print(clean["TrajetAllerRetour"].value_counts().head(10).to_string())

print("\n=== TYPEPARCOURS ===")
print(clean["TypeParcoursId"].value_counts().head(10).to_string())

print("\n=== CLASSEPHYSIQUE ===")
print(clean["ClassePhysiqueId"].value_counts().to_string())

print("\n=== CARTELIENT ===")
print(clean["CarteClientId"].value_counts().head(10).to_string())

print("\n=== TRAINING SET TEMPORAL SPLIT (80/20) ===")
feats_sorted = feats.sort_values("DateHeureDepartVoyageSegment")
cut = int(len(feats_sorted) * 0.8)
train = feats_sorted.iloc[:cut]
test  = feats_sorted.iloc[cut:]
print(f"Train: {len(train):,} rows, date up to {train['DateHeureDepartVoyageSegment'].max()}")
print(f"Test:  {len(test):,} rows,  date from {test['DateHeureDepartVoyageSegment'].min()}")
print(f"Train unique liaisons: {train['LiaisonId'].nunique()}")
print(f"Test  unique liaisons: {test['LiaisonId'].nunique()}")
print(f"Test liaisons not in train: {len(set(test['LiaisonId'])-set(train['LiaisonId']))}")

print("\n=== NULL SUMMARY (features) ===")
nulls = feats.isna().sum()
print(nulls[nulls>0].to_string())
