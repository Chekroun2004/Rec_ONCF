# Spec — Intégration test1.csv + Simulation retrain quotidien (sliding window 365j)

**Date** : 2026-05-20
**Auteur** : Omar Chekroun
**Statut** : Validé en brainstorming, prêt pour plan d'implémentation

---

## Contexte

- `horaire.csv` a été enrichi (ajout LGV Al Boraq + retours Marrakech). Cible : couverture O/D > 75 % (était 57 %).
- `test1.csv` (voyages 2021) reçu, sur le Desktop. Même structure que `oncf_data.csv` (18 colonnes, virgule).
- Objectif : intégrer test1.csv au pipeline existant, puis simuler 7 jours de retrain quotidien avec fenêtre glissante de 365 jours pour valider le scénario prod (Task Scheduler à 2h du matin).
- Pas d'intégration de `test2.csv` dans ce cycle (décision utilisateur).

## Objectifs

1. Valider que le nouveau `horaire.csv` améliore bien la couverture de l'index O/D et que tous les tests `local_schedule` + `schedule` passent.
2. Créer un nouveau **baseline post-test1** entraîné sur `oncf_data.csv + test1.csv (sauf 7 derniers jours)` et le promouvoir comme nouveau prod.
3. Simuler **7 retrains quotidiens** consécutifs avec sliding window de 365 jours, lancés à la demande de l'utilisateur (un jour à la fois).
4. Mesurer et documenter les métriques jour par jour, pour intégration au rapport PFA.

## Non-objectifs

- Ne pas intégrer `test2.csv` (sera fait dans un cycle ultérieur).
- Ne pas modifier les hyperparamètres du modèle (challenger promu le 2026-05-16 reste la référence).
- Ne pas modifier l'API : aucun changement dans `apps/api/` pour ce spec.
- Ne pas viser le déploiement automatique des modèles de simulation : ils restent dans `models/sim/day_N/` sans toucher au prod servi par l'API.

## Architecture globale (4 phases d'exécution + 2 transverses)

```
Phase 1 : Tests horaire enrichi              (~10 min)
Phase 2 : Extraction des 7 jours             (~5 min, script unique)
Phase 3 : Baseline post-test1                (~60-90 min, pipeline complet)
Phase 4 : Simulation jour par jour           (--day N, manuel, ~20 min/jour)
Phase 5 : Tests                              (transverse, TDD au fil de l'eau)
Phase 6 : Rapport                            (transverse, après chaque exécution)
```

Chaque phase est indépendante et testable ; les artefacts (CSVs jour-par-jour, baseline, modèles `sim/day_N/`) sont persistés sur disque pour reprise à tout moment.

---

## Phase 1 — Tests horaire enrichi

**But** : valider que le nouveau `horaire.csv` (LGV + Marrakech) améliore la couverture O/D.

**Actions** :
1. `python scripts/11_build_schedule_index.py` → reconstruction de `models/schedule_index.joblib`.
2. Comparer la couverture stdout : était 606/1067 LiaisonIds (57 %), cible > 800/1067 (> 75 %).
3. `pytest tests/test_local_schedule.py -v` (21 tests) + `pytest tests/test_schedule.py -v` (14 tests).
4. **Smoke test live** : 3 requêtes `/schedule/{liaison_id}` sur des routes LGV connues (Tanger ↔ Casa, Tanger ↔ Rabat, Tanger ↔ Kenitra) pour confirmer un retour non vide.

**Critère d'arrêt** : si la couverture < 75 %, analyser les LiaisonIds toujours manquants avant de continuer (probablement un problème de normalisation de nom de gare dans les ajouts).

**Artefacts** :
- `models/schedule_index.joblib` (mis à jour).
- Recap couverture dans stdout (copié dans CLAUDE.md).

---

## Phase 2 — Extraction des 7 jours

**But** : produire 7 CSVs jour-par-jour + un test1_base.csv (test1 sans ces 7 jours).

**Hypothèse à valider en début d'exécution** : `test1.csv` a la même structure que `oncf_data.csv` (18 colonnes, virgule, header, `DateHeureDepartVoyageSegment` au format datetime ISO). Vérification dans la première étape du script ; adaptation du parser si le format diffère.

**Nouveau script** : `scripts/12a_extract_test1_days.py`

```python
1. read_csv("C:/Users/omarc/Desktop/test1.csv")
2. Parser DateHeureDepartVoyageSegment → dates calendaires (jour de départ, pas date de réservation).
3. Identifier les 7 dernières dates uniques (du plus ancien au plus récent).
4. Pour chaque jour J → data/raw/daily/test1_day_<YYYY-MM-DD>.csv (header + lignes du jour).
5. test1 SANS ces 7 jours → data/raw/test1_base.csv (header + reste).
6. Imprimer récap : 7 dates, n_lignes par jour, n_lignes test1_base, total.
```

**Nommage** : `test1_day_2021-12-25.csv` (date dans le nom, ordre lexicographique = ordre chronologique).
**Emplacement** : `data/raw/daily/` (nouveau dossier, séparation raw/processed respectée).
**Idempotence** : le script écrase les fichiers existants à chaque lancement.

**Définition d'un "jour"** : date calendaire de `DateHeureDepartVoyageSegment` (date de départ du voyage), cohérent avec le split temporel existant.

---

## Phase 3 — Baseline post-test1

**But** : créer un nouveau modèle prod entraîné sur `oncf_data.csv + test1_base.csv` (4 ans + 2021 sans 7 jours).

**Modification** : `scripts/01_make_dataset.py` reçoit un nouvel argument optionnel `--extra-csv <path>` qui, s'il est présent, concatène ce CSV à `oncf_data.csv` avant `make_clean_dataset()`. Le module `cleaning.py` reste inchangé.

**Séquence d'exécution** :

```powershell
# 1. Archive de sécurité de l'actuel prod (commande PowerShell manuelle)
Copy-Item -Path "models\*" -Destination "models\archive\20260520T_pre_test1\" -Recurse -Force

# 2. Pipeline complet
python scripts/12a_extract_test1_days.py
python scripts/01_make_dataset.py --extra-csv data/raw/test1_base.csv
python scripts/02_build_features.py
python scripts/03_train_ranker.py
python scripts/05_build_cold_start.py
python scripts/06_export_onnx.py
python scripts/08_build_popularity.py
python scripts/11_build_schedule_index.py
```

**Modèle attendu** : ~570 000 lignes nettoyées, HR@1 ≥ baseline actuel (0.7691) si distribution stable. Une chute > 2pp serait suspecte.

**On n'utilise pas `07_retrain.py`** pour cette étape : son guardrail à 5pp bloquerait la promotion en cas de léger recul, alors qu'on accepte explicitement cette possibilité (plus de données = nouveau prod même si métriques légèrement inférieures).

**Archive** : étape 1 ci-dessus, **lancée à la main avant le pipeline**. Pas d'automatisation dans un script ; on garde le geste explicite pour éviter d'écraser un archive précédent par erreur.

---

## Phase 4 — Simulation jour par jour (sliding window 365j)

**Nouveau script** : `scripts/12_simulate_daily_retrain.py --day N` (N ∈ [1..7], à lancer manuellement, un jour à la fois).

**Logique pour `--day N`** :

```
1. Vérifier état : oncf_clean.parquet + features.parquet doivent exister
   (= baseline post-test1 doit être fait).
   Pour N >= 2 : vérifier que models/sim/day_(N-1)/ existe.

2. Charger data/raw/daily/test1_day_<dateN>.csv.
   Nettoyer avec la même logique que make_clean_dataset (sans Liaison.csv, mapping connu).
   Append à data/processed/oncf_clean.parquet (resauvegarde).

3. Reconstruire features.parquet via scripts/02_build_features.py.

4. Identifier la fenêtre : 365 jours se terminant au jour N inclus.
   Filtrer features.parquet sur [jour_N - 365j ; jour_N].
   ~80k lignes attendues.

5. Lancer le retrain sur cette fenêtre :
   - split temporel 80/20 interne
   - hyperparams prod actuels (max_depth=8, n_estimators=250)
   - ~20 min CPU attendues
   - sauvegarder dans models/sim/day_N/ (sans toucher models/ direct)

6. Évaluations :
   (a) Métriques split interne (80/20) sur la fenêtre.
   (b) Si N ∈ [1..6] : métriques sur jour N+1 (données vraies non vues).
       Inférence via Recommender.recommend() en boucle (~2.5 min pour ~10k lignes,
       acceptable, on optimise plus tard si besoin).
       Si N=7 : pas de J+1, on logge null pour cette section.

7. Logger dans reports/simulation_daily.json :
   {
     "day": N,
     "date": "<date_N>",
     "window_start": "<date_N - 365j>",
     "window_end": "<date_N>",
     "train_rows": <int>,
     "internal_split_metrics": {"hr@1": ..., "hr@3": ..., "mrr@3": ...},
     "next_day_metrics": {"hr@1": ..., "hr@3": ..., "mrr@3": ..., "n_eval": ...} | null,
     "duration_seconds": <float>,
     "guardrail_passes": <bool>,    # true si HR@1 chute < 5pp vs baseline
     "guardrail_reason": "<str>"
   }
```

**Guardrail** : seuil 5pp en cohérence avec `07_retrain.py`. Dans la simulation, c'est un **drapeau informatif** (le modèle est toujours sauvé dans `models/sim/day_N/`, le script n'arrête pas l'expérience). En production réelle (`07_retrain.py`), 5pp reste un **vrai bloqueur**.

**Référence pour le guardrail** : on compare HR@1 du jour N au HR@1 de la baseline post-test1 (point de départ), pas au jour N-1. Cohérent avec ce que fait `07_retrain.py` (compare au modèle prod actuel).

---

## Phase 5 — Tests (TDD au fil de l'eau)

**Nouveaux fichiers de tests** :

| Fichier | Tests | Couvre |
|---|---|---|
| `tests/test_extract_days.py` | ~5 | Phase 2 : extraction 7 derniers jours, conservation totale des lignes, nommage des fichiers, idempotence |
| `tests/test_simulate_daily.py` | ~6 | Phase 4 : ajout d'un jour, sliding window 365j correct, éval J+1, écriture JSON, cas --day 7 (pas de J+1), guardrail |
| `tests/test_dataset_extra_csv.py` | ~3 | Phase 3 : flag `--extra-csv`, concaténation, schéma préservé |

**Relance** : `tests/test_local_schedule.py` (21) + `tests/test_schedule.py` (14) après rebuild de l'index (Phase 1).

**Régression** : `pytest tests/ -v` → > 135 tests verts à la fin (135 actuels + ~14 nouveaux).

---

## Phase 6 — Rapport (`rapport_pfa_v2.tex`)

Modifications post-exécution (pas de pré-rédaction, on attend les chiffres) :

1. **Nouvelle sous-section** dans Sprint 2 ou Phase 3 — "Intégration des données 2021 et simulation de retrain quotidien" :
   - Tableau récap : baseline actuel vs baseline post-test1 (HR@1, HR@3, MRR@3, n_train, n_classes).
   - Justification du sliding window 365j (saisonnalité annuelle, temps constant, scénario prod).
   - Tableau des 7 jours : date, train_rows, HR@1 split interne, HR@1 sur J+1, durée.

2. **Figure** : courbe HR@1 sur 7 jours (split interne + J+1) — visualise la stabilité au fil des retrains. Ajoutée à `scripts/generate_extra_figures.py`.

3. **Mention horaire enrichi** : phrase courte dans la section `local_schedule.py` existante — "Le fichier horaire.csv a été enrichi pour intégrer les trajets LGV Al Boraq et Marrakech, faisant passer la couverture de 57 % à XX % des LiaisonId."

4. **CLAUDE.md** : mise à jour après chaque phase (état, fichiers créés, métriques, prochaines actions).

---

## Risques et plan de mitigation

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| Couverture horaire reste < 75 % après rebuild | Moyenne | Bloque Phase 1 | Analyse des LiaisonIds manquants ; demander à l'utilisateur si normalisation à ajouter |
| test1.csv a un format différent d'oncf_data.csv | Faible | Bloque Phase 2 | Validation de schéma au début du script 12a ; arrêt explicite si écart |
| Baseline post-test1 fait chuter HR@1 > 5pp | Faible | Décision à valider | Garder rollback prêt via models/archive/ ; analyser distribution test1 vs oncf |
| Un retrain quotidien dépasse 30 min | Moyenne | Allonge le total | Mesurer le Jour 1, ajuster hyperparams (Sprint 2 vs challenger) si besoin |
| Pas assez de données dans fenêtre 365j (jour ancien) | Très faible | Modèle dégradé | Le sliding window est appliqué sur 4+ ans de données, marge confortable |

---

## Critères de complétion

- [ ] Phase 1 : couverture horaire > 75 %, tous les tests `local_schedule` + `schedule` passent
- [ ] Phase 2 : 7 CSVs jour + test1_base.csv créés, somme des lignes = test1.csv total
- [ ] Phase 3 : pipeline complet relancé, nouveau prod en place, ancien archivé, métriques mesurées
- [ ] Phase 4 : 7 retrains exécutés (un par un), `reports/simulation_daily.json` rempli
- [ ] Phase 5 : > 135 tests verts (135 actuels + ~14 nouveaux)
- [ ] Phase 6 : rapport mis à jour, figure ajoutée, CLAUDE.md à jour

## Décisions clés (résumé)

- **Sélection 7 jours** : les 7 derniers jours de `test1.csv` par date de départ.
- **Hyperparamètres** : prod actuels (challenger, max_depth=8, n_estimators=250).
- **Orchestration** : script unique `--day N` lancé manuellement, un jour à la fois.
- **Évaluation** : split interne 80/20 + éval sur J+1 (sauf jour 7).
- **Promotion baseline** : oui, le baseline post-test1 remplace le prod (Option A, archive de sécurité avant).
- **Guardrail** : 5pp dans tous les cas (flag dans simulation, bloqueur en prod).
- **Stockage modèles simulation** : `models/sim/day_N/`, jamais dans `models/` direct.
