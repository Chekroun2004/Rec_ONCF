# Privacy & CNDP (Loi 09-08) — Notes de conception

## Principes appliqués
- **Minimisation** : n'utiliser que les colonnes nécessaires au modèle.
- **Pas de GPS** : aucune donnée de localisation.
- **Pseudonymisation** : `CodeClient` est un identifiant; en prod on utilise un identifiant pseudonymisé/stable (hash salé côté backend).
- **Rétention** : logs de monitoring limités dans le temps, agrégés si possible.

## Données sensibles
- `CodeClient` : identifiant — traiter comme donnée personnelle (indirecte).

## Monitoring en production
- Stocker (au minimum) : horodatage, user_id pseudonymisé, recommandation top-1/top-3, réservation réelle (liaison), indicateur hit@k.
- Éviter de stocker le détail complet des features si non requis.
