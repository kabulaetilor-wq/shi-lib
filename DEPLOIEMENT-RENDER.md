# Déployer le service de diagnostic SHI sur Render (sans carte)

Ce service ne fait que **conseiller** : il reçoit une description et des journaux, masque les secrets,
renvoie une catégorie, des hypothèses avec preuves et des pistes de vérification. Il n'exécute aucune commande.

1. Mettez le contenu de ce dossier dans un dépôt GitHub `shi` (à la racine : `pyproject.toml`, `src/`...).
2. Render → **New + → Web Service** (pas Blueprint) → dépôt `shi`.
3. Réglages : Language **Python 3** · Build Command `pip install .` · Start Command `shi-server` · Instance **Free** · Health Check Path `/health`.
4. Variable d'environnement : `SHI_API_KEY` = une longue suite au hasard (16 caractères minimum, mieux 40).
   Option : `SHI_RATE_PER_MIN` (60 par défaut) = nombre de requêtes autorisées par minute.
5. Ouvrez `https://votre-service.onrender.com/health` : `{"status":"ok","model":true}`.

## Utilisation
```
curl -X POST https://votre-service.onrender.com/diagnose \
  -H "Content-Type: application/json" -H "X-API-Key: VOTRE_CLE" \
  -d '{"description":"Les clients ne peuvent plus payer","logs":"ERROR stripe card_declined"}'
```
Réponse : `category`, `uncertain`, `hypotheses` (avec preuves), `advice` (vérifications sûres), `masked`, `suspicious_instructions`.

## Limites
- Le service gratuit s'endort après 15 minutes : le premier appel peut prendre une minute.
- Le classificateur est entraîné sur des phrases rédigées par une personne : voir le README pour les chiffres et leurs limites.
- Ne mettez pas d'informations de clients dans les journaux envoyés : le masquage des secrets est une aide, pas une garantie.
