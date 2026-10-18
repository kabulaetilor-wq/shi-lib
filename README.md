# shi-saash — SHI (SaaS H Intelligence), bibliothèque Python

```bash
pip install shi-saash-0.3.0-py3-none-any.whl     # ou, depuis les sources : pip install .
```

```python
import shi

r = shi.diagnose("Les clients ne peuvent plus payer", logs="ERROR stripe card_declined")
r["category"]      # 'paiement'
r["uncertain"]     # True si les preuves manquent
r["hypotheses"]    # causes possibles, chacune avec ses preuves (ligne de journal)
r["advice"]        # vérifications sûres, en lecture seule
r["masked"]        # secrets masqués avant l'analyse

shi.classify("Le disque est plein").label   # 'stockage'
```

Ligne de commande : `shi classify "texte"`, `shi-server` (service HTTP, variable `SHI_API_KEY`), `python -m shi ...`.
Le modèle d'exemple est fourni dans le paquet. Pour utiliser votre propre modèle entraîné sans toucher au paquet :
`SHI_HOME=/mon/dossier shi train-classifier`.
`pip install "shi-saash[api]"` ajoute FastAPI pour l'API complète (`shi/api`).

**Ce que la bibliothèque ne fait pas** : elle n'exécute aucune commande pour `diagnose`/`classify`, et ne remplace pas
un humain. Voir « Limites connues » plus bas.

**Vérification** : la structure du paquet a été vérifiée en construisant une roue, en l'installant dans un
environnement propre puis en lançant les tests contre le paquet installé. Le fichier `pyproject.toml` n'a pas pu être
compilé dans l'environnement de génération (outil `setuptools` incompatible) : lancez `pip install .` chez vous et
signalez toute erreur.

---

# SHI — SaaS H Intelligence (v0.1)

Implémentation de départ de l'architecture décrite dans le document technique SHI :
**observer → comprendre → diagnostiquer → proposer → tester → agir → vérifier**.

SHI n'est pas un chatbot : c'est une chaîne de composants (modèles d'IA minuscules + logiciels
classiques déterministes) où **le modèle ne lance jamais de commande**. Tout passe par une garde
de permissions, un exécuteur isolé et un journal d'audit.

## Démarrage rapide

```bash
pip install numpy            # seul prérequis du cœur (OK sur Pydroid 3)
pip install pytest fastapi uvicorn httpx   # optionnels : tests pytest + API

python -m shi.cli build-dataset      # génère datasets/{train,val,test}.jsonl
python -m shi.cli train-classifier    # SHI-0.2 : entraîne et évalue
python -m shi.cli classify "Les pages chargent très lentement"
python -m shi.cli train-transformer --steps 600   # SHI-0.3 : mini-Transformer
python -m shi.cli generate "Le serveur"
python -m shi.cli demo --approve      # cycle complet sur le dépôt de démonstration
python -m shi.cli demo --auto         # même cycle en mode automatique

python -m unittest discover -s tests -t .   # ou : pytest
```

API (nécessite FastAPI) :

```bash
export SHI_API_KEY="une-longue-clé-secrète"
export SHI_ALLOWED_ROOTS="/chemin/vers/mes/dépôts"
python -m shi.cli serve
```

## Service de diagnostic (conseil seulement)

`shi-server` (variable `SHI_API_KEY`) démarre un petit serveur HTTP sans dépendance autre que NumPy :
`POST /diagnose` renvoie catégorie, hypothèses avec preuves et pistes de vérification sûres. Voir `DEPLOIEMENT-RENDER.md`.
Les 11 tests de `tests/test_diagnose_service.py` passent (serveur réel lancé et interrogé).

## Ce qui est implémenté (phases du plan)

| Phase | Module | Contenu |
|---|---|---|
| 1-2 | `app/models/classifier` | SHI-0.2 (corpus manuel `corpus_libre.py`, `corpus_extra.py`, lexique `lexicon.py`, essais dans `experiments/`) : tokenizer français, TF-IDF (mots, bigrammes, 4-grammes), régression softmax NumPy + Adam, arrêt anticipé, métriques (précision, rappel, matrice de confusion), sauvegarde |
| 3 | `app/models/transformer` | SHI-0.3 : mini-Transformer caractère (attention causale, FFN, Adam), rétropropagation écrite à la main et **vérifiée par test de gradient numérique** |
| 4 | `app/ingestion` | Masquage des secrets/PII (clés, JWT, mots de passe, e-mails, cartes validées Luhn) **avant** analyse ; normalisation des logs en événements |
| 5 | `app/analysis`, `app/diagnosis`, `app/solutions`, `app/planner` | Faits vs hypothèses avec preuves ; analyse AST ; tracebacks → fichiers ; plusieurs solutions comparées (avantages, risques, score) ; correctifs candidats déterministes avec diff |
| 6 | `app/sandbox` | Copie temporaire, environnement épuré, limites CPU/mémoire/durée/fichiers, groupe de processus tué au délai, réseau coupé si `unshare` est disponible, liens symboliques ignorés, commande unique validée |
| 7 | `app/security`, `app/execution` | Mode manuel (diff → tests → approbation), garde de permissions, sauvegarde + retour arrière |
| 8 | `app/execution` | Mode automatique limité aux actions pré-autorisées ; s'arrête si diagnostic incertain, tests en échec ou action sensible |
| — | `app/audit`, `app/verification`, `app/reports` | Journal à chaînage de hachages ; verdict réussite / échec / incertain ; rapport Markdown |
| — | `app/api` | Endpoints du document (+ `/auto`, `/rollback`, `/api/audit/verify`), clé d'API, dépôts limités aux dossiers autorisés |

Correctifs automatiques disponibles : argument par défaut mutable, `except:` nu, `== None`,
appels `requests` sans `timeout`.

## Règles de sécurité appliquées

- Lecture seule par défaut ; `APPLY_FIX` doit être accordé au projet. Une approbation humaine ne crée jamais une permission absente.
- Aucune modification du dépôt réel sans tests isolés réussis ; sauvegarde avant, vérification après, retour arrière automatique si la vérification échoue.
- Une action destructive ne peut jamais être automatique.
- Le contenu du dépôt et des logs est **non fiable** : les instructions suspectes (README, commentaires, logs) sont signalées dans le rapport, jamais obéies.
- Un fichier modifié depuis l'analyse invalide le correctif (empreinte SHA-256).
- Le rapport ne déclare « résolu » que si les tests prouvent la correction ; un bug non reproduit donne « incertain ».

## Limites connues (à lire)

- **Précision du classificateur (v0.2.1)** : entraîné sur environ 420 phrases rédigées à la main, 12 catégories (10 d'incidents + `securite` + `inconnu` pour refuser les phrases hors sujet). Exactitude de **0,88** sur 60 phrases écrites à part et jamais utilisées pour entraîner ni régler (`datasets/test_final.jsonl`, 5 par catégorie). Précautions : un seul auteur et un seul style ; 60 phrases donnent une incertitude d'environ ±8 points ; ce n'est **pas** une mesure sur de vrais incidents. Historique mesuré : gabarits seuls 0,39 sur des phrases libres ; ajouter 264 phrases libres 0,51 ; ajouter 180 phrases + 60 phrases de développement 0,88. **La quantité et la variété des données comptent bien plus que le réglage du modèle** (régularisation, taux d'apprentissage, lexique de mots-clés : aucun gain sur le test final, le lexique `lexicon.py` est donc désactivé par défaut). Pour progresser : de vrais incidents anonymisés, de plusieurs auteurs.
- **Mini-Transformer** : modèle minuscule (un bloc, caractères). Il génère du texte plausible mais approximatif ; ce n'est pas un réparateur.
- **Isolation** : ce n'est pas un conteneur. Pour du code réellement hostile ou en production, exécuter le runner dans un conteneur/VM dédié (Docker rootless, gVisor, Firecracker) sans accès à la production. Le niveau d'isolation réellement obtenu (réseau coupé ou non) est écrit dans chaque résultat de test.
- **Non réalisé** : connecteurs GitHub/logs réels, PostgreSQL, file Redis/Celery, interface Android, déploiement. Le stockage de l'API est en mémoire.
- **Tests** : la suite complète passe en `unittest` (49 tests exécutés). Les 5 tests d'API sont ignorés si FastAPI/httpx sont absents ; l'API n'a donc pas été exécutée dans l'environnement de génération et doit être testée chez vous (`pip install fastapi httpx`, puis relancer les tests).
- Les corrections ne sont validées que par les tests du dépôt : des tests faibles donnent une validation faible.

## Organisation

```
app/            cœur de SHI (voir tableau)
datasets/       train/val/test.jsonl (générés) + test_manuel.jsonl (phrases libres)
demo_repo/      dépôt de démonstration avec un vrai bug (argument par défaut mutable)
artifacts/      modèles entraînés (classifier/, transformer/)
tests/          suite de tests
```

## Étapes suivantes

1. Remplacer le jeu synthétique par de vrais incidents anonymisés ; refaire l'évaluation sur un jeu de test jamais utilisé.
2. Connecteur GitHub en lecture (jetons à portée limitée), puis logs/métriques.
3. Runner en conteneur ; PostgreSQL ; file de tâches ; revue de sécurité avant toute action en production.
4. Application Android : écrans incidents, solutions, diff, approbation (elle appelle l'API ; aucun secret en clair sur le téléphone).
