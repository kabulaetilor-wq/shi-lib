"""Génération et chargement du jeu de données d'incidents (français).

Les ensembles sont séparés par GABARIT : les phrases de test utilisent des
formulations jamais vues à l'entraînement, ce qui mesure la généralisation et
non la mémorisation. Ce jeu est synthétique : il sert à apprendre le processus,
pas à prouver une précision sur de vrais incidents.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

SERVICES = [
    "le service de facturation", "l'API", "le site", "l'application",
    "le backend", "le portail client", "la plateforme", "le module d'export",
]
PREFIXES = [("", False), ("Urgent : ", False), ("Depuis hier, ", True), ("Signalement client : ", False)]

# 8 gabarits par catégorie : 0-4 entraînement, 5 validation, 6-7 test.
TEMPLATES: Dict[str, List[str]] = {
    "serveur": [
        "{Svc} ne répond plus", "Erreur 500 sur {svc}", "{Svc} plante au démarrage",
        "Le processus de {svc} s'arrête sans raison", "Impossible de joindre {svc}, le serveur semble hors service",
        "Le redémarrage de {svc} boucle indéfiniment", "Code 502 renvoyé par {svc}", "{Svc} est indisponible depuis ce matin",
    ],
    "performance": [
        "{Svc} charge très lentement", "Les temps de réponse de {svc} explosent", "Latence élevée constatée sur {svc}",
        "{Svc} est lent aux heures de pointe", "Le processeur est saturé à 100% sur {svc}",
        "Les pages de {svc} s'affichent après un long délai", "Forte consommation de mémoire sur {svc}",
        "Les requêtes vers {svc} mettent plusieurs secondes",
    ],
    "base_de_donnees": [
        "La connexion à la base de données de {svc} échoue", "Requête SQL en timeout sur {svc}",
        "Deadlock détecté dans la base de {svc}", "La migration de schéma a échoué pour {svc}",
        "Trop de connexions ouvertes sur la base de {svc}", "Les données de {svc} sont introuvables en base",
        "Le pool de connexions PostgreSQL de {svc} est épuisé", "Erreur d'intégrité sur la table des commandes de {svc}",
    ],
    "authentification": [
        "Les utilisateurs ne peuvent pas ouvrir leur session sur {svc}", "Le jeton d'accès de {svc} est refusé",
        "Mot de passe correct mais connexion refusée sur {svc}", "Erreur 401 sur {svc} pour tous les comptes",
        "L'authentification à deux facteurs échoue sur {svc}", "Les sessions de {svc} se déconnectent toutes seules",
        "Le jeton JWT de {svc} a expiré trop tôt", "La connexion SSO ne fonctionne plus sur {svc}",
    ],
    "reseau": [
        "Délai d'attente dépassé en contactant {svc}", "Le DNS ne résout plus le domaine de {svc}",
        "Certificat SSL expiré sur {svc}", "Perte de paquets entre {svc} et le répartiteur de charge",
        "Connexion refusée par le pare-feu pour {svc}", "Le proxy bloque les appels sortants de {svc}",
        "Erreur TLS lors de l'appel à {svc}", "Le réseau est coupé entre les datacenters de {svc}",
    ],
    "paiement": [
        "Les paiements échouent sur {svc}", "Le webhook Stripe de {svc} n'est plus reçu",
        "Les cartes bancaires sont refusées sur {svc}", "Doublons de facturation détectés sur {svc}",
        "Le remboursement client ne passe pas sur {svc}", "Les factures de {svc} ne sont pas générées",
        "La passerelle de paiement de {svc} renvoie une erreur", "Transactions bloquées en attente sur {svc}",
    ],
    "deploiement": [
        "Le déploiement de {svc} a échoué", "La pipeline CI est en échec pour {svc}",
        "La nouvelle version de {svc} casse la production", "Le build de {svc} ne compile plus",
        "Retour arrière nécessaire après la mise en production de {svc}", "L'image Docker de {svc} ne se construit pas",
        "Une dépendance manquante empêche le build de {svc}", "Le script de livraison de {svc} s'interrompt en route",
    ],
    "stockage": [
        "Le disque est plein sur {svc}", "Plus d'espace de stockage disponible pour {svc}",
        "Les fichiers téléversés sur {svc} sont perdus", "Le bucket de sauvegarde de {svc} est inaccessible",
        "Les journaux de {svc} saturent le volume", "La sauvegarde nocturne de {svc} a échoué",
        "Quota de stockage dépassé pour {svc}", "Erreur d'écriture sur le système de fichiers de {svc}",
    ],
    "interface": [
        "Le bouton d'enregistrement de {svc} ne fait rien", "L'affichage de {svc} est cassé sur mobile",
        "Page blanche à l'ouverture de {svc}", "Les formulaires de {svc} n'envoient pas les données",
        "Erreur JavaScript dans le tableau de bord de {svc}", "Les images de {svc} ne s'affichent plus",
        "Le menu de {svc} est illisible sur petit écran", "La traduction est incorrecte dans l'interface de {svc}",
    ],
    "bug_applicatif": [
        "Une exception non gérée interrompt {svc}", "Le calcul du total est faux sur {svc}",
        "Les données de deux clients sont mélangées dans {svc}", "Une régression fonctionnelle est apparue dans {svc}",
        "Valeur nulle inattendue provoque une erreur dans {svc}", "Un cas limite fait planter le calcul de {svc}",
        "Traceback Python dans les journaux de {svc}", "Le résultat affiché par {svc} ne correspond pas au résultat attendu",
    ],
}
EXTRA: Dict[str, List[str]] = {
    "serveur": ["Le serveur de {svc} est en panne", "{Svc} renvoie une erreur 503 en continu",
                "Les instances de {svc} redémarrent en boucle", "Aucune réponse du serveur hébergeant {svc}"],
    "performance": ["{Svc} est très lent pour tous les utilisateurs", "Temps de chargement trop long sur {svc}",
                    "Le processeur et la mémoire de {svc} sont surchargés", "Ralentissements importants constatés sur {svc}"],
    "base_de_donnees": ["La base de données de {svc} est inaccessible", "Les requêtes SQL de {svc} sont bloquées par un verrou",
                        "Erreur de connexion à PostgreSQL depuis {svc}", "La base de {svc} refuse de nouvelles connexions"],
    "authentification": ["Impossible de se connecter à {svc} avec un compte valide", "Le token d'authentification de {svc} est invalide",
                         "Les utilisateurs sont déconnectés en permanence de {svc}", "Le login sur {svc} est refusé malgré le bon mot de passe"],
    "reseau": ["Le domaine de {svc} ne répond plus sur le réseau", "Le certificat de {svc} a expiré",
               "Perte de connexion réseau vers {svc}", "Appels réseau vers {svc} bloqués par le pare-feu"],
    "paiement": ["Le paiement par carte échoue sur {svc}", "Les clients ne peuvent pas payer leur commande sur {svc}",
                 "Les webhooks de paiement de {svc} sont perdus", "La facturation de {svc} produit des erreurs"],
    "deploiement": ["La mise en production de {svc} a échoué", "Le pipeline de livraison de {svc} est bloqué",
                    "La compilation de {svc} échoue après la dernière version", "Retour à la version précédente de {svc} nécessaire après le déploiement"],
    "stockage": ["L'espace disque de {svc} est saturé", "Impossible d'écrire des fichiers sur {svc}, volume plein",
                 "La sauvegarde de {svc} échoue faute d'espace", "Le stockage de {svc} est presque plein"],
    "interface": ["L'interface de {svc} ne s'affiche pas correctement", "Un bouton de {svc} ne répond pas au clic",
                  "Le formulaire de {svc} est cassé sur téléphone", "Page blanche et erreur JavaScript sur {svc}"],
    "bug_applicatif": ["Une erreur dans le code de {svc} provoque une exception", "Le calcul du montant est incorrect dans {svc}",
                       "Un bug empêche {svc} de gérer un cas particulier", "Les paniers de plusieurs clients sont mélangés dans {svc}"],
}
for _k, _v in EXTRA.items():
    TEMPLATES[_k] = TEMPLATES[_k] + _v

# 12 gabarits par catégorie. Les gabarits de test (6-7) restent inchangés.
SPLIT_INDEXES = {"train": [0, 1, 2, 3, 4, 8, 9, 10], "val": [5, 11], "test": [6, 7]}


def _fill(template: str, svc: str, prefix: Tuple[str, bool]) -> str:
    body = template.format(svc=svc, Svc=svc[0].upper() + svc[1:])
    pre, lower = prefix
    if lower:
        body = body[0].lower() + body[1:]
    return pre + body


def build(seed: int = 0) -> Dict[str, List[dict]]:
    rng = random.Random(seed)
    out: Dict[str, List[dict]] = {k: [] for k in SPLIT_INDEXES}
    for split, idxs in SPLIT_INDEXES.items():
        seen = set()
        for label, templates in TEMPLATES.items():
            for i in idxs:
                for svc in SERVICES:
                    for prefix in PREFIXES:
                        text = _fill(templates[i], svc, prefix)
                        if text not in seen:
                            seen.add(text)
                            out[split].append({"text": text, "label": label})
        rng.shuffle(out[split])
    return out


def write_jsonl(path, rows: List[dict]) -> None:
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def load_jsonl(path) -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            texts.append(r["text"])
            labels.append(r["label"])
    return texts, labels


def check_no_overlap(*splits: List[str]) -> bool:
    """Vrai si aucune phrase n'apparaît dans deux ensembles différents."""
    seen = set()
    for s in splits:
        cur = set(s)
        if seen & cur:
            return False
        seen |= cur
    return True
