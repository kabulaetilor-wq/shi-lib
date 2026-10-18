"""Génération et comparaison de plusieurs solutions (avantages, risques)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from shi.analysis.code_analyzer import RepoScan
from shi.diagnosis.engine import Diagnosis
from shi.security.guard import Action
from shi.solutions.patches import Patch, propose_patches


@dataclass
class Solution:
    id: str
    title: str
    description: str
    pros: List[str]
    risks: List[str]
    risk_level: str  # low | medium | high
    score: float
    required_action: Action
    patch: Optional[Patch] = None
    status: str = "proposee"
    test: Optional[dict] = None


# (titre, description, avantages, risques, niveau de risque, action requise)
_GENERIC = {
    "serveur": [("Redémarrage contrôlé du service", "Redémarrer une instance à la fois après avoir sauvegardé les logs.",
                 ["Rétablit vite le service"], ["Coupure brève", "Peut masquer la cause racine"], "medium", Action.DEPLOY)],
    "performance": [("Profiler puis ajuster les ressources/caches", "Mesurer CPU, mémoire et requêtes lentes avant d'augmenter les ressources.",
                     ["Cible la vraie cause"], ["Coût d'infrastructure", "Résultat non immédiat"], "medium", Action.DEPLOY)],
    "base_de_donnees": [("Vérifier pool, verrous et requêtes lentes", "Examiner les connexions ouvertes, les verrous et le plan des requêtes.",
                         ["Lecture seule, sans risque"], ["Ne corrige pas seul"], "low", Action.READ_LOGS)],
    "authentification": [("Contrôler horloge, durée de vie et clés des jetons", "Comparer l'heure serveur, l'expiration et la rotation des clés.",
                          ["Lecture seule"], ["Peut demander un redéploiement"], "low", Action.READ_LOGS)],
    "reseau": [("Vérifier DNS, certificats et règles de pare-feu", "Contrôler la résolution, la validité des certificats et les règles.",
                ["Lecture seule"], ["Correction éventuelle chez un tiers"], "low", Action.READ_LOGS)],
    "paiement": [("Vérifier clés API, webhooks et rejouer les événements", "Contrôler les clés, la réception des webhooks et rejouer les événements perdus.",
                  ["Cible la chaîne de paiement"], ["Risque de doublons si mal rejoués"], "medium", Action.DEPLOY)],
    "deploiement": [("Retour arrière vers la dernière version saine", "Revenir à la version précédente puis corriger en branche.",
                     ["Rétablit rapidement"], ["Perd les nouveautés de la version"], "medium", Action.DEPLOY)],
    "stockage": [("Libérer de l'espace ou étendre le volume", "Archiver/rotater les journaux ou agrandir le volume.",
                  ["Débloque les écritures"], ["Suppression = destructive : approbation humaine obligatoire"], "high", Action.DESTRUCTIVE)],
    "interface": [("Reproduire sur navigateur et mobile, puis corriger le front", "Reproduire avec la console du navigateur ouverte.",
                   ["Sans risque pour la production"], ["Correction à écrire"], "low", Action.READ_LOGS)],
    "bug_applicatif": [("Écrire un test qui reproduit le bug", "Transformer l'incident en test automatique avant de corriger.",
                        ["Évite les régressions"], ["Demande du temps"], "low", Action.RUN_TESTS)],
}


def build_solutions(diag: Diagnosis, scan: Optional[RepoScan], zone_files: List[str]) -> List[Solution]:
    sols: List[Solution] = []
    hyp_cats = {h.id for h in diag.hypotheses}
    if scan:
        for rel, source in scan.files.items():
            if rel not in scan.findings:
                continue
            for p in propose_patches(rel, source):
                score = 0.55 + (0.2 if rel in zone_files else 0.0)
                if (p.rule == "mutable_default" and "etat_partage" in hyp_cats) or (p.rule == "requests_no_timeout" and "timeout" in hyp_cats):
                    score += 0.2
                if p.risk == "medium":
                    score -= 0.1
                sols.append(Solution("", "Correctif automatique : %s (%s)" % (p.rule, rel), p.description,
                                     ["Testable en environnement isolé", "Diff inspectable", "Réversible"],
                                     ["Ne couvre que ce que les tests vérifient"], p.risk, round(score, 2),
                                     Action.APPLY_FIX, p))
    for title, desc, pros, risks, level, action in _GENERIC.get(diag.category, []):
        sols.append(Solution("", title, desc, pros, risks, level, 0.35 if level != "low" else 0.4, action))
    evidence_score = 0.7 if diag.uncertain else 0.3
    sols.append(Solution("", "Collecter davantage de preuves", "Ajouter logs/métriques ciblés et reproduire le problème.",
                         ["Aucun risque", "Réduit l'incertitude"], ["Ne résout rien directement"], "low",
                         evidence_score, Action.READ_LOGS))
    sols.sort(key=lambda s: -s.score)
    return sols
