"""Planificateur : étapes candidates, chacune soumise à la garde."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from shi.security.guard import Action, Guard
from shi.solutions.proposer import Solution


@dataclass
class PlanStep:
    order: int
    text: str
    action: Action
    decision: str  # autorisé | approbation requise | bloqué
    reason: str


def build_plan(solution: Solution, guard: Guard) -> List[PlanStep]:
    if solution.patch:
        raw = [
            ("Reproduire le problème (tests de référence) dans la copie isolée", Action.RUN_TESTS),
            ("Appliquer le correctif dans la copie isolée (%s)" % solution.patch.path, Action.WRITE_BRANCH),
            ("Exécuter les tests sur la copie corrigée", Action.RUN_TESTS),
            ("Appliquer le correctif au dépôt réel (sauvegarde pour retour arrière)", Action.APPLY_FIX),
            ("Vérifier après action et signaler le résultat réel", Action.RUN_TESTS),
        ]
    else:
        raw = [("Mise en œuvre manuelle : %s" % solution.title, solution.required_action)]
    steps = []
    for i, (text, action) in enumerate(raw, start=1):
        d = guard.check(action)
        steps.append(PlanStep(i, text, action, d.label, d.reason))
    return steps
