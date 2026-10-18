"""Vérification avant/après : réussite, échec ou résultat incertain (jamais de fausse victoire)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

from shi.sandbox.runner import RunResult


class Outcome(str, Enum):
    SUCCESS = "reussite"
    FAILURE = "echec"
    UNCERTAIN = "incertain"


@dataclass
class Verification:
    outcome: Outcome
    reasons: List[str]


def verify(before: RunResult, after: RunResult) -> Verification:
    reasons: List[str] = []
    if after.timed_out:
        return Verification(Outcome.UNCERTAIN, ["les tests après correctif ont dépassé le délai"])
    if before.timed_out:
        return Verification(Outcome.UNCERTAIN, ["les tests de référence ont dépassé le délai"])
    if after.passed + after.failed + after.errors == 0:
        return Verification(Outcome.UNCERTAIN, ["aucun test exécuté après le correctif"])
    if before.failed + before.errors == 0:
        return Verification(Outcome.UNCERTAIN, ["le problème n'est pas reproduit par les tests de référence : impossible de prouver la correction"])
    if after.failed + after.errors > 0:
        reasons.append("%d test(s) échouent encore après le correctif" % (after.failed + after.errors))
        return Verification(Outcome.FAILURE, reasons)
    if after.passed < before.passed:
        return Verification(Outcome.FAILURE, ["régression : %d test(s) réussis avant, %d après" % (before.passed, after.passed)])
    return Verification(
        Outcome.SUCCESS,
        ["%d test(s) en échec avant, 0 après ; %d réussis (avant : %d)" % (before.failed + before.errors, after.passed, before.passed)],
    )
