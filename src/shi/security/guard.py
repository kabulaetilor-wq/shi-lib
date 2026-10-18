"""Garde de sécurité : modes, permissions, validation de commandes, contenu suspect.

Règle centrale : les données analysées (logs, code, README) ne sont jamais des
autorisations. Seule la `Policy` définie par l'utilisateur décide.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import List


class Mode(str, Enum):
    MANUAL = "manual"
    AUTO = "auto"


class Action(str, Enum):
    READ_CODE = "read_code"
    READ_LOGS = "read_logs"
    WRITE_BRANCH = "write_branch"  # écriture dans la copie isolée uniquement
    RUN_TESTS = "run_tests"
    APPLY_FIX = "apply_fix"  # modification du dépôt réel
    DEPLOY = "deploy"
    DESTRUCTIVE = "destructive"


# Lecture seule par défaut (+ tests et copie isolée).
DEFAULT_GRANTED = frozenset(
    {Action.READ_CODE, Action.READ_LOGS, Action.WRITE_BRANCH, Action.RUN_TESTS}
)


@dataclass(frozen=True)
class Policy:
    mode: Mode = Mode.MANUAL
    granted: frozenset = DEFAULT_GRANTED
    # Actions que SHI peut faire seul en mode automatique (jamais DESTRUCTIVE).
    auto_allowed: frozenset = frozenset()

    def __post_init__(self):
        if Action.DESTRUCTIVE in self.auto_allowed:
            raise ValueError("Une action destructive ne peut jamais être automatique.")


@dataclass(frozen=True)
class Decision:
    allowed: bool
    needs_approval: bool
    reason: str

    @property
    def label(self) -> str:
        if self.allowed:
            return "autorisé"
        return "approbation requise" if self.needs_approval else "bloqué"


class Guard:
    """Composant déterministe : le modèle d'IA ne peut pas le contourner."""

    def __init__(self, policy: Policy):
        self.policy = policy

    def check(self, action: Action, approved: bool = False) -> Decision:
        p = self.policy
        if action not in p.granted:
            return Decision(False, False, f"permission « {action.value} » non accordée (lecture seule par défaut)")
        if action in (Action.READ_CODE, Action.READ_LOGS, Action.WRITE_BRANCH, Action.RUN_TESTS):
            return Decision(True, False, "action sans effet sur la production")
        if action == Action.DESTRUCTIVE:
            if approved:
                return Decision(True, False, "action destructive approuvée par un humain")
            return Decision(False, True, "action destructive : approbation humaine obligatoire")
        # APPLY_FIX, DEPLOY
        if approved:
            return Decision(True, False, "approuvé par un humain")
        if p.mode == Mode.AUTO and action in p.auto_allowed:
            return Decision(True, False, "action pré-autorisée en mode automatique")
        return Decision(False, True, "approbation requise")


_PYTEST_FLAG = re.compile(r"^(-q|-x|-v|--maxfail=\d+|--tb=(short|line|no)|no:cacheprovider|-p)$")
_UNITTEST_FLAG = re.compile(r"^(-q|-v|-f|discover|-s|-t)$")
_PATH_RE = re.compile(r"^[\w./-]+$")


def validate_test_command(argv: List[str]) -> None:
    """Accepte uniquement `python -m pytest|unittest` avec options et chemins relatifs sûrs."""
    if len(argv) < 3 or argv[0] != sys.executable or argv[1] != "-m" or argv[2] not in ("pytest", "unittest"):
        raise ValueError("commande non autorisée : seuls « python -m pytest » et « python -m unittest » sont permis")
    flag_re = _PYTEST_FLAG if argv[2] == "pytest" else _UNITTEST_FLAG
    for arg in argv[3:]:
        if flag_re.match(arg):
            continue
        if arg.startswith("-") or arg.startswith("/") or not _PATH_RE.match(arg) or ".." in arg.split("/"):
            raise ValueError(f"argument non autorisé : {arg!r}")


_SUSPICIOUS = [
    ("ignorer les instructions", re.compile(r"(?i)ignore[rz]?\s+(all\s+|any\s+|toutes\s+)?(the\s+|les\s+)?(previous|prior|above|précédentes|precedentes|ci-dessus)?\s*(instructions|rules|règles|regles)")),
    ("oublier les règles", re.compile(r"(?i)(disregard|forget|oublie[rz]?)\s+(the\s+|your\s+|les\s+|tes\s+|vos\s+)?(rules|instructions|guidelines|règles|regles|consignes)")),
    ("désactiver la sécurité", re.compile(r"(?i)(désactiv|desactiv|disable|bypass|contourn)\w*\s+(la\s+|le\s+|the\s+|les\s+)?(sécurité|securite|security|guard|garde|permissions?)")),
    ("déploiement forcé", re.compile(r"(?i)(approve|approuve|deploy|déploie|deploie)\w*\s+(automatically|automatiquement|immediately|immédiatement|en production|to production)")),
    ("commande destructive", re.compile(r"(?i)\brm\s+-rf\s+/")),
    ("téléchargement exécuté", re.compile(r"(?i)(curl|wget)[^|\n]*\|\s*(ba)?sh")),
    ("changement de rôle", re.compile(r"(?i)\byou are now\b|\btu es maintenant\b")),
]


def scan_untrusted(text: str, source: str = "") -> List[dict]:
    """Signale les instructions suspectes dans du contenu non fiable.

    Le résultat est informatif : il n'ouvre ni ne ferme aucune permission.
    """
    findings = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for name, rx in _SUSPICIOUS:
            if rx.search(line):
                findings.append(
                    {"source": source, "line": line_no, "motif": name, "extrait": line.strip()[:100]}
                )
    return findings
