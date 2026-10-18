"""Masquage des secrets et données personnelles AVANT toute analyse."""
from __future__ import annotations

import re
from collections import Counter
from typing import Tuple

_RULES = [
    ("cle_privee", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[CLE_PRIVEE]"),
    ("aws", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[SECRET]"),
    ("github", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"), "[SECRET]"),
    ("stripe", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{10,}\b"), "[SECRET]"),
    ("slack", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"), "[SECRET]"),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b"), "[JWT]"),
    ("bearer", re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9\-._~+/]{8,}=*"), r"\1 [SECRET]"),
    ("url_identifiants", re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://[^\s:/@]+):([^\s@/]+)@"), r"\1:[SECRET]@"),
    (
        "cle_valeur",
        re.compile(
            r"(?i)\b(password|passwd|pwd|mot_de_passe|secret|token|api[_-]?key|apikey|access[_-]?key)"
            r"(\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
        ),
        r"\1\2[SECRET]",
    ),
]
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def mask_report(text: str, mask_pii: bool = True) -> Tuple[str, Counter]:
    """Retourne (texte masqué, compteur par type de donnée masquée)."""
    counts: Counter = Counter()
    for name, rx, repl in _RULES:
        text, n = rx.subn(repl, text)
        if n:
            counts[name] += n
    if mask_pii:
        text, n = _EMAIL.subn("[EMAIL]", text)
        if n:
            counts["email"] += n

        def card(m):
            if _luhn(m.group(0)):
                counts["carte"] += 1
                return "[CARTE]"
            return m.group(0)

        text = _CARD.sub(card, text)
    return text, counts


def mask(text: str, mask_pii: bool = True) -> str:
    return mask_report(text, mask_pii)[0]
