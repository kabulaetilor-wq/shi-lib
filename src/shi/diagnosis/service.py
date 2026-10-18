"""Diagnostic « conseil seulement » : texte et journaux en entrée, hypothèses et pistes en sortie.

Aucune commande n'est exécutée, aucun dépôt n'est lu. Les secrets sont masqués AVANT l'analyse.
Fonction pure, sans framework : utilisable depuis l'API HTTP, la ligne de commande ou des tests."""
from __future__ import annotations

from typing import Any, Dict, Optional

from shi.diagnosis.advice import ADVICE
from shi.diagnosis.engine import diagnose
from shi.ingestion.masking import mask_report
from shi.ingestion.normalize import max_severity, parse_logs
from shi.security.guard import scan_untrusted

MAX_DESCRIPTION = 2000
MAX_LOGS = 200_000


def diagnose_text(description: str, logs: str = "", classifier=None) -> Dict[str, Any]:
    description = (description or "")[:MAX_DESCRIPTION]
    logs = (logs or "")[:MAX_LOGS]
    d, m1 = mask_report(description)
    l, m2 = mask_report(logs)
    events = parse_logs(l)
    diag = diagnose(d, events, classifier)
    suspicious = scan_untrusted(l, "logs") + scan_untrusted(d, "description")
    category = diag.category if diag.category in ADVICE else "inconnu"
    top = diag.hypotheses[0] if diag.hypotheses else None
    return {
        "category": category,
        "uncertain": bool(diag.uncertain),
        "confidence": top.confidence if top else 0.0,
        "severity": max_severity(events) if events else "inconnue",
        "hypotheses": [{"id": h.id, "label": h.label, "category": h.category, "confidence": h.confidence,
                        "evidence": [{"source": e.source, "line": e.line_no, "text": e.text} for e in h.evidence]}
                       for h in diag.hypotheses[:5]],
        "facts": diag.facts,
        "notes": diag.notes,
        "advice": ADVICE[category],
        "masked": dict(m1 + m2),
        "suspicious_instructions": suspicious,
        "disclaimer": "Hypothèses à vérifier, pas des certitudes. SHI n'a exécuté aucune action.",
    }
