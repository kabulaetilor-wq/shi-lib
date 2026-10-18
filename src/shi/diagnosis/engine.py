"""Moteur de diagnostic : sépare les FAITS observés des HYPOTHÈSES, avec preuves."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from shi.ingestion.normalize import Event, max_severity

CONFIDENCE_THRESHOLD = 0.45

_RX = lambda s: re.compile(s, re.I)  # noqa: E731

PATTERNS = [
    dict(id="timeout", rx=_RX(r"time[ds]?[ -]?out|timeout|délai d'attente|delai d'attente"), cat="reseau", w=0.8,
         label="Appel réseau ou requête en dépassement de délai (dépendance lente ou injoignable)"),
    dict(id="conn_refused", rx=_RX(r"connection refused|econnrefused|connexion refus"), cat="serveur", w=0.85,
         label="Service cible arrêté, port fermé ou pare-feu bloquant"),
    dict(id="oom", rx=_RX(r"out of memory|\boom\b|memoryerror|oom-kill|killed process|mémoire saturée"), cat="performance", w=0.9,
         label="Épuisement de la mémoire (fuite ou charge excessive)"),
    dict(id="disk", rx=_RX(r"no space left|enospc|disk full|disque plein|quota.*dépass"), cat="stockage", w=0.95,
         label="Volume de stockage saturé"),
    dict(id="db", rx=_RX(r"operationalerror|psycopg2|too many connections|deadlock|sqlstate|database is locked|pool.*(épuis|exhaust)|base de donn"),
         cat="base_de_donnees", w=0.85, label="Problème de base de données (connexions, verrous ou requêtes)"),
    dict(id="auth", rx=_RX(r"\b401\b|\b403\b|unauthorized|invalid token|token expired|jwt|authentication failed|jeton .*(expir|refus)|authentification"),
         cat="authentification", w=0.8, label="Échec d'authentification ou jeton invalide/expiré"),
    dict(id="tls", rx=_RX(r"certificate|\bssl\b|\btls\b|x509|certificat"), cat="reseau", w=0.85,
         label="Problème de certificat ou de chiffrement TLS/SSL"),
    dict(id="http5xx", rx=_RX(r"\b50[0-4]\b|internal server error"), cat="serveur", w=0.6,
         label="Erreurs serveur HTTP 5xx"),
    dict(id="paiement", rx=_RX(r"stripe|card_declined|paiement|payment|webhook"), cat="paiement", w=0.7,
         label="Dysfonctionnement de la chaîne de paiement"),
    dict(id="build", rx=_RX(r"build failed|ci failed|npm err|compilation|modulenotfounderror|importerror|dépendance manquante"),
         cat="deploiement", w=0.85, label="Échec de build ou dépendance manquante"),
    dict(id="exception", rx=_RX(r"\b(keyerror|typeerror|attributeerror|valueerror|indexerror|zerodivisionerror|nameerror|assertionerror)\b|traceback \(most recent|exception non gérée"),
         cat="bug_applicatif", w=0.75, label="Exception ou assertion non satisfaite dans le code applicatif"),
    dict(id="etat_partage", rx=_RX(r"mélang|melang|partag[eé]s? entre|contamin|shared state|données? d'un autre (client|utilisateur)"),
         cat="bug_applicatif", w=0.85,
         label="État partagé entre requêtes ou clients (variable globale, cache, argument par défaut mutable)"),
]

# Une règle de code qui renforce une hypothèse.
CODE_LINKS = {"mutable_default": "etat_partage", "requests_no_timeout": "timeout"}


@dataclass
class Evidence:
    source: str
    line_no: int
    text: str


@dataclass
class Hypothesis:
    id: str
    label: str
    category: str
    confidence: float
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class Diagnosis:
    facts: List[str]
    category: str
    category_probs: Dict[str, float]
    hypotheses: List[Hypothesis]
    uncertain: bool
    notes: List[str] = field(default_factory=list)


def _facts(events: List[Event]) -> List[str]:
    facts = ["%d événement(s) analysé(s), gravité maximale : %s" % (len(events), max_severity(events))]
    errors = [e for e in events if e.rank >= 3]
    if errors:
        ts = next((e.timestamp for e in errors if e.timestamp), None)
        facts.append("%d erreur(s) observée(s)%s" % (len(errors), (", première à %s" % ts) if ts else ""))
    norm = Counter(re.sub(r"\d+", "N", e.message.splitlines()[0])[:90] for e in errors)
    for msg, n in norm.most_common(3):
        if n > 1:
            facts.append("message répété %d fois : %s" % (n, msg))
    excs = sorted({m for e in events for m in re.findall(r"\b([A-Z]\w*(?:Error|Exception))\b", e.message)})
    if excs:
        facts.append("exceptions citées : " + ", ".join(excs))
    return facts


def diagnose(description: str, events: List[Event], classifier=None,
             code_findings: Optional[List[Tuple[str, object]]] = None) -> Diagnosis:
    probs: Dict[str, float] = {}
    top_cat, top_p, notes = "inconnue", 0.0, []
    if classifier is not None:
        pred = classifier.predict(description)
        probs, top_cat, top_p = pred.probs, pred.label, pred.confidence
        if pred.uncertain:
            notes.append("classificateur peu sûr (%.2f) : catégorie indicative" % pred.confidence)

    texts = [("description", 0, description)] + [(e.source, e.line_no, e.message) for e in events]
    hyps: Dict[str, Hypothesis] = {}
    weights = {p["id"]: p for p in PATTERNS}

    def confidence(h: Hypothesis) -> float:
        # Chaque preuve indépendante rapproche la confiance du poids de la règle.
        strength = weights[h.id]["w"] * (1 - 0.5 ** len(h.evidence))
        return round(min(0.95, 0.8 * strength + 0.2 * probs.get(h.category, 0.0)), 2)

    for pat in PATTERNS:
        ev = [Evidence(src, ln, msg.splitlines()[0][:160]) for src, ln, msg in texts if pat["rx"].search(msg)]
        if ev:
            h = Hypothesis(pat["id"], pat["label"], pat["cat"], 0.0, ev)
            h.confidence = confidence(h)
            hyps[pat["id"]] = h

    for rel, f in code_findings or []:
        target = hyps.get(CODE_LINKS.get(f.rule, ""))
        if target:  # le code est une preuve supplémentaire, pas une certitude
            target.evidence.append(Evidence("code", f.lineno, "%s:%d %s" % (rel, f.lineno, f.message)))
            target.confidence = confidence(target)

    ranked = sorted(hyps.values(), key=lambda h: -h.confidence)
    if not ranked and classifier is not None:
        ranked = [Hypothesis("categorie", "Problème de type « %s » déduit de la description seule, non confirmé par les logs" % top_cat,
                             top_cat, round(0.5 * top_p, 2), [Evidence("description", 0, description[:160])])]
    category = ranked[0].category if ranked else top_cat
    uncertain = not ranked or ranked[0].confidence < CONFIDENCE_THRESHOLD
    if uncertain:
        notes.append("diagnostic incertain : collecter plus de preuves avant d'agir")
    return Diagnosis(_facts(events), category, probs, ranked, uncertain, notes)
