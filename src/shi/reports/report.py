"""Rapport d'incident (Markdown) : faits, hypothèses, solutions, tests, décision, résultat réel."""
from __future__ import annotations

from shi.execution.pipeline import Incident

_RESULT = {
    "resolu": "RÉUSSITE — correctif appliqué et vérifié",
    "echec_verification": "ÉCHEC — vérification après action non concluante (retour arrière effectué)",
    "echec_tests": "ÉCHEC — aucun correctif ne passe les tests, rien n'a été déployé",
    "retour_arriere": "ANNULÉ — retour arrière effectué",
    "intervention_requise": "NON RÉSOLU — intervention humaine requise",
    "en_attente_approbation": "NON RÉSOLU — correctif validé en test, en attente d'approbation",
}


def render_report(inc: Incident, audit_ok: bool = True) -> str:
    out = ["# Rapport d'incident %s" % inc.id, "",
           "- Service : %s — gravité : %s — statut : `%s`" % (inc.service, inc.severity, inc.status),
           "- Description : %s" % inc.description]
    if inc.masked:
        out.append("- Données masquées avant analyse : " + ", ".join("%s ×%d" % kv for kv in inc.masked.items()))
    d = inc.diagnosis
    if d:
        out += ["", "## Faits observés"] + ["- " + f for f in d.facts]
        out += ["", "## Hypothèses (non confirmées tant que les tests ne les valident pas)"]
        for h in d.hypotheses:
            out.append("- **%s** (confiance %.2f, catégorie %s)" % (h.label, h.confidence, h.category))
            out += ["    - preuve [%s:%d] %s" % (e.source, e.line_no, e.text) for e in h.evidence[:4]]
        if not d.hypotheses:
            out.append("- aucune hypothèse fondée sur des preuves")
        out += ["- " + n for n in d.notes]
    if inc.suspicious:
        out += ["", "## Contenu suspect ignoré"]
        out += ["- %s ligne %d : %s (« %s »)" % (s["source"], s["line"], s["motif"], s["extrait"]) for s in inc.suspicious[:8]]
    if inc.solutions:
        out += ["", "## Solutions comparées"]
        for s in inc.solutions:
            out.append("### %s — risque %s, score %.2f, statut `%s`" % (s.title, s.risk_level, s.score, s.status))
            out.append("%s\n- Avantages : %s\n- Risques : %s" % (s.description, "; ".join(s.pros), "; ".join(s.risks)))
            if s.test:
                out.append("- Test isolé : %s — %s" % (s.test["outcome"], "; ".join(s.test["reasons"])))
            if s.patch:
                out.append("```diff\n%s```" % s.patch.diff)
            plan = inc.plans.get(s.id, [])
            if plan:
                out.append("Plan : " + " → ".join("%s [%s]" % (p.text, p.decision) for p in plan))
    out += ["", "## Décision", inc.decision or "Aucune décision prise."]
    if inc.verification:
        out += ["", "## Vérification après action", "- %s : %s" % (inc.verification["outcome"], "; ".join(inc.verification["reasons"]))]
    out += ["", "## Résultat", _RESULT.get(inc.status, "INCERTAIN — analyse en cours ou incomplète"),
            "", "_Journal d'audit : chaîne de hachages %s._" % ("valide" if audit_ok else "INVALIDE")]
    return "\n".join(out) + "\n"
