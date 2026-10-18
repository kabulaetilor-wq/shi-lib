"""Orchestrateur : OBSERVER → COMPRENDRE → DIAGNOSTIQUER → PROPOSER → TESTER → AGIR → VÉRIFIER.

Le modèle ne lance jamais de commande : tout passe par la garde (permissions),
l'exécuteur isolé (commande unique et validée) et le journal d'audit.
"""
from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from shi.analysis.code_analyzer import RepoScan, match_frames, parse_tracebacks, scan_repo
from shi.audit.log import AuditLog
from shi.diagnosis.engine import Diagnosis, diagnose
from shi.ingestion.masking import mask_report
from shi.ingestion.normalize import main_service, max_severity, parse_logs
from shi.planner.planner import PlanStep, build_plan
from shi.sandbox.runner import RunResult, SandboxRunner
from shi.security.guard import Action, Guard, Mode, Policy, scan_untrusted
from shi.solutions.patches import sha256
from shi.solutions.proposer import Solution, build_solutions
from shi.verification.verify import Outcome, verify


class PipelineError(Exception):
    """Erreur métier (état invalide, solution non validée…)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Incident:
    id: str
    project_id: str
    description: str
    logs: str  # déjà masqués
    repo_path: Optional[str]
    created_at: str
    status: str = "ouvert"
    service: str = "inconnu"
    severity: str = "UNKNOWN"
    events_count: int = 0
    masked: dict = field(default_factory=dict)
    suspicious: List[dict] = field(default_factory=list)
    diagnosis: Optional[Diagnosis] = None
    code_zones: List[tuple] = field(default_factory=list)
    solutions: List[Solution] = field(default_factory=list)
    plans: dict = field(default_factory=dict)
    baseline: Optional[RunResult] = None
    decision: str = ""
    verification: Optional[dict] = None
    applied_solution: Optional[str] = None
    notes: List[str] = field(default_factory=list)


class Pipeline:
    def __init__(self, policy: Policy, audit: Optional[AuditLog] = None,
                 runner: Optional[SandboxRunner] = None, classifier=None):
        self.policy = policy
        self.guard = Guard(policy)
        self.audit = audit or AuditLog()
        self.runner = runner or SandboxRunner()
        self.classifier = classifier

    # ------------------------------------------------------------------ 1-2
    def new_incident(self, project_id: str, description: str, logs: str, repo_path: Optional[str] = None) -> Incident:
        """Détection + collecte : les secrets sont masqués avant toute analyse."""
        d, c1 = mask_report(description)
        l, c2 = mask_report(logs)
        inc = Incident("inc-" + uuid.uuid4().hex[:8], project_id, d, l, repo_path, _now(), masked=dict(c1 + c2))
        self.audit.record("systeme", "incident.cree", {"masques": inc.masked}, inc.id)
        return inc

    # ------------------------------------------------------------------ 3-6
    def analyze(self, inc: Incident) -> Incident:
        if not self.guard.check(Action.READ_LOGS).allowed:
            raise PermissionError("lecture des logs non autorisée")
        events = parse_logs(inc.logs, incident_id=inc.id)
        inc.events_count, inc.severity, inc.service = len(events), max_severity(events), main_service(events)
        inc.suspicious = scan_untrusted(inc.logs, "logs") + scan_untrusted(inc.description, "description")

        scan: Optional[RepoScan] = None
        findings = []
        zone_files: List[str] = []
        if inc.repo_path and self.guard.check(Action.READ_CODE).allowed:
            scan = scan_repo(inc.repo_path)
            inc.suspicious += scan.suspicious
            inc.code_zones = match_frames(parse_tracebacks(inc.logs), scan.files.keys())
            zone_files = [z[0] for z in inc.code_zones]
            findings = [(rel, f) for rel, fl in scan.findings.items() for f in fl]
        if inc.suspicious:
            inc.notes.append("%d instruction(s) suspecte(s) détectée(s) dans les données : ignorées, aucune règle modifiée" % len(inc.suspicious))
            self.audit.record("garde", "contenu.suspect", {"n": len(inc.suspicious)}, inc.id)

        inc.diagnosis = diagnose(inc.description, events, self.classifier, findings)
        inc.solutions = build_solutions(inc.diagnosis, scan, zone_files)
        for n, s in enumerate(inc.solutions, start=1):
            s.id = "%s-s%d" % (inc.id, n)
            inc.plans[s.id] = build_plan(s, self.guard)
        inc.status = "analyse"
        self.audit.record("shi", "incident.analyse", {"categorie": inc.diagnosis.category, "incertain": inc.diagnosis.uncertain,
                                                      "solutions": len(inc.solutions)}, inc.id)
        return inc

    # ------------------------------------------------------------------ 8
    def _solution(self, inc: Incident, solution_id: str) -> Solution:
        for s in inc.solutions:
            if s.id == solution_id:
                return s
        raise PipelineError("solution inconnue : %s" % solution_id)

    def test_solution(self, inc: Incident, solution_id: str) -> Solution:
        sol = self._solution(inc, solution_id)
        if sol.patch is None:
            sol.status = "non_testable"
            sol.test = {"outcome": "incertain", "reasons": ["recommandation manuelle : pas de test automatique"]}
            return sol
        for a in (Action.RUN_TESTS, Action.WRITE_BRANCH):
            d = self.guard.check(a)
            if not d.allowed:
                raise PermissionError(d.reason)
        if not inc.repo_path:
            raise PipelineError("aucun dépôt associé à l'incident")
        with self.runner.workspace(inc.repo_path) as ws:
            current = (ws / sol.patch.path).read_text(encoding="utf-8")
            if sha256(current) != sol.patch.base_sha256:
                sol.status = "obsolete"
                sol.test = {"outcome": "incertain", "reasons": ["le fichier a changé depuis l'analyse : relancer l'analyse"]}
                return sol
            if inc.baseline is None:
                inc.baseline = self.runner.run_tests(ws)
                self.audit.record("sandbox", "tests.reference", {"passed": inc.baseline.passed, "failed": inc.baseline.failed}, inc.id)
            self.runner.write_files(ws, {sol.patch.path: sol.patch.new_source})
            after = self.runner.run_tests(ws)
        v = verify(inc.baseline, after)
        sol.test = {"outcome": v.outcome.value, "reasons": v.reasons, "before": _brief(inc.baseline), "after": _brief(after),
                    "isolation": after.isolation}
        sol.status = {Outcome.SUCCESS: "teste_ok", Outcome.FAILURE: "teste_echec", Outcome.UNCERTAIN: "teste_incertain"}[v.outcome]
        if v.outcome == Outcome.SUCCESS:
            sol.score = round(min(0.99, sol.score + 0.25), 2)
            inc.status = "en_attente_approbation"
        else:
            sol.score = 0.0
            inc.status = "echec_tests" if v.outcome == Outcome.FAILURE else inc.status
            inc.notes.append("solution %s non déployable : %s" % (sol.id, "; ".join(v.reasons)))
        self.audit.record("sandbox", "solution.testee", {"solution": sol.id, "resultat": v.outcome.value}, inc.id)
        inc.solutions.sort(key=lambda s: -s.score)
        return sol

    # ------------------------------------------------------------------ 9-10
    def _apply(self, inc: Incident, sol: Solution, actor: str) -> Incident:
        repo = Path(inc.repo_path).resolve()
        target = (repo / sol.patch.path).resolve()
        if repo not in target.parents:
            raise PipelineError("chemin hors du dépôt")
        if sha256(target.read_text(encoding="utf-8")) != sol.patch.base_sha256:
            raise PipelineError("le fichier a changé depuis l'analyse : relancer l'analyse")
        backup = repo / ".shi_backups" / inc.id / sol.patch.path
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
        target.write_text(sol.patch.new_source, encoding="utf-8")
        inc.applied_solution, sol.status = sol.id, "applique"
        self.audit.record(actor, "correctif.applique", {"solution": sol.id, "fichier": sol.patch.path}, inc.id)

        with self.runner.workspace(repo) as ws:  # vérification après action
            after = self.runner.run_tests(ws)
        v = verify(inc.baseline, after)
        inc.verification = {"outcome": v.outcome.value, "reasons": v.reasons, "after": _brief(after)}
        if v.outcome == Outcome.SUCCESS:
            inc.status, inc.decision = "resolu", "Correctif appliqué et vérifié par les tests."
        else:
            self.rollback(inc, "la vérification après action n'est pas concluante : " + "; ".join(v.reasons))
            inc.status = "echec_verification"
        self.audit.record("verification", "verification.apres_action", {"resultat": v.outcome.value}, inc.id)
        return inc

    def approve(self, inc: Incident, solution_id: str, approver: str) -> Incident:
        sol = self._solution(inc, solution_id)
        if sol.patch is None:
            raise PipelineError("recommandation manuelle : SHI n'applique rien automatiquement pour cette solution")
        if sol.status != "teste_ok":
            raise PipelineError("solution non validée par les tests isolés (statut : %s)" % sol.status)
        d = self.guard.check(Action.APPLY_FIX, approved=True)
        if not d.allowed:
            raise PermissionError(d.reason)
        return self._apply(inc, sol, "humain:" + approver)

    def rollback(self, inc: Incident, reason: str = "demandé par l'utilisateur") -> Incident:
        if not inc.applied_solution:
            raise PipelineError("aucun correctif appliqué")
        sol = self._solution(inc, inc.applied_solution)
        repo = Path(inc.repo_path).resolve()
        backup = repo / ".shi_backups" / inc.id / sol.patch.path
        if not backup.exists():
            raise PipelineError("sauvegarde introuvable")
        shutil.copy2(backup, repo / sol.patch.path)
        sol.status, inc.applied_solution, inc.status = "retour_arriere", None, "retour_arriere"
        inc.decision = "Retour arrière effectué : " + reason
        self.audit.record("shi", "correctif.annule", {"solution": sol.id, "raison": reason}, inc.id)
        return inc

    # ------------------------------------------------------------------ mode automatique
    def run_auto(self, inc: Incident) -> Incident:
        if self.policy.mode != Mode.AUTO:
            raise PermissionError("le mode automatique n'est pas activé pour ce projet")
        self.analyze(inc)
        diag = inc.diagnosis
        if diag.uncertain:
            inc.status, inc.decision = "intervention_requise", "Arrêt : diagnostic trop incertain, collecte de preuves ou aide humaine nécessaire."
            return self._done(inc)
        candidates = [s for s in inc.solutions if s.patch and s.risk_level == "low"]
        if not candidates:
            inc.status, inc.decision = "intervention_requise", "Arrêt : aucune action à faible risque disponible."
            return self._done(inc)
        for sol in candidates:
            self.test_solution(inc, sol.id)
            if sol.status != "teste_ok":
                continue
            d = self.guard.check(Action.APPLY_FIX)
            if not d.allowed:
                inc.status, inc.decision = "en_attente_approbation", "Tests réussis mais %s : %s." % (d.label, d.reason)
                return self._done(inc)
            return self._done(self._apply(inc, sol, "shi-auto"))
        inc.status, inc.decision = "echec_tests", "Aucun correctif candidat ne réussit les tests : rien n'est déployé, intervention requise."
        return self._done(inc)

    def _done(self, inc: Incident) -> Incident:
        self.audit.record("shi", "decision", {"statut": inc.status, "decision": inc.decision}, inc.id)
        return inc


def _brief(r: Optional[RunResult]) -> Optional[dict]:
    if r is None:
        return None
    return {"passed": r.passed, "failed": r.failed, "errors": r.errors, "timed_out": r.timed_out, "duration": r.duration}
