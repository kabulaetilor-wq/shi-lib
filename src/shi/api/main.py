"""API FastAPI : endpoints versionnés, authentifiés, validés et journalisés.

Stockage en mémoire pour le prototype ; à remplacer par PostgreSQL (phase 9).
Les tâches longues (analyse, tests) s'exécutent en arrière-plan ; l'état est consultable.
"""
from __future__ import annotations

import hmac
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from shi.audit.log import AuditLog
from shi.execution.pipeline import Incident, Pipeline, PipelineError
from shi.models.classifier import load_default_classifier
from shi.reports.report import render_report
from shi.security.guard import DEFAULT_GRANTED, Action, Mode, Policy
from shi.utils import to_jsonable


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    repo_path: str
    mode: Mode = Mode.MANUAL
    granted: List[Action] = []        # permissions en plus du défaut (lecture seule + tests)
    auto_allowed: List[Action] = []   # actions que SHI peut faire seul (mode auto)


class IncidentIn(BaseModel):
    description: str = Field(min_length=3, max_length=2000)
    logs: str = Field(default="", max_length=500_000)


class ApproveIn(BaseModel):
    approver: str = Field(min_length=1, max_length=100)


def create_app(api_key: str, allowed_roots: List[Path], classifier=None, audit: Optional[AuditLog] = None) -> FastAPI:
    if not api_key:
        raise ValueError("une clé d'API est obligatoire")
    roots = [Path(r).resolve() for r in allowed_roots]
    audit = audit or AuditLog()
    classifier = classifier if classifier is not None else load_default_classifier()
    projects: Dict[str, dict] = {}
    incidents: Dict[str, Incident] = {}
    locks: Dict[str, threading.Lock] = {}
    app = FastAPI(title="SHI — SaaS H Intelligence", version="0.1.0")

    def auth(x_api_key: str = Header(default="")):
        if not hmac.compare_digest(x_api_key.encode(), api_key.encode()):
            raise HTTPException(401, "authentification requise")

    def get_incident(incident_id: str) -> Incident:
        if incident_id not in incidents:
            raise HTTPException(404, "incident introuvable")
        return incidents[incident_id]

    def pipeline_for(inc: Incident) -> Pipeline:
        return projects[inc.project_id]["pipeline"]

    def find_solution(solution_id: str):
        for inc in incidents.values():
            for s in inc.solutions:
                if s.id == solution_id:
                    return inc, s
        raise HTTPException(404, "solution introuvable")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/projects", status_code=201, dependencies=[Depends(auth)])
    def create_project(body: ProjectIn):
        repo = Path(body.repo_path).resolve()
        if not repo.is_dir() or not any(r == repo or r in repo.parents for r in roots):
            raise HTTPException(400, "dépôt introuvable ou hors des dossiers autorisés")
        if Action.DESTRUCTIVE in body.auto_allowed:
            raise HTTPException(422, "une action destructive ne peut jamais être automatique")
        policy = Policy(body.mode, DEFAULT_GRANTED | frozenset(body.granted), frozenset(body.auto_allowed))
        pid = "prj-" + uuid.uuid4().hex[:8]
        projects[pid] = {"id": pid, "name": body.name, "repo_path": str(repo), "mode": body.mode.value,
                         "pipeline": Pipeline(policy, audit, classifier=classifier)}
        audit.record("api", "projet.cree", {"id": pid, "mode": body.mode.value})
        return {k: v for k, v in projects[pid].items() if k != "pipeline"}

    @app.post("/api/projects/{project_id}/incidents", status_code=201, dependencies=[Depends(auth)])
    def create_incident(project_id: str, body: IncidentIn):
        if project_id not in projects:
            raise HTTPException(404, "projet introuvable")
        p = projects[project_id]
        inc = p["pipeline"].new_incident(project_id, body.description, body.logs, p["repo_path"])
        incidents[inc.id] = inc
        locks[inc.id] = threading.Lock()
        return {"id": inc.id, "status": inc.status}

    def _run(inc_id: str, job):
        inc = incidents[inc_id]
        with locks[inc_id]:
            try:
                job(inc)
            except Exception as exc:  # l'état d'erreur reste consultable
                inc.status = "erreur"
                inc.notes.append("erreur interne : %s" % type(exc).__name__)

    @app.post("/api/incidents/{incident_id}/analyze", status_code=202, dependencies=[Depends(auth)])
    def analyze(incident_id: str, tasks: BackgroundTasks):
        inc = get_incident(incident_id)
        inc.status = "analyse_en_cours"
        tasks.add_task(_run, incident_id, pipeline_for(inc).analyze)
        return {"id": incident_id, "status": inc.status}

    @app.get("/api/incidents/{incident_id}", dependencies=[Depends(auth)])
    def read_incident(incident_id: str):
        inc = get_incident(incident_id)
        return to_jsonable({"id": inc.id, "status": inc.status, "service": inc.service, "severity": inc.severity,
                            "diagnosis": inc.diagnosis, "decision": inc.decision, "verification": inc.verification,
                            "suspicious": inc.suspicious, "notes": inc.notes, "masked": inc.masked})

    @app.post("/api/incidents/{incident_id}/solutions", dependencies=[Depends(auth)])
    def compare_solutions(incident_id: str):
        inc = get_incident(incident_id)
        if not inc.solutions:
            raise HTTPException(409, "analyse non terminée")
        return to_jsonable({"solutions": inc.solutions, "plans": inc.plans})

    @app.post("/api/solutions/{solution_id}/test", status_code=202, dependencies=[Depends(auth)])
    def test_solution(solution_id: str, tasks: BackgroundTasks):
        inc, sol = find_solution(solution_id)
        sol.status = "test_en_cours"
        tasks.add_task(_run, inc.id, lambda i: pipeline_for(i).test_solution(i, solution_id))
        return {"id": solution_id, "status": sol.status}

    @app.post("/api/solutions/{solution_id}/approve", dependencies=[Depends(auth)])
    def approve(solution_id: str, body: ApproveIn):
        inc, _ = find_solution(solution_id)
        with locks[inc.id]:
            try:
                pipeline_for(inc).approve(inc, solution_id, body.approver)
            except PermissionError as exc:
                raise HTTPException(403, str(exc))
            except PipelineError as exc:
                raise HTTPException(409, str(exc))
        return to_jsonable({"id": inc.id, "status": inc.status, "decision": inc.decision, "verification": inc.verification})

    @app.post("/api/incidents/{incident_id}/auto", status_code=202, dependencies=[Depends(auth)])
    def run_auto(incident_id: str, tasks: BackgroundTasks):
        inc = get_incident(incident_id)
        if projects[inc.project_id]["mode"] != Mode.AUTO.value:
            raise HTTPException(403, "le mode automatique n'est pas activé pour ce projet")
        tasks.add_task(_run, incident_id, pipeline_for(inc).run_auto)
        return {"id": incident_id, "status": "automatique_en_cours"}

    @app.post("/api/incidents/{incident_id}/rollback", dependencies=[Depends(auth)])
    def rollback(incident_id: str):
        inc = get_incident(incident_id)
        with locks[incident_id]:
            try:
                pipeline_for(inc).rollback(inc)
            except PipelineError as exc:
                raise HTTPException(409, str(exc))
        return {"id": incident_id, "status": inc.status}

    @app.get("/api/incidents/{incident_id}/report", dependencies=[Depends(auth)])
    def report(incident_id: str):
        inc = get_incident(incident_id)
        return {"id": incident_id, "markdown": render_report(inc, audit.verify())}

    @app.get("/api/audit/verify", dependencies=[Depends(auth)])
    def audit_verify():
        return {"valid": audit.verify(), "entries": len(audit.entries)}

    return app
