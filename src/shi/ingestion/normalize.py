"""Normalisation des journaux en événements cohérents.

Format reconnu (souple) :
    2026-10-17 10:02:11 ERROR panier/prod: message
    2026-10-17T10:02:11Z [WARN] message
Le service et l'environnement s'écrivent « service/env: ». Les lignes sans
niveau (tracebacks) sont rattachées à l'événement précédent.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

SEVERITY_RANK = {"DEBUG": 0, "INFO": 1, "UNKNOWN": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
_ALIASES = {"WARN": "WARNING", "ERR": "ERROR", "FATAL": "CRITICAL", "TRACE": "DEBUG"}

_LINE = re.compile(
    r"^\s*(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?)?\s*"
    r"[\[(]?(?P<lvl>CRITICAL|FATAL|ERROR|ERR|WARNING|WARN|INFO|DEBUG|TRACE)\b[\])]?:?\s*"
    r"(?:(?P<svc>[\w.-]+)/(?P<env>[\w-]+)\s*[:|-]\s+)?(?P<msg>.*)$",
    re.I,
)


@dataclass
class Event:
    timestamp: Optional[str]
    service: str
    environment: str
    severity: str
    message: str
    source: str
    incident_id: Optional[str]
    line_no: int

    @property
    def rank(self) -> int:
        return SEVERITY_RANK.get(self.severity, 1)


def parse_logs(text: str, service: str = "inconnu", environment: str = "inconnu",
               source: str = "logs", incident_id: Optional[str] = None) -> List[Event]:
    events: List[Event] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        m = _LINE.match(line)
        if m and m.group("lvl"):
            lvl = m.group("lvl").upper()
            lvl = _ALIASES.get(lvl, lvl)
            events.append(
                Event(
                    timestamp=m.group("ts"),
                    service=m.group("svc") or service,
                    environment=m.group("env") or environment,
                    severity=lvl,
                    message=m.group("msg").strip(),
                    source=source,
                    incident_id=incident_id,
                    line_no=line_no,
                )
            )
        elif events:  # ligne sans niveau (traceback…) : rattachée à l'événement précédent
            events[-1].message += "\n" + line.rstrip()
        else:
            events.append(Event(None, service, environment, "UNKNOWN", line.strip(), source, incident_id, line_no))
    return events


def max_severity(events: List[Event]) -> str:
    if not events:
        return "UNKNOWN"
    return max(events, key=lambda e: e.rank).severity


def main_service(events: List[Event]) -> str:
    c = Counter(e.service for e in events if e.service != "inconnu")
    return c.most_common(1)[0][0] if c else "inconnu"
