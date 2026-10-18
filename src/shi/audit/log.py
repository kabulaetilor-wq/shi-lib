"""Journal d'audit à chaînage de hachages (détection de falsification)."""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

GENESIS = "0" * 64


def _digest(entry: Dict[str, Any]) -> str:
    payload = {k: v for k, v in entry.items() if k != "hash"}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else None
        self.entries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.entries.append(json.loads(line))

    def record(self, actor: str, action: str, details: Optional[dict] = None,
               incident_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            prev = self.entries[-1]["hash"] if self.entries else GENESIS
            entry = {
                "seq": len(self.entries) + 1,
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "actor": actor,
                "action": action,
                "incident_id": incident_id,
                "details": details or {},
                "prev": prev,
            }
            entry["hash"] = _digest(entry)
            self.entries.append(entry)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            return entry

    def verify(self) -> bool:
        prev = GENESIS
        for i, e in enumerate(self.entries, start=1):
            if e.get("seq") != i or e.get("prev") != prev or e.get("hash") != _digest(e):
                return False
            prev = e["hash"]
        return True

    def for_incident(self, incident_id: str) -> List[Dict[str, Any]]:
        return [e for e in self.entries if e.get("incident_id") == incident_id]
