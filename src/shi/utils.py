"""Utilitaires : conversion JSON des objets SHI (les champs « hidden » sont omis)."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(o: Any) -> Any:
    if is_dataclass(o) and not isinstance(o, type):
        return {f.name: to_jsonable(getattr(o, f.name)) for f in fields(o) if not f.metadata.get("hidden")}
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o
