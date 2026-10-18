"""Configuration centrale de SHI."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import os

ROOT = Path(__file__).resolve().parent            # dossier du paquet installé
DATASETS_DIR = ROOT / "datasets"
BUNDLED_ARTIFACTS_DIR = ROOT / "artifacts"        # modèle livré avec le paquet (lecture seule)
# SHI_HOME : dossier modifiable de l'utilisateur pour ses propres modèles entraînés
ARTIFACTS_DIR = Path(os.environ["SHI_HOME"]) / "artifacts" if os.environ.get("SHI_HOME") else BUNDLED_ARTIFACTS_DIR
DEMO_REPO = ROOT / "demo_repo"

# Dossiers jamais copiés ni analysés.
IGNORE_DIRS = frozenset(
    {".git", "__pycache__", ".venv", "venv", "node_modules", ".shi_backups", ".pytest_cache"}
)


@dataclass(frozen=True)
class Limits:
    """Limites appliquées à l'exécuteur isolé."""

    cpu_seconds: int = 30
    wall_seconds: int = 90
    memory_mb: int = 1024
    max_file_mb: int = 10
    max_repo_mb: int = 50
    max_files: int = 2000
