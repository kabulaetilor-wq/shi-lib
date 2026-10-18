"""Exécuteur isolé : copie du dépôt, limites CPU/mémoire/durée/fichiers, réseau coupé si possible.

Niveau d'isolation HONNÊTE : ce n'est pas un conteneur. Il combine un répertoire
temporaire, un environnement épuré, des limites `resource`, un groupe de
processus tué au délai, et un espace de noms réseau (`unshare -rn`) lorsque le
système l'autorise. Pour du code réellement hostile, utiliser un conteneur ou
une VM dédiée (voir README).
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence

from shi.config import IGNORE_DIRS, Limits
from shi.security.guard import validate_test_command

try:  # absent hors POSIX
    import resource
except ImportError:  # pragma: no cover
    resource = None


@dataclass
class RunResult:
    ok: bool
    returncode: Optional[int]
    passed: int
    failed: int
    errors: int
    duration: float
    timed_out: bool
    stdout_tail: str
    stderr_tail: str
    isolation: Dict[str, str] = field(default_factory=dict)


class SandboxError(Exception):
    pass


def _copy_repo(src: Path, dst: Path, limits: Limits) -> None:
    total, count = 0, 0
    for root, dirs, files in os.walk(src, followlinks=False):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not os.path.islink(os.path.join(root, d))]
        target = dst / os.path.relpath(root, src)
        target.mkdir(parents=True, exist_ok=True)
        for f in files:
            p = os.path.join(root, f)
            if os.path.islink(p):
                continue
            count += 1
            total += os.path.getsize(p)
            if count > limits.max_files or total > limits.max_repo_mb * 1024 * 1024:
                raise SandboxError("dépôt trop volumineux pour l'environnement isolé")
            shutil.copy2(p, target / f)


class SandboxRunner:
    def __init__(self, limits: Limits = Limits(), framework: Optional[str] = None):
        self.limits = limits
        self._netns: Optional[bool] = None
        # pytest si disponible, sinon unittest (bibliothèque standard, utile sur Pydroid 3).
        self.framework = framework or ("pytest" if importlib.util.find_spec("pytest") else "unittest")

    def _network_isolated(self) -> bool:
        if self._netns is None:
            ok = False
            if shutil.which("unshare"):
                try:
                    ok = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=5).returncode == 0
                except (OSError, subprocess.TimeoutExpired):
                    ok = False
            self._netns = ok
        return self._netns

    @contextmanager
    def workspace(self, repo_path) -> Iterator[Path]:
        repo = Path(repo_path).resolve()
        if not repo.is_dir():
            raise SandboxError("dépôt introuvable : %s" % repo)
        tmp = Path(tempfile.mkdtemp(prefix="shi-sandbox-"))
        try:
            _copy_repo(repo, tmp, self.limits)
            yield tmp
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @staticmethod
    def write_files(ws: Path, files: Dict[str, str]) -> None:
        base = ws.resolve()
        for rel, content in files.items():
            target = (base / rel).resolve()
            if base != target and base not in target.parents:
                raise SandboxError("chemin hors de l'espace isolé : %s" % rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    def _preexec(self):
        lim = self.limits

        def apply():
            if resource is None:
                return
            resource.setrlimit(resource.RLIMIT_CPU, (lim.cpu_seconds, lim.cpu_seconds))
            resource.setrlimit(resource.RLIMIT_AS, (lim.memory_mb * 1024 * 1024,) * 2)
            resource.setrlimit(resource.RLIMIT_FSIZE, (lim.max_file_mb * 1024 * 1024,) * 2)
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        return apply

    def _parse_counts(self, text: str):
        def count(pattern):
            m = re.search(pattern, text)
            return int(m.group(1)) if m else 0

        if self.framework == "pytest":
            return count(r"(\d+) passed"), count(r"(\d+) failed"), count(r"(\d+) error")
        ran = count(r"Ran (\d+) test")
        failed, errors = count(r"failures=(\d+)"), count(r"errors=(\d+)")
        return max(ran - failed - errors, 0), failed, errors

    def run_tests(self, ws: Path, extra_args: Sequence[str] = ()) -> RunResult:
        if self.framework == "pytest":
            inner = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *extra_args]
        else:
            inner = [sys.executable, "-m", "unittest", "discover", "-s", "tests", *extra_args]
        validate_test_command(inner)  # composant déterministe : seule commande permise
        netns = self._network_isolated()
        argv = (["unshare", "-rn"] + inner) if netns else inner
        env = {
            "PATH": os.pathsep.join([os.path.dirname(sys.executable), "/usr/bin", "/bin"]),
            "HOME": str(ws), "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ws),
        }
        iso = {
            "repertoire": "copie temporaire", "environnement": "épuré",
            "limites": "cpu=%ss mémoire=%sMo durée=%ss" % (self.limits.cpu_seconds, self.limits.memory_mb, self.limits.wall_seconds),
            "reseau": "coupé (espace de noms)" if netns else "NON isolé (meilleur effort)",
        }
        start = time.time()
        timed_out = False
        proc = subprocess.Popen(argv, cwd=str(ws), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True, preexec_fn=self._preexec() if resource else None)
        try:
            out, err = proc.communicate(timeout=self.limits.wall_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            out, err = proc.communicate()
        text = (out or "") + ("\n" + err if self.framework == "unittest" and err else "")
        passed, failed, errors = self._parse_counts(text)

        return RunResult(
            ok=(proc.returncode == 0 and not timed_out), returncode=proc.returncode,
            passed=passed, failed=failed, errors=errors,
            duration=round(time.time() - start, 2), timed_out=timed_out,
            stdout_tail=text[-1500:], stderr_tail=(err or "")[-800:], isolation=iso,
        )
