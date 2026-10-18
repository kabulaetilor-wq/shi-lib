"""Analyse de code : symboles, zones citées par les tracebacks, signaux statiques.

La syntaxe ne prouve pas la cause d'un incident : ces résultats sont des indices.
"""
from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from shi.config import IGNORE_DIRS
from shi.security.guard import scan_untrusted


@dataclass
class Symbol:
    kind: str
    name: str
    lineno: int


@dataclass
class Finding:
    rule: str
    lineno: int
    message: str


@dataclass
class Frame:
    file: str
    line: int
    func: str


_FRAME = re.compile(r'File "(?P<f>[^"]+)", line (?P<l>\d+)(?:, in (?P<fn>[\w<>]+))?')
_REQUESTS = {"get", "post", "put", "delete", "patch", "head", "request"}


def parse_tracebacks(text: str) -> List[Frame]:
    return [Frame(m.group("f"), int(m.group("l")), m.group("fn") or "") for m in _FRAME.finditer(text)]


def _is_mutable_default(d) -> bool:
    if isinstance(d, (ast.List, ast.Dict, ast.Set)):
        return True
    return (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id in {"list", "dict", "set"}
            and not d.args and not d.keywords)


def analyze_source(source: str) -> Tuple[List[Symbol], List[Finding]]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [], [Finding("syntax_error", exc.lineno or 0, "erreur de syntaxe : %s" % exc.msg)]
    symbols: List[Symbol] = []
    findings: List[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(Symbol("fonction", node.name, node.lineno))
            a = node.args
            defaults = list(a.defaults) + [d for d in a.kw_defaults if d is not None]
            if any(_is_mutable_default(d) for d in defaults):
                findings.append(Finding("mutable_default", node.lineno,
                                        "argument par défaut mutable dans %s() : l'objet est partagé entre les appels" % node.name))
        elif isinstance(node, ast.ClassDef):
            symbols.append(Symbol("classe", node.name, node.lineno))
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            findings.append(Finding("bare_except", node.lineno, "« except: » nu : masque toutes les erreurs"))
        elif isinstance(node, ast.Compare):
            if any(isinstance(o, (ast.Eq, ast.NotEq)) for o in node.ops) and any(
                isinstance(c, ast.Constant) and c.value is None for c in node.comparators
            ):
                findings.append(Finding("eq_none", node.lineno, "comparaison « == None » (préférer « is None »)"))
        elif isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "requests"
                    and f.attr in _REQUESTS and not any(k.arg in ("timeout", None) for k in node.keywords)):
                findings.append(Finding("requests_no_timeout", node.lineno, "appel requests sans délai d'attente (timeout)"))
            elif isinstance(f, ast.Name) and f.id in ("eval", "exec"):
                findings.append(Finding("eval_exec", node.lineno, "usage de %s() : risque d'exécution de code arbitraire" % f.id))
    return symbols, findings


@dataclass
class RepoScan:
    files: Dict[str, str] = field(default_factory=dict)  # sources .py analysables (hors tests)
    findings: Dict[str, List[Finding]] = field(default_factory=dict)
    symbols: Dict[str, List[Symbol]] = field(default_factory=dict)
    suspicious: List[dict] = field(default_factory=list)  # instructions suspectes (informatif)


_TEXT_SUFFIXES = {".py", ".md", ".txt", ".log", ".cfg", ".ini", ".toml", ".yml", ".yaml", ".json"}


def _is_test(rel: str) -> bool:
    name = os.path.basename(rel)
    return rel.startswith("tests/") or "/tests/" in rel or name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


def scan_repo(root, max_files: int = 300, max_bytes: int = 200_000) -> RepoScan:
    """Lit le dépôt SANS suivre les liens symboliques ; le contenu est non fiable."""
    root = Path(root).resolve()
    scan = RepoScan()
    count = 0
    for dirpath, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in IGNORE_DIRS and not os.path.islink(os.path.join(dirpath, d)))
        for name in sorted(files):
            full = os.path.join(dirpath, name)
            if os.path.islink(full) or Path(name).suffix not in _TEXT_SUFFIXES:
                continue
            if os.path.getsize(full) > max_bytes:
                continue
            count += 1
            if count > max_files:
                return scan
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            text = Path(full).read_text(encoding="utf-8", errors="replace")
            scan.suspicious += scan_untrusted(text, rel)
            if name.endswith(".py") and not _is_test(rel):
                scan.files[rel] = text
                syms, finds = analyze_source(text)
                scan.symbols[rel] = syms
                if finds:
                    scan.findings[rel] = finds
    return scan


def match_frames(frames: List[Frame], known_files) -> List[Tuple[str, int, str]]:
    """Associe les fichiers cités dans un traceback aux fichiers du dépôt (par suffixe)."""
    out = []
    for fr in frames:
        path = fr.file.replace("\\", "/")
        for rel in known_files:
            if path == rel or path.endswith("/" + rel):
                out.append((rel, fr.line, fr.func))
                break
    return out
