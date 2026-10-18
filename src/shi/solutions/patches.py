"""Correctifs candidats déterministes (AST) pour des défauts Python simples.

Chaque correctif est une PROPOSITION : il est testé en environnement isolé et
n'est appliqué qu'après les contrôles de la garde.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

_REQUESTS = {"get", "post", "put", "delete", "patch", "head", "request"}


@dataclass
class Patch:
    path: str
    rule: str
    description: str
    risk: str
    diff: str
    base_sha256: str
    new_source: str = field(default="", repr=False, metadata={"hidden": True})


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_mutable(d) -> bool:
    if isinstance(d, (ast.List, ast.Dict, ast.Set)):
        return True
    return (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id in {"list", "dict", "set"}
            and not d.args and not d.keywords)


def fix_mutable_defaults(source: str) -> Optional[str]:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    funcs.sort(key=lambda n: n.lineno, reverse=True)
    changed = False
    for fn in funcs:
        a = fn.args
        pos = a.posonlyargs + a.args
        pairs = list(zip(pos[len(pos) - len(a.defaults):], a.defaults))
        pairs += [(k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
        bad = [(arg.arg, d) for arg, d in pairs if _is_mutable(d)]
        if not bad or any(d.lineno != d.end_lineno for _, d in bad):
            continue
        first = fn.body[0]
        if first.lineno == fn.lineno:  # fonction sur une seule ligne
            continue
        guards = [(name, ast.get_source_segment(source, d)) for name, d in bad]  # ordre d'origine
        for name, d in sorted(bad, key=lambda t: (t[1].lineno, -t[1].col_offset)):  # de droite à gauche
            raw = lines[d.lineno - 1].encode("utf-8")
            raw = raw[: d.col_offset] + b"None" + raw[d.end_col_offset :]
            lines[d.lineno - 1] = raw.decode("utf-8")
        is_doc = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
        insert_at = first.end_lineno if is_doc else first.lineno - 1
        indent = re.match(r"[ \t]*", lines[first.lineno - 1]).group(0)
        block = "".join("%sif %s is None:\n%s    %s = %s\n" % (indent, n, indent, n, o) for n, o in guards)
        lines[insert_at:insert_at] = [block]
        changed = True
    return "".join(lines) if changed else None


def fix_bare_except(source: str) -> Optional[str]:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    changed = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            new = re.sub(r"except\s*:", "except Exception:", lines[node.lineno - 1], count=1)
            changed |= new != lines[node.lineno - 1]
            lines[node.lineno - 1] = new
    return "".join(lines) if changed else None


def fix_eq_none(source: str) -> Optional[str]:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    changed = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or node.lineno != node.end_lineno:
            continue
        if not any(isinstance(c, ast.Constant) and c.value is None for c in node.comparators):
            continue
        raw = lines[node.lineno - 1].encode("utf-8")
        seg = raw[node.col_offset : node.end_col_offset].decode("utf-8")
        new = re.sub(r"!=(\s*None\b)", r"is not\1", re.sub(r"==(\s*None\b)", r"is\1", seg))
        if new != seg:
            raw = raw[: node.col_offset] + new.encode("utf-8") + raw[node.end_col_offset :]
            lines[node.lineno - 1] = raw.decode("utf-8")
            changed = True
    return "".join(lines) if changed else None


def fix_requests_timeout(source: str, timeout: int = 10) -> Optional[str]:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    calls = []
    for node in ast.walk(tree):
        f = getattr(node, "func", None)
        if (isinstance(node, ast.Call) and isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and f.value.id == "requests" and f.attr in _REQUESTS and node.lineno == node.end_lineno
                and not any(k.arg in ("timeout", None) for k in node.keywords)):
            calls.append(node)
    calls.sort(key=lambda n: (n.lineno, n.end_col_offset), reverse=True)
    for node in calls:
        raw = lines[node.lineno - 1].encode("utf-8")
        end = node.end_col_offset - 1  # position de la parenthèse fermante
        extra = (", timeout=%d" if (node.args or node.keywords) else "timeout=%d") % timeout
        lines[node.lineno - 1] = (raw[:end] + extra.encode() + raw[end:]).decode("utf-8")
    return "".join(lines) if calls else None


FIXERS: List[tuple] = [
    ("mutable_default", "Remplacer l'argument par défaut mutable par None (objet créé à chaque appel)", "low", fix_mutable_defaults),
    ("bare_except", "Remplacer « except: » par « except Exception: »", "low", fix_bare_except),
    ("eq_none", "Remplacer « == None » par « is None »", "low", fix_eq_none),
    ("requests_no_timeout", "Ajouter un délai d'attente (timeout=10) aux appels requests", "medium", fix_requests_timeout),
]


def propose_patches(path: str, source: str) -> List[Patch]:
    """Chaque correctif est calculé indépendamment sur la source d'origine."""
    patches = []
    for rule, desc, risk, fixer in FIXERS:
        try:
            new = fixer(source)
            if not new or new == source:
                continue
            ast.parse(new)  # le résultat doit rester du Python valide
        except SyntaxError:
            continue
        diff = "".join(difflib.unified_diff(source.splitlines(keepends=True), new.splitlines(keepends=True),
                                            fromfile="a/" + path, tofile="b/" + path))
        patches.append(Patch(path, rule, desc, risk, diff, sha256(source), new))
    return patches
