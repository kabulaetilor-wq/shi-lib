"""Métriques : exactitude, précision, rappel, F1, matrice de confusion."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: List[str]) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t], idx[p]] += 1
    return m


def classification_report(y_true: Sequence[str], y_pred: Sequence[str], labels: List[str]) -> Dict:
    cm = confusion_matrix(y_true, y_pred, labels)
    per = {}
    for i, l in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[l] = {"precision": float(prec), "recall": float(rec), "f1": float(f1), "support": int(cm[i, :].sum())}
    total = len(y_true)
    acc = float(sum(1 for t, p in zip(y_true, y_pred) if t == p) / total) if total else 0.0
    return {"accuracy": acc, "per_class": per, "confusion": cm, "labels": labels, "n": total}


def format_report(rep: Dict) -> str:
    lines = ["exactitude : %.3f sur %d exemples" % (rep["accuracy"], rep["n"]), ""]
    lines.append("%-18s %9s %7s %6s %8s" % ("catégorie", "précision", "rappel", "F1", "support"))
    for l, s in rep["per_class"].items():
        lines.append("%-18s %9.2f %7.2f %6.2f %8d" % (l, s["precision"], s["recall"], s["f1"], s["support"]))
    lines.append("")
    lines.append("matrice de confusion (lignes = réel, colonnes = prédit)")
    labels = rep["labels"]
    lines.append(" " * 18 + " ".join("%4s" % l[:4] for l in labels))
    for i, l in enumerate(labels):
        lines.append("%-18s" % l + " ".join("%4d" % v for v in rep["confusion"][i]))
    return "\n".join(lines)
