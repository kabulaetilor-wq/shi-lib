"""SHI-0.2 : classificateur de descriptions d'incidents (régression softmax, NumPy)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from shi.config import ARTIFACTS_DIR, BUNDLED_ARTIFACTS_DIR
from shi.models.classifier.tokenizer import Vectorizer


@dataclass
class Prediction:
    label: str
    confidence: float
    probs: Dict[str, float]
    uncertain: bool


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SHIClassifier:
    def __init__(self, l2: float = 1e-4, lr: float = 0.05, epochs: int = 400,
                 seed: int = 0, min_confidence: float = 0.35):
        self.l2, self.lr, self.epochs, self.seed = l2, lr, epochs, seed
        self.min_confidence = min_confidence
        self.vectorizer = Vectorizer()
        self.labels: List[str] = []
        self.W = np.zeros((0, 0))
        self.b = np.zeros(0)

    # --- entraînement ---------------------------------------------------
    def _loss(self, X, Y):
        P = _softmax(X @ self.W + self.b)
        ce = -np.log(np.clip((P * Y).sum(axis=1), 1e-12, None)).mean()
        return ce + 0.5 * self.l2 * float((self.W ** 2).sum())

    def fit(self, texts: Sequence[str], labels: Sequence[str],
            val_texts: Optional[Sequence[str]] = None, val_labels: Optional[Sequence[str]] = None,
            patience: int = 40) -> Dict[str, list]:
        self.labels = sorted(set(labels))
        idx = {l: i for i, l in enumerate(self.labels)}
        self.vectorizer = Vectorizer().fit(texts)
        X = self.vectorizer.transform(texts)
        Y = np.eye(len(self.labels))[[idx[l] for l in labels]]
        Xv = Yv = None
        if val_texts is not None:
            Xv = self.vectorizer.transform(val_texts)
            Yv = np.eye(len(self.labels))[[idx[l] for l in val_labels]]

        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0, 0.01, (X.shape[1], len(self.labels)))
        self.b = np.zeros(len(self.labels))
        mW = np.zeros_like(self.W); vW = np.zeros_like(self.W)
        mb = np.zeros_like(self.b); vb = np.zeros_like(self.b)
        b1, b2, eps = 0.9, 0.999, 1e-8
        hist = {"train_loss": [], "val_loss": []}
        best, best_state, wait = np.inf, None, 0

        for t in range(1, self.epochs + 1):
            P = _softmax(X @ self.W + self.b)
            G = (P - Y) / len(X)
            gW = X.T @ G + self.l2 * self.W
            gb = G.sum(axis=0)
            mW = b1 * mW + (1 - b1) * gW; vW = b2 * vW + (1 - b2) * gW ** 2
            mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb ** 2
            self.W -= self.lr * (mW / (1 - b1 ** t)) / (np.sqrt(vW / (1 - b2 ** t)) + eps)
            self.b -= self.lr * (mb / (1 - b1 ** t)) / (np.sqrt(vb / (1 - b2 ** t)) + eps)
            hist["train_loss"].append(float(self._loss(X, Y)))
            if Xv is not None:
                vl = float(self._loss(Xv, Yv))
                hist["val_loss"].append(vl)
                if vl < best - 1e-6:
                    best, best_state, wait = vl, (self.W.copy(), self.b.copy()), 0
                else:
                    wait += 1
                    if wait >= patience:
                        break
        if best_state is not None:
            self.W, self.b = best_state
        return hist

    # --- prédiction -----------------------------------------------------
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return _softmax(self.vectorizer.transform(texts) @ self.W + self.b)

    def predict_labels(self, texts: Sequence[str]) -> List[str]:
        return [self.labels[i] for i in self.predict_proba(texts).argmax(axis=1)]

    def predict(self, text: str) -> Prediction:
        p = self.predict_proba([text])[0]
        i = int(p.argmax())
        return Prediction(
            label=self.labels[i],
            confidence=float(p[i]),
            probs={l: float(v) for l, v in zip(self.labels, p)},
            uncertain=bool(p[i] < self.min_confidence or self.vectorizer.transform([text]).sum() == 0),
        )

    # --- persistance ----------------------------------------------------
    def save(self, directory) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "weights.npz", W=self.W, b=self.b)
        meta = {"labels": self.labels, "vectorizer": self.vectorizer.to_dict(),
                "min_confidence": self.min_confidence}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory) -> "SHIClassifier":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        w = np.load(d / "weights.npz")
        clf = cls(min_confidence=meta["min_confidence"])
        clf.labels = meta["labels"]
        clf.vectorizer = Vectorizer.from_dict(meta["vectorizer"])
        clf.W, clf.b = w["W"], w["b"]
        return clf


def load_default_classifier() -> Optional[SHIClassifier]:
    """Modèle de l'utilisateur (SHI_HOME) s'il existe, sinon le modèle livré avec le paquet."""
    for base in (ARTIFACTS_DIR, BUNDLED_ARTIFACTS_DIR):
        d = base / "classifier"
        if (d / "meta.json").exists():
            return SHIClassifier.load(d)
    return None
