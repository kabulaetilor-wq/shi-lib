"""Tokenizer et vectorisation TF-IDF pour le français (NumPy uniquement)."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List

import numpy as np

STOPWORDS = frozenset(
    "le la les un une des de du d l et ou en au aux a à est sont sur pour par que qui dans ce cet "
    "cette ces se s y il elle on son sa ses leur leurs avec sans vers depuis apres avant".split()
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.replace("'", " ").replace("’", " ")


def words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", normalize(text)) if w not in STOPWORDS]


LEXICON_WEIGHT = 0  # mesuré : aucun gain sur le test final ; le lexique reste disponible (mettre 2 pour l'activer)


def features(text: str) -> List[str]:
    """Mots, bigrammes, 4-grammes de caractères et indices du lexique de domaine."""
    from .lexicon import lexicon_hits

    ws = words(text)
    feats = ["w:" + w for w in ws]
    for cat in lexicon_hits(normalize(text)):
        feats += ["k:" + cat] * LEXICON_WEIGHT
    feats += ["b:%s_%s" % (a, b) for a, b in zip(ws, ws[1:])]
    for w in ws:
        if len(w) >= 5:
            feats += ["c:" + w[i : i + 4] for i in range(len(w) - 3)]
    return feats


class Vectorizer:
    def __init__(self, min_df: int = 1):
        self.min_df = min_df
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0)

    def fit(self, texts: Iterable[str]) -> "Vectorizer":
        texts = list(texts)
        df: Counter = Counter()
        for t in texts:
            df.update(set(features(t)))
        kept = sorted(f for f, c in df.items() if c >= self.min_df)
        self.vocab = {f: i for i, f in enumerate(kept)}
        counts = np.array([df[f] for f in kept], dtype=np.float64)
        n = len(texts)
        self.idf = np.log((1 + n) / (1 + counts)) + 1.0
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        X = np.zeros((len(texts), len(self.vocab)), dtype=np.float64)
        for i, t in enumerate(texts):
            for f in features(t):
                j = self.vocab.get(f)
                if j is not None:
                    X[i, j] += 1.0
        X *= self.idf
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def to_dict(self) -> dict:
        return {"min_df": self.min_df, "vocab": self.vocab, "idf": self.idf.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Vectorizer":
        v = cls(d["min_df"])
        v.vocab = {k: int(i) for k, i in d["vocab"].items()}
        v.idf = np.array(d["idf"], dtype=np.float64)
        return v
