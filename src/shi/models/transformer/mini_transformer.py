"""SHI-0.3 : mini-Transformer expérimental au niveau caractère (NumPy pur).

Un seul bloc : embeddings + positions -> attention causale à une tête ->
réseau feed-forward (ReLU) -> projection sur le vocabulaire. La rétropropagation
est écrite à la main et vérifiée par un test de gradient numérique.

C'est un objectif d'apprentissage : un modèle minuscule, pas un réparateur
autonome. Il ne doit jamais lancer de commande.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class CharTokenizer:
    def __init__(self, chars: str):
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls("".join(sorted(set(text))))

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.stoi[c] for c in text if c in self.stoi], dtype=np.int64)

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)

    @property
    def size(self) -> int:
        return len(self.chars)


class MiniTransformer:
    NAMES = ["E", "P", "Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "b2", "Wout", "bout"]

    def __init__(self, vocab_size: int, context: int = 48, dim: int = 32, seed: int = 0):
        self.V, self.T, self.D, self.H = vocab_size, context, dim, 4 * dim
        rng = np.random.default_rng(seed)
        D, H, V, T = self.D, self.H, self.V, self.T

        def w(r, c, scale=1.0):
            return rng.normal(0, scale / np.sqrt(r), (r, c))

        self.p: Dict[str, np.ndarray] = {
            "E": rng.normal(0, 0.1, (V, D)), "P": rng.normal(0, 0.1, (T, D)),
            "Wq": w(D, D), "Wk": w(D, D), "Wv": w(D, D), "Wo": w(D, D, 0.5),
            "W1": w(D, H), "b1": np.zeros(H), "W2": w(H, D, 0.5), "b2": np.zeros(D),
            "Wout": w(D, V), "bout": np.zeros(V),
        }
        self._m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._t = 0

    # --- propagation avant ---------------------------------------------
    def forward(self, X: np.ndarray):
        p, D = self.p, self.D
        B, T = X.shape
        h0 = p["E"][X] + p["P"][:T][None]
        q, k, v = h0 @ p["Wq"], h0 @ p["Wk"], h0 @ p["Wv"]
        s = (q @ k.transpose(0, 2, 1)) / np.sqrt(D)
        s = np.where(np.tril(np.ones((T, T), dtype=bool)), s, -1e30)
        s = s - s.max(-1, keepdims=True)
        a = np.exp(s)
        a /= a.sum(-1, keepdims=True)
        ctx = a @ v
        h1 = h0 + ctx @ p["Wo"]
        f1 = h1 @ p["W1"] + p["b1"]
        r = np.maximum(f1, 0)
        h2 = h1 + r @ p["W2"] + p["b2"]
        logits = h2 @ p["Wout"] + p["bout"]
        return logits, (X, h0, q, k, v, a, ctx, h1, f1, r, h2)

    @staticmethod
    def _softmax(z):
        z = z - z.max(-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(-1, keepdims=True)

    # --- perte et gradients --------------------------------------------
    def loss_and_grads(self, X: np.ndarray, Y: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        p, D = self.p, self.D
        logits, (X, h0, q, k, v, a, ctx, h1, f1, r, h2) = self.forward(X)
        B, T = X.shape
        probs = self._softmax(logits)
        bi, ti = np.meshgrid(np.arange(B), np.arange(T), indexing="ij")
        loss = float(-np.log(np.clip(probs[bi, ti, Y], 1e-12, None)).mean())

        g: Dict[str, np.ndarray] = {}
        dl = probs.copy()
        dl[bi, ti, Y] -= 1.0
        dl /= B * T
        g["Wout"] = np.einsum("btd,btv->dv", h2, dl)
        g["bout"] = dl.sum((0, 1))
        dh2 = dl @ p["Wout"].T
        g["W2"] = np.einsum("bth,btd->hd", r, dh2)
        g["b2"] = dh2.sum((0, 1))
        df1 = (dh2 @ p["W2"].T) * (f1 > 0)
        g["W1"] = np.einsum("btd,bth->dh", h1, df1)
        g["b1"] = df1.sum((0, 1))
        dh1 = dh2 + df1 @ p["W1"].T
        g["Wo"] = np.einsum("btd,bte->de", ctx, dh1)
        dctx = dh1 @ p["Wo"].T
        dA = dctx @ v.transpose(0, 2, 1)
        dv = a.transpose(0, 2, 1) @ dctx
        dS = a * (dA - (dA * a).sum(-1, keepdims=True)) / np.sqrt(D)
        dq = dS @ k
        dk = dS.transpose(0, 2, 1) @ q
        g["Wq"] = np.einsum("btd,bte->de", h0, dq)
        g["Wk"] = np.einsum("btd,bte->de", h0, dk)
        g["Wv"] = np.einsum("btd,bte->de", h0, dv)
        dh0 = dh1 + dq @ p["Wq"].T + dk @ p["Wk"].T + dv @ p["Wv"].T
        g["P"] = np.zeros_like(p["P"])
        g["P"][:T] = dh0.sum(0)
        g["E"] = np.zeros_like(p["E"])
        np.add.at(g["E"], X, dh0)
        return loss, g

    def loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        return self.loss_and_grads(X, Y)[0]

    # --- entraînement ---------------------------------------------------
    def _adam(self, grads: Dict[str, np.ndarray], lr: float, clip: float = 1.0):
        total = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
        scale = min(1.0, clip / (total + 1e-12))
        self._t += 1
        b1, b2 = 0.9, 0.999
        for k in self.NAMES:
            gk = grads[k] * scale
            self._m[k] = b1 * self._m[k] + (1 - b1) * gk
            self._v[k] = b2 * self._v[k] + (1 - b2) * gk ** 2
            mh = self._m[k] / (1 - b1 ** self._t)
            vh = self._v[k] / (1 - b2 ** self._t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)

    def train(self, data: np.ndarray, steps: int = 500, batch: int = 16, lr: float = 3e-3,
              seed: int = 0, log_every: int = 0) -> List[float]:
        rng = np.random.default_rng(seed)
        T = self.T
        if len(data) <= T + 1:
            raise ValueError("corpus trop court pour le contexte choisi")
        losses = []
        for step in range(1, steps + 1):
            starts = rng.integers(0, len(data) - T - 1, size=batch)
            X = np.stack([data[s : s + T] for s in starts])
            Y = np.stack([data[s + 1 : s + T + 1] for s in starts])
            loss, grads = self.loss_and_grads(X, Y)
            self._adam(grads, lr)
            losses.append(loss)
            if log_every and step % log_every == 0:
                print("étape %d  perte %.3f" % (step, np.mean(losses[-log_every:])))
        return losses

    def evaluate(self, data: np.ndarray, n_batches: int = 20, batch: int = 16, seed: int = 123) -> float:
        rng = np.random.default_rng(seed)
        T, out = self.T, []
        for _ in range(n_batches):
            starts = rng.integers(0, len(data) - T - 1, size=batch)
            X = np.stack([data[s : s + T] for s in starts])
            Y = np.stack([data[s + 1 : s + T + 1] for s in starts])
            out.append(self.loss(X, Y))
        return float(np.mean(out))

    # --- génération -----------------------------------------------------
    def generate(self, prompt_ids: np.ndarray, n: int, temperature: float = 0.8, seed: int = 0) -> List[int]:
        rng = np.random.default_rng(seed)
        ids = list(prompt_ids) or [0]
        for _ in range(n):
            ctx = np.array(ids[-self.T :])[None]
            logits, _ = self.forward(ctx)
            z = logits[0, -1] / max(temperature, 1e-3)
            pr = self._softmax(z)
            ids.append(int(rng.choice(len(pr), p=pr)))
        return ids

    # --- persistance ----------------------------------------------------
    def save(self, directory, tokenizer: Optional[CharTokenizer] = None) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "params.npz", **self.p)
        meta = {"V": self.V, "T": self.T, "D": self.D, "chars": tokenizer.chars if tokenizer else None}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory) -> Tuple["MiniTransformer", Optional[CharTokenizer]]:
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        m = cls(meta["V"], meta["T"], meta["D"])
        data = np.load(d / "params.npz")
        m.p = {k: data[k] for k in cls.NAMES}
        tok = CharTokenizer(meta["chars"]) if meta.get("chars") else None
        return m, tok
