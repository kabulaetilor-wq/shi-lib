import tempfile
import unittest

import numpy as np

from shi.models.classifier import SHIClassifier, classification_report
from shi.models.classifier import dataset
from shi.models.classifier.metrics import confusion_matrix
from shi.models.transformer import CharTokenizer, MiniTransformer


class TestDataset(unittest.TestCase):
    def test_splits_do_not_overlap(self):
        data = dataset.build(0)
        self.assertTrue(dataset.check_no_overlap(*[[r["text"] for r in rows] for rows in data.values()]))
        self.assertEqual({r["label"] for r in data["test"]}, set(dataset.TEMPLATES))


class TestClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = dataset.build(0)
        cls.data = data
        cls.clf = SHIClassifier(epochs=150)
        cls.clf.fit([r["text"] for r in data["train"]], [r["label"] for r in data["train"]],
                    [r["text"] for r in data["val"]], [r["label"] for r in data["val"]])

    def test_learns_training_distribution(self):
        val = self.data["val"]
        pred = self.clf.predict_labels([r["text"] for r in val])
        acc = np.mean([p == r["label"] for p, r in zip(pred, val)])
        self.assertGreater(acc, 0.5)  # bien au-dessus du hasard (0.1)

    def test_obvious_sentences(self):
        for text, label in [("Le disque est plein", "stockage"), ("Le déploiement a échoué", "deploiement"),
                            ("Certificat SSL expiré", "reseau")]:
            self.assertEqual(self.clf.predict(text).label, label, text)

    def test_unknown_text_is_uncertain(self):
        self.assertTrue(self.clf.predict("zzzz qqqq").uncertain)

    def test_save_load_roundtrip(self):
        d = tempfile.mkdtemp()
        self.clf.save(d)
        other = SHIClassifier.load(d)
        texts = ["Les pages chargent très lentement", "Erreur 401 pour tous les comptes"]
        np.testing.assert_allclose(self.clf.predict_proba(texts), other.predict_proba(texts))

    def test_metrics(self):
        labels = ["a", "b"]
        rep = classification_report(["a", "a", "b", "b"], ["a", "b", "b", "b"], labels)
        self.assertAlmostEqual(rep["accuracy"], 0.75)
        self.assertAlmostEqual(rep["per_class"]["b"]["recall"], 1.0)
        self.assertAlmostEqual(rep["per_class"]["b"]["precision"], 2 / 3)
        self.assertEqual(confusion_matrix(["a"], ["b"], labels).tolist(), [[0, 1], [0, 0]])


class TestTransformer(unittest.TestCase):
    def test_gradients_match_numerical(self):
        rng = np.random.default_rng(1)
        m = MiniTransformer(vocab_size=7, context=5, dim=6, seed=3)
        X = rng.integers(0, 7, (2, 5))
        Y = rng.integers(0, 7, (2, 5))
        _, grads = m.loss_and_grads(X, Y)
        eps = 1e-6
        for name in MiniTransformer.NAMES:
            flat = m.p[name].ravel()
            for idx in rng.choice(flat.size, size=min(4, flat.size), replace=False):
                old = flat[idx]
                flat[idx] = old + eps
                up = m.loss(X, Y)
                flat[idx] = old - eps
                down = m.loss(X, Y)
                flat[idx] = old
                num = (up - down) / (2 * eps)
                ana = grads[name].ravel()[idx]
                self.assertAlmostEqual(num, ana, delta=1e-5 + 1e-4 * abs(num), msg="%s[%d]" % (name, idx))

    def test_attention_is_causal(self):
        m = MiniTransformer(vocab_size=5, context=6, dim=8, seed=0)
        X = np.array([[1, 2, 3, 4, 0, 1]])
        X2 = X.copy()
        X2[0, -1] = 3  # change seulement le dernier jeton
        a, b = m.forward(X)[0], m.forward(X2)[0]
        np.testing.assert_allclose(a[0, :-1], b[0, :-1])

    def test_training_reduces_loss_and_generates(self):
        text = "le serveur ne répond plus\n" * 40
        tok = CharTokenizer.from_text(text)
        data = tok.encode(text)
        m = MiniTransformer(tok.size, context=16, dim=16, seed=0)
        before = m.evaluate(data, n_batches=5)
        m.train(data, steps=120, batch=8, lr=1e-2)
        after = m.evaluate(data, n_batches=5)
        self.assertLess(after, before * 0.6)
        out = tok.decode(m.generate(tok.encode("le ser"), 20, temperature=0.5, seed=1))
        self.assertEqual(len(out), 26)
        d = tempfile.mkdtemp()
        m.save(d, tok)
        m2, tok2 = MiniTransformer.load(d)
        np.testing.assert_allclose(m.forward(data[None, :16])[0], m2.forward(data[None, :16])[0])
        self.assertEqual(tok2.chars, tok.chars)


if __name__ == "__main__":
    unittest.main()
