import sys, json, random
sys.path.insert(0, '.')
import numpy as np
from shi.models.classifier import SHIClassifier
from shi.models.classifier.dataset import load_jsonl
from shi.models.classifier.corpus_libre import CORPUS, N_TRAIN, split_libre

D = 'src/shi/datasets/'
tt, tl = load_jsonl(D + 'train.jsonl'); vt, vl = load_jsonl(D + 'val.jsonl')
xt, yt, xs, ys = split_libre()
# validation interne : 4 dernières phrases d'entraînement de chaque catégorie
tr_x, tr_y, va_x, va_y = [], [], [], []
for lab, rows in CORPUS.items():
    for i, t in enumerate(rows[:N_TRAIN]):
        (va_x if i >= N_TRAIN - 4 else tr_x).append(t); (va_y if i >= N_TRAIN - 4 else tr_y).append(lab)

def acc(clf, X, Y, only=None):
    P = clf.predict_labels(X)
    idx = [i for i, y in enumerate(Y) if only is None or y in only]
    return sum(P[i] == Y[i] for i in idx) / len(idx)

def run(name, X, Y, VX, VY, **kw):
    clf = SHIClassifier(**kw); clf.fit(X, Y, VX, VY)
    print('%-34s libre-test: %.3f   (10 catégories communes: %.3f)' % (name, acc(clf, xs, ys), acc(clf, xs, ys, only=set(tl))))
    return clf

print('phrases libres de test :', len(xs))
run('A  gabarits seuls (référence)', tt, tl, vt, vl)
run('B  libre seul', tr_x, tr_y, va_x, va_y)
for rep in (1, 3):
    run('C  gabarits + libre x%d' % rep, tt + tr_x * rep, tl + tr_y * rep, vt + va_x, vl + va_y)
