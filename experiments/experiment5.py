import sys; sys.path.insert(0,'.')
exec(open('experiment.py').read().split("print('phrases libres")[0])
sys.path.insert(0,'src')
from shi.models.classifier.corpus_extra import EXTRA
from shi.models.classifier import tokenizer
dx, dy = load_jsonl('src/shi/datasets/test_aveugle.jsonl')       # maintenant utilisé comme données de développement
fx, fy = load_jsonl('src/shi/datasets/test_final.jsonl')          # jamais consulté avant ce test
ex, ey = [], []
for l, rows in EXTRA.items():
    ex += rows; ey += [l]*len(rows)
# apprentissage : phrases libres (train+val) + extra + ancien test aveugle ; validation = 1 phrase sur 6
allx = tr_x + va_x + ex + dx; ally = tr_y + va_y + ey + dy
vx = allx[::6]; vy = ally[::6]
px = [t for i, t in enumerate(allx) if i % 6]; py = [t for i, t in enumerate(ally) if i % 6]
print('apprentissage:', len(px), ' validation:', len(vx), ' test final:', len(fx))
res = {}
for w in (0, 2):
    tokenizer.LEXICON_WEIGHT = w
    for rep, use_tpl in ((1, False), (1, True)):
        X = px * rep + (tt if use_tpl else []); Y = py * rep + (tl if use_tpl else [])
        # les gabarits pèsent peu : on répète les phrases libres 3 fois quand on les ajoute
        if use_tpl: X = px * 3 + tt; Y = py * 3 + tl
        clf = SHIClassifier(); clf.fit(X, Y, vx + vt[:200], vy + vl[:200])
        a_lib = acc(clf, xs, ys)  # (ces phrases font partie de l'apprentissage ici : à ne pas lire comme un test)
        print('lexique=%d gabarits=%s  test final: %.3f' % (w, use_tpl, acc(clf, fx, fy)))
