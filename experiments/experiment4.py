import sys; sys.path.insert(0,'.')
exec(open('experiment.py').read().split("print('phrases libres")[0])
from shi.models.classifier import tokenizer
bx, by = load_jsonl('src/shi/datasets/test_aveugle.jsonl')
for w in (0, 2):
    tokenizer.LEXICON_WEIGHT = w
    clf = SHIClassifier(); clf.fit(tt + tr_x * 3, tl + tr_y * 3, vt + va_x, vl + va_y)
    print('poids lexique=%d  test aveugle (%d phrases): %.3f' % (w, len(bx), acc(clf, bx, by)))
    if w == 2:
        P = clf.predict_labels(bx)
        from collections import Counter
        err = Counter((y, p) for y, p in zip(by, P) if y != p)
        for (y, p), c in err.most_common(): print('   %s -> %s : %d' % (y, p, c))
