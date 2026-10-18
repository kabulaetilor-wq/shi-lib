import sys; sys.path.insert(0,'.')
exec(open('experiment.py').read().split("print('phrases libres")[0])
from shi.models.classifier import tokenizer
for w in (0,1,2,4):
    tokenizer.LEXICON_WEIGHT=w
    for name,(X,Y,VX,VY) in {'libre seul':(tr_x,tr_y,va_x,va_y),'gabarits+libre x3':(tt+tr_x*3,tl+tr_y*3,vt+va_x,vl+va_y)}.items():
        clf=SHIClassifier(); clf.fit(X,Y,VX,VY)
        print('poids lexique=%d  %-20s test=%.3f'%(w,name,acc(clf,xs,ys)))
