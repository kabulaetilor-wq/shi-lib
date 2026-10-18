import sys; sys.path.insert(0,'.')
exec(open('experiment.py').read().split("print('phrases libres")[0])
import itertools
X=tt+tr_x*3; Y=tl+tr_y*3; VX=vt+va_x; VY=vl+va_y
for l2,lr,ep in [(1e-4,0.05,400),(1e-3,0.05,400),(1e-2,0.05,400),(1e-5,0.1,600),(1e-3,0.2,600),(3e-3,0.1,800)]:
    clf=SHIClassifier(l2=l2,lr=lr,epochs=ep); clf.fit(X,Y,VX,VY)
    print('l2=%g lr=%g ep=%d  test=%.3f'%(l2,lr,ep,acc(clf,xs,ys)))
# libre seul, plus fort poids
for l2 in (1e-3,1e-2):
    clf=SHIClassifier(l2=l2,lr=0.1,epochs=600); clf.fit(tr_x,tr_y,va_x,va_y)
    print('libre seul l2=%g test=%.3f'%(l2,acc(clf,xs,ys)))
