import sys, glob, os
import numpy as np
from collections import Counter
ROOT=sys.argv[1]
def rle(col):
    out=[];start=0
    for i in range(1,len(col)+1):
        if i==len(col) or abs(col[i]-col[start])>1.0:
            out.append((start,i-start,int(round(col[start])), int(round(col[i-1]))));start=i
    return out
def runs_with_grad(col):
    # segments split at jumps > 30 m/s
    d=np.diff(col); cut=np.nonzero(np.abs(d)>30)[0]+1
    b=np.concatenate([[0],cut,[len(col)]])
    return [(int(b[k]),int(b[k+1]-b[k]),int(col[b[k]]),int(col[b[k+1]-1])) for k in range(len(b)-1)]
for fam in ["flatfault-a","flatfault-b","curvefault-a","curvefault-b"]:
    f=sorted(glob.glob(os.path.join(ROOT,fam,"vel*.npy")))[0]
    v=np.asarray(np.load(f,allow_pickle=False,mmap_mode="r"),dtype=float)[:,0]
    print("==",fam)
    for i in [0,1,2]:
        print(" model",i,"col0 segments(start,len,vtop,vbot):",runs_with_grad(v[i,:,0]))
        print("          col69:",runs_with_grad(v[i,:,69]))
    # number of 'major' layers: segments split by jumps>30 in centre col
    nl=[len(runs_with_grad(v[i,:,35])) for i in range(len(v))]
    print(" major segments (jumps>30 m/s) centre col hist:",sorted(Counter(nl).items()))
    # within-layer gradient: median slope inside segments of len>=5
    sl=[]
    for i in range(len(v)):
        for (s,l,a,b) in runs_with_grad(v[i,:,35]):
            if l>=5: sl.append((b-a)/(l-1))
    sl=np.array(sl); print(" within-layer gradient m/s per cell p5/50/95:",np.percentile(sl,[5,50,95]).round(2), " frac segments with gradient>1:",np.mean(np.abs(sl)>1).round(3))
    # lateral jump connected components (faults)
    try:
        from scipy import ndimage
        nf=[]
        for i in range(len(v)):
            m=np.abs(np.diff(v[i],axis=1))>30
            m=ndimage.binary_dilation(m,iterations=2)
            nf.append(ndimage.label(m)[1])
        print(" connected lateral-jump components (dilated 2):",sorted(Counter(nf).items()))
    except Exception as e: print("no scipy",e)
for fam in ["flatvel-a","flatvel-b","curvevel-a","curvevel-b"]:
    v=np.asarray(np.load(os.path.join(ROOT,fam,"model1.npy"),allow_pickle=False,mmap_mode="r"),dtype=float)[:,0]
    print("==",fam,"model0 col35:",rle(v[0,:,35]))
    print("    model1 col35:",rle(v[1,:,35]))
for fam in ["style-a","style-b"]:
    v=np.asarray(np.load(os.path.join(ROOT,fam,"model1.npy"),allow_pickle=False,mmap_mode="r"),dtype=float)[:,0]
    # depth trend: mean over x per depth, averaged over models; fraction of models where row-mean increases monotonic
    rm=v.mean(axis=2)
    print("==",fam,"dataset mean v at depth rows 0,10,...,69:",rm.mean(0)[::10].round(0), rm.mean(0)[-1].round(0))
    mono=np.mean([np.all(np.diff(r)>=-1) for r in rm]); print("  models with laterally-averaged v monotone in depth:",mono)
    # correlation length laterally vs vertically: autocorr at lag 5
    def ac(a,axis,lag):
        a=a-a.mean(axis=(1,2),keepdims=True)
        if axis==1: x,y=a[:,:-lag,:],a[:,lag:,:]
        else: x,y=a[:,:,:-lag],a[:,:,lag:]
        return (x*y).mean()/np.sqrt((x*x).mean()*(y*y).mean())
    dt=v-rm[:,:,None]
    print("  autocorr of detrended v at lag5: vertical",round(ac(dt,1,5),3),"lateral",round(ac(dt,2,5),3))
