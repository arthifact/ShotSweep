import sys, glob, os
import numpy as np
from collections import Counter
ROOT=sys.argv[1]
for fam in ["flatvel-a","flatvel-b","curvevel-a","curvevel-b","flatfault-a","flatfault-b","curvefault-a","curvefault-b"]:
    f=sorted(glob.glob(os.path.join(ROOT,fam,"model*.npy"))+glob.glob(os.path.join(ROOT,fam,"vel*.npy")))[0]
    v=np.asarray(np.load(f,allow_pickle=False,mmap_mode="r"),dtype=float)[:,0]
    nl=[];neg=0;tot=0;ramp=0
    for i in range(len(v)):
        col=v[i,:,0]  # left edge column, away from fault in most models
        d=np.diff(col);cut=np.nonzero(np.abs(d)>30)[0]+1
        b=np.concatenate([[0],cut,[70]]);L=np.diff(b)
        segs=[(b[k],L[k],col[b[k]]) for k in range(len(L)) if L[k]>=3]
        nl.append(len(segs)); ramp+= int(np.any(L<=2))
        for a,c in zip(segs[:-1],segs[1:]):
            tot+=1; neg+= c[2]<a[2]-30
    c=Counter(nl)
    print(f"{fam}: layers(>=3 cells, col0) hist {sorted(c.items())}; median {np.median(nl)}; inversions {neg}/{tot}={neg/max(tot,1):.1%}; models w/ <=2-cell transition segs {ramp/len(v):.0%}")
