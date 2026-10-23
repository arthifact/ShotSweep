err ratio = fraction of the initial velocity error left on 96 held-out models (1.0 = no better than the 1D start).

| run | err ratio | ellipse | fault | misfit ratio | train s |
|---|---|---|---|---|---|
| fast pixel in=both    steps=3000 stage1 | 0.318 | 0.420 | 0.256 | 0.632 | 319 |
| fast pixel in=gathers steps=3000 stage0 | 0.341 | 0.432 | 0.287 | 0.035 | 390 |
| fast pixel in=both    steps=3000 stage0 | 0.345 | 0.433 | 0.292 | 0.032 | 393 |
| unrolled oneshot ntrain=1000 T=4 | 0.420 | 0.516 | 0.362 | 0.051 | 46 |
| fast gauss in=both    steps=3000 stage1 | 0.502 | 0.581 | 0.454 | 0.878 | 229 |
| fast gauss in=both    steps=3000 stage0 | 0.513 | 0.593 | 0.465 | 0.305 | 246 |
| unrolled oneshot ntrain= 100 T=4 | 0.527 | 0.629 | 0.466 | 0.177 | 46 |
| fast pixel in=grad    steps=3000 stage1 | 0.537 | 0.624 | 0.485 | 0.535 | 254 |
| fast pixel in=grad    steps= 600 stage1 | 0.569 | 0.611 | 0.544 | 0.522 | 54 |
| fast pixel in=grad    steps=3000 stage0 | 0.577 | 0.661 | 0.527 | 0.188 | 279 |
| unrolled gauss   ntrain=1000 T=4 | 0.672 | 0.715 | 0.646 | 0.212 | 1162 |
| fast pixel in=grad    steps= 600 stage0 | 0.678 | 0.723 | 0.650 | 0.280 | 61 |
| unrolled pixel   ntrain= 100 T=4 | 0.767 | 0.814 | 0.739 | 0.169 | 1204 |
| unrolled gauss   ntrain= 100 T=4 | 0.796 | 0.847 | 0.766 | 0.153 | 1166 |
| unrolled pixel   ntrain=1000 T=4 | 0.808 | 0.870 | 0.771 | 0.708 | 1201 |
| no-learning gauss_T30 | 0.809 | 0.808 | 0.809 | 0.014 | 0 |
| no-learning gauss_T4 | 0.883 | 0.881 | 0.885 | 0.179 | 0 |
| no-learning dataset_mean | 0.918 | nan | nan | nan | 0 |
| no-learning pixel_T4 | 1.003 | 1.000 | 1.005 | 0.553 | 0 |
| no-learning pixel_T30 | 1.011 | 1.002 | 1.017 | 0.304 | 0 |
