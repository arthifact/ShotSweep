#!/bin/zsh
# One model for every family: joint training, transfer, data efficiency. Appends to toy/slopes/train2_results.jsonl.
cd /Users/torus/Desktop/Research/veronicae_betifulae
t() { uv run python -I toy/slopes/train.py --save toy/slopes/preds2 "$@" 2>&1 | grep -v Warning | grep "^{"; }
B=flatvel-a,flatfault-a
for seed in 0 1; do for i in both physics; do t --inputs $i --train $B --test $B --seed $seed; done; done
t --inputs raw --train $B --test $B
for i in both physics raw; do t --inputs $i --train flatvel-a --test $B; t --inputs $i --train flatfault-a --test $B; done
for n in 25 100; do for i in both physics raw; do t --inputs $i --train $B --test $B --ntrain $n; done; done
