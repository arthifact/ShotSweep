#!/bin/zsh
# FlatFault-A rerun with the refraction channels; appends to toy/slopes/train_results.jsonl (phys/both/raw rows already there).
cd /Users/torus/Desktop/Research/veronicae_betifulae
t() { uv run python -I toy/slopes/train_channels.py --threads 2 "$@" 2>&1 | grep -v Warning | tail -1; }
t --train flatfault-a --test flatfault-a --variant refr
for seed in 0 1; do for n in 452 100 25; do for v in physR bothR; do
  t --train flatfault-a --test flatfault-a --variant $v --ntrain $n --seed $seed; done; done; done
for n in 452 100; do for v in physL physP; do t --train flatfault-a --test flatfault-a --variant $v --ntrain $n --seed 0; done; done
for v in physR bothR; do t --train flatfault-a --test flatvel-a --variant $v --ntrain 452; done
