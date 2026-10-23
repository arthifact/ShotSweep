#!/bin/zsh
# All training runs for the physics-channel comparison; results append to toy/slopes/train_results.jsonl.
cd /Users/torus/Desktop/Research/veronicae_betifulae
t() { uv run python -I toy/slopes/train_channels.py --threads 2 "$@" 2>&1 | grep -v Warning | tail -1; }
for fam in flatvel-a flatfault-a; do t --train $fam --test $fam --variant geo; done
for seed in 0 1; do for n in 452 100 25; do for fam in flatvel-a flatfault-a; do for v in phys both raw; do
  t --train $fam --test $fam --variant $v --ntrain $n --seed $seed; done; done; done; done
for v in phys both raw; do
  t --train flatvel-a --test flatfault-a --variant $v --ntrain 452
  t --train flatfault-a --test flatvel-a --variant $v --ntrain 452
done
