"""Generate synthetic models and their observed data once; cache to disk."""
import os, sys, torch
torch.set_num_threads(6)
from wave import forward
from models import make_models

os.makedirs("data", exist_ok=True)
for name, n, seed in [("test", 96, 999), ("train", 2000, 1)]:
    vt, fam = make_models(n, seed)
    d = []
    with torch.no_grad():
        for i in range(0, n, 16):
            d.append(forward(vt[i:i + 16]))
            if i % 320 == 0:
                print(name, i, flush=True)
    torch.save({"v": vt, "fam": fam, "d": torch.cat(d)}, f"data/{name}.pt")
    print("saved", name, vt.shape)
