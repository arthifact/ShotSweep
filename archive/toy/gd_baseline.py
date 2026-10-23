"""No learning: plain Adam on the pixel field or on the Gaussian scene using the same exact adjoint."""
import json, sys, torch
torch.set_num_threads(4)
from wave import gradient, misfit, start_model
from scene import init_state, render, render_pixel

te = torch.load("data/test.pt"); tr = torch.load("data/train.pt")


def run(state, vt, d_obs, T, lr):
    B = vt.shape[0]
    v0 = start_model(B)
    if state == "pixel":
        params = [(v0 / 1000.0).clone().requires_grad_(True)]
        rend = lambda: render_pixel(params[0])
    else:
        bg, slots = init_state(v0)
        params = [bg.clone().requires_grad_(True), slots.clone().requires_grad_(True)]
        rend = lambda: render(params[0], params[1])
    opt = torch.optim.Adam(params, lr=lr)
    for t in range(T):
        v = rend()
        J, g = gradient(v, d_obs)
        g = g / (g.abs().amax(dim=(1, 2), keepdim=True) + 1e-12)   # per-sample normalised, like the painter sees
        opt.zero_grad()
        v.backward(g)
        opt.step()
    with torch.no_grad():
        vT = rend()
        err = (vT - vt).abs().mean(dim=(1, 2)) / (v0 - vt).abs().mean(dim=(1, 2))
        mr = misfit(vT, d_obs) / misfit(v0, d_obs)
    return err, mr


out = {}
for state in ["pixel", "gauss"]:
    for T in [4, 30]:
        best = None
        for lr in [0.003, 0.01, 0.03, 0.1]:
            e, _ = run(state, tr["v"][:16], tr["d"][:16], T, lr)
            if best is None or e.mean() < best[0]:
                best = (e.mean().item(), lr)
        lr = best[1]
        errs, mrs = [], []
        for i in range(0, 96, 16):
            e, m = run(state, te["v"][i:i + 16], te["d"][i:i + 16], T, lr)
            errs.append(e); mrs.append(m)
        e, m = torch.cat(errs), torch.cat(mrs)
        fam = te["fam"]
        out[f"{state}_T{T}"] = {"lr": lr, "err_ratio": e.mean().item(), "err_ratio_ellipse": e[fam == 0].mean().item(),
                                "err_ratio_fault": e[fam == 1].mean().item(), "misfit_ratio": m.mean().item()}
        print(state, T, out[f"{state}_T{T}"], flush=True)
# dataset-mean baseline: what a painter that ignores the physics could do at best
vm = tr["v"][:1000].mean(0, keepdim=True)
v0 = start_model(96)
e = (vm - te["v"]).abs().mean(dim=(1, 2)) / (v0 - te["v"]).abs().mean(dim=(1, 2))
out["dataset_mean"] = {"err_ratio": e.mean().item()}
print("dataset_mean", out["dataset_mean"])
json.dump(out, open("runs_gd.json", "w"), indent=1)
