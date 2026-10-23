"""Fast, minimal, watchable inversion learner.
The wave physics runs ONCE per stage over the whole dataset (cached adjoint images); the optimizer loop
contains only the network, so a stage trains in about a minute and you can watch it at toy/live/index.html.
A hypernet maps (adjoint image, current field[, gathers]) -> velocity pixels, or -> Gaussian scene parameters.
Stage s+1 is trained on adjoints recomputed at stage s's predictions (deep unfolding, stage-wise)."""
import argparse, json, os, time
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wave import gradient, misfit, start_model, NZ, NX, NS
from scene import init_state, render, render_pixel, K, BGZ, BGX

p = argparse.ArgumentParser()
p.add_argument("--input", choices=["grad", "gathers", "both"], default="grad")
p.add_argument("--state", choices=["pixel", "gauss"], default="pixel")
p.add_argument("--stages", type=int, default=1)
p.add_argument("--steps", type=int, default=600)
p.add_argument("--ntrain", type=int, default=1000)
p.add_argument("--bs", type=int, default=16)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--name", default=None)
p.add_argument("--nolive", action="store_true")
args = p.parse_args()
torch.set_num_threads(args.threads); torch.manual_seed(0)
name = args.name or f"{args.state}_{args.input}_s{args.stages}_n{args.ntrain}"
out_dir = f"runs/fast_{name}"; os.makedirs(out_dir, exist_ok=True); os.makedirs("live", exist_ok=True)
MUTE = 5


def status(msg):
    print(msg, flush=True)
    if not args.nolive: open("live/status.txt", "w").write(msg)


def prep(g):
    """Illumination compensation: mute the source rows, normalise each depth row by its RMS, clip."""
    g = g.clone(); g[:, :MUTE] = 0
    rms = g.pow(2).mean(dim=2, keepdim=True).sqrt() + 1e-12
    return (g / rms).clamp(-5, 5)


def grads_for(v, d, bs=16):
    return torch.cat([gradient(v[i:i + bs], d[i:i + bs])[1] for i in range(0, v.shape[0], bs)])


tr, te = torch.load("data/train.pt"), torch.load("data/test.pt")
vtr, dtr = tr["v"][:args.ntrain], tr["d"][:args.ntrain]
vte, dte, fte = te["v"], te["d"], te["fam"]
if not os.path.exists("data/g0.pt"):
    status("caching first adjoint for all models (one pass of physics)...")
    torch.save({"train": grads_for(start_model(tr["v"].shape[0]), tr["d"]), "test": grads_for(start_model(96), dte)}, "data/g0.pt")
g0 = torch.load("data/g0.pt")
Gtr, Gte = g0["train"][:args.ntrain], g0["test"]
Vtr, Vte = start_model(args.ntrain), start_model(96)


class Net(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        cin = 2 if args.input != "gathers" else 1
        blk = lambda i, o, s=1: nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1), nn.GELU())
        self.e1 = nn.Sequential(blk(cin, c), blk(c, c))
        self.e2 = nn.Sequential(blk(c, 2 * c, 2), blk(2 * c, 2 * c))
        self.e3 = nn.Sequential(blk(2 * c, 2 * c, 2), blk(2 * c, 2 * c))
        if args.input != "grad":
            self.gath = nn.Sequential(nn.Conv2d(NS, 16, 5, 2, 2), nn.GELU(), nn.Conv2d(16, 32, 5, 2, 2), nn.GELU(),
                                      nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(), nn.AdaptiveAvgPool2d((8, 4)), nn.Flatten(),
                                      nn.Linear(64 * 32, 2 * c * 8 * 10), nn.GELU())
        if args.state == "pixel":
            self.d2 = blk(4 * c, 2 * c); self.d1 = blk(3 * c, c); self.out = nn.Conv2d(c, 1, 1)
        else:  # hypernet head: scene parameters
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d((4, 5)), nn.Flatten(), nn.Linear(2 * c * 20, 256), nn.GELU(),
                                      nn.Linear(256, BGZ * BGX + K * 5))
            self.register_buffer("slot_scale", torch.tensor([0.15, 0.15, 0.5, 0.5, 0.5]))
        last = self.out if args.state == "pixel" else self.head[-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def forward(self, g, v, d):
        B = v.shape[0]
        vn = (v - 2500) / 1000
        x = torch.stack([prep(g), vn], 1) if args.input != "gathers" else vn[:, None]
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2)
        if args.input != "grad":
            dn = d / (d.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-12)
            e3 = e3 + self.gath(dn).reshape(B, -1, 8, 10)
        if args.state == "pixel":
            u2 = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
            u1 = F.interpolate(self.d2(torch.cat([u2, e2], 1)), size=e1.shape[-2:], mode="bilinear", align_corners=False)
            return render_pixel(v / 1000 + self.out(self.d1(torch.cat([u1, e1], 1)))[:, 0])
        h = self.head(e3)
        bg, slots = init_state(v)
        bg = bg + 0.5 * h[:, :BGZ * BGX].reshape(B, BGZ, BGX)
        slots = slots + h[:, BGZ * BGX:].reshape(B, K, 5) * self.slot_scale
        return render(bg, slots)


def err_ratio(vp, vt):
    return (vp - vt).abs().mean(dim=(1, 2)) / (start_model(vt.shape[0]) - vt).abs().mean(dim=(1, 2))


def live_plot(net, stage, curve, t0):
    idx = [0, 1, 2, 5]
    with torch.no_grad():
        vp = net(Gte[idx], Vte[idx], dte[idx])
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(4, 5, width_ratios=[1, 1, 1, 1, 1.6])
    for i, j in enumerate(idx):
        for c, (title, img, kw) in enumerate([("input: adjoint (prepped)", prep(Gte[j:j + 1])[0], dict(cmap="RdBu_r", vmin=-3, vmax=3)),
                                               ("stage input field", Vte[j], dict(cmap="viridis", vmin=1500, vmax=4200)),
                                               ("prediction", vp[i], dict(cmap="viridis", vmin=1500, vmax=4200)),
                                               ("true", vte[j], dict(cmap="viridis", vmin=1500, vmax=4200))]):
            a = fig.add_subplot(gs[i, c]); a.imshow(img, **kw); a.set_xticks([]); a.set_yticks([])
            if i == 0: a.set_title(title, fontsize=9)
    a = fig.add_subplot(gs[:, 4])
    if curve:
        it, tl, tst = zip(*curve)
        a.plot(it, tl, label="train err ratio"); a.plot(it, tst, label="test err ratio")
        a.set_ylim(0, 1.05); a.set_xlabel("optimizer step"); a.legend(fontsize=8); a.grid(alpha=.3)
        a.set_title(f"stage {stage}  test err {tst[-1]:.3f}  {time.time()-t0:.0f}s", fontsize=9)
    plt.tight_layout(); plt.savefig("live/live_tmp.png", dpi=90); plt.close(fig)
    os.replace("live/live_tmp.png", "live/live.png")


result = {"args": vars(args), "stages": []}
T0 = time.time()
for stage in range(args.stages):
    net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.1)
    curve, t0 = [], time.time()
    for it in range(args.steps):
        idx = torch.randint(0, args.ntrain, (args.bs,))
        vp = net(Gtr[idx], Vtr[idx], dtr[idx])
        loss = (vp - vtr[idx]).abs().mean() / 1000
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if it % 20 == 0 or it == args.steps - 1:
            with torch.no_grad():
                tr_err = err_ratio(vp, vtr[idx]).mean().item()
                te_err = err_ratio(net(Gte, Vte, dte), vte).mean().item()
            curve.append((it, tr_err, te_err))
            status(f"stage {stage}  step {it}/{args.steps}  train err {tr_err:.3f}  test err {te_err:.3f}  "
                   f"{time.time()-t0:.0f}s  ({(it+1)/(time.time()-t0+1e-9):.1f} steps/s)")
            if not args.nolive: live_plot(net, stage, curve, t0)
    with torch.no_grad():
        pte = net(Gte, Vte, dte); e = err_ratio(pte, vte)
        result["stages"].append({"stage": stage, "test_err_ratio": e.mean().item(), "ellipse": e[fte == 0].mean().item(),
                                 "fault": e[fte == 1].mean().item(), "misfit_ratio": (misfit(pte, dte) / misfit(Vte, dte)).mean().item(),
                                 "train_s": time.time() - t0})
    torch.save(net.state_dict(), f"{out_dir}/net_stage{stage}.pt")
    print(json.dumps(result["stages"][-1]), flush=True)
    if stage + 1 < args.stages:
        status(f"stage {stage} done, test err {e.mean().item():.3f}. Recomputing adjoints at the predictions for stage {stage+1}...")
        with torch.no_grad():
            Vtr = torch.cat([net(Gtr[i:i + 64], Vtr[i:i + 64], dtr[i:i + 64]) for i in range(0, args.ntrain, 64)]); Vte = pte
        Gtr, Gte = grads_for(Vtr, dtr), grads_for(Vte, dte)
result["total_s"] = time.time() - T0
json.dump(result, open(f"{out_dir}/result.json", "w"), indent=1)
status(f"done {name}: " + "  ".join(f"stage{s['stage']} test err {s['test_err_ratio']:.3f}" for s in result["stages"]) + f"  total {result['total_s']:.0f}s")
