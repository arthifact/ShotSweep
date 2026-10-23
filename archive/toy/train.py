"""Unrolled training: T painter steps, each conditioned on the exact adjoint of the current estimate.
The adjoint is an input feature (not differentiated through); supervision is L1 on the rendered field
after every step (deep supervision)."""
import argparse, json, math, os, time
import torch
import torch.nn.functional as F
from wave import gradient, misfit, start_model
from scene import init_state, render, param_grads, render_pixel, K
from painters import GaussPainter, PixelPainter, OneShotNet

p = argparse.ArgumentParser()
p.add_argument("--state", choices=["gauss", "pixel", "oneshot"], required=True)
p.add_argument("--ntrain", type=int, default=1000)
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--T", type=int, default=4)
p.add_argument("--bs", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--threads", type=int, default=3)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
args = p.parse_args()
torch.set_num_threads(args.threads)
torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)

tr = torch.load("data/train.pt")
te = torch.load("data/test.pt")
vtr, dtr = tr["v"][: args.ntrain], tr["d"][: args.ntrain]

net = {"gauss": GaussPainter, "pixel": PixelPainter, "oneshot": OneShotNet}[args.state]()
nparam = sum(q.numel() for q in net.parameters())
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr * 0.1)


def unroll(vt, d_obs, T, record=False):
    """Returns list of rendered estimates v_1..v_T (m/s), differentiable w.r.t. net."""
    B = vt.shape[0]
    v0 = start_model(B)
    outs = []
    if args.state == "oneshot":
        return [render_pixel(net(d_obs))], None
    if args.state == "gauss":
        bg, slots = init_state(v0)
        for t in range(T):
            v = render(bg, slots)
            J, g = gradient(v, d_obs)
            gb, gs = param_grads(bg, slots, g)
            d_bg, d_slots = net(bg, slots, gb, gs, g, v, t)
            bg, slots = bg + d_bg, slots + d_slots
            outs.append(render(bg, slots))
        return outs, slots.detach()
    vkm = v0 / 1000.0
    for t in range(T):
        v = render_pixel(vkm)
        J, g = gradient(v, d_obs)
        vkm = vkm + net(g, v, t)
        outs.append(render_pixel(vkm))
    return outs, None


def loss_fn(outs, vt):
    T = len(outs)
    return sum((t + 1) / T * (outs[t] - vt).abs().mean() / 1000.0 for t in range(T))


@torch.no_grad()
def evaluate(T):
    net.eval()
    errs, fams, J0s, JTs, slots_all = [], [], [], [], []
    for i in range(0, te["v"].shape[0], 16):
        vt, d_obs, fam = te["v"][i:i + 16], te["d"][i:i + 16], te["fam"][i:i + 16]
        with torch.enable_grad():
            outs, slots = unroll(vt, d_obs, T)
        vT = outs[-1].detach()
        v0 = start_model(vt.shape[0])
        errs.append(((vT - vt).abs().mean(dim=(1, 2)) / (v0 - vt).abs().mean(dim=(1, 2))))
        J0s.append(misfit(v0, d_obs)); JTs.append(misfit(vT, d_obs))
        fams.append(fam)
        if slots is not None:
            slots_all.append(slots)
    net.train()
    err, fam = torch.cat(errs), torch.cat(fams)
    J0, JT = torch.cat(J0s), torch.cat(JTs)
    res = {"err_ratio": err.mean().item(),
           "err_ratio_ellipse": err[fam == 0].mean().item(),
           "err_ratio_fault": err[fam == 1].mean().item(),
           "misfit_ratio": (JT / J0).mean().item()}
    if slots_all:
        s = torch.cat(slots_all)
        res["active_slots_mean"] = (s[..., 4].abs() > 0.05).float().sum(1).mean().item()
        res["slot_amp_abs_mean"] = s[..., 4].abs().mean().item()
    return res, (torch.cat(errs))


log = []
t0 = time.time()
g = torch.Generator().manual_seed(args.seed)
for it in range(args.steps):
    idx = torch.randint(0, args.ntrain, (args.bs,), generator=g)
    outs, _ = unroll(vtr[idx], dtr[idx], args.T)
    loss = loss_fn(outs, vtr[idx])
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step(); sched.step()
    if it % 50 == 0 or it == args.steps - 1:
        final_err = ((outs[-1] - vtr[idx]).abs().mean() / (start_model(args.bs) - vtr[idx]).abs().mean()).item()
        log.append({"it": it, "loss": loss.item(), "train_err_ratio": final_err, "t": time.time() - t0})
        print(f"[{args.out}] it {it} loss {loss.item():.4f} train_err_ratio {final_err:.3f} {time.time()-t0:.0f}s", flush=True)

res = {"args": vars(args), "nparam": nparam, "train_time_s": time.time() - t0, "log": log}
res["test_T"] = evaluate(args.T)[0]
if args.state != "oneshot":
    res["test_T+2"] = evaluate(args.T + 2)[0]
json.dump(res, open(f"{args.out}/result.json", "w"), indent=1)
torch.save(net.state_dict(), f"{args.out}/net.pt")
print(json.dumps({k: v for k, v in res.items() if k.startswith("test")}, indent=1))
