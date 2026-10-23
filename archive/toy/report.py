"""Collect results into a table and a figure of test examples."""
import glob, json, os, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.set_num_threads(4)
from wave import gradient, start_model
from scene import init_state, render, param_grads, render_pixel
from painters import GaussPainter, PixelPainter, OneShotNet

rows = []
for f in sorted(glob.glob("runs/*/result.json")):
    r = json.load(open(f)); a = r["args"]
    for key in ["test_T", "test_T+2"]:
        if key in r:
            t = r[key]
            rows.append((f"{a['state']:8s} ntrain={a['ntrain']:5d} {key:8s}", t["err_ratio"], t["err_ratio_ellipse"], t["err_ratio_fault"], t["misfit_ratio"], t.get("active_slots_mean", float('nan'))))
if os.path.exists("runs_gd.json"):
    for k, t in json.load(open("runs_gd.json")).items():
        rows.append((f"GD {k:14s} (no learning)", t["err_ratio"], t.get("err_ratio_ellipse", float('nan')), t.get("err_ratio_fault", float('nan')), t.get("misfit_ratio", float('nan')), float('nan')))
print(f"{'run':38s} {'err':>6s} {'ellip':>6s} {'fault':>6s} {'misfit':>7s} {'slots':>6s}")
for r in rows:
    print(f"{r[0]:38s} {r[1]:6.3f} {r[2]:6.3f} {r[3]:6.3f} {r[4]:7.3f} {r[5]:6.1f}")
with open("results.md", "w") as fh:
    fh.write("| run | err ratio | ellipse | fault | misfit ratio | active slots |\n|---|---|---|---|---|---|\n")
    for r in rows:
        fh.write(f"| {r[0]} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.1f} |\n")

# figure
te = torch.load("data/test.pt")
idx = [0, 1, 2, 3, 5, 8]
vt, d_obs = te["v"][idx], te["d"][idx]
v0 = start_model(len(idx))
cols = {"true": vt, "start": v0}
J, g = gradient(v0, d_obs); cols["adjoint @ start"] = g / g.abs().amax(dim=(1, 2), keepdim=True)
for name, cls in [("gauss", GaussPainter), ("pixel", PixelPainter), ("oneshot", OneShotNet)]:
    d = f"runs/{name}_n1000"
    if not os.path.exists(f"{d}/net.pt"):
        continue
    net = cls(); net.load_state_dict(torch.load(f"{d}/net.pt")); net.eval()
    T = 4
    if name == "oneshot":
        with torch.no_grad():
            cols["one-shot (gathers)"] = render_pixel(net(d_obs))
        continue
    if name == "gauss":
        bg, slots = init_state(v0)
        for t in range(T):
            v = render(bg, slots); J, g = gradient(v, d_obs); gb, gs = param_grads(bg, slots, g)
            with torch.no_grad():
                dbg, ds = net(bg, slots, gb, gs, g, v, t)
            bg, slots = bg + dbg, slots + ds
        cols["gauss painter T=4"] = render(bg, slots).detach()
    else:
        vkm = v0 / 1000
        for t in range(T):
            v = render_pixel(vkm); J, g = gradient(v, d_obs)
            with torch.no_grad():
                vkm = vkm + net(g, v, t)
        cols["pixel painter T=4"] = render_pixel(vkm).detach()
fig, ax = plt.subplots(len(idx), len(cols), figsize=(2.2 * len(cols), 1.8 * len(idx)))
for j, (name, arr) in enumerate(cols.items()):
    for i in range(len(idx)):
        a = ax[i, j]
        if "adjoint" in name:
            a.imshow(arr[i], cmap="RdBu_r", vmin=-1, vmax=1)
        else:
            a.imshow(arr[i], cmap="viridis", vmin=1500, vmax=4200)
        a.set_xticks([]); a.set_yticks([])
        if i == 0: a.set_title(name, fontsize=9)
plt.tight_layout(); plt.savefig("results.png", dpi=110)
print("wrote results.md results.png")
