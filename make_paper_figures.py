"""Reproduce the paper's figures and the matched-target / matched-movement analysis.

Reads results/all_runs.csv and results/baseline.json (both included in this repo), so it runs
on any machine without a GPU:  python make_paper_figures.py
Outputs figures/fig1_lr_sweep.png, fig2_forgetting_vs_weight_change.png, fig3_tradeoff.png
and prints Table 2 and the correlation statistics.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

base = json.load(open(os.path.join(RES, "baseline.json")))
d = pd.read_csv(os.path.join(RES, "all_runs.csv"))
d["forget"] = d.fineweb_loss - base["fineweb_loss"]
d["target"] = np.where(d.setting == "narrow", d.wikitext2_val_loss, d.tulu_val_loss)
g = (d.groupby(["setting", "opt", "lr"])
     .agg(target=("target", "mean"), forget=("forget", "mean"), rel=("rel_change_matrices", "mean"),
          n=("seed", "count")).reset_index())

# best rate per (setting, optimizer): lowest seed-0 target loss (the rule used in the study)
best = {}
for (s, o), x in d[d.seed == 0].groupby(["setting", "opt"]):
    best[(s, o)] = x.loc[x.target.idxmin(), "lr"]

C = {"adamw": "#1f5fa8", "muon": "#c8442f"}
L = {"adamw": "AdamW (matched)", "muon": "Muon (switched)"}
T = {"narrow": "Narrow: WikiText-2", "broad": "Broad: Tulu-3 subset"}
YT = {"narrow": "WikiText-2 validation loss", "broad": "Tulu-3 held-out loss"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "serif"})

# Figure 1: target loss vs learning rate (1e-4 omitted: off scale)
fig, ax = plt.subplots(1, 2, figsize=(7, 2.7))
for a, s in zip(ax, ["narrow", "broad"]):
    for o in ["adamw", "muon"]:
        x = g[(g.setting == s) & (g.opt == o) & (g.lr < 1e-4)].sort_values("lr")
        a.plot(x.lr, x.target, "o-", color=C[o], label=L[o], ms=3.5, lw=1.3)
        b = x[np.isclose(x.lr, best[(s, o)])]
        a.plot(b.lr, b.target, "o", ms=8, mfc="none", mec=C[o], mew=1.3)
    a.set_xscale("log"); a.set_xlabel("learning rate"); a.set_ylabel(YT[s]); a.set_title(T[s], fontsize=9)
ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_lr_sweep.png"), dpi=300)

# Figure 2: forgetting vs relative weight change, every run
fig, ax = plt.subplots(1, 2, figsize=(7, 2.7))
for a, s in zip(ax, ["narrow", "broad"]):
    for o in ["adamw", "muon"]:
        x = d[(d.setting == s) & (d.opt == o)]
        a.scatter(x.rel_change_matrices, x.forget, s=16, color=C[o], label=L[o], alpha=.85, edgecolor="none")
    x = d[d.setting == s]
    r = np.corrcoef(x.rel_change_matrices, x.forget)[0, 1]
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"relative weight change $\|W-W_0\|_F/\|W_0\|_F$")
    a.set_ylabel("forgetting (FineWeb-Edu loss increase)")
    a.set_title(f"{T[s]}  (Pearson r = {r:.3f})", fontsize=9)
ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_forgetting_vs_weight_change.png"), dpi=300)

# Figure 3: learning-forgetting trade-off along each sweep
fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
for a, s in zip(ax, ["narrow", "broad"]):
    for o in ["adamw", "muon"]:
        x = g[(g.setting == s) & (g.opt == o) & (g.lr <= 3e-5)].sort_values("lr")
        a.plot(x.target, x.forget, "o-", color=C[o], label=L[o], ms=3.5, lw=1.2)
        for _, r in x[x.lr < best[(s, o)]].iterrows():
            a.annotate(f"{r.lr:.0e}".replace("e-0", "e-"), (r.target, r.forget), fontsize=6, color=C[o],
                       xytext=(4, -3), textcoords="offset points")
        b = x[np.isclose(x.lr, best[(s, o)])]
        a.plot(b.target, b.forget, "o", ms=8, mfc="none", mec=C[o], mew=1.3)
    a.set_yscale("log"); a.set_xlabel(YT[s] + " (lower is better)"); a.set_ylabel("forgetting (lower is better)")
    a.set_title(T[s], fontsize=9)
ax[0].set_xlim(2.515, 2.60); ax[1].set_xlim(0.9165, 0.935)
ax[1].legend(frameon=False, fontsize=8, loc="upper center")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_tradeoff.png"), dpi=300)


def interp(xs, ys, x):
    o = np.argsort(xs)
    return float(np.interp(x, np.asarray(xs)[o], np.asarray(ys)[o]))


print("Best learning rates:", {f"{s}/{o}": f"{lr:.0e}" for (s, o), lr in best.items()})
print("\nTable 2: forgetting at equal target loss (AdamW interpolated on its low-LR branch)")
for s in ["narrow", "broad"]:
    a = g[(g.setting == s) & (g.opt == "adamw")].sort_values("lr")
    m = g[(g.setting == s) & (g.opt == "muon") & (g.lr <= best[(s, "muon")])].sort_values("lr")
    a_low = a.loc[:a.target.idxmin()]
    for _, r in m.iterrows():
        if a_low.target.min() <= r.target <= a_low.target.max():
            fa = interp(a_low.target.values, a_low.forget.values, r.target)
            print(f"  {s:6s} muon lr {r.lr:.0e}: target {r.target:.4f}  forgetting muon {r.forget:.4f}"
                  f"  adamw {fa:.4f}  ratio {r.forget / fa:.2f}")

print("\nMuon excess forgetting at equal weight change (AdamW interpolated)")
for s in ["narrow", "broad"]:
    a = g[(g.setting == s) & (g.opt == "adamw")]
    m = g[(g.setting == s) & (g.opt == "muon") & (g.lr <= 3e-5)]
    ex = [r.forget / interp(a.rel.values, a.forget.values, r.rel) - 1 for _, r in m.iterrows()
          if a.rel.min() <= r.rel <= a.rel.max()]
    print(f"  {s:6s} median {100 * np.median(ex):+.0f}%  range {100 * min(ex):+.0f}% to {100 * max(ex):+.0f}%")

print("\nCorrelation of forgetting with relative weight change (all runs)")
for s in ["narrow", "broad"]:
    x = d[d.setting == s]
    print(f"  {s:6s} Pearson {np.corrcoef(x.rel_change_matrices, x.forget)[0, 1]:.4f}  "
          f"Spearman {x[['rel_change_matrices', 'forget']].corr(method='spearman').iloc[0, 1]:.4f}")
print("\nFigures written to", FIG)
