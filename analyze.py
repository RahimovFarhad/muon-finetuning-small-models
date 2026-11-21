"""Make the paper's tables and figures from results/*.json.

Outputs: tables/*.md, tables/*.csv and figures/*.pdf (+ .png previews) under STUDY_ROOT.
"""
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS_DIR, ROOT
from pick_best import TARGET, best_lrs

FIG, TAB = f"{ROOT}/figures", f"{ROOT}/tables"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
COL = {"adamw": "#1f77b4", "muon": "#d62728"}
LABEL = {"narrow": "Narrow (WikiText-2)", "broad": "Broad (Tulu-3 subset)"}
TLABEL = {"narrow": "WikiText-2 val loss", "broad": "Tulu-3 held-out loss"}
BENCH = ["arc_easy", "hellaswag", "mmlu"]

runs = [json.load(open(f)) for f in glob.glob(f"{RESULTS_DIR}/*.json")
        if not f.endswith("baseline.json") and "_smoke" not in f]
runs = [r for r in runs if r.get("setting")]
base = json.load(open(f"{RESULTS_DIR}/baseline.json")) if os.path.exists(
    f"{RESULTS_DIR}/baseline.json") else {}
print(f"{len(runs)} runs loaded; baseline {'found' if base else 'MISSING'}")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.pdf")
    fig.savefig(f"{FIG}/{name}.png", dpi=150)
    plt.close(fig)


def forgetting(r):
    return r["fineweb_loss"] - base["fineweb_loss"] if base else r["fineweb_loss"]


def bench_avg(r):
    v = [r[b] for b in BENCH if b in r]
    return float(np.mean(v)) if v else float("nan")


# ---------------- Fig 1 + 2: learning-rate sweeps (seed 0) ----------------
for metric_fn, ylab, fname in [
        (lambda r, s: r[TARGET[s]], None, "fig1_lr_sweep_target"),
        (lambda r, s: forgetting(r), "Forgetting: FineWeb-Edu loss increase", "fig2_lr_sweep_forgetting")]:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, s in zip(axes, ["narrow", "broad"]):
        for o in ["adamw", "muon"]:
            d = sorted([r for r in runs if r["setting"] == s and r["opt"] == o and r["seed"] == 0
                        and not r.get("diverged")], key=lambda r: r["lr"])
            if d:
                ax.plot([r["lr"] for r in d], [metric_fn(r, s) for r in d], "o-",
                        color=COL[o], label=o.upper() if o == "muon" else "AdamW")
        ax.set_xscale("log")
        ax.set_xlabel("learning rate")
        ax.set_ylabel(ylab or TLABEL[s])
        ax.set_title(LABEL[s])
        if ylab:
            ax.axhline(0, color="grey", lw=0.8)
    axes[0].legend()
    save(fig, fname)

# ---------------- best-LR summary over seeds ----------------
best = best_lrs([r for r in runs if r["seed"] == 0])
rows, summary = [], {}
for (s, o), lr in best.items():
    d = [r for r in runs if r["setting"] == s and r["opt"] == o and math.isclose(r["lr"], lr)]
    vals = {
        "target": [r[TARGET[s]] for r in d],
        "forgetting": [forgetting(r) for r in d],
        "bench_avg": [bench_avg(r) for r in d],
        "rel_change": [r["rel_change_matrices"] for r in d],
        **{b: [r[b] for r in d if b in r] for b in BENCH},
    }
    summary[(s, o)] = vals
    rows.append((s, o, lr, len(d), vals))


def ms(v, k=4):
    v = [x for x in v if x == x]
    if not v:
        return "-"
    if len(v) == 1:
        return f"{v[0]:.{k}f}"
    return f"{np.mean(v):.{k}f} ± {np.std(v, ddof=1):.{k}f}"


hdr = ("| Setting | Optimizer | Best LR | Seeds | Target loss | Forgetting | Bench avg | "
       + " | ".join(BENCH) + " | Rel. weight change |")
md = [hdr, "|" + "---|" * (8 + len(BENCH) - 1)]
for s, o, lr, n, v in sorted(rows):
    md.append(f"| {s} | {o} | {lr:.0e} | {n} | {ms(v['target'])} | {ms(v['forgetting'])} | "
              f"{ms(v['bench_avg'], 3)} | " + " | ".join(ms(v[b], 3) for b in BENCH)
              + f" | {ms(v['rel_change'])} |")
if base:
    md.append(f"| baseline | (no fine-tuning) | - | - | wt2 {base.get('wikitext2_val_loss', float('nan')):.4f}"
              f" / tulu {base.get('tulu_val_loss', float('nan')):.4f} | 0 | {bench_avg(base):.3f} | "
              + " | ".join(f"{base[b]:.3f}" if b in base else "-" for b in BENCH) + " | 0 |")
open(f"{TAB}/table1_best_lr.md", "w").write("\n".join(md) + "\n")

# ---------------- the key comparison: Muon minus AdamW ----------------
lines = ["| Setting | Measure | Muon − AdamW | Within noise? |", "|---|---|---|---|"]
for s in ["narrow", "broad"]:
    if (s, "adamw") not in summary or (s, "muon") not in summary:
        continue
    for k, nice, better in [("target", "Target loss", "lower"), ("forgetting", "Forgetting", "lower"),
                            ("bench_avg", "Benchmark avg", "higher")]:
        a = [x for x in summary[(s, "adamw")][k] if x == x]
        m = [x for x in summary[(s, "muon")][k] if x == x]
        if not a or not m:
            continue
        diff = np.mean(m) - np.mean(a)
        if len(a) > 1 and len(m) > 1:
            se = math.sqrt(np.var(a, ddof=1) / len(a) + np.var(m, ddof=1) / len(m))
            noise = "yes" if abs(diff) <= 2 * se else "no"
            txt = f"{diff:+.4f} (±{2*se:.4f}, ~95%)"
        else:
            noise, txt = "need seeds", f"{diff:+.4f}"
        lines.append(f"| {s} | {nice} ({better} is better) | {txt} | {noise} |")
open(f"{TAB}/table2_muon_minus_adamw.md", "w").write("\n".join(lines) + "\n")

# ---------------- Fig 3: target vs forgetting trade-off (all runs) ----------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, s in zip(axes, ["narrow", "broad"]):
    for o in ["adamw", "muon"]:
        d = [r for r in runs if r["setting"] == s and r["opt"] == o and not r.get("diverged")]
        ax.scatter([r[TARGET[s]] for r in d], [forgetting(r) for r in d], color=COL[o],
                   label="Muon" if o == "muon" else "AdamW", s=28, alpha=0.8)
    ax.set_xlabel(TLABEL[s] + " (lower is better)")
    ax.set_ylabel("Forgetting (lower is better)")
    ax.set_title(LABEL[s])
axes[0].legend()
save(fig, "fig3_target_vs_forgetting")

# ---------------- Fig 4: training curves at best LR (seed 0) ----------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, s in zip(axes, ["narrow", "broad"]):
    for o in ["adamw", "muon"]:
        if (s, o) not in best:
            continue
        d = [r for r in runs if r["setting"] == s and r["opt"] == o and r["seed"] == 0
             and math.isclose(r["lr"], best[(s, o)])]
        if d and d[0]["train_log"]:
            lg = d[0]["train_log"]
            ax.plot([x["step"] for x in lg], [x["loss"] for x in lg], color=COL[o],
                    label=f"{'Muon' if o == 'muon' else 'AdamW'} (lr {best[(s, o)]:.0e})")
    ax.set_xlabel("step")
    ax.set_ylabel("training loss")
    ax.set_title(LABEL[s])
    ax.legend()
save(fig, "fig4_training_curves")

# ---------------- flat CSV of every run ----------------
keys = ["run", "setting", "opt", "lr", "seed", "wikitext2_val_loss", "tulu_val_loss",
        "fineweb_loss", *BENCH, "rel_change_matrices", "train_minutes", "diverged"]
with open(f"{TAB}/all_runs.csv", "w") as f:
    f.write(",".join(keys) + "\n")
    for r in sorted(runs, key=lambda r: (r["setting"], r["opt"], r["lr"], r["seed"])):
        f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

print(open(f"{TAB}/table1_best_lr.md").read())
print(open(f"{TAB}/table2_muon_minus_adamw.md").read())
print("Figures in", FIG)
