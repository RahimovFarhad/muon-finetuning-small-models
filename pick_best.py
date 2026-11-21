"""After the learning-rate sweep: pick each optimizer's best rate per setting (by the TARGET
metric only, as the SSOT fairness rules require) and write the seed runs to do next."""
import glob
import json

from common import RESULTS_DIR, ROOT

TARGET = {"narrow": "wikitext2_val_loss", "broad": "tulu_val_loss"}


def load_sweep():
    runs = []
    for f in glob.glob(f"{RESULTS_DIR}/*_s0.json"):
        r = json.load(open(f))
        if r.get("setting"):
            runs.append(r)
    return runs


def best_lrs(runs):
    best = {}
    for setting, metric in TARGET.items():
        for opt in ["adamw", "muon"]:
            cand = [r for r in runs if r["setting"] == setting and r["opt"] == opt
                    and not r.get("diverged") and r.get(metric) is not None]
            if not cand:
                continue
            b = min(cand, key=lambda r: r[metric])
            best[(setting, opt)] = b["lr"]
            tried = [r["lr"] for r in cand]
            edge = b["lr"] in (min(tried), max(tried))
            print(f"{setting:6s} {opt:5s}: best lr {b['lr']:.0e}  {metric}={b[metric]:.4f}"
                  f"  ({len(cand)} runs){'  <-- AT GRID EDGE: extend the grid!' if edge else ''}")
    return best


def main():
    best = best_lrs(load_sweep())
    lines = [f"{s} {o} {lr:.0e} {seed}" for (s, o), lr in best.items() for seed in (1, 2)]
    with open(f"{ROOT}/seeds_todo.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {len(lines)} seed runs to {ROOT}/seeds_todo.txt")


if __name__ == "__main__":
    main()
