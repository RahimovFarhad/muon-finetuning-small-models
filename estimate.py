"""Use the smoke-test speed to project how long the full study will take."""
import glob
import json

import torch

from common import DATA_DIR, LRS, RESULTS_DIR, TRAIN_CFG

speed, evalm = {}, []
for f in glob.glob(f"{RESULTS_DIR}/*_smoke.json"):
    r = json.load(open(f))
    speed[r["setting"]] = r["tokens_per_sec"]
    evalm.append(r["eval_minutes"])
    print(f"{r['setting']:6s} {r['opt']:5s}: {r['tokens_per_sec']:,.0f} tok/s, "
          f"peak GPU memory {r['config']['peak_mem_gb']:.1f} GB, eval {r['eval_minutes']:.1f} min, "
          f"Muon params {r['muon_params']/1e6:.0f}M / AdamW params {r['adamw_params']/1e6:.0f}M")

wt = torch.load(f"{DATA_DIR}/wikitext2.pt")["train"]
tu = torch.load(f"{DATA_DIR}/tulu.pt")["train"]
tok = {"narrow": sum(len(x) for x in wt) * TRAIN_CFG["narrow"]["epochs"],
       "broad": sum(len(x[0]) for x in tu) * TRAIN_CFG["broad"]["epochs"]}
ev = max(evalm) if evalm else 3
total = 0
for s in ["narrow", "broad"]:
    if s not in speed:
        continue
    m = tok[s] / speed[s] / 60 + ev
    runs = 2 * len(LRS) + 4  # sweep + 2 extra seeds x 2 optimizers
    total += m * runs
    print(f"{s}: ~{m:.0f} min per run x {runs} runs = ~{m*runs/60:.1f} h")
print(f"TOTAL for the whole study: ~{total/60:.1f} hours (+ baseline ~{ev:.0f} min)")
