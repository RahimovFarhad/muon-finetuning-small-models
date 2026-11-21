"""Fine-tune the base model with AdamW or Muon in one setting, then evaluate.

Examples
  python train.py --baseline                                   # evaluate the untouched model
  python train.py --setting narrow --opt muon --lr 1e-5 --seed 0
  python train.py --setting broad --opt adamw --lr 1e-5 --max-steps 30 --tag smoke
"""
import argparse
import json
import math
import os
import time

import torch

from common import (ADAM_BETAS, DATA_DIR, GRAD_CLIP, MODEL_ID, MUON_MOMENTUM, RESULTS_DIR,
                    TRAIN_CFG, WEIGHT_DECAY, autocast, choice_logprobs, device, lr_lambda,
                    pad_batch, run_name, save_json, set_seed, token_loss)


# ------------------------------------------------------------------ optimizer
def is_muon_param(name, p):
    """Muon only for 2-D weight matrices inside the transformer blocks (attention + MLP).
    Embeddings / (tied) output head, norms and biases stay on AdamW, as in both papers."""
    return p.ndim == 2 and ".layers." in f".{name}" and "embed" not in name


def build_optimizers(model, opt, lr):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if opt == "adamw":
        groups = {"adamw": [p for _, p in named], "muon": []}
    else:
        groups = {"muon": [p for n, p in named if is_muon_param(n, p)],
                  "adamw": [p for n, p in named if not is_muon_param(n, p)]}
    opts = [torch.optim.AdamW(groups["adamw"], lr=lr, betas=ADAM_BETAS, eps=1e-8,
                              weight_decay=WEIGHT_DECAY)]
    if groups["muon"]:
        # match_rms_adamw: scales Muon's update to AdamW's typical size (Moonshot,
        # "Muon is Scalable"), so both optimizers can share one learning-rate grid.
        opts.append(torch.optim.Muon(groups["muon"], lr=lr, momentum=MUON_MOMENTUM,
                                     nesterov=True, weight_decay=WEIGHT_DECAY,
                                     adjust_lr_fn="match_rms_adamw"))
    n_muon = sum(p.numel() for p in groups["muon"])
    n_adam = sum(p.numel() for p in groups["adamw"])
    return opts, {"muon_params": n_muon, "adamw_params": n_adam}


# ------------------------------------------------------------------ data
def load_split(setting, split):
    if setting == "narrow":
        d = torch.load(f"{DATA_DIR}/wikitext2.pt")
        return d[split], None
    d = torch.load(f"{DATA_DIR}/tulu.pt")
    return [x[0] for x in d[split]], [x[1] for x in d[split]]


# ------------------------------------------------------------------ evaluation
def eval_bench(model, pad_id):
    path = f"{DATA_DIR}/bench.json"
    if not os.path.exists(path):
        return {}
    bench = json.load(open(path))
    res = {}
    for name, qs in bench.items():
        items = [tuple(it) for q in qs for it in q["items"]]
        lps = choice_logprobs(model, items, pad_id=pad_id)
        k, acc, accn = 0, 0, 0
        for q in qs:
            n = len(q["items"])
            s = lps[k:k + n]
            k += n
            acc += int(max(range(n), key=lambda j: s[j]) == q["gold"])
            accn += int(max(range(n), key=lambda j: s[j] / max(1, q["char_len"][j])) == q["gold"])
        key = "acc_norm" if name == "hellaswag" else "acc"
        res[name] = (accn if key == "acc_norm" else acc) / len(qs)
        res[name + "_metric"] = key
        res[name + "_n"] = len(qs)
    return res


def evaluate(model, pad_id, which=("narrow", "broad")):
    out = {}
    if "narrow" in which:
        val, _ = load_split("narrow", "val")
        out["wikitext2_val_loss"] = token_loss(model, val, pad_id=pad_id)
        out["wikitext2_val_ppl"] = math.exp(out["wikitext2_val_loss"])
    if "broad" in which:
        val, lab = load_split("broad", "val")
        out["tulu_val_loss"] = token_loss(model, val, lab, pad_id=pad_id)
    fw = torch.load(f"{DATA_DIR}/fineweb.pt")
    out["fineweb_loss"] = token_loss(model, fw, pad_id=pad_id)  # forgetting probe
    out.update(eval_bench(model, pad_id))
    return out


@torch.no_grad()
def weight_change(model, ref):
    """Relative L2 distance from the pretrained weights (all params / Muon-eligible only)."""
    num_all = den_all = num_m = den_m = 0.0
    for n, p in model.named_parameters():
        d = (p.detach().float().cpu() - ref[n]).pow(2).sum().item()
        r = ref[n].pow(2).sum().item()
        num_all += d
        den_all += r
        if is_muon_param(n, p):
            num_m += d
            den_m += r
    return {"rel_change_all": math.sqrt(num_all / den_all),
            "rel_change_matrices": math.sqrt(num_m / max(den_m, 1e-12))}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["narrow", "broad"])
    ap.add_argument("--opt", choices=["adamw", "muon"])
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", action="store_true", help="evaluate the pretrained model only")
    ap.add_argument("--max-steps", type=int, default=None, help="for smoke tests")
    ap.add_argument("--tag", default="", help="suffix for the result file (e.g. smoke)")
    ap.add_argument("--grad-ckpt", action="store_true", help="save memory (needed on <24 GB GPUs)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = device()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(dev)

    if args.baseline:
        res = {"run": "baseline", "model": MODEL_ID, **evaluate(model, pad_id)}
        save_json(res, f"{RESULTS_DIR}/baseline.json")
        print(json.dumps(res, indent=2))
        return

    assert args.setting and args.opt and args.lr, "need --setting, --opt and --lr"
    name = run_name(args.setting, args.opt, args.lr, args.seed) + (f"_{args.tag}" if args.tag else "")
    out_path = f"{RESULTS_DIR}/{name}.json"
    if os.path.exists(out_path):
        print("exists, skipping:", out_path)
        return

    set_seed(args.seed)
    cfg = TRAIN_CFG[args.setting]
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    ref = {n: p.detach().float().cpu().clone() for n, p in model.named_parameters()}

    seqs, labels = load_split(args.setting, "train")
    n = len(seqs)
    eff_bs = cfg["micro_bs"] * cfg["grad_accum"]
    steps_per_epoch = n // eff_bs
    total = cfg["epochs"] * steps_per_epoch
    if args.max_steps:
        total = min(total, args.max_steps)

    opts, pinfo = build_optimizers(model, args.opt, args.lr)
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lr_lambda(total)) for o in opts]
    params = [p for p in model.parameters() if p.requires_grad]
    g = torch.Generator().manual_seed(args.seed)  # same data order for both optimizers
    order = []
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(n, generator=g).tolist()
        if labels is not None:
            # Broad setting: group similar lengths into the same micro-batch (less padding,
            # faster), then shuffle those micro-batches. Same order for both optimizers.
            mb = cfg["micro_bs"]
            batches = []
            for c in range(0, n, 256):
                chunk = sorted(perm[c:c + 256], key=lambda i: len(seqs[i]))
                batches += [chunk[j:j + mb] for j in range(0, len(chunk), mb)]
            shuffle = torch.randperm(len(batches), generator=g).tolist()
            perm = [i for b in shuffle for i in batches[b]]
        order += perm

    log, t0, tokens, step, pos = [], time.time(), 0, 0, 0
    model.train()
    while step < total:
        loss_acc = 0.0
        for _ in range(cfg["grad_accum"]):
            idx = order[pos:pos + cfg["micro_bs"]]
            pos += cfg["micro_bs"]
            lab = None if labels is None else [labels[i] for i in idx]
            ids, att, tgt = pad_batch([seqs[i] for i in idx], lab, pad_id)
            ids, att, tgt = ids.to(dev), att.to(dev), tgt.to(dev)
            with autocast():
                loss = model(input_ids=ids, attention_mask=att, labels=tgt).loss
            (loss / cfg["grad_accum"]).backward()
            loss_acc += loss.item() / cfg["grad_accum"]
            tokens += int(att.sum())
        gnorm = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP).item()
        for o in opts:
            o.step()
        for s in scheds:
            s.step()
        for o in opts:
            o.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0 or step == total:
            el = time.time() - t0
            log.append({"step": step, "loss": loss_acc, "grad_norm": gnorm,
                        "lr": scheds[0].get_last_lr()[0]})
            print(f"[{name}] step {step}/{total} loss {loss_acc:.4f} gnorm {gnorm:.2f} "
                  f"{tokens/el:,.0f} tok/s  eta {(total-step)*el/step/60:.1f} min", flush=True)
            if not math.isfinite(loss_acc):
                print("loss is not finite, stopping run early")
                break

    train_time = time.time() - t0
    peak_train_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    del opts, scheds          # free optimizer state before evaluation
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    t1 = time.time()
    metrics = evaluate(model, pad_id)
    res = {"run": name, "setting": args.setting, "opt": args.opt, "lr": args.lr,
           "seed": args.seed, "steps": step, "total_steps": total, "train_tokens": tokens,
           "train_minutes": train_time / 60, "eval_minutes": (time.time() - t1) / 60,
           "tokens_per_sec": tokens / train_time, "final_train_loss": log[-1]["loss"] if log else None,
           "diverged": not all(math.isfinite(x["loss"]) for x in log),
           **pinfo, **weight_change(model, ref), **metrics, "train_log": log,
           "config": {**cfg, "warmup_frac": 0.03, "weight_decay": 0.0, "grad_clip": GRAD_CLIP,
                      "adam_betas": list(ADAM_BETAS), "muon_momentum": MUON_MOMENTUM,
                      "model": MODEL_ID, "torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                      "peak_mem_gb": peak_train_gb}}
    save_json(res, out_path)
    print(f"saved {out_path}  (train {train_time/60:.1f} min, eval {res['eval_minutes']:.1f} min)")


if __name__ == "__main__":
    main()
