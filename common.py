"""Shared settings and helpers for the AdamW-vs-Muon fine-tuning study."""
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- paths
ROOT = os.environ.get("STUDY_ROOT", os.path.dirname(os.path.abspath(__file__)))  # default: the repo folder
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B")  # BASE model, not Instruct

# ---------------------------------------------------------------- experiment grid
LRS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
SETTINGS = ["narrow", "broad"]
OPTIMIZERS = ["adamw", "muon"]

# Per-setting training config (identical for both optimizers -> fairness rules).
TRAIN_CFG = {
    # WikiText-2 packed into 1024-token blocks; 2 epochs; 16 blocks per optimizer step.
    "narrow": dict(epochs=2, micro_bs=4, grad_accum=4, max_len=1024),
    # Tulu-3 subset; 2 epochs (as in Zhao et al.); 16 examples per optimizer step.
    "broad": dict(epochs=2, micro_bs=2, grad_accum=8, max_len=2048),
}
WARMUP_FRAC = 0.03
WEIGHT_DECAY = 0.0          # same for both optimizers
GRAD_CLIP = 1.0
ADAM_BETAS = (0.9, 0.999)
MUON_MOMENTUM = 0.95

N_TULU_TRAIN = int(os.environ.get("N_TULU_TRAIN", 20000))
N_TULU_VAL = 500
N_FINEWEB = 500             # held-out pretraining-style docs for forgetting
FINEWEB_LEN = 1024
N_HELLASWAG = 2000
N_MMLU = 2000


def run_name(setting, opt, lr, seed):
    return f"{setting}_{opt}_lr{lr:.0e}_s{seed}"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)  # atomic: a half-written file never looks finished


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast():
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)


# ---------------------------------------------------------------- batching
def pad_batch(seqs, labels=None, pad_id=0):
    """Right-pad a list of 1-D LongTensors. Labels are padded with -100."""
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    lab = torch.full((len(seqs), L), -100, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = s
        att[i, : len(s)] = 1
        lab[i, : len(s)] = s if labels is None else labels[i]
    return ids, att, lab


@torch.no_grad()
def token_loss(model, seqs, labels=None, bs=4, pad_id=0):
    """Token-weighted mean next-token loss over a dataset (labels -100 are ignored)."""
    model.eval()
    dev = device()
    tot, n = 0.0, 0
    for i in range(0, len(seqs), bs):
        lab_chunk = None if labels is None else labels[i : i + bs]
        ids, att, lab = pad_batch(seqs[i : i + bs], lab_chunk, pad_id)
        ids, att, lab = ids.to(dev), att.to(dev), lab.to(dev)
        with autocast():
            logits = model(input_ids=ids, attention_mask=att).logits
        logits = logits[:, :-1].float()
        tgt = lab[:, 1:]
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                               ignore_index=-100, reduction="sum").item()
        n += (tgt != -100).sum().item()
    return tot / max(n, 1)


@torch.no_grad()
def choice_logprobs(model, items, bs=16, pad_id=0):
    """items: list of (ctx_ids, cont_ids). Returns summed log-prob of each continuation."""
    model.eval()
    dev = device()
    out = []
    for i in range(0, len(items), bs):
        chunk = items[i : i + bs]
        seqs = [torch.tensor(c + k, dtype=torch.long) for c, k in chunk]
        ids, att, _ = pad_batch(seqs, pad_id=pad_id)
        ids, att = ids.to(dev), att.to(dev)
        with autocast():
            logits = model(input_ids=ids, attention_mask=att).logits
        for j, (c, k) in enumerate(chunk):
            # token at position p is predicted by logits at p-1; only those rows need softmax
            pos = torch.arange(len(c), len(c) + len(k), device=dev)
            tgt = torch.tensor(k, device=dev)
            rows = torch.log_softmax(logits[j, pos - 1].float(), dim=-1)
            out.append(rows[torch.arange(len(k), device=dev), tgt].sum().item())
        del logits
    return out


def lr_lambda(total_steps):
    warm = max(1, int(WARMUP_FRAC * total_steps))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    return f
