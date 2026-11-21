"""Download the model and all datasets once, and save tokenized caches to DATA_DIR.

Run this ONCE before any training. It needs internet access (Hugging Face).
"""
import json
import os
import re

import torch

from common import (DATA_DIR, FINEWEB_LEN, MODEL_ID, N_FINEWEB, N_HELLASWAG, N_MMLU,
                    N_TULU_TRAIN, N_TULU_VAL, TRAIN_CFG)

IM_START, IM_END = "<|im_start|>", "<|im_end|>"


# ------------------------------------------------------------------ builders (pure functions)
def build_wikitext_blocks(texts, tok, block=1024):
    """Concatenate all text and cut it into fixed-length blocks (standard LM fine-tuning)."""
    eos = tok.eos_token_id
    ids = []
    for t in texts:
        if t.strip():
            ids.extend(tok(t, add_special_tokens=False)["input_ids"])
    ids.append(eos)
    n = len(ids) // block
    return [torch.tensor(ids[i * block:(i + 1) * block], dtype=torch.long) for i in range(n)]


def build_chat_example(messages, tok, max_len):
    """ChatML formatting; loss only on assistant replies (+ their end token)."""
    ids, labels = [], []
    for m in messages:
        role, content = m["role"], m["content"]
        head = tok(f"{IM_START}{role}\n", add_special_tokens=False)["input_ids"]
        body = tok(f"{content}{IM_END}\n", add_special_tokens=False)["input_ids"]
        ids += head + body
        labels += [-100] * len(head) + (body if role == "assistant" else [-100] * len(body))
    ids, labels = ids[:max_len], labels[:max_len]
    if all(l == -100 for l in labels):
        return None  # nothing to learn after truncation
    return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def hellaswag_preprocess(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def mc_question(ctx, choices, gold, tok):
    """One multiple-choice question -> token items for log-likelihood scoring."""
    c_ids = tok(ctx, add_special_tokens=False)["input_ids"]
    items = [(c_ids, tok(ch, add_special_tokens=False)["input_ids"]) for ch in choices]
    return {"items": items, "gold": gold, "char_len": [len(ch) for ch in choices]}


def build_arc(rows, tok):
    out = []
    for r in rows:
        ctx = f"Question: {r['question']}\nAnswer:"
        choices = [" " + t for t in r["choices"]["text"]]
        out.append(mc_question(ctx, choices, r["choices"]["label"].index(r["answerKey"]), tok))
    return out


def build_hellaswag(rows, tok):
    out = []
    for r in rows:
        ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
        query = hellaswag_preprocess(r["activity_label"] + ": " + ctx)
        choices = [" " + hellaswag_preprocess(e) for e in r["endings"]]
        out.append(mc_question(query, choices, int(r["label"]), tok))
    return out


def build_mmlu(rows, tok):
    out = []
    for r in rows:
        subj = r["subject"].replace("_", " ")
        opts = "".join(f"\n{l}. {c}" for l, c in zip("ABCD", r["choices"]))
        ctx = (f"The following are multiple choice questions (with answers) about {subj}.\n\n"
               f"{r['question'].strip()}{opts}\nAnswer:")
        out.append(mc_question(ctx, [" A", " B", " C", " D"], int(r["answer"]), tok))
    return out


# ------------------------------------------------------------------ main
def main():
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading model", MODEL_ID)
    snapshot_download(MODEL_ID)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    print("WikiText-2 ...")
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    blk = TRAIN_CFG["narrow"]["max_len"]
    torch.save({"train": build_wikitext_blocks(wt["train"]["text"], tok, blk),
                "val": build_wikitext_blocks(wt["validation"]["text"], tok, blk)},
               f"{DATA_DIR}/wikitext2.pt")

    print("Tulu-3 SFT mixture (downloads ~1-2 GB once) ...")
    tulu = load_dataset("allenai/tulu-3-sft-mixture", split="train").shuffle(seed=0)
    max_len = TRAIN_CFG["broad"]["max_len"]
    train, val, i = [], [], 0
    while len(train) < N_TULU_TRAIN or len(val) < N_TULU_VAL:
        ex = build_chat_example(tulu[i]["messages"], tok, max_len)
        i += 1
        if ex is None:
            continue
        (train if len(train) < N_TULU_TRAIN else val).append(ex)
    torch.save({"train": train, "val": val}, f"{DATA_DIR}/tulu.pt")
    print(f"  {len(train)} train / {len(val)} val, "
          f"{sum(len(x[0]) for x in train)/1e6:.1f}M train tokens per epoch")

    print("FineWeb-Edu held-out sample (streaming) ...")
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    docs = []
    for r in fw:
        ids = tok(r["text"], add_special_tokens=False)["input_ids"]
        if len(ids) >= 512:
            docs.append(torch.tensor(ids[:FINEWEB_LEN], dtype=torch.long))
        if len(docs) >= N_FINEWEB:
            break
    torch.save(docs, f"{DATA_DIR}/fineweb.pt")

    print("Benchmarks ...")
    bench = {}
    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    bench["arc_easy"] = build_arc(arc, tok)
    try:
        hs = load_dataset("Rowan/hellaswag", split="validation").shuffle(seed=0)
        bench["hellaswag"] = build_hellaswag(hs.select(range(N_HELLASWAG)), tok)
    except Exception as e:  # dataset format changes sometimes; the study works without it
        print("  HellaSwag skipped:", e)
    mm = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=0)
    bench["mmlu"] = build_mmlu(mm.select(range(N_MMLU)), tok)
    with open(f"{DATA_DIR}/bench.json", "w") as f:
        json.dump(bench, f)
    print("Done. Data in", DATA_DIR)


if __name__ == "__main__":
    main()
