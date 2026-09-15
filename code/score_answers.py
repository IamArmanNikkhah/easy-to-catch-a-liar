#!/usr/bin/env python3
"""Score frozen prompts by comparing the two answer tokens. No training, no adapters, no
generation beyond one position.

The load pattern (fp16 straight to VRAM, device_map={"":0}, nothing cloned to CPU first)
is lifted from research/unlearning-membership/code/score_nll.py, which was written after
a 7B hit a cgroup OOM on a Colab VM. The loss function is not lifted: here we want the
last-position logits restricted to two token ids, which is a different measurement.

What gets kept, per the plan's instruction not to filter anything away silently: both
label probabilities, their renormalised split, and the top-5 tokens the model actually
wanted. If a model is trying to say something else entirely, that shows up in the top-5
rather than being quietly rounded into a forced binary choice.

  python3 score_answers.py --prompts data/prompts.jsonl --out data/runs/pilot.jsonl \
      --model Qwen/Qwen2.5-1.5B-Instruct --limit 32
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import time

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def label_token_ids(tok, labels, chat: bool):
    """One id per label under the convention the prompt actually ends with. Refuses rather
    than guessing: a multi-token label makes the two-way comparison ill-defined, and that
    is a CPU-time discovery (see check_answer_tokens.py), never a run-time surprise."""
    ids = []
    for lab in labels:
        # A chat turn starts fresh, so the assistant's first token is the bare label.
        # A raw completion follows "Answer:" and so carries a leading space.
        enc = tok.encode(lab if chat else " " + lab, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError("label %r is %d tokens under chat=%s" % (lab, len(enc), chat))
        ids.append(enc[0])
    if len(set(ids)) != len(ids):
        raise ValueError("labels collide on the same token: %r" % (labels,))
    return ids


def build_inputs(tok, prompts, chat: bool):
    if chat:
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True)
                 for p in prompts]
    else:
        texts = list(prompts)
    return tok(texts, return_tensors="pt", padding=True, add_special_tokens=not chat)


def done_ids(path: pathlib.Path):
    """Restart without duplicate rows. The file is append-only and one JSON object per
    line, so a run killed mid-write loses at most the final partial line."""
    if not path.exists():
        return set()
    out = set()
    with path.open() as fh:
        for line in fh:
            try:
                out.add(json.loads(line)["episode_id"])
            except Exception:
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all remaining")
    ap.add_argument("--raw", action="store_true", help="raw completion instead of chat")
    ap.add_argument("--seed", type=int, default=0)
    # "single" pins every weight to GPU 0, the pattern this file was born with and the one
    # every run through 14B used. "auto" shards across all visible GPUs and exists only
    # for the 72B, which at fp16 is 144GB and fits no single card we can reach. Added
    # 2026-09-10; nothing about the single-GPU path changed.
    ap.add_argument("--device-map", choices=("single", "auto"), default="single")
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    chat = not args.raw
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already = done_ids(out_path)

    records = []
    with open(args.prompts) as fh:
        for line in fh:
            r = json.loads(line)
            if r["episode_id"] not in already:
                records.append(r)
    if args.limit:
        records = records[:args.limit]
    print("%d to score (%d already done)" % (len(records), len(already)))

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                        padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision,
        dtype=torch.float16 if dev == "cuda" else torch.float32,
        device_map=(("auto" if args.device_map == "auto" else {"": 0})
                    if dev == "cuda" else None)).eval()

    meta = {"model": args.model, "revision": args.revision, "device": dev,
            "precision": "fp16" if dev == "cuda" else "fp32",
            "chat_template": chat, "batch_size": args.batch_size, "seed": args.seed,
            "transformers": transformers.__version__, "torch": torch.__version__,
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None}
    print(json.dumps(meta, indent=2))

    t0 = time.time()
    n = 0
    with out_path.open("a") as fh:
        for i in range(0, len(records), args.batch_size):
            batch = records[i:i + args.batch_size]
            enc = build_inputs(tok, [b["prompt"] for b in batch], chat)
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits[:, -1, :].float()
            logprobs = torch.log_softmax(logits, dim=-1)
            topv, topi = logprobs.topk(5, dim=-1)
            for j, rec in enumerate(batch):
                ids = label_token_ids(tok, rec["labels"], chat)
                lp = [logprobs[j, t].item() for t in ids]
                m = max(lp)
                ex = [pow(2.718281828459045, x - m) for x in lp]
                split = [e / sum(ex) for e in ex]
                fh.write(json.dumps({
                    "episode_id": rec["episode_id"],
                    "prompt_sha1": rec["prompt_sha1"],
                    "labels": rec["labels"],
                    "label_token_ids": ids,
                    "logprobs": lp,
                    "p_first_label": split[0],
                    "choice": rec["labels"][0] if lp[0] >= lp[1] else rec["labels"][1],
                    "top5": [{"token": tok.decode([int(t)]), "id": int(t),
                              "logprob": float(v)}
                             for t, v in zip(topi[j].tolist(), topv[j].tolist())],
                    "n_input_tokens": int(enc["attention_mask"][j].sum().item()),
                    "meta": meta if n == 0 else None,
                }) + "\n")
                fh.flush()
                n += 1
            if i == 0:
                print("first batch of %d in %.1fs" % (len(batch), time.time() - t0))

    dt = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / 1e9) if dev == "cuda" else 0.0
    print("scored %d in %.1fs (%.3f s/episode), peak GPU %.2f GB" % (n, dt, dt / max(n, 1), peak))
    print("BENCHMARK LINE: %.4f s/episode, %.2f GB peak, %s, batch %d"
          % (dt / max(n, 1), peak, meta["gpu"], args.batch_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
