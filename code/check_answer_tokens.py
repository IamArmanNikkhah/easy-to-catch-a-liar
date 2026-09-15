#!/usr/bin/env python3
"""Validate the answer alphabet against the real tokenizer, on CPU, before any GPU time.

We compare the model's probability of one label against the other at the answer position.
That comparison is only well formed if each label is exactly one token under the exact
string convention the prompt ends with. Finding out otherwise inside a Colab run would
cost a session; finding out here costs a tokenizer download.

Downloads tokenizer files only (a few MB of public config), no weights, no API key.

  python3 check_answer_tokens.py --model Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from histories import LABEL_POOL  # noqa: E402

# The prompt ends with "Answer:" so a generated label arrives with a leading space.
# Both conventions are checked because the chat template may strip it.
CONVENTIONS = {"leading_space": " %s", "bare": "%s"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--labels", default=",".join(LABEL_POOL))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]

    failures = []
    for name, fmt in CONVENTIONS.items():
        ids_seen = {}
        for label in labels:
            ids = tok.encode(fmt % label, add_special_tokens=False)
            ok = len(ids) == 1
            print("%-14s %-3s -> %-18s %s" % (name, label, ids, "ok" if ok else "MULTI-TOKEN"))
            if not ok:
                failures.append((name, label, ids))
            else:
                if ids[0] in ids_seen:
                    failures.append((name, label, "collides with %s" % ids_seen[ids[0]]))
                ids_seen[ids[0]] = label

    if failures:
        print("\nFAIL: %d problem(s). Shrink LABEL_POOL in histories.py to the survivors."
              % len(failures))
        for f in failures:
            print("  ", f)
        return 1
    print("\nOK: every label is a single distinct token under both conventions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
