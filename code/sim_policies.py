#!/usr/bin/env python3
"""Fake run files from KNOWN policies, so the analysis can be checked before a GPU bill.

An estimator written before the run is only half the discipline; an estimator that has
never been fed data is a guess with a docstring. This emits answer files in the exact
shape score_answers.py writes, from policies whose correct output is known in advance, so
`analyze_paired.py` can be held to what it must report.

The policies are chosen to be the live rival explanations of the 2026-09-10 results:

  oracle        always right where an answer exists. The analysis must show no
                dissociation. If it reports one here, the estimator is broken.
  chance        uniform over the two labels. Everything near 0.5, no structure anywhere.
  flip          the heuristic the 72B actually runs on no-evidence items: reports flipped,
                so the source lied. On the winner arm, follow the report. The OLD leak
                control passes this at exactly 0.500; the new one must catch it at 1.00
                versus 0.00. This is the policy the whole §4 statistic exists for.
  knows_not_acts  right about the source, and still follows the report when asked to act.
                This is the hypothesis the run is being bought to test, so the analysis
                must report a large dissociation and a low P(winner right | source right).
                If it cannot see it here, a real one would be missed too.

  python3 sim_policies.py --key ../data/paired_s9001_n64_key.jsonl --policy flip \
      --out ../data/runs/simpaired_flip.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random

POLICIES = ("oracle", "chance", "flip", "knows_not_acts")


def other(labels, pick):
    return next(l for l in labels if l != pick)


def answer(k, labels, rng, policy):
    """One choice, from the key alone. Everything these policies read is visible in the
    prompt: the labels, whether the reports flipped, and which option the reports name."""
    flipped_label = other(labels, k["accurate_label"]) if k["question"] == "source" else None

    if policy == "chance":
        return rng.choice(labels)

    if policy == "oracle":
        return k["correct"] if k["correct"] in labels else rng.choice(labels)

    if k["question"] == "source":
        if policy in ("flip", "knows_not_acts"):
            if policy == "knows_not_acts" and k["correct"] in labels:
                return k["correct"]
            return flipped_label if k["reports_flipped"] else k["accurate_label"]

    # winner arm for flip and knows_not_acts: take the report at face value
    rw = k.get("reported_winner")
    if rw in labels:
        return rw
    return k["correct"] if k["correct"] in labels else rng.choice(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--prompts", help="only needed to read labels; defaults beside the key")
    ap.add_argument("--policy", required=True, choices=POLICIES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    keyp = pathlib.Path(args.key)
    promptp = pathlib.Path(args.prompts) if args.prompts else \
        keyp.with_name(keyp.name.replace("_key", ""))
    labels = {json.loads(l)["episode_id"]: json.loads(l)["labels"]
              for l in promptp.open()}

    rng = random.Random(args.seed)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as fh:
        for line in keyp.open():
            k = json.loads(line)
            lab = labels[k["episode_id"]]
            ch = answer(k, lab, rng, args.policy)
            fh.write(json.dumps({"episode_id": k["episode_id"], "labels": lab,
                                 "choice": ch, "p_first_label": 1.0 if ch == lab[0] else 0.0,
                                 "meta": None}) + "\n")
            n += 1
    print("%s: %d simulated answers -> %s" % (args.policy, n, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
