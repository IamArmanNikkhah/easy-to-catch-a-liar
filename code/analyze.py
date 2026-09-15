#!/usr/bin/env python3
"""Join a run against the answer key and report condition-level results.

One thing here is a design decision rather than a detail: the resampling unit is the
FAMILY, not the episode. A family's sixteen or thirty-two episodes share a history, a
label assignment and a source; treating them as independent draws would shrink every
interval by a factor of four for free. Label swaps and re-queries of one history are
dependent variants, and the plan says so explicitly.

No significance test and no threshold. This reports values and intervals; whether they
mean anything is a separate conversation that has to happen before a confirmatory study,
not after this one.

  python3 analyze.py --answers data/answers.jsonl --run data/runs/pilot.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random


def load(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def bootstrap_ci(by_family, n=2000, seed=0, alpha=0.05):
    """Percentile bootstrap over families. Returns (point, lo, hi) or None if empty."""
    fams = list(by_family)
    if not fams:
        return None
    flat = [x for f in fams for x in by_family[f]]
    point = sum(flat) / len(flat)
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        pick = [by_family[rng.choice(fams)] for _ in fams]
        vals = [x for grp in pick for x in grp]
        draws.append(sum(vals) / len(vals))
    draws.sort()
    return point, draws[int(alpha / 2 * n)], draws[int((1 - alpha / 2) * n)]


def fmt(ci, n):
    if ci is None:
        return "   n/a"
    return "%.3f [%.3f, %.3f]  n=%d" % (ci[0], ci[1], ci[2], n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    key = {r["episode_id"]: r for r in load(args.answers)}
    run = load(args.run)
    joined = []
    for r in run:
        k = key.get(r["episode_id"])
        if k is None:
            continue
        joined.append((k, r))
    print("joined %d of %d scored rows against the key\n" % (len(joined), len(run)))
    if not joined:
        return 1

    # Answerable rows only. An ambiguous episode has no correct answer by construction,
    # so scoring it as right or wrong would be inventing a result; its own measurement is
    # how close to 0.5 the split sits, reported separately below.
    def group(pred, value):
        acc = collections.defaultdict(lambda: collections.defaultdict(list))
        for k, r in joined:
            if not pred(k, r):
                continue
            acc[k["condition"]][k["family_id"]].append(value(k, r))
        return acc

    print("== correct-answer rate, answerable episodes (bootstrap over families) ==")
    acc = group(lambda k, r: k["oracle_status"] == "definite",
                lambda k, r: 1.0 if r["choice"] == k["oracle_action"] else 0.0)
    for cond in sorted(acc):
        n = sum(len(v) for v in acc[cond].values())
        print("  %-34s %s" % (cond, fmt(bootstrap_ci(acc[cond], args.boot), n)))

    print("\n== probability mass on the correct answer ==")
    pm = group(lambda k, r: k["oracle_status"] == "definite",
               lambda k, r: (r["p_first_label"] if k["oracle_action"] == r["labels"][0]
                             else 1 - r["p_first_label"]))
    for cond in sorted(pm):
        n = sum(len(v) for v in pm[cond].values())
        print("  %-34s %s" % (cond, fmt(bootstrap_ci(pm[cond], args.boot), n)))

    print("\n== unanswerable episodes: distance of the split from 0.5 ==")
    print("   (a model reading only the evidence has nothing to go on here; large values"
          " mean something other than the evidence is driving the choice)")
    amb = group(lambda k, r: k["oracle_status"] == "ambiguous",
                lambda k, r: abs(r["p_first_label"] - 0.5))
    for cond in sorted(amb):
        n = sum(len(v) for v in amb[cond].values())
        print("  %-34s %s" % (cond, fmt(bootstrap_ci(amb[cond], args.boot), n)))

    print("\n== THE PRIMARY CONTRAST: specificity ==")
    print("   Both conditions have a clean transfer reporter. Source-specific learning"
          " predicts no gap; blanket distrust predicts the second collapses to zero.")
    disc = collections.defaultdict(lambda: collections.defaultdict(list))
    for k, r in joined:
        if not k.get("is_discriminating"):
            continue
        disc[k["discriminating_group"]][k["family_id"]].append(
            1.0 if r["choice"] == k["oracle_action"] else 0.0)
    for g in ("control", "other_source_caught"):
        n = sum(len(v) for v in disc[g].values())
        print("  %-34s %s" % (g, fmt(bootstrap_ci(disc[g], args.boot), n)))

    print("\n== rival accounts scored against what the model actually chose ==")
    tally = collections.Counter()
    total = 0
    for k, r in joined:
        pred = k.get("predicted")
        if not pred or not k.get("is_primary"):
            continue
        total += 1
        for account, a in pred.items():
            if r["choice"] == a:
                tally[account] += 1
    if total:
        for account in ("source_specific", "blanket", "latest"):
            print("  %-34s %.3f  (%d/%d)"
                  % (account, tally[account] / total, tally[account], total))
    else:
        print("  (no specificity-arm primary rows in this run)")

    print("\n== answer validity: was a label even the model's top token? ==")
    bad = [(k, r) for k, r in joined
           if r["top5"] and r["top5"][0]["id"] not in r["label_token_ids"]]
    print("  %d of %d rows had a non-label token in first place" % (len(bad), len(joined)))
    if bad:
        ex = collections.Counter(r["top5"][0]["token"] for _, r in bad).most_common(5)
        print("  most common: %r" % (ex,))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
