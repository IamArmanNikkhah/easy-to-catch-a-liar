#!/usr/bin/env python3
"""Estimators for the paired measure. WRITTEN BEFORE THE RUN, ON PURPOSE.

This corridor's own rule forbids inventing a threshold after seeing the numbers. An
estimator is the same thing: choosing how to slice after the data lands is how 0.770 got
reported as a result when it was the average of a perfect cell and a below-chance one.
So every number this file prints was decided while the answer was still unknown.

WHAT IT REPORTS, and why each one is here:

1. PER-CELL, BESIDE ITS OWN FLOOR. The aggregate is banned as a headline. On 2026-09-10 the
   72B's 0.770 hid `reporter` 1.000 and `world` 0.422 — perfect and worse-than-guessing,
   averaged. `world` is the cell that IS the research question (the payouts moved and the
   source is honest), so it never gets folded into a mean again.

2. DIFFERENCE IN DIFFERENCES against the byte-matched uncalibrated item. The two arms use
   different answer alphabets (the source arm is concentrated on K/M/Q/R while the winner
   arm draws the whole pool), so their raw accuracies are not on one scale. Differencing
   each arm against its own no-evidence baseline removes the token prior, the positional
   bias, and any constant-label policy. This is the primary estimator.

3. THE ITEM-CONDITIONAL STATISTIC, P(winner correct | source correct) on the same prompt.
   This, and only this, licenses "it states the channel is corrupted and still acts on the
   corrupted reading". A marginal gap does not: in `world` and `both` a wrong winner answer
   can come from a wrong state rather than from failing to apply a right one.

4. THE FLIP-HEURISTIC STATISTIC on the no-evidence items. The existing leak control reports
   how often the model says "accurate" and wants ~0.5. The 72B passes it at exactly 0.500
   while running a perfectly deterministic "the reports flipped, so the source lied" policy,
   because the design pairs one honest cell with one flipped cell and the heuristic averages
   out. The honest control is the gap between the two conditions, not the marginal.

5. THE ROUND MODERATOR. Which post-change round the verified record names moves accuracy by
   +0.13 to +0.26, and restatement does not remove it. Any claim has to survive at both
   levels or it is a claim about round 5.

6. THE STOPPING RULE. Inferring a source's state cannot beat being told it. If a calibrated
   arm outscores its own ceiling, something is leaking and the run is void.

  python3 analyze_paired.py --key ../data/paired_s9001_n64_key.jsonl --run ../data/runs/<f>
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random

CELLS = ("stable", "world", "reporter", "both")
# What the true state is in each cell, and whether the reports visibly flip. The flip
# heuristic ("they flipped, so it lied") is right in stable and reporter, wrong in the other
# two, which is exactly why it scores 0.500 over a balanced design and hides.
TRUE_STATE = {"stable": "accurate", "world": "accurate",
              "reporter": "inverted", "both": "inverted"}


def load(path):
    with pathlib.Path(path).open() as fh:
        return [json.loads(l) for l in fh]


def boot(units, n=4000, seed=0):
    """Bootstrap over FAMILIES, never items. Four surface forms and both question kinds
    share a family, so resampling items would treat one history as up to eight."""
    by = collections.defaultdict(list)
    for fam, ok in units:
        by[fam].append(ok)
    fams = list(by)
    if not fams:
        return float("nan"), float("nan"), float("nan"), 0
    rng = random.Random(seed)
    tot = sum(len(v) for v in by.values())
    pt = sum(sum(v) for v in by.values()) / tot
    d = []
    for _ in range(n):
        pick = [by[rng.choice(fams)] for _ in fams]
        d.append(sum(sum(v) for v in pick) / sum(len(v) for v in pick))
    d.sort()
    return pt, d[int(0.025 * n)], d[int(0.975 * n)], tot


def fmt(t):
    pt, lo, hi, n = t
    return "%.3f [%.3f, %.3f] n=%d" % (pt, lo, hi, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--label", default="model")
    args = ap.parse_args()

    key = {k["episode_id"]: k for k in load(args.key)}
    got = {r["episode_id"]: r for r in load(args.run)}
    rows = [(key[i], got[i]) for i in key if i in got]
    print("%s: %d of %d items scored\n" % (args.label, len(rows), len(key)))

    def cut(pred, correct_only=True):
        u = []
        for k, r in rows:
            if not pred(k):
                continue
            if correct_only and k["correct"] is None:
                continue
            u.append((k["family_id"], r["choice"] == k["correct"]))
        return boot(u)

    # ---- 1. per cell, beside its own floor -------------------------------------------
    print("ACCURACY BY CELL, each variant beside the floor for the same question.")
    print("A ceiling is only readable if its floor cleared; `world` is the research question.")
    for q in ("source", "winner"):
        print("\n  %s question" % q.upper())
        for v in ("floor", "ceiling", "cal", "cal_prior"):
            line = "    %-8s" % v
            for c in CELLS:
                t = cut(lambda k, q=q, v=v, c=c: k["question"] == q and k["variant"] == v
                        and k["cell"] == c)
                line += "  %s %.3f" % (c[:4], t[0])
            print(line + "   | all %s" % fmt(cut(
                lambda k, q=q, v=v: k["question"] == q and k["variant"] == v)))

    # ---- 6. the stopping rule (checked early; a void run should not be read further) --
    print("\nSTOPPING RULE: a calibrated arm may not outscore its own ceiling.")
    void = False
    for q in ("source", "winner"):
        cal = cut(lambda k, q=q: k["question"] == q and k["variant"] == "cal")
        ceil = cut(lambda k, q=q: k["question"] == q and k["variant"] == "ceiling")
        bad = cal[0] > ceil[0]
        void = void or bad
        print("  %-6s cal %.3f vs ceiling %.3f  %s"
              % (q, cal[0], ceil[0], "*** VOID, something leaks ***" if bad else "ok"))
    if void:
        print("\n  Run is void. Do not read anything below.")

    # ---- 2. difference in differences -------------------------------------------------
    print("\nDIFFERENCE IN DIFFERENCES, each arm against its own no-evidence baseline.")
    print("Removes the answer-alphabet difference between the arms. PRIMARY ESTIMATOR.")
    deltas = {}
    for q in ("source", "winner"):
        cal = cut(lambda k, q=q: k["question"] == q and k["variant"] == "cal")
        # nocal has no correct answer, so the baseline is agreement with the truth the
        # matched item cannot support: how often it lands on the answer that WOULD be right.
        u = [(k["family_id"], r["choice"] == (k["accurate_label"] if q == "source"
                                              else k["correct"]))
             for k, r in rows if k["question"] == q and k["variant"] == "nocal"
             and (q == "source" or k["correct"] is not None)]
        base = boot(u) if u else (0.5, 0.5, 0.5, 0)
        deltas[q] = cal[0] - base[0]
        print("  %-6s cal %.3f  nocal-baseline %.3f  delta %+.3f"
              % (q, cal[0], base[0], deltas[q]))
    print("  DISSOCIATION (source delta minus winner delta): %+.3f"
          % (deltas["source"] - deltas["winner"]))

    # ---- 3. the item-conditional statistic --------------------------------------------
    print("\nITEM-CONDITIONAL, on byte-matched pairs only. This is the claim-bearing number.")
    arms = collections.defaultdict(dict)
    for k, r in rows:
        if k["variant"] == "cal" and k["prefix_paired"]:
            arms[k["pair_id"]][k["question"]] = (k, r)
    both = [v for v in arms.values() if len(v) == 2]
    for c in CELLS + ("ALL",):
        sel = [v for v in both if c == "ALL" or v["source"][0]["cell"] == c]
        sc = [v for v in sel if v["source"][1]["choice"] == v["source"][0]["correct"]]
        if not sc:
            continue
        u = [(v["winner"][0]["family_id"],
              v["winner"][1]["choice"] == v["winner"][0]["correct"]) for v in sc]
        print("  P(winner right | source right)  %-7s %s" % (c, fmt(boot(u))))

    # ---- 4. the flip heuristic --------------------------------------------------------
    print("\nNO-EVIDENCE ITEMS: is the model reading evidence it does not have?")
    print("The marginal is the old control and it passes a deterministic heuristic at 0.500.")
    for flip in (True, False):
        u = [(k["family_id"], r["choice"] != k["accurate_label"])
             for k, r in rows if k["question"] == "source" and k["variant"] == "nocal"
             and k["reports_flipped"] is flip]
        print("  says FLIPPED when reports %s flip: %s"
              % ("DO" if flip else "do NOT", fmt(boot(u))))
    marg = [(k["family_id"], k["accurate_label"] == r["choice"])
            for k, r in rows if k["question"] == "source" and k["variant"] == "nocal"]
    print("  (old marginal, says 'accurate': %s)" % fmt(boot(marg)))

    # ---- 4b. what the calibrated answers are actually tracking -------------------------
    # Registered rule 3 says that if the no-evidence items separate by more than 0.25, every
    # calibrated number gets reported against the heuristic rather than against 0.5. They do
    # separate at 32B, so these two lines are not optional.
    print("\nWHAT THE CALIBRATED ANSWERS TRACK (required when the no-evidence items separate).")
    u = []
    for k, r in rows:
        if k["question"] != "source" or k["variant"] != "cal":
            continue
        flip_label = next(x for x in r["labels"] if x != k["accurate_label"])
        heur = flip_label if k["reports_flipped"] else k["accurate_label"]
        u.append((k["family_id"], r["choice"] == heur))
    print("  source cal agrees with the flip heuristic: %s" % fmt(boot(u)))
    print("  (a pure heuristic would score 0.500 accuracy; agreement near 0.5 means the")
    print("   calibrated answers are not the heuristic wearing a better score)")
    for v in ("ceiling", "cal", "cal_prior"):
        line = "  winner %-9s picks the REPORTED winner:" % v
        for c in CELLS + ("ALL",):
            u = [(k["family_id"], r["choice"] == k["reported_winner"])
                 for k, r in rows if k["question"] == "winner" and k["variant"] == v
                 and k.get("reported_winner") and (c == "ALL" or k["cell"] == c)]
            if u:
                line += "  %s %.3f" % (c[:4], boot(u)[0])
        print(line)

    # ---- 5. the round moderator -------------------------------------------------------
    print("\nROUND MODERATOR: which post-change round the verified record names.")
    for q in ("source", "winner"):
        line = "  %-6s" % q
        for rnd in (5, 6):
            t = cut(lambda k, q=q, rnd=rnd: k["question"] == q and k["variant"] == "cal"
                    and k["audit_round"] == rnd)
            line += "  round %d %.3f" % (rnd, t[0])
        print(line)
    return 1 if void else 0


if __name__ == "__main__":
    raise SystemExit(main())
