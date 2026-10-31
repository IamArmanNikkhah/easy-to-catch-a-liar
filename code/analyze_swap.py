#!/usr/bin/env python3
"""Letter x round, crossed. Written 2026-09-13 while the swap run was on the box, before any
of its rows landed.

The original file ties the honest letter to family parity, and so does the audited round;
the swap file reverses the letter pairing. Read together, each (round, letter-index) cell has
128 calibrated source prompts per situation. Two rival explanations for the honest-cell
collapse make opposite predictions here:
  ROUND:  accuracy is low whenever the record names round 6, on both files.
  LETTER: accuracy is low whenever the honest answer is the second free letter, on both files.
The table prints both margins with paired intervals over worlds. A third possibility, that
both matter, shows up as both margins excluding zero.

LETTER here is the honest letter's position in the prompt (listed first or second), not its
pool order, which is what the paper's Table 5 uses (paper_numbers.py, crossed()); the two
definitions give different letter margins. The resampling unit is the world: a world and its
swapped copy share the history (unit fixed 2026-10-30; it was world/file before).

  python3 analyze_swap.py
"""
import json, collections, random

def load(p): return [json.loads(l) for l in open(p)]
FILES = {"orig": ("../data/paired_s9001_n64", {"qwen32b": "paired_32b", "qwen72b": "paired_72b", "llama70b": "paired_llama70b"}),
         "swap": ("../data/paired_s9001_n64_swap", {"qwen32b": "swap_qwen32b", "qwen72b": "swap_qwen72b", "llama70b": "swap_llama70b"})}
PLAIN = {"stable": "nothing moved, honest", "world": "points moved, honest",
         "reporter": "nothing moved, lying", "both": "points moved, lying"}

def gather(model):
    """-> list of (family, cell, variant, round, honest_is_second_letter, correct?)"""
    out = []
    for tag, (stem, runs) in FILES.items():
        key = {k["episode_id"]: k for k in load(stem + "_key.jsonl")}
        labels = {r["episode_id"]: r["labels"] for r in load(stem + ".jsonl")}
        try:
            got = {r["episode_id"]: r for r in load("../data/runs/%s.jsonl" % runs[model])}
        except FileNotFoundError:
            continue
        for i, k in key.items():
            if k["question"] != "source" or i not in got or k["correct"] is None: continue
            second = labels[i][1] == k["accurate_label"]
            out.append((k["family_id"], k["cell"], k["variant"], k["audit_round"], second, got[i]["choice"] == k["correct"]))
    return out

def boot(rows, n=3000, seed=0):
    by = collections.defaultdict(list)
    for fam, ok in rows: by[fam].append(ok)
    fams = sorted(by); rng = random.Random(seed)
    tot = sum(len(v) for v in by.values()); pt = sum(sum(v) for v in by.values()) / tot
    d = sorted(sum(sum(by[f]) for f in p) / sum(len(by[f]) for f in p) for p in ([rng.choice(fams) for _ in fams] for _ in range(n)))
    return pt, d[int(.025 * n)], d[int(.975 * n)], tot

def margin(rows, n=3000, seed=0):
    """paired difference (flag False minus flag True), resampling worlds once for both sides"""
    A = collections.defaultdict(list); B = collections.defaultdict(list)
    for fam, flag, ok in rows: (B if flag else A)[fam].append(ok)
    fams = sorted(set(A) | set(B)); rng = random.Random(seed)
    def acc(D, p):
        x = [b for f in p for b in D.get(f, [])]; return sum(x) / len(x)
    pt = acc(A, fams) - acc(B, fams)
    d = sorted(acc(A, p) - acc(B, p) for p in ([rng.choice(fams) for _ in fams] for _ in range(n)))
    return pt, d[int(.025 * n)], d[int(.975 * n)]

def main():
    for model in ("qwen72b", "llama70b", "qwen32b"):
        rows = gather(model)
        if not rows:
            print(model, ": no rows yet"); continue
        print("=== %s  (%d source rows across both files)" % (model, len(rows)))
        for v in ("floor", "cal"):
            print("  %s" % v)
            for cell in ("stable", "world", "reporter", "both"):
                sel = [(f, r, s, ok) for f, c, vv, r, s, ok in rows if c == cell and vv == v]
                line = "    %-22s" % PLAIN[cell]
                for r in (5, 6):
                    for s in (False, True):
                        t = boot([(f, ok) for f, rr, ss, ok in sel if rr == r and ss == s])
                        line += "  r%d/%s %.3f(n=%d)" % (r, "2nd" if s else "1st", t[0], t[3])
                round_m = margin([(f, r == 6, ok) for f, r, s, ok in sel])
                letter_m = margin([(f, s, ok) for f, r, s, ok in sel])
                print(line)
                print("    %-22s  ROUND margin (r5 - r6) %+.3f [%+.3f,%+.3f]   LETTER margin (1st - 2nd) %+.3f [%+.3f,%+.3f]"
                      % ("", *round_m, *letter_m))
        print()

if __name__ == "__main__":
    main()
