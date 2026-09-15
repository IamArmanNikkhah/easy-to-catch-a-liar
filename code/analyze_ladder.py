#!/usr/bin/env python3
"""Ceiling/floor curve across a model-size ladder, on the four-form dev item set.

Every model here is scored on byte-identical prompts (dev_forms.py at DEV_SEED 9001), so a
change down a column is a change in the model and nothing else. That was checked by diffing
regenerated prompt_sha1 against the already-scored runs, not assumed.

Two things this reports that a bare accuracy table would hide:

  THE GUARD. A wiring fault and a genuine null both land on 0.5, and the fault is invisible
  precisely because it produces the result you predicted. The floor question prints its own
  answer in the prompt, so floor-at-chance is not a finding, it is a broken harness. Every
  ceiling number below is reported next to its floor for exactly this reason.

  BOOTSTRAP OVER FAMILIES, not items. Four forms and two question kinds are drawn from one
  family, so they share a history and labels. Resampling items would treat those as
  independent and quote an interval far too tight.

  python3 analyze_ladder.py
"""
from __future__ import annotations

import collections
import json
import pathlib
import random

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data"
KEY = DATA / "devforms_key_s9001.jsonl"

# (label, run file). The 7B and 14B A100 rows are the runs that already existed; the L4 rows
# are this ladder. Keeping both 7B entries is the cross-hardware bridge: identical bytes,
# identical library versions, different GPU.
LADDER = [
    ("0.5B", "ladder_0.5b.jsonl", "L4"),
    ("1.5B", "ladder_1.5b.jsonl", "L4"),
    ("3B", "ladder_3b.jsonl", "L4"),
    ("7B", "ladder_7b.jsonl", "L4"),
    ("7B", "rw_7b.jsonl", "A100"),
    ("14B", "rw_14b.jsonl", "A100"),
    ("32B", "big_forms_32b.jsonl", "2xA100-80"),
    ("72B", "big_forms_72b.jsonl", "2xA100-80"),
]
FORMS = ("log", "minutes", "prose", "commentary")


def load_key():
    k = {}
    with KEY.open() as fh:
        for line in fh:
            r = json.loads(line)
            k[r["episode_id"]] = r
    return k


def load_run(path):
    out = {}
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            out[r["episode_id"]] = r
    return out


def boot_ci(units, n=4000, seed=0):
    """units: list of (family_id, correct_bool). Resample FAMILIES with replacement."""
    by_fam = collections.defaultdict(list)
    for fam, ok in units:
        by_fam[fam].append(ok)
    fams = list(by_fam)
    if not fams:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    point = sum(sum(v) for v in by_fam.values()) / sum(len(v) for v in by_fam.values())
    draws = []
    for _ in range(n):
        pick = [by_fam[rng.choice(fams)] for _ in fams]
        num = sum(sum(v) for v in pick)
        den = sum(len(v) for v in pick)
        draws.append(num / den)
    draws.sort()
    return point, draws[int(0.025 * n)], draws[int(0.975 * n)]


def main():
    key = load_key()
    print("item set: %s  (%d items)" % (KEY.name, len(key)))
    print()

    rows = []
    for label, fname, gpu in LADDER:
        path = DATA / "runs" / fname
        if not path.exists():
            print("-- %-5s %-6s MISSING (%s)" % (label, gpu, fname))
            continue
        run = load_run(path)
        shared = set(run) & set(key)
        if len(shared) != len(key):
            print("-- %-5s %-6s PARTIAL: %d/%d items" % (label, gpu, len(shared), len(key)))
        # Format sanity: did the model try to say something that is not a label?
        bad = sum(1 for e in shared
                  if run[e]["top5"] and run[e]["top5"][0]["token"].strip()
                  not in [l.strip() for l in run[e]["labels"]])
        stat = {}
        for kind in ("floor", "ceiling"):
            units = [(key[e]["family_id"], run[e]["choice"] == key[e]["correct"])
                     for e in shared if key[e]["kind"] == kind]
            stat[kind] = boot_ci(units)
            for form in FORMS:
                u = [(key[e]["family_id"], run[e]["choice"] == key[e]["correct"])
                     for e in shared if key[e]["kind"] == kind and key[e]["form"] == form]
                stat[(kind, form)] = sum(ok for _, ok in u) / max(len(u), 1)
        # Agreement gate: all four forms give the same answer for one underlying item.
        groups = collections.defaultdict(dict)
        for e in shared:
            groups[key[e]["variant_group"]][key[e]["form"]] = (
                run[e]["choice"], key[e]["correct"])
        agree = {}
        for kind in ("floor", "ceiling"):
            gs = [g for gname, g in groups.items() if gname.endswith("-" + kind)
                  and len(g) == 4]
            unanimous = [g for g in gs if len({v[0] for v in g.values()}) == 1]
            right = [g for g in unanimous if list(g.values())[0][0] == list(g.values())[0][1]]
            agree[kind] = (len(unanimous), len(gs), len(right))
        rows.append((label, gpu, stat, agree, bad, len(shared)))

    hdr = "%-6s %-5s | %-22s %-22s | %-10s %-10s | %s"
    print(hdr % ("model", "gpu", "FLOOR (answer printed)", "CEILING (cause stated)",
                 "floor agr", "ceil agr", "bad tok"))
    print("-" * 118)
    for label, gpu, stat, agree, bad, n in rows:
        f = stat["floor"]
        c = stat["ceiling"]
        fa, fn, fr = agree["floor"]
        ca, cn, cr = agree["ceiling"]
        print(hdr % (
            label, gpu,
            "%.3f [%.3f, %.3f]" % f,
            "%.3f [%.3f, %.3f]" % c,
            "%2d/%2d %s" % (fr, fa, ("%d%%" % round(100 * fr / fa)) if fa else "-"),
            "%2d/%2d %s" % (cr, ca, ("%d%%" % round(100 * cr / ca)) if ca else "-"),
            "%d/%d" % (bad, n)))

    print()
    print("per-form accuracy")
    print("%-6s %-5s | %-30s | %s" % ("model", "gpu",
                                      "FLOOR  " + "  ".join("%-9s" % f for f in FORMS),
                                      "CEILING  " + "  ".join("%-9s" % f for f in FORMS)))
    for label, gpu, stat, _a, _b, _n in rows:
        fl = "  ".join("%-9.3f" % stat[("floor", f)] for f in FORMS)
        ce = "  ".join("%-9.3f" % stat[("ceiling", f)] for f in FORMS)
        print("%-6s %-5s |        %s |          %s" % (label, gpu, fl, ce))

    print()
    print("GUARD: a model at chance on the FLOOR is a broken harness, not a null. The floor")
    print("answer is printed in the prompt. Only read a ceiling number whose floor cleared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
