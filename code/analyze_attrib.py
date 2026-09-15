#!/usr/bin/env python3
"""Score the attribution question (dev_attrib.py items) across a model ladder.

Three kinds of item and each is read differently:
  floor  the cause is printed in the prompt. Accuracy here is a reading check.
  cal    one verified record settles the source's state. Accuracy here IS the measurement:
         can the model compare a verified result to a report on the same round.
  nocal  no correct answer exists (the oracle is ambiguous by construction). Two things
         are reported instead: how often the model says "accurate", which should sit near
         0.5 with no systematic lean, and whether matched pairs get the SAME answer, which
         they must, because they are the same bytes.

The decomposition that mattered on the winner question is repeated here: cal split by the
TRUE source state. On the winner question the told-flipping half collapsed while the
told-honest half soared, because "follow the report" was right in one and wrong in the
other. Here there is no report to follow for the answer, so if the split stays balanced
the redesign has escaped that anchor; if it does not, it has not.

  python3 analyze_attrib.py
"""
from __future__ import annotations

import collections
import json
import pathlib
import random

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data"
KEY = DATA / "devattrib_key_s9001.jsonl"
LADDER = [("3B", "attrib_3b.jsonl"), ("7B", "attrib_7b.jsonl"), ("14B", "attrib_14b.jsonl"),
          ("32B", "big_attrib_32b.jsonl"), ("72B", "big_attrib_72b.jsonl")]
# The one-line diagnostic: same cal items with the audited round's report restated beside
# the verified result. Scored separately; merged by label below.
RESTATED = {"32B": "diag_restated_32b.jsonl", "72B": "diag_restated_72b.jsonl"}
FORMS = ("log", "minutes", "prose", "commentary")


def load(path):
    out = {}
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            out[r["episode_id"]] = r
    return out


def boot(units, n=4000, seed=0):
    by = collections.defaultdict(list)
    for fam, ok in units:
        by[fam].append(ok)
    fams = list(by)
    if not fams:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    pt = sum(sum(v) for v in by.values()) / sum(len(v) for v in by.values())
    d = []
    for _ in range(n):
        pick = [by[rng.choice(fams)] for _ in fams]
        d.append(sum(sum(v) for v in pick) / sum(len(v) for v in pick))
    d.sort()
    return pt, d[int(0.025 * n)], d[int(0.975 * n)]


def fmt(t):
    return "%.3f [%.3f, %.3f]" % t


def main():
    key = load(KEY)
    print("item set: %s (%d items)\n" % (KEY.name, len(key)))
    for label, fname in LADDER:
        path = DATA / "runs" / fname
        if not path.exists():
            print("== %s  MISSING\n" % label)
            continue
        run = load(path)
        rp = DATA / "runs" / RESTATED.get(label, "")
        if label in RESTATED and rp.exists():
            run.update(load(rp))
        sh = [e for e in key if e in run]
        bad = sum(1 for e in sh if run[e]["top5"] and run[e]["top5"][0]["token"].strip()
                  not in [l.strip() for l in run[e]["labels"]])
        print("== %s   (%d/%d items, %d non-label top tokens)" % (label, len(sh), len(key), bad))

        floor = boot([(key[e]["family_id"], run[e]["choice"] == key[e]["correct"])
                      for e in sh if key[e]["kind"] == "floor"])
        cal = boot([(key[e]["family_id"], run[e]["choice"] == key[e]["correct"])
                    for e in sh if key[e]["kind"] == "cal"])
        print("   FLOOR (cause printed)      %s" % fmt(floor))
        print("   CAL   (one verified record) %s   <- the measurement" % fmt(cal))

        rest = [(key[e]["family_id"], run[e]["choice"] == key[e]["correct"])
                for e in sh if key[e]["kind"] == "cal_restated"]
        if rest:
            print("   CAL_RESTATED (report on the same line) %s   <- the diagnostic" % fmt(boot(rest)))
            for state in ("accurate", "inverted"):
                u = [(key[e]["family_id"], run[e]["choice"] == key[e]["correct"]) for e in sh
                     if key[e]["kind"] == "cal_restated" and key[e]["true_state"] == state]
                print("      restated, source truly %-8s %s" % (state, fmt(boot(u))))

        # The decomposition: cal by true source state.
        for state in ("accurate", "inverted"):
            u = [(key[e]["family_id"], run[e]["choice"] == key[e]["correct"])
                 for e in sh if key[e]["kind"] == "cal" and key[e]["true_state"] == state]
            print("      cal, source truly %-8s %s" % (state, fmt(boot(u))))

        # nocal: lean and pair-consistency.
        nocal = [e for e in sh if key[e]["kind"] == "nocal"]
        says_acc = boot([(key[e]["family_id"], run[e]["choice"] == key[e]["accurate_label"])
                         for e in nocal])
        by_prompt = collections.defaultdict(set)
        for e in nocal:
            by_prompt[run[e]["prompt_sha1"]].add(run[e]["choice"])
        consistent = sum(1 for v in by_prompt.values() if len(v) == 1)
        print("   NOCAL says 'accurate'       %s   (no right answer; want ~0.5)" % fmt(says_acc))
        print("   NOCAL matched pairs agree   %d/%d unique prompts" % (consistent, len(by_prompt)))

        print("   per-form CAL: " + "  ".join(
            "%s %.3f" % (f, sum(run[e]["choice"] == key[e]["correct"]
                                for e in sh if key[e]["kind"] == "cal" and key[e]["form"] == f)
                         / max(1, sum(1 for e in sh
                                      if key[e]["kind"] == "cal" and key[e]["form"] == f)))
            for f in FORMS))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
