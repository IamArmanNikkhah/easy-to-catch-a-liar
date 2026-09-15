#!/usr/bin/env python3
"""Development set. Hand-written, deliberately disjoint from the frozen evaluation set.

The first ladder found that the same logical problem scores 13/16 in one wording and 0/16
in another, which is a finding about our presentation rather than about the model. But
those two wordings differed in three ways at once, so it could not say WHICH way carried
the break. This crosses all three, one factor at a time, which is the only version of the
question that has an answer.

  T  temporal    the premise says "after round 4" and the rounds are numbered 5 and 6
  I  indirect    "you chose W. Source 3 reported 1 point"  vs  "Source 3 reported: W got 1 point"
  L  log         a "Round log:" header with "Round N -" prefixes

T0 I0 L0 is the wording that works. T1 I1 L1 is the wording the real instrument uses.

In every item the source is stated to be inverting, so the correct answer is always the
option the source did NOT call the winner. Below 0.5 therefore means the model is taking
the report at face value, not that it is guessing.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pathlib

# Twelve label pairs rather than four. The first ladder ran sixteen items a cell and one
# cell came back 13/16, which I read as a real effect; the rerun put the same wording at
# 9/16. Sixteen items cannot tell those apart, so the item count is the fix, not another
# interpretation.
PAIRS = (("W", "Z"), ("K", "M"), ("G", "J"), ("P", "Q"),
         ("B", "C"), ("D", "H"), ("L", "R"), ("V", "X"),
         ("K", "R"), ("G", "Q"), ("B", "X"), ("D", "M"))


def report_line(n, option, points, indirect, log):
    unit = "point" if points == 1 else "points"
    if indirect:
        body = "you chose %s. Source 3 reported %d %s." % (option, points, unit)
    else:
        body = "Source 3 reported: %s got %d %s." % (option, points, unit)
    if not log:
        return body[0].upper() + body[1:]
    return "Round %d - %s" % (n, body)


def build(temporal, indirect, log, reported_winner, other, order):
    premise = ("Source 3 started reporting the opposite of each actual result"
               + (" after round 4." if temporal else "."))
    n1, n2 = (5, 6) if temporal else (1, 2)
    lines = [report_line(n1, reported_winner, 1, indirect, log),
             report_line(n2, other, 0, indirect, log)]
    block = ("Round log:\n" + "\n".join(lines)) if log else "\n".join(lines)
    return ("%s\n\n%s\n\nWhich of %s and %s is actually worth 1 point? "
            "Reply with a single letter.\nAnswer:" % (premise, block, order[0], order[1]))


def build_b_exact(reported_winner, other, order):
    """The exact wording that scored 13/16 in the first ladder, carried forward verbatim so
    the replication is of that string and not of my paraphrase of it. Both reports sit on
    one line here; the factorial's T0 I0 L0 splits them across two."""
    return ("Source 3 started reporting the opposite of each actual result.\n"
            "Source 3 reported: %s got 1 point. %s got 0 points.\n"
            "Which of %s and %s is actually worth 1 point? Reply with a single letter.\n"
            "Answer:" % (reported_winner, other, order[0], order[1]))


def generate():
    rows = []
    for a, b in PAIRS:
        for winner_first in (True, False):
            for order in ((a, b), (b, a)):
                reported_winner = a if winner_first else b
                other = b if winner_first else a
                text = build_b_exact(reported_winner, other, order)
                digest = hashlib.sha1(text.encode()).hexdigest()
                rows.append({"episode_id": "dev2-Bexact-%s" % digest[:8],
                             "prompt": text, "labels": list(order),
                             "prompt_sha1": digest,
                             "cell": "B-exact", "correct": other})
    for temporal, indirect, log in itertools.product((False, True), repeat=3):
        cell = "T%d I%d L%d" % (temporal, indirect, log)
        for a, b in PAIRS:
            for winner_first in (True, False):
                for order in ((a, b), (b, a)):
                    reported_winner = a if winner_first else b
                    other = b if winner_first else a
                    text = build(temporal, indirect, log, reported_winner, other, order)
                    digest = hashlib.sha1(text.encode()).hexdigest()
                    rows.append({"episode_id": "dev2-%s-%s" % (cell.replace(" ", ""), digest[:8]),
                                 "prompt": text, "labels": list(order),
                                 "prompt_sha1": digest,
                                 "cell": cell, "correct": other})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = generate()
    ids = {r["episode_id"] for r in rows}
    assert len(ids) == len(rows), "collision: %d rows, %d ids" % (len(rows), len(ids))
    with (out / "dev2.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps({k: r[k] for k in
                                 ("episode_id", "prompt", "labels", "prompt_sha1")}) + "\n")
    with (out / "dev2_key.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    per = len(rows) // 9
    print("%d items, 9 cells, %d per cell" % (len(rows), per))
    for cell in ("T0 I0 L0", "T1 I1 L1"):
        print("\n--- %s ---" % cell)
        print(next(r for r in rows if r["cell"] == cell)["prompt"])


if __name__ == "__main__":
    raise SystemExit(main())
