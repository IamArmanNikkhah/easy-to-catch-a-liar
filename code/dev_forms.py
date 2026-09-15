#!/usr/bin/env python3
"""Development check: are the four surface forms equally faithful, or a difficulty ladder?

This has to run BEFORE the evaluation set is frozen. The agreement gate only means
something if a competent reader would find all four forms equally clear; if one of them is
degenerate for the target model, the gate bakes in a guaranteed disagreement and the
agreement rate stops being a measurement. Freezing first and finding that out afterwards
kills the run for that model.

Two questions, both answered from the same families:
  FLOOR   the payout table is stated outright. Any form a model cannot pass here is not a
          faithful presentation of the task, and the form is the problem, not the model.
  CEILING the cause is stated outright and the model still has to apply it. This is the one
          that failed at 0.516 in the single-form instrument; running it per form says
          whether any form recovers it, which is what decides the next step.

Seed 9001 is RESERVED for development and must never be used to build an evaluation set.
Different seed means different label assignments, so these items are disjoint from the
frozen ones by construction.

  python3 dev_forms.py --out-dir <dir> --families 16
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import histories as H
import prompts as P

DEV_SEED = 9001
CELLS = ("stable", "world", "reporter", "both")


def rows(families):
    for fam in families:
        for cell in CELLS:
            variants = [
                ("floor", H.disclosed_payouts(fam.by(cell, False, H.MAIN_Q)),
                 lambda ep: ep.hidden.post.winner),
                ("ceiling", H.disclosed_cause(fam.by(cell, False, H.TRANSFER_Q)),
                 lambda ep: ep.hidden.transfer.winner),
            ]
            for kind, ep, answer in variants:
                correct = answer(ep)
                group = "%s-%s-%s" % (fam.family_id, cell, kind)
                for form in P.FORMS:
                    text = P.render(ep.visible, form)
                    digest = hashlib.sha1(text.encode()).hexdigest()
                    yield ({"episode_id": "devf-%s-%s" % (form, digest[:10]),
                            "prompt": text,
                            "labels": list(P.question_options(ep.visible, form)),
                            "prompt_sha1": digest},
                           {"episode_id": "devf-%s-%s" % (form, digest[:10]),
                            "kind": kind, "form": form, "cell": cell,
                            "variant_group": group, "correct": correct,
                            "family_id": fam.family_id})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--families", type=int, default=16)
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fams = H.generate(args.families, seed=DEV_SEED)
    pairs = list(rows(fams))
    ids = {p["episode_id"] for p, _ in pairs}
    if len(ids) != len(pairs):
        raise SystemExit("id collision: %d rows, %d ids" % (len(pairs), len(ids)))

    with (out / "devforms.jsonl").open("w") as pf, (out / "devforms_key.jsonl").open("w") as kf:
        for p, k in pairs:
            pf.write(json.dumps(p) + "\n")
            kf.write(json.dumps(k) + "\n")

    n = collections.Counter((k["kind"], k["form"]) for _, k in pairs)
    print("%d items -> %s" % (len(pairs), out / "devforms.jsonl"))
    for kind in ("floor", "ceiling"):
        print("  %-8s %s" % (kind, ", ".join("%s %d" % (f, n[(kind, f)]) for f in P.FORMS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
