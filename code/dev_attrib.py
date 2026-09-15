#!/usr/bin/env python3
"""Development check for the ATTRIBUTION question: ask the model what happened to the
source, not which option wins.

Why this exists. The winner question makes the model do two jobs in one forward pass:
decide whether the reporter is broken, then apply that to a report and flip it. The size
ladder of 2026-09-10 showed the second job is what fails (told-flipping 0.109 at 7B while
the model picks the reported winner 0.90 of the time), and Arman's hand test showed the
same model gets it right the moment it is allowed scratch space. Only the FIRST job was
ever the research question. This asks it alone.

Three variants per family and cell, on the MAIN history (no transfer trials needed):
  floor   the cause is stated outright, then we ask the source's state. The answer is
          printed in the prompt; this is a reading check and nothing else.
  cal     one verified record on a post-change round, no disclosure. The oracle settles
          the source's state from that one audit. THIS is the measurement.
  nocal   no audit, no disclosure. The oracle is ambiguous by construction (matched pairs
          render byte-identically), so a model above chance here is reading something
          other than the evidence.

Seed 9001 is the development seed and is never used to build an evaluation set.

  python3 dev_attrib.py --out-dir <dir> --families 16
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
import oracle as O
import prompts as P
from world import ACCURATE, INVERTED

DEV_SEED = 9001
CELLS = ("stable", "world", "reporter", "both")


def source_labels(fam: H.Family, index: int):
    used = set(fam.main_options) | set(fam.transfer_options)
    free = [l for l in H.LABEL_POOL if l not in used]
    labels = (free[0], free[1])
    accurate = labels[index % 2]
    return labels, accurate


def oracle_state(ep: H.Episode) -> str:
    trials, audits = H.observations(ep)
    surv = O.survivors(ep.visible.main_options, ep.visible.sources, trials, audits,
                       ep.visible.transfer_options)
    return O.source_verdict(surv, ep.visible.question.source)


def rows(families):
    for index, fam in enumerate(families):
        labels, accurate = source_labels(fam, index)
        opposite = [l for l in labels if l != accurate][0]
        to_label = {ACCURATE: accurate, INVERTED: opposite}
        for cell in CELLS:
            nocal = fam.by(cell, False, H.MAIN_Q)
            cal = fam.by(cell, True, H.MAIN_Q)
            true_state = nocal.hidden.as_hypothesis().state_of(nocal.visible.question.source)
            variants = [
                ("floor", H.as_source_question(H.disclosed_cause(nocal), labels, accurate),
                 to_label[true_state], "definite"),
                ("cal", H.as_source_question(cal, labels, accurate), None, None),
                # Diagnostic: same item, the audited round's report restated beside the
                # verified result. If cal fails and this clears, the cross-reference from
                # record to log line is the bottleneck, not the compare.
                ("cal_restated",
                 H.as_source_question(H.restated_report(cal), labels, accurate), None, None),
                ("nocal", H.as_source_question(nocal, labels, accurate), None, None),
            ]
            for kind, ep, correct, status in variants:
                if kind != "floor":
                    verdict = oracle_state(ep)
                    status = "ambiguous" if verdict == O.AMBIGUOUS else "definite"
                    correct = to_label.get(verdict)
                group = "%s-%s-%s" % (fam.family_id, cell, kind)
                for form in P.FORMS:
                    text = P.render(ep.visible, form)
                    digest = hashlib.sha1(text.encode()).hexdigest()
                    # The id carries the cell because matched-pair nocal items are the SAME
                    # prompt by construction (that is the leak guarantee), so a prompt hash
                    # alone collides. prompt_sha1 stays the true hash so the identity is
                    # visible in the data rather than hidden by a salted id.
                    eid = "deva-%s-%s-%s" % (form, cell, digest[:10])
                    yield ({"episode_id": eid, "prompt": text,
                            "labels": list(P.question_options(ep.visible, form)),
                            "prompt_sha1": digest},
                           {"episode_id": eid, "kind": kind, "form": form, "cell": cell,
                            "variant_group": group, "correct": correct,
                            "oracle_status": status, "true_state": true_state,
                            "accurate_label": accurate, "family_id": fam.family_id})


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

    with (out / "devattrib.jsonl").open("w") as pf, \
            (out / "devattrib_key.jsonl").open("w") as kf:
        for p, k in pairs:
            pf.write(json.dumps(p) + "\n")
            kf.write(json.dumps(k) + "\n")

    n = collections.Counter((k["kind"], k["oracle_status"]) for _, k in pairs)
    print("%d items -> %s" % (len(pairs), out / "devattrib.jsonl"))
    for kind in ("floor", "cal", "nocal"):
        print("  %-6s definite %3d  ambiguous %3d"
              % (kind, n[(kind, "definite")], n[(kind, "ambiguous")]))
    print()
    print(next(p for p, k in pairs if k["kind"] == "cal" and k["form"] == "log")["prompt"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
