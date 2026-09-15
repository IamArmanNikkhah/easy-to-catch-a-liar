#!/usr/bin/env python3
"""The PAIRED measure: one history, two questions, byte-identical up to the question line.

Why this file exists rather than another variant inside dev_attrib.py. dev_attrib.py is the
record of what produced the 2026-09-10 numbers (32B cal_restated 0.848) and it stays frozen
so those stay reproducible. This asks a different question and it asks it on a different
substrate.

THE SUBSTRATE CHANGE, and it is the whole reason for the rewrite. dev_attrib builds on MAIN
histories, where the audited round uses the same two options the winner question asks about.
Measured 2026-09-11: on a main history the audited action plus its verified outcome names
the post winner 64 times out of 64, with the source never entering. So a winner/source pair
built there is not one evidence set and two questions; it is a two-fact comparison against a
one-fact lookup, and neither outcome would mean anything. On a TRANSFER history the audit is
on a main-option round and the question is about a fresh pair, so the only route to the
answer runs through the source's state. test_design.PairedQuestionTests asserts this.

FIVE VARIANTS, EACH ASKED BOTH WAYS:
  floor    the answer is printed AND nothing visible contradicts it. For the source question
           that is the stated cause; for the winner question it is reading_floor, which
           strips the post-change reports before printing the payout table. A model below
           ceiling here is a broken harness, not a result.
  ceiling  the cause is stated and the model must apply it. For the source question this
           coincides with the floor, because the cause IS the answer; for the winner
           question it is the genuine upper bound on using a known source state.
  cal      one verified record, restated beside the report it covers, no disclosure. THIS IS
           THE MEASUREMENT. The restatement is carried because without it the attribution
           question does not clear at any size we can afford (0.637 vs 0.848 at 32B).
  cal_prior  cal plus one rules line stating that the two causes occur about equally often,
           naming neither for this episode. Arman's call, 2026-09-11. It exists to split two
           explanations of the `world` cell (payouts moved, source honest), where everything
           collapses and the 72B is WORSE than the 32B, 0.422 against 0.672, while both are
           perfect where the source really did lie. Either the model holds a prior that a
           surprising report means a lying reporter, or it never entertains world-change as
           a candidate at all. Salience moves the first and not the second, so a lift in
           `world` and nowhere else says prior.
  nocal    no audit, no disclosure. Matched pairs render byte-identically, so the oracle is
           ambiguous by construction and any structure here is not evidence-driven.

WHAT THE KEY CARRIES THAT THE OLD ONE DID NOT: `pair_id` links the two arms of one prompt so
P(winner correct | source correct) is computable; `audit_round` and `audit_target` make the
round-5/6 moderator (a +0.13 to +0.26 swing that survives restatement) reportable from
shipped data instead of recomputed from the generator; `reports_flipped` is what the flip
heuristic keys on, and without it the leak control cannot see the policy the 72B actually
runs on zero-evidence items (deterministic 1.00/0.00/0.00/1.00 by cell, averaging to exactly
the 0.500 the old control passes).

  python3 paired_attrib.py --out-dir ../data --families 64
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
from world import ACCURATE, INVERTED, POST, PRE

DEV_SEED = 9001
CELLS = ("stable", "world", "reporter", "both")
WINNER, SOURCE = "winner", "source"


LETTER_ROTATION = "parity"   # set from --letter-rotation; see main()


def source_labels(fam: H.Family, index: int):
    """Two answer letters disjoint from every option label in the family. Chosen here rather
    than left to as_source_question, whose own check reads transfer_options and so cannot
    exclude them on a main episode. Rotating which letter means 'accurate' by family index
    stops a constant-letter policy from scoring."""
    used = set(fam.main_options) | set(fam.transfer_options)
    free = [l for l in H.LABEL_POOL if l not in used]
    labels = (free[0], free[1])
    # 2026-09-13: histories.py assigns the audited round by the same parity, so on the
    # original file "honest is the later letter" and "record names round 6" are one
    # variable. "swapped" reverses the pairing; scored beside the original it crosses
    # letter x round exactly, which no analysis of either file alone can do.
    pick = index % 2 if LETTER_ROTATION == "parity" else (index + 1) % 2
    return labels, labels[pick]


def reports_flipped(ep: H.Episode) -> bool:
    """Did the reported winner change across the change point, as the model can see it?

    This is the input to the one heuristic we know a model runs: reports flipped, so the
    source lied. It is right in `stable` and `reporter` and wrong in `world` and `both`, so
    it scores exactly 0.500 over a balanced design — which is why a marginal leak control
    cannot detect it and why this field has to ship with the data."""
    pre = [t for t in ep.visible.trials if t.phase == PRE]
    post = [t for t in ep.visible.trials if t.phase == POST]
    pre_winner = next((t.action for t in pre if t.report == 1), None)
    post_winner = next((t.action for t in post if t.report == 1), None)
    if pre_winner is None or post_winner is None:
        return False
    return pre_winner != post_winner


def reported_winner(ep: H.Episode):
    """The option the question phase's reports name as the winner, taken at face value.

    This is the answer a pure follow-the-report policy gives, and it is the honest lower
    bound for the winner arm: on the winner question the ceiling measured 0.234 at 32B
    while picks-reported-winner ran 0.88, so the baseline that matters is not 0.5, it is
    this. Shipping it in the key means the baseline is computable from the data instead of
    recovered from the generator later."""
    phase = ep.visible.question.phase
    if ep.visible.withhold_reports and phase != PRE:
        return None
    hits = [t.action for t in ep.visible.trials if t.phase == phase and t.report == 1]
    return hits[0] if hits else None


def oracle_source_state(ep: H.Episode) -> str:
    trials, audits = H.observations(ep)
    surv = O.survivors(ep.visible.main_options, ep.visible.sources, trials, audits,
                       ep.visible.transfer_options)
    return O.source_verdict(surv, ep.visible.question.source)


def oracle_winner(ep: H.Episode):
    trials, audits = H.observations(ep)
    ans = O.solve(main_options=ep.visible.main_options, sources=ep.visible.sources,
                  trials=trials, audits=audits, question=ep.visible.question,
                  transfer_options=ep.visible.transfer_options or ep.hidden.transfer.options)
    return ans.status, ans.action


def variants(fam: H.Family, cell: str):
    """The four variants as (name, winner_episode, source_base_episode) triples. The source
    arm is built by wrapping the SAME base episode, which is what keeps the prefixes equal."""
    cal = fam.by(cell, True, H.TRANSFER_Q)
    nocal = fam.by(cell, False, H.TRANSFER_Q)
    return [
        # floor: answer printed, nothing contradicting it. Different constructor per arm,
        # because what counts as "the answer printed" differs between the two questions.
        ("floor", H.reading_floor(cal), H.disclosed_cause(nocal)),
        # ceiling: cause stated, model must apply it. Same construction both arms.
        ("ceiling", H.disclosed_cause(cal), H.disclosed_cause(cal)),
        # cal: the measurement.
        ("cal", H.restated_report(cal), H.restated_report(cal)),
        # cal_prior: cal plus one rules line making world-change as salient as
        # reporter-fault, naming neither. Separates "the model holds a blame-the-messenger
        # prior" from "the model never considers the world changing at all". Only the
        # `world` cell should move if it is a prior.
        ("cal_prior", H.with_world_prior(H.restated_report(cal)),
         H.with_world_prior(H.restated_report(cal))),
        # nocal: the leak control.
        ("nocal", nocal, nocal),
    ]


def rows(families):
    for index, fam in enumerate(families):
        labels, accurate = source_labels(fam, index)
        opposite = next(l for l in labels if l != accurate)
        to_label = {ACCURATE: accurate, INVERTED: opposite}
        for cell in CELLS:
            base_cal = fam.by(cell, True, H.TRANSFER_Q)
            audit = base_cal.visible.audits[0]
            flipped = reports_flipped(base_cal)
            truth = base_cal.hidden.as_hypothesis().state_of(
                base_cal.visible.question.source)
            for vname, win_ep, src_base in variants(fam, cell):
                src_ep = H.as_source_question(src_base, labels, accurate)
                # The answer key. Floor and ceiling hand the fact over, so the truth is the
                # key; cal and nocal must be settled by the oracle, which is entitled to say
                # ambiguous and does on every nocal item by construction.
                if vname in ("floor", "ceiling"):
                    win_status, win_correct = "definite", win_ep.hidden.transfer.winner
                    src_status, src_correct = "definite", to_label[truth]
                else:
                    win_status, win_correct = oracle_winner(win_ep)
                    verdict = oracle_source_state(src_ep)
                    src_status = "ambiguous" if verdict == O.AMBIGUOUS else "definite"
                    src_correct = to_label.get(verdict)
                for form in P.FORMS:
                    for qname, ep, correct, status in (
                            (WINNER, win_ep, win_correct, win_status),
                            (SOURCE, src_ep, src_correct, src_status)):
                        text = P.render(ep.visible, form)
                        digest = hashlib.sha1(text.encode()).hexdigest()
                        # The cell goes in the id because matched-pair nocal items ARE the
                        # same prompt by construction — that is the leak guarantee — so a
                        # prompt hash alone collides. prompt_sha1 stays the true hash so the
                        # identity stays visible in the data rather than hidden by the id.
                        eid = "pair-%s-%s-%s-%s-%s" % (form, qname, vname, cell, digest[:10])
                        yield ({"episode_id": eid, "prompt": text,
                                "labels": list(P.question_options(ep.visible, form)),
                                "prompt_sha1": digest},
                               {"episode_id": eid, "question": qname, "variant": vname,
                                "form": form, "cell": cell, "family_id": fam.family_id,
                                # the two arms of one prompt share this and nothing else
                                "pair_id": "%s-%s-%s-%s" % (fam.family_id, cell, vname, form),
                                "correct": correct, "oracle_status": status,
                                "true_state": truth, "accurate_label": accurate,
                                "audit_round": audit.round, "audit_target": audit.action,
                                "reports_flipped": flipped,
                                "reported_winner": reported_winner(ep),
                                # Floors are the one variant whose two arms legitimately
                                # differ upstream of the question, because "the answer is
                                # printed with nothing contradicting it" means a stated
                                # cause on one arm and a stripped-then-disclosed payout
                                # table on the other. They are per-arm baselines, not a
                                # paired item, and the key says so rather than letting an
                                # analysis assume byte-matching it does not have.
                                "prefix_paired": vname != "floor"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--families", type=int, default=64)
    ap.add_argument("--letter-rotation", choices=("parity", "swapped"), default="parity")
    args = ap.parse_args()
    global LETTER_ROTATION
    LETTER_ROTATION = args.letter_rotation
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fams = H.generate(args.families, seed=DEV_SEED)
    pairs = list(rows(fams))
    ids = {p["episode_id"] for p, _ in pairs}
    if len(ids) != len(pairs):
        raise SystemExit("id collision: %d rows, %d ids" % (len(pairs), len(ids)))

    # Every pair_id must name exactly two items, one per question, or the item-conditional
    # statistic silently drops or double-counts.
    by_pair = collections.Counter(k["pair_id"] for _, k in pairs)
    bad = [p for p, n in by_pair.items() if n != 2]
    if bad:
        raise SystemExit("%d pair_ids do not hold exactly 2 items, e.g. %r" % (len(bad), bad[:3]))

    # The paper's central claim rests on the two arms of a pair differing ONLY after the
    # question line. Asserting it here, against the shipped prompts rather than against the
    # generator, is what makes it a property of the artifact. This caught a real mismatch on
    # first run: 128 of 512 pairs, all of them floors, which is how prefix_paired got added.
    prompt_of = {p["episode_id"]: p["prompt"] for p, _ in pairs}
    arms = collections.defaultdict(dict)
    for _, k in pairs:
        arms[k["pair_id"]][k["question"]] = k
    unpaired = []
    for pid, d in arms.items():
        if not d[WINNER]["prefix_paired"]:
            continue
        a = prompt_of[d[WINNER]["episode_id"]].rsplit("\n\n", 1)[0]
        b = prompt_of[d[SOURCE]["episode_id"]].rsplit("\n\n", 1)[0]
        if a != b:
            unpaired.append(pid)
    if unpaired:
        raise SystemExit("%d pairs marked prefix_paired do not share a prefix, e.g. %r"
                         % (len(unpaired), unpaired[:3]))

    stem = "paired_s%d_n%d" % (DEV_SEED, args.families)
    if LETTER_ROTATION == "swapped":
        stem += "_swap"
    with (out / (stem + ".jsonl")).open("w") as pf, \
            (out / (stem + "_key.jsonl")).open("w") as kf:
        for p, k in pairs:
            pf.write(json.dumps(p) + "\n")
            kf.write(json.dumps(k) + "\n")

    n = collections.Counter((k["variant"], k["question"], k["oracle_status"])
                            for _, k in pairs)
    print("%d items -> %s" % (len(pairs), out / (stem + ".jsonl")))
    print("%d pairs, each holding one winner item and one source item" % len(by_pair))
    for v in ("floor", "ceiling", "cal", "cal_prior", "nocal"):
        for q in (WINNER, SOURCE):
            print("  %-8s %-6s definite %4d  ambiguous %4d"
                  % (v, q, n[(v, q, "definite")], n[(v, q, "ambiguous")]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
