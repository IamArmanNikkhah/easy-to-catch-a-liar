#!/usr/bin/env python3
"""Freeze the evaluation set to disk, prompts and answer key in SEPARATE files.

The split is a structural guard, not tidiness. The runner opens prompts.jsonl and nothing
else, so a mistake in the notebook cannot put a hidden field in front of the model. The
answer key joins back on episode_id at analysis time.

Prompts are frozen here, before anything is run, because the plan requires it: a prompt
edited after seeing a result is no longer the prompt the result belongs to.

  python3 build_dataset.py --out ../data --single-families 32 --spec-families 64
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
import specificity as S
from world import INVERTED, TRANSFER


def _solve(ep, transfer_options):
    trials, audits = H.observations(ep)
    return O.solve(main_options=ep.visible.main_options, sources=ep.visible.sources,
                   trials=trials, audits=audits, question=ep.visible.question,
                   transfer_options=transfer_options)


def _control_rows(families):
    """Comprehension controls, which the answer key cannot derive on its own: a disclosure
    is prose in the prompt, not an observation the oracle can condition on.

    These decide whether a null result is interpretable. Without them, a model at chance
    on the calibrated transfer question leaves us unable to separate "cannot infer
    reliability" from "did not understand the task", and those need very different
    responses."""
    for fam in families:
        for cell in ("stable", "world", "reporter", "both"):
            base_t = fam.by(cell, False, H.TRANSFER_Q)
            base_m = fam.by(cell, False, H.MAIN_Q)
            variants = [
                # Cause handed over outright: the ceiling for the transfer question.
                (H.disclosed_cause(base_t), "disclosed_cause", O.DEFINITE,
                 base_t.hidden.transfer.winner, H.TRANSFER_Q),
                # Payout table handed over: the floor-level "can it read a table" gate.
                (H.disclosed_payouts(base_m), "disclosed_payouts", O.DEFINITE,
                 base_m.hidden.post.winner, H.MAIN_Q),
                # Same history, reports withheld: nothing is learnable, so a strong
                # preference here is a preference for something other than the evidence.
                (H.without_feedback(base_t), "no_feedback", O.AMBIGUOUS, None,
                 H.TRANSFER_Q),
            ]
            for ep, name, status, action, qkind in variants:
                text = P.render(ep.visible)
                yield (
                    {"episode_id": ep.episode_id, "prompt": text,
                     "labels": list(ep.visible.question.options),
                     "prompt_sha1": hashlib.sha1(text.encode()).hexdigest()},
                    {"episode_id": ep.episode_id, "arm": "control",
                     "family_id": ep.family_id, "cell": ep.cell, "calibrated": False,
                     "question_kind": qkind,
                     "condition": "control/%s/%s" % (name, ep.cell),
                     "control": name,
                     "pair_id": "%s:control-%s" % (ep.family_id, name),
                     "oracle_status": status, "oracle_action": action,
                     "n_survivors": None,
                     "labels": list(ep.visible.question.options),
                     "is_primary": False, "is_discriminating": False},
                )


def _rows_single(families):
    for fam in families:
        pair_of = {c: "%s|%s" % tuple(sorted(p))
                   for p in H.MATCHED_PAIRS for c in p}
        for ep in fam.episodes:
            ans = _solve(ep, ep.hidden.transfer.options)
            text = P.render(ep.visible)
            yield (
                {"episode_id": ep.episode_id,
                 "prompt": text,
                 "labels": list(ep.visible.question.options),
                 "prompt_sha1": hashlib.sha1(text.encode()).hexdigest()},
                {"episode_id": ep.episode_id, "arm": "single", "family_id": ep.family_id,
                 "cell": ep.cell, "calibrated": ep.calibrated,
                 "question_kind": ep.question_kind,
                 "condition": "single/%s/%s/%s" % (ep.question_kind,
                                                   "cal" if ep.calibrated else "nocal",
                                                   ep.cell),
                 "pair_id": "%s:%s" % (ep.family_id, pair_of[ep.cell]),
                 "oracle_status": ans.status, "oracle_action": ans.action,
                 "n_survivors": len(ans.survivors),
                 "source_inverted": ep.hidden.as_hypothesis().state_of(fam.source)
                 == INVERTED,
                 "payouts_moved": ep.hidden.pre.winner != ep.hidden.post.winner,
                 "reported_transfer_winner": fam.transfer_reported_winner,
                 "audit_target": ep.audit_target,
                 "audit_position": fam.audit_position,
                 "labels": list(ep.visible.question.options),
                 "is_primary": ep.question_kind == H.TRANSFER_Q and ep.calibrated,
                 "is_discriminating": False},
            )


def _rows_spec(families):
    for fam in families:
        groups = S.discriminating_cells(fam)
        disc = {c: label for label, cells in groups.items() for c in cells}
        pair_of = {c: "%s|%s" % tuple(sorted(p))
                   for p in S.MATCHED_PAIRS8 for c in p}
        for ep in fam.episodes:
            ans = _solve(ep, ep.hidden.transfer.options)
            text = P.render(ep.visible)
            reporter_inv = (ep.hidden.as_hypothesis().state_of(fam.transfer_reporter)
                            == INVERTED)
            yield (
                {"episode_id": ep.episode_id,
                 "prompt": text,
                 "labels": list(ep.visible.question.options),
                 "prompt_sha1": hashlib.sha1(text.encode()).hexdigest()},
                {"episode_id": ep.episode_id, "arm": "spec", "family_id": ep.family_id,
                 "cell": ep.cell, "calibrated": ep.calibrated,
                 "question_kind": ep.question_kind,
                 "condition": "spec/%s/%s/%s" % (ep.question_kind,
                                                 "cal" if ep.calibrated else "nocal",
                                                 ep.cell),
                 "pair_id": "%s:%s" % (ep.family_id, pair_of[ep.cell]),
                 "oracle_status": ans.status, "oracle_action": ans.action,
                 "n_survivors": len(ans.survivors),
                 "reporter_inverted": reporter_inv,
                 "other_source_inverted": any(
                     ep.hidden.as_hypothesis().state_of(s) == INVERTED
                     for s in fam.sources if s != fam.transfer_reporter),
                 "reported_transfer_winner": fam.transfer_reported_winner,
                 "transfer_reporter_is_main": fam.transfer_reporter == fam.sources[0],
                 "audit_position": fam.audit_position,
                 "labels": list(ep.visible.question.options),
                 "predicted": {a: S.predicted_answer(fam, ep.cell, a)
                               for a in ("source_specific", "blanket", "latest")}
                 if ep.question_kind == H.TRANSFER_Q else None,
                 "is_primary": ep.question_kind == H.TRANSFER_Q and ep.calibrated,
                 "is_discriminating": (ep.question_kind == H.TRANSFER_Q
                                       and ep.calibrated and ep.cell in disc),
                 "discriminating_group": disc.get(ep.cell) if ep.calibrated else None},
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--single-families", type=int, default=32,
                    help="multiple of 16 closes the counterbalance")
    ap.add_argument("--spec-families", type=int, default=64,
                    help="multiple of 64 closes the counterbalance")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.single_families % 16 or args.spec_families % 64:
        print("refusing: family counts must close the counterbalance"
              " (single multiple of 16, spec multiple of 64)", file=sys.stderr)
        return 2

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts_path, answers_path = out / "prompts.jsonl", out / "answers.jsonl"

    single_fams = H.generate(args.single_families, seed=args.seed)
    rows = list(_rows_single(single_fams))
    rows += list(_rows_spec(S.generate(args.spec_families, seed=args.seed + 1)))
    rows += list(_control_rows(single_fams))

    # The readable id names the cell ("...-reporter-cal-transfer"). It never reaches the
    # model, because score_answers.py reads only rec["prompt"] -- but the shipped file
    # should not depend on that staying true through a future careless edit. The prompt
    # file carries an opaque id; the key keeps the readable one for analysis.
    seen = set()
    opaque_rows = []
    for p, a in rows:
        readable = p["episode_id"]
        if readable in seen:
            raise SystemExit("duplicate episode_id %s" % readable)
        seen.add(readable)
        opaque = hashlib.sha1(("rl-trust|" + readable).encode()).hexdigest()[:16]
        opaque_rows.append((dict(p, episode_id=opaque),
                            dict(a, episode_id=opaque, readable_id=readable)))
    rows = opaque_rows

    with prompts_path.open("w") as pf, answers_path.open("w") as af:
        for p, a in rows:
            pf.write(json.dumps(p) + "\n")
            af.write(json.dumps(a) + "\n")

    # A last independent check that no answer-key field reached the prompt file. Cheap,
    # and it is the one mistake in this pipeline that would silently invalidate everything.
    banned_fields = {"cell", "oracle_action", "source_inverted", "reporter_inverted",
                     "payouts_moved", "predicted", "condition"}
    with prompts_path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            assert not (set(rec) & banned_fields), rec.keys()

    # Two families can draw the same labels, source and counterbalance and so produce the
    # same prompt. Those are dependent samples that the family-level bootstrap would count
    # as independent, so they get reported. The same prompt carrying two DIFFERENT correct
    # answers would be a genuine contradiction in the key, and that is a hard stop.
    by_hash = collections.defaultdict(list)
    for p, a in rows:
        by_hash[p["prompt_sha1"]].append(a)
    cross = [h for h, aa in by_hash.items() if len({a["family_id"] for a in aa}) > 1]
    for h in cross:
        keys = {(a["oracle_status"], a["oracle_action"]) for a in by_hash[h]}
        if len(keys) > 1:
            raise SystemExit("identical prompt %s carries conflicting answers %r" % (h, keys))

    counts = collections.Counter(a["condition"].rsplit("/", 1)[0] for _, a in rows)
    chars = sum(len(p["prompt"]) for p, _ in rows)
    print("wrote %d episodes" % len(rows))
    print("  %s" % prompts_path)
    print("  %s" % answers_path)
    print("\nforward evaluations by condition group:")
    for k in sorted(counts):
        print("  %-28s %5d" % (k, counts[k]))
    primary = sum(1 for _, a in rows if a["is_primary"])
    disc = sum(1 for _, a in rows if a["is_discriminating"])
    print("\n  primary (calibrated transfer)   %5d" % primary)
    print("  discriminating subset           %5d" % disc)
    print("  cross-family identical prompts  %5d  (dependent samples; 0 is ideal)"
          % len(cross))
    print("\nmean prompt length %.0f chars (~%.0f tokens at 4 chars/token)"
          % (chars / len(rows), chars / len(rows) / 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
