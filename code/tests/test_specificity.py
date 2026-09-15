#!/usr/bin/env python3
"""Battery for the two-source arm. The load-bearing test is the last one: the three rival
accounts must actually score differently on this design, or the arm measures nothing.

Run:  python3 \
          code/tests/test_specificity.py
"""
from __future__ import annotations

import collections
import difflib
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import histories as H          # noqa: E402
import oracle as O             # noqa: E402
import prompts as P            # noqa: E402
import specificity as S        # noqa: E402
from world import ACCURATE, INVERTED, TRANSFER   # noqa: E402

CORPUS = S.generate(64, seed=11)


def solve_ep(ep):
    trials, audits = H.observations(ep)
    return O.solve(main_options=ep.visible.main_options,
                   sources=ep.visible.sources,
                   trials=trials, audits=audits,
                   question=ep.visible.question,
                   transfer_options=ep.hidden.transfer.options)


def true_winner(ep):
    return ep.hidden.as_hypothesis().payouts_for(ep.visible.question.phase).winner


class StructureTests(unittest.TestCase):

    def test_eight_cells_form_four_matched_pairs(self):
        self.assertEqual(len(S.CELLS8), 8)
        self.assertEqual(len(S.MATCHED_PAIRS8), 4)
        for a, b in S.MATCHED_PAIRS8:
            self.assertEqual(S.matched_partner(a), b)
            self.assertNotEqual(a, b)

    def test_matched_pairs_render_byte_identical_without_calibration(self):
        for fam in CORPUS:
            for a, b in S.MATCHED_PAIRS8:
                for qkind in (H.MAIN_Q, H.TRANSFER_Q):
                    self.assertEqual(P.render(fam.by(a, False, qkind).visible),
                                     P.render(fam.by(b, False, qkind).visible),
                                     "%s/%s leaked in %s" % (a, b, fam.family_id))

    def test_calibrated_pairs_differ_only_in_verified_results(self):
        for fam in CORPUS:
            for a, b in S.MATCHED_PAIRS8:
                la = P.render(fam.by(a, True, H.TRANSFER_Q).visible).splitlines()
                lb = P.render(fam.by(b, True, H.TRANSFER_Q).visible).splitlines()
                changed = [l for l in difflib.unified_diff(la, lb, n=0)
                           if l.startswith(("+", "-"))
                           and not l.startswith(("+++", "---"))]
                self.assertTrue(changed)
                for line in changed:
                    self.assertIn("actual result", line)

    def test_transfer_evidence_is_pinned_across_all_eight_cells(self):
        for fam in CORPUS:
            seen = {tuple((t.action, t.report)
                          for t in fam.by(c, True, H.TRANSFER_Q).visible.trials
                          if t.phase == TRANSFER)
                    for c in S.CELLS8}
            self.assertEqual(len(seen), 1)

    def test_no_evaluative_vocabulary(self):
        banned = ("faulty", "unreliable", "trust", "lie", "broken", "invert", "reliab")
        for ep in CORPUS[0].episodes:
            text = P.render(ep.visible).lower()
            for w in banned:
                self.assertNotIn(w, text)


class OracleTests(unittest.TestCase):

    def test_truth_never_eliminated(self):
        for fam in CORPUS[:16]:
            for ep in fam.episodes:
                trials, audits = H.observations(ep)
                self.assertTrue(O.consistent(ep.hidden.as_hypothesis(), trials, audits),
                                ep.episode_id)

    def test_uncalibrated_is_ambiguous(self):
        for fam in CORPUS[:16]:
            for ep in fam.episodes:
                if not ep.calibrated:
                    self.assertEqual(solve_ep(ep).status, O.AMBIGUOUS, ep.episode_id)

    def test_calibrated_is_definite_and_correct(self):
        for fam in CORPUS[:16]:
            for ep in fam.episodes:
                if ep.calibrated:
                    ans = solve_ep(ep)
                    self.assertEqual(ans.status, O.DEFINITE, ep.episode_id)
                    self.assertEqual(ans.action, true_winner(ep), ep.episode_id)

    def test_each_source_verdict_is_recovered_independently(self):
        for fam in CORPUS[:16]:
            for cell in S.CELLS8:
                ep = fam.by(cell, True, H.MAIN_Q)
                trials, audits = H.observations(ep)
                surv = O.survivors(ep.visible.main_options, ep.visible.sources,
                                   trials, audits, ep.hidden.transfer.options)
                for src in fam.sources:
                    self.assertEqual(O.source_verdict(surv, src), fam.state_of(cell, src))


class SpecificityTests(unittest.TestCase):

    def test_oracle_matches_the_source_specific_account_everywhere(self):
        for fam in CORPUS:
            for cell in S.CELLS8:
                self.assertEqual(solve_ep(fam.by(cell, True, H.TRANSFER_Q)).action,
                                 S.predicted_answer(fam, cell, "source_specific"))

    def test_catching_the_other_source_does_not_move_the_answer(self):
        """The specificity claim stated as a comparison: hold the transfer reporter's
        state fixed, flip the OTHER source's state, and the correct answer must not
        move. Any model whose answer moves here has generalised its distrust."""
        for fam in CORPUS:
            reporter_is_main = fam.transfer_reporter == fam.sources[0]
            for moved in ("0", "1"):
                for reporter_state in ("a", "i"):
                    answers = set()
                    for other_state in ("a", "i"):
                        s, t = ((reporter_state, other_state) if reporter_is_main
                                else (other_state, reporter_state))
                        cell = "m%s_s%s_t%s" % (moved, s, t)
                        answers.add(solve_ep(fam.by(cell, True, H.TRANSFER_Q)).action)
                    self.assertEqual(len(answers), 1,
                                     "answer moved with the other source in %s"
                                     % fam.family_id)

    def test_the_three_accounts_are_actually_distinguishable(self):
        """Overall accuracy across all eight cells. Blanket distrust lands at 0.75, not
        chance, because it is wrong only where the reporter is clean and the other source
        is not. That gap is real but thin, which is exactly why it is not the primary
        measure; see the discriminating-subset test below."""
        score = collections.Counter()
        total = 0
        for fam in CORPUS:
            for cell in S.CELLS8:
                truth = solve_ep(fam.by(cell, True, H.TRANSFER_Q)).action
                total += 1
                for account in ("source_specific", "blanket", "latest"):
                    if S.predicted_answer(fam, cell, account) == truth:
                        score[account] += 1
        self.assertEqual(score["source_specific"], total)
        self.assertAlmostEqual(score["blanket"] / total, 0.75, places=6)
        self.assertAlmostEqual(score["latest"] / total, 0.5, places=6)

    def test_discriminating_subset_separates_the_accounts_completely(self):
        """The primary measure. Clean reporter throughout; the only thing that moves is
        whether the other source was caught. Source-specific is right in both conditions,
        blanket distrust is wrong in every case of the second."""
        hit = collections.Counter()
        n = collections.Counter()
        for fam in CORPUS:
            groups = S.discriminating_cells(fam)
            self.assertTrue(groups["control"] and groups["other_source_caught"])
            for label, cells in groups.items():
                for cell in cells:
                    truth = solve_ep(fam.by(cell, True, H.TRANSFER_Q)).action
                    n[label] += 1
                    for account in ("source_specific", "blanket", "latest"):
                        if S.predicted_answer(fam, cell, account) == truth:
                            hit[(label, account)] += 1
        for label in ("control", "other_source_caught"):
            self.assertEqual(hit[(label, "source_specific")], n[label], label)
        self.assertEqual(hit[("control", "blanket")], n["control"])
        self.assertEqual(hit[("other_source_caught", "blanket")], 0)


class CounterbalanceTests(unittest.TestCase):

    def test_verified_values_carry_no_information_about_the_answer(self):
        by_flip = collections.defaultdict(collections.Counter)
        for fam in CORPUS:
            for cell in S.CELLS8:
                ep = fam.by(cell, True, H.TRANSFER_Q)
                flipped = solve_ep(ep).action != fam.transfer_reported_winner
                for a in ep.visible.audits:
                    by_flip[flipped][a.verified_outcome] += 1
        self.assertEqual(dict(by_flip[True]), dict(by_flip[False]))

    def test_which_source_is_faulty_is_balanced(self):
        counts = collections.Counter()
        for fam in CORPUS:
            for cell in S.CELLS8:
                counts[(fam.state_of(cell, fam.sources[0]) == INVERTED,
                        fam.state_of(cell, fam.sources[1]) == INVERTED)] += 1
        self.assertEqual(len(set(counts.values())), 1)

    def test_transfer_reporter_role_is_balanced(self):
        roles = collections.Counter(f.transfer_reporter == f.sources[0] for f in CORPUS)
        self.assertEqual(roles[True], roles[False])


if __name__ == "__main__":
    unittest.main(verbosity=2)
