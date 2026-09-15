#!/usr/bin/env python3
"""The CPU battery. This is the receipt for the three things Arman called non-negotiable:
hidden causes never reach a prompt, ambiguous histories are genuinely ambiguous, and the
transfer question carries enough signal to be answerable.

Run:  python3 \
          code/tests/test_design.py
"""
from __future__ import annotations

import collections
import difflib
import pathlib
import random
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import histories as H            # noqa: E402
import oracle as O               # noqa: E402
import prompts as P              # noqa: E402
from world import ACCURATE, INVERTED, POST, PRE, TRANSFER   # noqa: E402

CORPUS = H.generate(32, seed=7)


def solve_ep(ep):
    trials, audits = H.observations(ep)
    return O.solve(main_options=ep.visible.main_options,
                   sources=ep.visible.sources,
                   trials=trials, audits=audits,
                   question=ep.visible.question,
                   transfer_options=ep.visible.transfer_options
                   or ep.hidden.transfer.options)


def true_winner(ep):
    return ep.hidden.as_hypothesis().payouts_for(ep.visible.question.phase).winner


class LeakTests(unittest.TestCase):
    """Sanity check 1: hidden causes must never reach a prompt."""

    def test_matched_pairs_render_byte_identical(self):
        for fam in CORPUS:
            for a, b in H.MATCHED_PAIRS:
                for qkind in (H.MAIN_Q, H.TRANSFER_Q):
                    pa = P.render(fam.by(a, False, qkind).visible)
                    pb = P.render(fam.by(b, False, qkind).visible)
                    self.assertEqual(pa, pb,
                                     "%s vs %s (%s) leaked in family %s"
                                     % (a, b, qkind, fam.family_id))

    def test_matched_pairs_differ_only_in_the_audit_line(self):
        """With calibration the prompts MUST differ, and in exactly one line: the verified
        result. Anything else differing would mean history length, ordering or wording is
        carrying the cause."""
        for fam in CORPUS:
            for a, b in H.MATCHED_PAIRS:
                for qkind in (H.MAIN_Q, H.TRANSFER_Q):
                    la = P.render(fam.by(a, True, qkind).visible).splitlines()
                    lb = P.render(fam.by(b, True, qkind).visible).splitlines()
                    changed = [l for l in difflib.unified_diff(la, lb, n=0)
                               if l.startswith(("+", "-"))
                               and not l.startswith(("+++", "---"))]
                    self.assertEqual(len(changed), 2, "expected one changed line, got %r"
                                     % (changed,))
                    for line in changed:
                        self.assertIn("actual result", line)

    def test_visible_type_carries_no_hidden_state(self):
        names = set(H.VisibleEpisode.__dataclass_fields__)
        for banned in ("hidden", "cell", "source_states", "post", "pre"):
            self.assertNotIn(banned, names)

    def test_no_evaluative_vocabulary_in_any_prompt(self):
        """Competing explanation (3): the model responding to loaded words rather than to
        evidence. The rules describe a mechanism and never characterise a source."""
        banned = ("faulty", "unreliable", "trust", "distrust", "lie", "lying", "broken",
                  "malfunction", "deceiv", "honest", "invert", "corrupt", "reliab")
        for fam in CORPUS:
            for ep in fam.episodes:
                text = P.render(ep.visible).lower()
                for w in banned:
                    self.assertNotIn(w, text, "%r in %s" % (w, ep.episode_id))


class PerFormLeakTests(unittest.TestCase):
    """Every invariant that matters holds PER FORM, or it does not hold. Three of the four
    forms ship, and until this class existed only `log` was covered — the same gap as
    LESSON 3e4c05a9, where a control existed as a tested function and never reached the
    artifact."""

    def test_matched_pairs_are_byte_identical_in_every_form(self):
        for fam in CORPUS[:12]:
            for form in P.FORMS:
                for a, b in H.MATCHED_PAIRS:
                    for qkind in (H.MAIN_Q, H.TRANSFER_Q):
                        self.assertEqual(P.render(fam.by(a, False, qkind).visible, form),
                                         P.render(fam.by(b, False, qkind).visible, form),
                                         "%s/%s leaked in form %s" % (a, b, form))

    def test_calibrated_pairs_differ_only_where_the_form_states_a_verified_result(self):
        for fam in CORPUS[:12]:
            for form in P.FORMS:
                marker = P.AUDIT_MARKER[form]
                for a, b in H.MATCHED_PAIRS:
                    la = P.render(fam.by(a, True, H.TRANSFER_Q).visible, form).splitlines()
                    lb = P.render(fam.by(b, True, H.TRANSFER_Q).visible, form).splitlines()
                    changed = [l for l in difflib.unified_diff(la, lb, n=0)
                               if l.startswith(("+", "-"))
                               and not l.startswith(("+++", "---"))]
                    self.assertTrue(changed, "form %s: calibrated pair is identical" % form)
                    for line in changed:
                        self.assertIn(marker, line,
                                      "form %s changed a line that is not the verified"
                                      " result: %r" % (form, line))

    def test_no_evaluative_vocabulary_in_any_form(self):
        banned = ("faulty", "unreliable", "trust", "distrust", "lie", "lying", "broken",
                  "malfunction", "deceiv", "honest", "invert", "corrupt", "reliab")
        for fam in CORPUS[:6]:
            for ep in fam.episodes:
                for form in P.FORMS:
                    text = P.render(ep.visible, form).lower()
                    for w in banned:
                        self.assertNotIn(w, text, "%r in %s form %s"
                                         % (w, ep.episode_id, form))

    def test_every_form_states_the_five_facts_the_answer_key_assumes(self):
        """A form missing one of these is a different task, not a rephrasing."""
        vis = CORPUS[0].by("world", True, H.TRANSFER_Q).visible
        needles = [
            ("exactly one pays", ("worth 1 point", "worth the point", "carries a point",
                                  "pays 1 point", "worth a point")),
            ("accurate before the change", ("correctly", "accurate", "called straight",
                                            "accurately")),
            ("the four possibilities", ("neither", )),
            ("persistence", ("continues", "keeps", "for good", "keeps doing")),
            ("a new pair has its own values", ("different pair", "new pair", "fresh pair",
                                               "another pair")),
        ]
        for form in P.FORMS:
            text = P.render(vis, form).lower()
            for name, alts in needles:
                self.assertTrue(any(a in text for a in alts),
                                "form %s never states: %s" % (form, name))

    def test_option_order_is_split_two_and_two(self):
        vis = CORPUS[0].by("world", True, H.TRANSFER_Q).visible
        given = vis.question.options
        same = [f for f in P.FORMS if P.question_options(vis, f) == given]
        self.assertEqual(len(same), 2, "order must split 2/2 so a position-biased model"
                                       " disagrees with itself: %r" % (same,))


class AmbiguityTests(unittest.TestCase):
    """Sanity check 2: ambiguous histories stay actually ambiguous."""

    def test_no_calibration_is_ambiguous_everywhere(self):
        for fam in CORPUS:
            for ep in fam.episodes:
                if ep.calibrated:
                    continue
                ans = solve_ep(ep)
                self.assertEqual(ans.status, O.AMBIGUOUS,
                                 "%s should be unanswerable, got %s"
                                 % (ep.episode_id, ans.status))

    def test_ambiguity_is_realised_by_a_built_world_not_by_a_weak_oracle(self):
        """The strong form. For each unanswerable episode, the rival hypothesis that
        disagrees with the truth is itself the ground truth of the PAIRED episode, whose
        prompt is the same string. Ambiguity here is two worlds we actually constructed
        looking identical, not the solver failing to be clever."""
        for fam in CORPUS:
            for a, b in H.MATCHED_PAIRS:
                for qkind in (H.MAIN_Q, H.TRANSFER_Q):
                    ea, eb = fam.by(a, False, qkind), fam.by(b, False, qkind)
                    surv = {h for h in solve_ep(ea).survivors}
                    self.assertIn(ea.hidden.as_hypothesis(), surv)
                    self.assertIn(eb.hidden.as_hypothesis(), surv)
                    self.assertNotEqual(true_winner(ea), true_winner(eb),
                                        "pair %s/%s has the same correct answer, so it is"
                                        " not a real ambiguity" % (a, b))

    def test_no_feedback_control_is_ambiguous(self):
        for fam in CORPUS[:8]:
            for ep in fam.episodes:
                ans = solve_ep(H.without_feedback(ep))
                self.assertEqual(ans.status, O.AMBIGUOUS, ep.episode_id)


class TransferSignalTests(unittest.TestCase):
    """Sanity check 3: the transfer question has enough signal to be answerable."""

    def test_calibrated_transfer_is_definite(self):
        for fam in CORPUS:
            for cell in ("stable", "world", "reporter", "both"):
                ep = fam.by(cell, True, H.TRANSFER_Q)
                ans = solve_ep(ep)
                self.assertEqual(ans.status, O.DEFINITE, ep.episode_id)
                self.assertEqual(ans.action, true_winner(ep))

    def test_calibration_flips_the_transfer_answer_within_a_pair(self):
        """The measurement itself. Identical transfer evidence, identical everything but
        one verified result, opposite correct answers."""
        for fam in CORPUS:
            for a, b in H.MATCHED_PAIRS:
                ea, eb = fam.by(a, True, H.TRANSFER_Q), fam.by(b, True, H.TRANSFER_Q)
                sa, sb = solve_ep(ea), solve_ep(eb)
                self.assertEqual(sa.status, O.DEFINITE)
                self.assertEqual(sb.status, O.DEFINITE)
                self.assertNotEqual(sa.action, sb.action, "%s/%s did not flip" % (a, b))

    def test_transfer_reports_are_pinned_across_all_four_cells(self):
        """If the transfer reports themselves moved with the cell, the transfer result
        would be explained by the new evidence rather than by the carried-over judgement."""
        for fam in CORPUS:
            seen = set()
            for cell in ("stable", "world", "reporter", "both"):
                ep = fam.by(cell, True, H.TRANSFER_Q)
                seen.add(tuple((t.action, t.report) for t in ep.visible.trials
                               if t.phase == TRANSFER))
            self.assertEqual(len(seen), 1, "transfer evidence moved with the cell")

    def test_transfer_answer_tracks_the_source_state_only(self):
        for fam in CORPUS:
            for cell in ("stable", "world", "reporter", "both"):
                ep = fam.by(cell, True, H.TRANSFER_Q)
                inverted = ep.hidden.as_hypothesis().state_of(fam.source) == INVERTED
                expected = ([o for o in fam.transfer_options
                             if o != fam.transfer_reported_winner][0]
                            if inverted else fam.transfer_reported_winner)
                self.assertEqual(solve_ep(ep).action, expected, ep.episode_id)


class OracleTests(unittest.TestCase):

    def test_truth_is_never_eliminated(self):
        for fam in CORPUS:
            for ep in fam.episodes:
                trials, audits = H.observations(ep)
                self.assertTrue(
                    O.consistent(ep.hidden.as_hypothesis(), trials, audits),
                    "oracle eliminated the true world in %s" % ep.episode_id)

    def test_soundness_definite_answers_are_correct(self):
        for fam in CORPUS:
            for ep in fam.episodes:
                ans = solve_ep(ep)
                if ans.is_definite:
                    self.assertEqual(ans.action, true_winner(ep), ep.episode_id)

    def test_source_verdict_matches_ground_truth_when_calibrated(self):
        for fam in CORPUS:
            for cell in ("stable", "world", "reporter", "both"):
                ep = fam.by(cell, True, H.MAIN_Q)
                trials, audits = H.observations(ep)
                surv = O.survivors(ep.visible.main_options, ep.visible.sources,
                                   trials, audits, ep.hidden.transfer.options)
                self.assertEqual(O.source_verdict(surv, fam.source),
                                 ep.hidden.as_hypothesis().state_of(fam.source))

    def test_hand_written_expectation_table(self):
        """Sixteen cases written out by hand from the design, not derived from the code
        under test. If the implementation and this table ever disagree, one of them is
        wrong and the argument has to be redone on paper."""
        fam = CORPUS[0]
        pre_w = fam.main_options[0] if fam.by("stable", True, H.MAIN_Q).hidden.pre.winner \
            == fam.main_options[0] else fam.main_options[1]
        other_main = [o for o in fam.main_options if o != pre_w][0]
        rep_w = fam.transfer_reported_winner
        other_tr = [o for o in fam.transfer_options if o != rep_w][0]
        table = {
            # (cell, calibrated, question): (status, expected answer)
            ("stable", False, H.MAIN_Q): (O.AMBIGUOUS, None),
            ("world", False, H.MAIN_Q): (O.AMBIGUOUS, None),
            ("reporter", False, H.MAIN_Q): (O.AMBIGUOUS, None),
            ("both", False, H.MAIN_Q): (O.AMBIGUOUS, None),
            ("stable", False, H.TRANSFER_Q): (O.AMBIGUOUS, None),
            ("world", False, H.TRANSFER_Q): (O.AMBIGUOUS, None),
            ("reporter", False, H.TRANSFER_Q): (O.AMBIGUOUS, None),
            ("both", False, H.TRANSFER_Q): (O.AMBIGUOUS, None),
            ("stable", True, H.MAIN_Q): (O.DEFINITE, pre_w),
            ("world", True, H.MAIN_Q): (O.DEFINITE, other_main),
            ("reporter", True, H.MAIN_Q): (O.DEFINITE, pre_w),
            ("both", True, H.MAIN_Q): (O.DEFINITE, other_main),
            ("stable", True, H.TRANSFER_Q): (O.DEFINITE, rep_w),
            ("world", True, H.TRANSFER_Q): (O.DEFINITE, rep_w),
            ("reporter", True, H.TRANSFER_Q): (O.DEFINITE, other_tr),
            ("both", True, H.TRANSFER_Q): (O.DEFINITE, other_tr),
        }
        self.assertEqual(len(table), 16)
        for (cell, cal, qkind), (status, action) in table.items():
            ans = solve_ep(fam.by(cell, cal, qkind))
            self.assertEqual(ans.status, status, (cell, cal, qkind))
            self.assertEqual(ans.action, action, (cell, cal, qkind))


class CounterbalanceTests(unittest.TestCase):

    def test_audit_value_carries_no_information_about_the_cause(self):
        """Without this, a model could answer by reading the verified NUMBER rather than
        by noticing whether it agrees with the report, and we would score that as
        reliability learning."""
        by_state = collections.defaultdict(collections.Counter)
        for fam in CORPUS:
            for cell in ("stable", "world", "reporter", "both"):
                ep = fam.by(cell, True, H.MAIN_Q)
                state = ep.hidden.as_hypothesis().state_of(fam.source)
                by_state[state][ep.visible.audits[0].verified_outcome] += 1
        self.assertEqual(dict(by_state[ACCURATE]), dict(by_state[INVERTED]),
                         "verified value is correlated with the true source state")

    def test_audit_value_carries_no_information_about_the_answer(self):
        by_answer = collections.defaultdict(collections.Counter)
        for fam in CORPUS:
            for cell in ("stable", "world", "reporter", "both"):
                ep = fam.by(cell, True, H.TRANSFER_Q)
                flipped = (solve_ep(ep).action != fam.transfer_reported_winner)
                by_answer[flipped][ep.visible.audits[0].verified_outcome] += 1
        self.assertEqual(dict(by_answer[True]), dict(by_answer[False]))

    def test_factors_are_fully_crossed(self):
        counts = collections.Counter(
            (f.audit_target_index, f.audit_position,
             f.transfer_reported_winner == f.transfer_options[0],
             f.by("stable", False, H.MAIN_Q).hidden.pre.winner == f.main_options[0])
            for f in CORPUS)
        self.assertEqual(len(counts), 16)
        self.assertEqual(set(counts.values()), {len(CORPUS) // 16})

    def test_option_order_and_labels_vary(self):
        self.assertGreater(len({f.main_options for f in CORPUS}), 4)
        self.assertGreater(len({f.source for f in CORPUS}), 1)


class ControlTests(unittest.TestCase):

    def test_disclosed_cause_states_the_actual_truth(self):
        """The upper bound only works if what it discloses is true. Asserting merely that
        a disclosure line appears would pass on a disclosure that says the opposite, and
        that would make the comprehension gate silently wrong in the one direction nobody
        would check."""
        for fam in CORPUS[:8]:
            for cell in ("stable", "world", "reporter", "both"):
                for qkind in (H.MAIN_Q, H.TRANSFER_Q):
                    base = fam.by(cell, False, qkind)
                    ep = H.disclosed_cause(base)
                    text = P.render(ep.visible)
                    moved = base.hidden.pre.winner != base.hidden.post.winner
                    inverted = (base.hidden.as_hypothesis().state_of(fam.source)
                                == INVERTED)
                    self.assertIn("were swapped" if moved else "were not changed", text)
                    self.assertIn("began reporting" if inverted
                                  else "did not begin reporting", text)

    def test_disclosed_payouts_names_the_true_winner_and_refuses_transfer(self):
        ep = H.disclosed_payouts(CORPUS[0].by("world", False, H.MAIN_Q))
        self.assertIn("%s pays 1 point" % ep.hidden.post.winner, P.render(ep.visible))
        with self.assertRaises(ValueError):
            H.disclosed_payouts(CORPUS[0].by("world", False, H.TRANSFER_Q))

    def test_no_feedback_hides_reports_and_the_verified_record_together(self):
        """A control whose prompt shows a fact its own answer key withholds is worse than
        no control at all."""
        for cal in (False, True):
            ep = H.without_feedback(CORPUS[0].by("world", cal, H.MAIN_Q))
            text = P.render(ep.visible)
            self.assertIn("No result was reported", text)
            self.assertNotIn("Verified records", text)
            self.assertEqual(ep.visible.audits, ())
            trials, audits = H.observations(ep)
            self.assertEqual(audits, ())
            self.assertEqual(solve_ep(ep).status, O.AMBIGUOUS)


class SourceQuestionTests(unittest.TestCase):
    """The attribution question (2026-09-10): ask what happened to the source instead of
    which option wins. Every invariant the winner question had must survive the swap."""

    LABELS = ("K", "M")

    def _srcq(self, fam, cell, calibrated):
        used = set(fam.main_options) | set(fam.transfer_options)
        free = [l for l in H.LABEL_POOL if l not in used]
        return H.as_source_question(fam.by(cell, calibrated, H.MAIN_Q),
                                    (free[0], free[1]), free[0])

    def test_uncalibrated_pairs_stay_byte_identical_in_every_form(self):
        for fam in CORPUS[:12]:
            for form in P.FORMS:
                for a, b in H.MATCHED_PAIRS:
                    self.assertEqual(P.render(self._srcq(fam, a, False).visible, form),
                                     P.render(self._srcq(fam, b, False).visible, form),
                                     "%s/%s leaked via the source question, form %s"
                                     % (a, b, form))

    def test_uncalibrated_oracle_is_ambiguous_in_every_cell(self):
        import oracle as O
        for fam in CORPUS[:12]:
            for cell in ("stable", "world", "reporter", "both"):
                ep = self._srcq(fam, cell, False)
                trials, audits = H.observations(ep)
                surv = O.survivors(ep.visible.main_options, ep.visible.sources, trials,
                                   audits, ep.visible.transfer_options)
                self.assertEqual(O.source_verdict(surv, ep.visible.question.source),
                                 O.AMBIGUOUS, cell)

    def test_calibrated_oracle_is_definite_and_true_in_every_cell(self):
        import oracle as O
        for fam in CORPUS[:12]:
            for cell in ("stable", "world", "reporter", "both"):
                ep = self._srcq(fam, cell, True)
                trials, audits = H.observations(ep)
                surv = O.survivors(ep.visible.main_options, ep.visible.sources, trials,
                                   audits, ep.visible.transfer_options)
                verdict = O.source_verdict(surv, ep.visible.question.source)
                truth = ep.hidden.as_hypothesis().state_of(ep.visible.question.source)
                self.assertEqual(verdict, truth, cell)

    def test_source_labels_never_collide_with_option_labels(self):
        for fam in CORPUS[:12]:
            with self.assertRaises(ValueError):
                H.as_source_question(fam.by("stable", False, H.MAIN_Q),
                                     fam.main_options, fam.main_options[0])

    def test_answer_labels_follow_form_order(self):
        fam = CORPUS[0]
        ep = self._srcq(fam, "stable", False)
        fwd = P.question_options(ep.visible, "log")
        rev = P.question_options(ep.visible, "prose")
        self.assertEqual(fwd, tuple(reversed(rev)))
        for form in P.FORMS:
            text = P.render(ep.visible, form)
            for lab in ep.visible.source_query["labels"]:
                self.assertIn(lab, text.split("Answer:")[0].splitlines()[-2:][0]
                              + text.split("Answer:")[0].splitlines()[-1], form)


class FloorTests(unittest.TestCase):
    """Floors, and the difference between a comprehension test and a conflict test
    (2026-09-11). The old main-question floor measures 0.531 at 32B in the inverted cells
    while the source floor measures 0.93-0.96. That gap is not a broken harness; it is the
    old floor printing a payout table that the visible reports contradict. These tests pin
    the distinction so nobody reads a ceiling against the wrong baseline again."""

    def test_transfer_payout_disclosure_refuses_main_questions(self):
        for fam in CORPUS[:8]:
            with self.assertRaises(ValueError):
                H.disclosed_transfer_payouts(fam.by("world", True, H.MAIN_Q))

    def test_transfer_payout_disclosure_states_the_true_transfer_winner(self):
        for fam in CORPUS[:12]:
            for cell in ("stable", "world", "reporter", "both"):
                ep = H.disclosed_transfer_payouts(fam.by(cell, True, H.TRANSFER_Q))
                self.assertEqual(ep.visible.payout_disclosure["winner"],
                                 ep.hidden.transfer.winner, cell)

    def test_reading_floor_leaves_nothing_to_contradict_the_disclosure(self):
        """The point of the honest floor: after the change round every report is gone, so
        the disclosure is the only claim in the prompt about the phase being asked."""
        for fam in CORPUS[:12]:
            for cell in ("stable", "world", "reporter", "both"):
                ep = H.reading_floor(fam.by(cell, True, H.TRANSFER_Q))
                self.assertEqual(ep.visible.audits, ())
                trials, audits = H.observations(ep)
                self.assertTrue(all(t.phase == PRE for t in trials), cell)
                text = P.render(ep.visible, "log")
                for line in text.splitlines():
                    if line.startswith(("Round 5", "Round 6", "Round 7", "Round 8")):
                        self.assertIn("No result was reported", line, cell)

    def test_the_old_main_floor_really_does_contradict_itself(self):
        """The receipt for why reading_floor had to exist. On an inverted cell the printed
        table names one winner and the visible reports name the other, so the model is
        asked to resolve a conflict, not to read."""
        for fam in CORPUS[:12]:
            for cell in ("reporter", "both"):
                ep = H.disclosed_payouts(fam.by(cell, False, H.MAIN_Q))
                stated = ep.visible.payout_disclosure["winner"]
                post = [t for t in ep.visible.trials if t.phase == POST]
                reported_winner = next(t.action for t in post if t.report == 1)
                self.assertNotEqual(stated, reported_winner,
                                    "%s/%s was expected to conflict and did not"
                                    % (fam.family_id, cell))

    def test_the_honest_floor_does_not_contradict_itself_in_any_cell(self):
        for fam in CORPUS[:12]:
            for cell in ("stable", "world", "reporter", "both"):
                ep = H.reading_floor(fam.by(cell, False, H.TRANSFER_Q))
                trials, _ = H.observations(ep)
                self.assertFalse([t for t in trials if t.phase != PRE], cell)


class PairedQuestionTests(unittest.TestCase):
    """THE invariant of the paired measure (2026-09-11). One episode, two questions, and
    everything the model reads before the question line must be the same bytes.

    Why this has to be a test and not a comment. The claim the paired design licenses is
    "the model states the channel is corrupted and still acts on the corrupted reading".
    That claim dies the moment the two arms differ anywhere upstream of the question,
    because then the gap could be a reading difference. The property holds today only
    because render() appends the question block last and as_source_question() touches
    nothing but source_query and episode_id. Both are one refactor away from silently
    breaking, and nothing would fail.
    """

    def _pair(self, fam, cell, qkind, calibrated=True):
        """The winner and source renderings of ONE episode. Labels are drawn disjoint from
        every option label by hand: as_source_question's own check reads transfer_options,
        which is None on a MAIN episode, so it cannot exclude them there."""
        base = fam.by(cell, calibrated, qkind)
        used = set(fam.main_options) | set(fam.transfer_options)
        free = [l for l in H.LABEL_POOL if l not in used]
        return base, H.as_source_question(base, (free[0], free[1]), free[0])

    def test_both_questions_share_every_byte_before_the_question_block(self):
        for fam in CORPUS[:12]:
            for qkind in (H.MAIN_Q, H.TRANSFER_Q):
                for cell in ("stable", "world", "reporter", "both"):
                    win, src = self._pair(fam, cell, qkind)
                    for form in P.FORMS:
                        a, b = P.render(win.visible, form), P.render(src.visible, form)
                        self.assertEqual(a.rsplit("\n\n", 1)[0], b.rsplit("\n\n", 1)[0],
                                         "prefix diverged: %s %s %s form=%s"
                                         % (fam.family_id, qkind, cell, form))

    def test_the_two_questions_actually_differ(self):
        """Guards the test above against passing vacuously. If someone made the source
        question render the winner question, the prefix check would still be green."""
        for fam in CORPUS[:12]:
            for qkind in (H.MAIN_Q, H.TRANSFER_Q):
                win, src = self._pair(fam, "reporter", qkind)
                for form in P.FORMS:
                    a, b = P.render(win.visible, form), P.render(src.visible, form)
                    self.assertNotEqual(a.rsplit("\n\n", 1)[1], b.rsplit("\n\n", 1)[1],
                                        "%s %s form=%s" % (fam.family_id, qkind, form))

    def test_answer_alphabets_are_disjoint_across_the_pair(self):
        """A model must not be able to score on one arm by emitting the other arm's letter.
        Without this, a constant-label policy could look like a dissociation."""
        for fam in CORPUS[:12]:
            for qkind in (H.MAIN_Q, H.TRANSFER_Q):
                win, src = self._pair(fam, "world", qkind)
                for form in P.FORMS:
                    self.assertFalse(set(P.question_options(win.visible, form))
                                     & set(P.question_options(src.visible, form)),
                                     "%s %s form=%s" % (fam.family_id, qkind, form))

    def test_transfer_substrate_has_no_audit_shortcut(self):
        """The reason the paired measure moved off MAIN histories on 2026-09-11. On a MAIN
        history the audited round uses the same two options the winner question asks about,
        so the audit line alone names the winner (measured: 64/64) and the source never
        enters. On a TRANSFER history the audit is on a main-option round and the question
        is about a fresh pair, so the only route to the answer runs through the source."""
        for fam in CORPUS[:12]:
            for cell in ("stable", "world", "reporter", "both"):
                ep = fam.by(cell, True, H.TRANSFER_Q)
                self.assertTrue(ep.visible.audits, "transfer item lost its audit")
                for a in ep.visible.audits:
                    self.assertNotIn(a.action, ep.visible.question.options,
                                     "audited action %s is an option of the transfer "
                                     "question in %s/%s" % (a.action, fam.family_id, cell))


class RestatedReportTests(unittest.TestCase):
    """The one-line diagnostic (2026-09-10): the audited round's report printed beside the
    verified result. It must add nothing a matched pair does not already share."""

    def _cal(self, fam, cell):
        used = set(fam.main_options) | set(fam.transfer_options)
        free = [l for l in H.LABEL_POOL if l not in used]
        return H.as_source_question(H.restated_report(fam.by(cell, True, H.MAIN_Q)),
                                    (free[0], free[1]), free[0])

    def test_refuses_uncalibrated(self):
        with self.assertRaises(ValueError):
            H.restated_report(CORPUS[0].by("stable", False, H.MAIN_Q))

    def test_restated_pairs_differ_only_on_the_audit_line_in_every_form(self):
        for fam in CORPUS[:12]:
            for form in P.FORMS:
                marker = P.AUDIT_MARKER[form]
                for a, b in H.MATCHED_PAIRS:
                    la = P.render(self._cal(fam, a).visible, form).splitlines()
                    lb = P.render(self._cal(fam, b).visible, form).splitlines()
                    changed = [l for l in difflib.unified_diff(la, lb, n=0)
                               if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
                    self.assertTrue(changed, "%s: restated pair identical" % form)
                    for line in changed:
                        self.assertIn(marker, line, "%s changed a non-audit line: %r" % (form, line))

    def test_restated_line_carries_the_true_report_for_that_round(self):
        for fam in CORPUS[:12]:
            ep = self._cal(fam, "reporter")
            a = ep.visible.audits[0]
            rep = next(t for t in ep.visible.trials if t.round == a.round).report
            text = P.render(ep.visible, "log")
            line = next(l for l in text.splitlines() if l.startswith("Round %d - actual" % a.round))
            self.assertIn("had reported %s" % P._points(rep), line)
            # and on a reporter cell the restated report must DISAGREE with the verified result
            self.assertNotEqual(rep, a.verified_outcome)


# This block MUST stay at the bottom of the file. It sat at line 379 from 2026-09-10 until
# 2026-09-11, above SourceQuestionTests and RestatedReportTests, so running this file directly
# collected 27 of 35 tests and silently skipped the eight that guard the attribution design.
# `unittest discover` imported the module and ran all of them, which is why the battery looked
# green the whole time. A test that does not run is not a test.
if __name__ == "__main__":
    unittest.main(verbosity=2)
