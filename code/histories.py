#!/usr/bin/env python3
"""Paired-history generator.

A family is one fully counterbalanced scenario. Inside it, the four design cells produce
histories that pair off two by two: `world` matches `reporter`, and `stable` matches
`both`. Matched means the visible record is the same record — same rounds, same actions,
same numbers — so a prompt built from it cannot carry the hidden cause. The leak test
proves this by comparing rendered bytes rather than by scanning for suspicious words.

The transfer reports are pinned, not emitted freely: we fix WHICH new option gets the
point in the report, then choose the hidden transfer payouts so that the source's state
produces exactly that report. Same evidence in front of the model, opposite correct
answer depending on what calibration established about the source. That is the whole
transfer measurement, and pinning is what makes it a measurement instead of a confound.
"""
from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

from world import (ACCURATE, INVERTED, POST, PRE, TRANSFER, Audit, HiddenWorld, Payouts,
                   Question, Trial, cell_of)

# Arbitrary option labels, all validated as single distinct tokens in the real Qwen2.5
# vocabulary under both the chat and raw-completion conventions (check_answer_tokens.py).
# "A" is out because "option A" carries an ordering pull; "F", "T", "Y" and "N" are out
# because they read as false/true/yes/no next to a binary result. Sixteen rather than
# eight so that two families rarely draw the same four labels: identical prompts in
# different families are dependent samples that the family-level bootstrap would count
# as independent.
LABEL_POOL = ("K", "M", "Q", "R", "W", "Z", "H", "G",
              "J", "L", "P", "V", "X", "B", "C", "D")
SOURCE_POOL = ("Source 3", "Source 6", "Source 7", "Source 9")

ACQUISITION_ROUNDS = 4      # two observations of each main option
CHANGE_ROUND = ACQUISITION_ROUNDS
MAIN_Q = "main"
TRANSFER_Q = "transfer"


@dataclass(frozen=True)
class VisibleEpisode:
    """Everything the renderer is allowed to see. There is deliberately no hidden field
    and no cell label on this type: prompts.py takes this and only this, so a leak would
    have to be a deliberate code change rather than an oversight."""

    main_options: Tuple[str, str]        # in question order
    transfer_options: Optional[Tuple[str, str]]
    sources: Tuple[str, ...]
    trials: Tuple[Trial, ...]            # in render order
    audits: Tuple[Audit, ...]
    audit_position: str                  # "before" | "after"
    question: Question
    change_round: int
    # Control disclosures travel as FACTS, not as sentences. The first four-form check put
    # one prose sentence into all four registers, and the two forms it clashed with dropped
    # to chance on a question whose answer was printed in the prompt. Wording belongs to the
    # renderer; the generator owns only what is true.
    disclosure: Optional[dict] = None
    payout_disclosure: Optional[dict] = None
    withhold_reports: bool = False       # the no-feedback control
    # The attribution question. When set, the renderer asks "is this source reporting
    # accurately or the opposite" with these two labels instead of "which option wins".
    # {"labels": (a, b) in question order, "accurate": a-or-b, "opposite": the other}.
    # Added 2026-09-10 after the ladder showed the winner question makes the model do two
    # jobs (decide the source's state, then apply it to a report and flip) and only the
    # first is the research question. This asks the first job alone: one judgment, no
    # inversion step, so it fits a one-token answer.
    source_query: Optional[dict] = None
    # Diagnostic (2026-09-10): print what the source reported on the audited round on the
    # SAME line as the verified result, so the comparison needs no lookup. It lifts the 32B
    # from 0.637 to 0.848. The report is already in the visible log, so restating it adds no
    # information and cannot leak.
    #
    # CORRECTED 2026-09-11. This field's comment used to claim the lift showed "the cross-
    # reference from record to log line is what fails, not the compare". The data refutes
    # that. If the cross-reference were the bottleneck, removing it would collapse the
    # round-5-versus-round-6 gap. It does not: 32B goes +0.133 -> +0.117 and 72B +0.274 ->
    # +0.257. Restatement lifts both rounds by about the same amount, so whatever the round
    # moderator is, this variant does not remove it and we do not yet have a name for it.
    audit_with_report: bool = False
    # The salience intervention (2026-09-11, Arman's go). One rules line saying the two
    # causes occur about equally often, naming neither for this episode.
    #
    # It exists to separate two explanations of the `world` cell, where the payouts moved
    # and the source is honest and everything collapses (72B 0.422, WORSE than 32B's 0.672,
    # while both are perfect where the source really did lie). Either the model holds a
    # prior that a surprising report means a lying reporter, and evidence has to overcome
    # it, or it never entertains "the world changed" as a candidate at all. The first is a
    # prior that salience can move; the second is not. If the `world` cell lifts under this
    # and nothing else changes, it was a prior.
    #
    # The statement is TRUE of the generator: cells are crossed 2x2, so across the corpus
    # the payouts swap on half the episodes and a source starts inverting on half. It is
    # identical in every cell, so matched pairs stay byte-identical and it cannot leak.
    world_prior: bool = False


@dataclass(frozen=True)
class Episode:
    episode_id: str
    family_id: str
    arm: str
    cell: str
    calibrated: bool
    question_kind: str
    audit_target: Optional[str]
    visible: VisibleEpisode
    hidden: HiddenWorld


@dataclass(frozen=True)
class Family:
    family_id: str
    main_options: Tuple[str, str]
    transfer_options: Tuple[str, str]
    source: str
    audit_target_index: int
    audit_position: str
    transfer_reported_winner: str
    episodes: Tuple[Episode, ...]

    def by(self, cell: str, calibrated: bool, question_kind: str) -> Episode:
        for ep in self.episodes:
            if (ep.cell == cell and ep.calibrated == calibrated
                    and ep.question_kind == question_kind):
                return ep
        raise KeyError((cell, calibrated, question_kind))


# The two matched pairs. Members of a pair must render byte-identically without calibration.
MATCHED_PAIRS = (("world", "reporter"), ("stable", "both"))


def _family_id(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:10]


def _build_hidden(main_options: Tuple[str, str], pre_winner: str, cell: str,
                  transfer_options: Tuple[str, str], transfer_reported_winner: str,
                  source: str) -> HiddenWorld:
    other = [o for o in main_options if o != pre_winner][0]
    payouts_moved = cell in ("world", "both")
    faulty = cell in ("reporter", "both")
    pre = Payouts(main_options, pre_winner)
    post = Payouts(main_options, other if payouts_moved else pre_winner)
    state = INVERTED if faulty else ACCURATE
    # Pin the transfer REPORT, then back out the truth that produces it.
    if state == ACCURATE:
        true_transfer_winner = transfer_reported_winner
    else:
        true_transfer_winner = [o for o in transfer_options
                                if o != transfer_reported_winner][0]
    return HiddenWorld(pre=pre, post=post,
                       source_states=frozenset({(source, state)}),
                       transfer=Payouts(transfer_options, true_transfer_winner),
                       cell=cell_of(pre, post, faulty))


def _trials_for(hidden: HiddenWorld, main_options: Tuple[str, str],
                transfer_options: Tuple[str, str], source: str,
                include_transfer: bool) -> Tuple[Trial, ...]:
    trials: List[Trial] = []
    r = 1
    for _ in range(ACQUISITION_ROUNDS // 2):
        for opt in main_options:
            trials.append(hidden.emit(r, PRE, opt, source))
            r += 1
    for opt in main_options:
        trials.append(hidden.emit(r, POST, opt, source))
        r += 1
    if include_transfer:
        for opt in transfer_options:
            trials.append(hidden.emit(r, TRANSFER, opt, source))
            r += 1
    return tuple(trials)


def _post_trials(trials: Sequence[Trial]) -> List[Trial]:
    return [t for t in trials if t.phase == POST]


def build_family(rng: random.Random, index: int = 0, arm: str = "single") -> Family:
    """Labels are sampled; the four counterbalance factors are CROSSED on the family
    index, not sampled. Random assignment would only balance in expectation, and the
    audit-value neutrality claim needs to hold exactly: if verified values were even
    slightly correlated with the true cause, a model could read the number instead of
    the agreement and we would score that as reliability learning."""
    labels = rng.sample(LABEL_POOL, 4)
    main_options = (labels[0], labels[1])
    transfer_options = (labels[2], labels[3])
    source = SOURCE_POOL[rng.randrange(len(SOURCE_POOL))]
    audit_target_index = index % 2
    audit_position = ("before", "after")[(index // 2) % 2]
    transfer_reported_winner = transfer_options[(index // 4) % 2]
    pre_winner = main_options[(index // 8) % 2]
    fid = _family_id(main_options, transfer_options, source, pre_winner,
                     audit_target_index, audit_position, transfer_reported_winner)

    episodes: List[Episode] = []
    for cell, calibrated, qkind in itertools.product(
            ("stable", "world", "reporter", "both"), (False, True), (MAIN_Q, TRANSFER_Q)):
        hidden = _build_hidden(main_options, pre_winner, cell, transfer_options,
                               transfer_reported_winner, source)
        include_transfer = qkind == TRANSFER_Q
        trials = _trials_for(hidden, main_options, transfer_options, source,
                             include_transfer)
        audits: Tuple[Audit, ...] = ()
        audit_target = None
        if calibrated:
            target = _post_trials(trials)[audit_target_index]
            audits = (hidden.audit(target),)
            audit_target = target.action
        question = Question(
            phase=TRANSFER if qkind == TRANSFER_Q else POST,
            options=transfer_options if qkind == TRANSFER_Q else main_options,
            source=source)
        visible = VisibleEpisode(
            main_options=main_options,
            transfer_options=transfer_options if include_transfer else None,
            sources=(source,),
            trials=trials, audits=audits, audit_position=audit_position,
            question=question, change_round=CHANGE_ROUND)
        episodes.append(Episode(
            episode_id="%s-%s-%s-%s" % (fid, cell, "cal" if calibrated else "nocal", qkind),
            family_id=fid, arm=arm, cell=cell, calibrated=calibrated,
            question_kind=qkind, audit_target=audit_target,
            visible=visible, hidden=hidden))

    return Family(family_id=fid, main_options=main_options,
                  transfer_options=transfer_options, source=source,
                  audit_target_index=audit_target_index, audit_position=audit_position,
                  transfer_reported_winner=transfer_reported_winner,
                  episodes=tuple(episodes))


def generate(n: int, seed: int = 0) -> List[Family]:
    """n should be a multiple of 16 for the counterbalance to close exactly."""
    rng = random.Random(seed)
    return [build_family(rng, index=i) for i in range(n)]


# ---------------------------------------------------------------------------
# Control variants. These are applied to an existing episode so the base history is
# held fixed and only the manipulation moves. Adding them now rather than after the
# prompts are frozen is deliberate: a control bolted on later means two prompt versions.

def _state_of(hidden: HiddenWorld, source: str) -> str:
    return hidden.as_hypothesis().state_of(source)


def disclosed_cause(ep: Episode) -> Episode:
    """Upper-bound control: state the true cause outright. Never counted as inferred
    learning; it measures whether the model can use the fact once it is handed over."""
    source = ep.visible.question.source
    return replace(ep,
                   episode_id=ep.episode_id + "-disclosecause",
                   visible=replace(ep.visible, disclosure={
                       "source": source,
                       "inverting": _state_of(ep.hidden, source) == INVERTED,
                       "moved": ep.hidden.pre.winner != ep.hidden.post.winner,
                       "round": ep.visible.change_round}))


def disclosed_payouts(ep: Episode) -> Episode:
    """Comprehension control: hand over the current payout table directly. MAIN questions
    only. The table covers the two original options, so on a transfer question it does not
    contain the answer and would not be an upper bound at all."""
    if ep.question_kind != MAIN_Q:
        raise ValueError("disclosed_payouts is only an upper bound on the main question")
    post = ep.hidden.post
    return replace(ep,
                   episode_id=ep.episode_id + "-disclosepayouts",
                   visible=replace(ep.visible, payout_disclosure={
                       "winner": post.winner,
                       "loser": [o for o in post.options if o != post.winner][0],
                       "round": ep.visible.change_round}))


def disclosed_transfer_payouts(ep: Episode) -> Episode:
    """The transfer twin of disclosed_payouts, added 2026-09-11. Hands over the NEW pair's
    payout table, which is the only table that contains the transfer question's answer.

    Why it did not exist until now: the paired measure used to live on main histories, and
    disclosed_payouts covers the two original options, so it raises on a transfer question
    rather than pretending to be an upper bound. The consequence was that the transfer
    winner question had no floor at all, and a ceiling with no floor beside it is a number
    we are not allowed to read."""
    if ep.question_kind != TRANSFER_Q:
        raise ValueError("disclosed_transfer_payouts is only an upper bound on the "
                         "transfer question")
    tr = ep.hidden.transfer
    if tr is None:
        raise ValueError("episode carries no transfer regime")
    rounds = [t.round for t in ep.visible.trials if t.phase == TRANSFER]
    return replace(ep,
                   episode_id=ep.episode_id + "-disclosetransfer",
                   visible=replace(ep.visible, payout_disclosure={
                       "winner": tr.winner,
                       "loser": [o for o in tr.options if o != tr.winner][0],
                       "round": (min(rounds) - 1) if rounds else ep.visible.change_round}))


def with_world_prior(ep: Episode) -> Episode:
    """Make the world-change hypothesis as salient as the reporter-fault one, without
    naming either for this episode. See VisibleEpisode.world_prior for why."""
    return replace(ep,
                   episode_id=ep.episode_id + "-worldprior",
                   visible=replace(ep.visible, world_prior=True))


def reading_floor(ep: Episode) -> Episode:
    """An HONEST floor: the answer is printed AND nothing visible contradicts it.

    The distinction this exists to draw (2026-09-11). disclosed_payouts on a main question
    prints the payout table while the reports, in the inverted cells, say the opposite.
    That is a conflict-resolution test wearing a comprehension test's clothes, and it
    measures 0.531 at 32B where the honest floors measure 0.93-0.96. Reading a ceiling
    against it would charge the model twice for the same failure.

    Here without_feedback removes every post-change report first, so the disclosure is the
    only claim in the prompt about the phase being asked. A model below ceiling on THIS is
    a broken harness. A model below ceiling on the conflicted one is a result."""
    stripped = without_feedback(ep)
    if ep.question_kind == TRANSFER_Q:
        return disclosed_transfer_payouts(stripped)
    return disclosed_payouts(stripped)


def without_feedback(ep: Episode) -> Episode:
    """Matched history, reports withheld after the change point. Nothing post-change is
    learnable, so the oracle must go ambiguous; a model that still shows an effect here
    is responding to something other than the evidence.

    The verified record goes with them. Leaving it in would put a fact in the prompt that
    `observations()` withholds from the answer key, and a prompt that disagrees with its
    own key is worse than no control at all."""
    return replace(ep,
                   episode_id=ep.episode_id + "-nofeedback",
                   calibrated=False,
                   visible=replace(ep.visible, withhold_reports=True, audits=()))


SOURCE_Q = "source"


def as_source_question(ep: Episode, labels: Tuple[str, str], accurate: str) -> Episode:
    """Turn any episode into the attribution question about its source. The history is
    untouched, so every leak invariant that held for the winner question holds here: the
    question block depends on the source name and two labels, never on hidden state.

    `labels` must be disjoint from the family's option labels, or a model could answer the
    winner question by reflex and score by accident. `accurate` names which label means
    "reporting accurately"; the builder counterbalances it on the family index."""
    if accurate not in labels:
        raise ValueError("accurate label %r not in %r" % (accurate, labels))
    used = set(ep.visible.main_options) | set(ep.visible.transfer_options or ())
    if set(labels) & used:
        raise ValueError("source labels %r collide with option labels %r" % (labels, used))
    opposite = [l for l in labels if l != accurate][0]
    return replace(ep,
                   episode_id=ep.episode_id + "-srcq",
                   question_kind=SOURCE_Q,
                   visible=replace(ep.visible, source_query={
                       "labels": tuple(labels), "accurate": accurate,
                       "opposite": opposite}))


def restated_report(ep: Episode) -> Episode:
    """Diagnostic variant: the audited round's report is restated beside its verified
    result. Requires a calibrated episode; on an uncalibrated one there is nothing to
    restate and silently returning it unchanged would hide a builder bug."""
    if not ep.visible.audits:
        raise ValueError("restated_report needs a calibrated episode")
    return replace(ep,
                   episode_id=ep.episode_id + "-restated",
                   visible=replace(ep.visible, audit_with_report=True))


def observations(ep: Episode) -> Tuple[Tuple[Trial, ...], Tuple[Audit, ...]]:
    """What the oracle is entitled to condition on, mirroring exactly what the prompt
    shows. The no-feedback control drops post-change reports from both at once, which is
    the point of routing the oracle through this function instead of ep.visible.trials."""
    if ep.visible.withhold_reports:
        trials = tuple(t for t in ep.visible.trials if t.phase == PRE)
        return trials, ()
    return ep.visible.trials, ep.visible.audits
