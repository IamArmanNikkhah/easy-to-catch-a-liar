#!/usr/bin/env python3
"""The toy world: two options, deterministic binary payouts, a source that may invert.

The whole experiment rests on one arithmetic fact, so it is worth stating plainly. A
report is `outcome XOR inverting`. That means a payout swap and a source inversion are
the same bit flip seen from two sides, and a history that shows only reports cannot tell
them apart. Everything else here exists to keep that ambiguity honest.

Naming: hidden state uses internal words ("accurate", "inverted", "cell") that MUST NOT
reach a prompt. The renderer in prompts.py never sees a HiddenWorld, by type.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Optional, Tuple

ACCURATE = "accurate"
INVERTED = "inverted"
SOURCE_STATES = (ACCURATE, INVERTED)

# Phase tags. A trial belongs to exactly one payout regime.
PRE = "pre"            # acquisition, before the change point
POST = "post"          # after the change point, same two options
TRANSFER = "transfer"  # after the change point, two NEW options
PHASES = (PRE, POST, TRANSFER)


@dataclass(frozen=True)
class Payouts:
    """Which of two options pays a point. Exactly one does; that is a stated rule of the
    task, and stating it is what keeps the hypothesis space at two entries per regime."""

    options: Tuple[str, str]
    winner: str

    def __post_init__(self):
        if self.winner not in self.options:
            raise ValueError("winner %r not among options %r" % (self.winner, self.options))
        if len(set(self.options)) != 2:
            raise ValueError("options must be two distinct labels: %r" % (self.options,))

    def outcome(self, action: str) -> int:
        if action not in self.options:
            raise KeyError("action %r not in this regime %r" % (action, self.options))
        return 1 if action == self.winner else 0

    @staticmethod
    def both(options: Tuple[str, str]) -> Tuple["Payouts", "Payouts"]:
        """The two possible configurations over one option pair, in label order."""
        return (Payouts(options, options[0]), Payouts(options, options[1]))


def report_of(outcome: int, source_state: str) -> int:
    """The one line the whole design turns on."""
    if source_state == ACCURATE:
        return outcome
    if source_state == INVERTED:
        return 1 - outcome
    raise ValueError("unknown source state %r" % (source_state,))


@dataclass(frozen=True)
class Trial:
    """A visible record: on round r the agent chose `action`, and `source` said `report`."""

    round: int
    phase: str
    action: str
    source: str
    report: int


@dataclass(frozen=True)
class Audit:
    """The calibration record. An independent check of one round's ACTUAL result, sitting
    next to what the source said about that same round. Neutral by construction: it never
    characterises the source, it just puts two numbers side by side."""

    round: int
    phase: str
    action: str
    source: str
    verified_outcome: int


@dataclass(frozen=True)
class Question:
    """What we ask the model to choose between. `phase` fixes which payout regime is at
    stake, which is what makes the transfer question a different question and not a
    rephrasing of the main one."""

    phase: str
    options: Tuple[str, str]        # rendered in this order
    source: str                     # whose reports bear on it


@dataclass(frozen=True)
class Hypothesis:
    """One complete guess at the unknowns. Deliberately generic over sources and over the
    transfer regime: the specificity control and the transfer question then run through
    the same oracle instead of getting special cases."""

    pre: Payouts
    post: Payouts
    source_states: FrozenSet[Tuple[str, str]]   # (source label, state after the change)
    transfer: Optional[Payouts] = None

    def state_of(self, source: str) -> str:
        for label, state in self.source_states:
            if label == source:
                return state
        raise KeyError("no state for source %r" % (source,))

    def payouts_for(self, phase: str) -> Payouts:
        if phase == PRE:
            return self.pre
        if phase == POST:
            return self.post
        if phase == TRANSFER:
            if self.transfer is None:
                raise ValueError("hypothesis carries no transfer regime")
            return self.transfer
        raise ValueError("unknown phase %r" % (phase,))

    def predicted_report(self, trial: Trial) -> int:
        outcome = self.payouts_for(trial.phase).outcome(trial.action)
        # Before the change point every source is stipulated accurate; the rules block
        # tells the model this, so the oracle is entitled to assume it too.
        state = ACCURATE if trial.phase == PRE else self.state_of(trial.source)
        return report_of(outcome, state)

    def predicted_outcome(self, audit: Audit) -> int:
        return self.payouts_for(audit.phase).outcome(audit.action)


@dataclass(frozen=True)
class HiddenWorld:
    """Ground truth. Never rendered, never handed to prompts.py, never serialised into a
    prompt-bearing record. It exists to generate observations and to grade the oracle."""

    pre: Payouts
    post: Payouts
    source_states: FrozenSet[Tuple[str, str]]
    transfer: Optional[Payouts]
    cell: str  # stable | world | reporter | both — an analysis label, not task content

    def as_hypothesis(self) -> Hypothesis:
        return Hypothesis(self.pre, self.post, self.source_states, self.transfer)

    def emit(self, round: int, phase: str, action: str, source: str) -> Trial:
        report = self.as_hypothesis().predicted_report(
            Trial(round=round, phase=phase, action=action, source=source, report=-1))
        return Trial(round=round, phase=phase, action=action, source=source, report=report)

    def audit(self, trial: Trial) -> Audit:
        outcome = self.as_hypothesis().payouts_for(trial.phase).outcome(trial.action)
        return Audit(round=trial.round, phase=trial.phase, action=trial.action,
                     source=trial.source, verified_outcome=outcome)


CELLS = ("stable", "world", "reporter", "both")


def cell_of(pre: Payouts, post: Payouts, faulty: bool) -> str:
    """The 2x2 of the design table: did the payouts move, did the source start inverting."""
    moved = pre.winner != post.winner
    if moved and faulty:
        return "both"
    if moved:
        return "world"
    if faulty:
        return "reporter"
    return "stable"


def enumerate_hypotheses(main_options: Tuple[str, str],
                         sources: Iterable[str],
                         transfer_options: Optional[Tuple[str, str]] = None):
    """The full product over everything the model is not told. Small on purpose: two
    payout configs for each regime, two states per source. Two sources with a transfer
    regime is 32 hypotheses, which enumerates in microseconds and needs no cleverness."""
    sources = tuple(sources)
    pres = Payouts.both(main_options)
    posts = Payouts.both(main_options)
    transfers = (None,) if transfer_options is None else Payouts.both(transfer_options)
    for pre, post, transfer in itertools.product(pres, posts, transfers):
        for states in itertools.product(SOURCE_STATES, repeat=len(sources)):
            yield Hypothesis(pre=pre, post=post,
                             source_states=frozenset(zip(sources, states)),
                             transfer=transfer)
