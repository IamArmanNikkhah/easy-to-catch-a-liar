#!/usr/bin/env python3
"""The exact answer key: enumerate every hypothesis, keep the ones the observations allow,
answer only if the survivors agree.

This is the piece that lets us be wrong cheaply. If the oracle says a question is
unanswerable, no model result on that question means anything, and we find that out on a
laptop instead of on a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from world import (Audit, Hypothesis, Question, Trial, enumerate_hypotheses)

DEFINITE = "definite"
AMBIGUOUS = "ambiguous"
IMPOSSIBLE = "impossible"   # no hypothesis survives: the generator built something broken


@dataclass(frozen=True)
class Answer:
    status: str
    action: Optional[str]
    survivors: Tuple[Hypothesis, ...]

    @property
    def is_definite(self) -> bool:
        return self.status == DEFINITE


def consistent(h: Hypothesis, trials: Sequence[Trial], audits: Sequence[Audit]) -> bool:
    for t in trials:
        if h.predicted_report(t) != t.report:
            return False
    for a in audits:
        if h.predicted_outcome(a) != a.verified_outcome:
            return False
    return True


def survivors(main_options: Tuple[str, str],
              sources: Sequence[str],
              trials: Sequence[Trial],
              audits: Sequence[Audit],
              transfer_options: Optional[Tuple[str, str]] = None) -> List[Hypothesis]:
    space = enumerate_hypotheses(main_options, sources, transfer_options)
    return [h for h in space if consistent(h, trials, audits)]


def solve(main_options: Tuple[str, str],
          sources: Sequence[str],
          trials: Sequence[Trial],
          audits: Sequence[Audit],
          question: Question,
          transfer_options: Optional[Tuple[str, str]] = None) -> Answer:
    """Definite iff every surviving hypothesis names the same best option for the phase
    the question is about. Note what this does NOT do: it never breaks a tie by
    likelihood or by a prior. An ambiguous history has equal behaviour as its correct
    null, and smuggling a prior in here would quietly manufacture a right answer."""
    surv = survivors(main_options, sources, trials, audits, transfer_options)
    if not surv:
        return Answer(IMPOSSIBLE, None, ())
    picks = {h.payouts_for(question.phase).winner for h in surv}
    if len(picks) == 1:
        return Answer(DEFINITE, picks.pop(), tuple(surv))
    return Answer(AMBIGUOUS, None, tuple(surv))


def source_verdict(surv: Sequence[Hypothesis], source: str) -> str:
    """What the observations settle about ONE source, independent of any action choice.
    Used by the specificity control: learning that source S inverts must leave the
    verdict on source T untouched."""
    states = {h.state_of(source) for h in surv}
    if len(states) == 1:
        return states.pop()
    return AMBIGUOUS
