#!/usr/bin/env python3
"""The two-source arm: is the judgement about THAT source, or about reports in general?

Without this arm the headline result has a boring rival. A model that answers the
transfer question correctly after catching a source inverting might have learned
something about that source, or it might simply have started flipping every number it
sees. Those two accounts make opposite predictions here, so this is the arm that decides
between them, and it is the reason the instrument is worth running at all.

Both sources report on the SAME original options, which is what lets one audit chain
through to a verdict on the other. Eight cells: payouts moved or not, crossed with each
source accurate or inverting. They pair off four ways, because flipping the payouts and
flipping BOTH sources leaves every visible number exactly where it was.
"""
from __future__ import annotations

import hashlib
import itertools
import random
from typing import Dict, List, Optional, Tuple

from histories import (ACQUISITION_ROUNDS, CHANGE_ROUND, LABEL_POOL, MAIN_Q, SOURCE_POOL,
                       TRANSFER_Q, Episode, VisibleEpisode)
from world import (ACCURATE, INVERTED, POST, PRE, TRANSFER, Audit, HiddenWorld, Payouts,
                   Question, Trial)


def cell_name(moved: bool, inv_main: bool, inv_alt: bool) -> str:
    return "m%d_s%s_t%s" % (int(moved), "i" if inv_main else "a", "i" if inv_alt else "a")


def parse_cell(cell: str) -> Tuple[bool, bool, bool]:
    """Split, never index. Reading these flags off character positions cost five test
    failures on the first run: the states sit at 4 and 7, and indices 3 and 5 are the
    literals 's' and '_', so every cell silently built with both sources accurate."""
    moved, main, alt = cell.split("_")
    return moved[1] == "1", main[1] == "i", alt[1] == "i"


CELLS8 = tuple(cell_name(*c) for c in itertools.product((False, True), repeat=3))


def matched_partner(cell: str) -> str:
    """(payouts, S, T) pairs with (payouts flipped, S flipped, T flipped). Flipping the
    payouts and both sources is the identity on every visible number."""
    moved, inv_main, inv_alt = parse_cell(cell)
    return cell_name(not moved, not inv_main, not inv_alt)


MATCHED_PAIRS8 = tuple(sorted({tuple(sorted((c, matched_partner(c)))) for c in CELLS8}))


class SpecFamily:
    def __init__(self, family_id, main_options, transfer_options, sources,
                 transfer_reporter, transfer_reported_winner, pre_winner,
                 audit_targets, audit_position, episodes):
        self.family_id = family_id
        self.main_options = main_options
        self.transfer_options = transfer_options
        self.sources = sources                      # (s_main, s_alt)
        self.transfer_reporter = transfer_reporter
        self.transfer_reported_winner = transfer_reported_winner
        self.pre_winner = pre_winner
        self.audit_targets = audit_targets          # (index into s_main rounds, into s_alt)
        self.audit_position = audit_position
        self.episodes = tuple(episodes)

    def by(self, cell: str, calibrated: bool, question_kind: str) -> Episode:
        for ep in self.episodes:
            if (ep.cell == cell and ep.calibrated == calibrated
                    and ep.question_kind == question_kind):
                return ep
        raise KeyError((cell, calibrated, question_kind))

    def state_of(self, cell: str, source: str) -> str:
        return self.by(cell, False, MAIN_Q).hidden.as_hypothesis().state_of(source)


def _hidden(main_options, transfer_options, pre_winner, cell, sources,
            transfer_reporter, transfer_reported_winner) -> HiddenWorld:
    s_main, s_alt = sources
    moved, inv_main, inv_alt = parse_cell(cell)
    other_main = [o for o in main_options if o != pre_winner][0]
    pre = Payouts(main_options, pre_winner)
    post = Payouts(main_options, other_main if moved else pre_winner)
    states = {(s_main, INVERTED if inv_main else ACCURATE),
              (s_alt, INVERTED if inv_alt else ACCURATE)}
    reporter_inverted = inv_main if transfer_reporter == s_main else inv_alt
    true_transfer = ([o for o in transfer_options if o != transfer_reported_winner][0]
                     if reporter_inverted else transfer_reported_winner)
    return HiddenWorld(pre=pre, post=post, source_states=frozenset(states),
                       transfer=Payouts(transfer_options, true_transfer), cell=cell)


def _trials(hidden, main_options, transfer_options, sources, transfer_reporter,
            include_transfer) -> Tuple[Trial, ...]:
    s_main, s_alt = sources
    out: List[Trial] = []
    r = 1
    for _ in range(ACQUISITION_ROUNDS // 2):
        for opt in main_options:
            out.append(hidden.emit(r, PRE, opt, s_main))
            r += 1
    for src in (s_main, s_alt):
        for opt in main_options:
            out.append(hidden.emit(r, POST, opt, src))
            r += 1
    if include_transfer:
        for opt in transfer_options:
            out.append(hidden.emit(r, TRANSFER, opt, transfer_reporter))
            r += 1
    return tuple(out)


def build_family(rng: random.Random, index: int = 0) -> SpecFamily:
    labels = rng.sample(LABEL_POOL, 4)
    main_options = (labels[0], labels[1])
    transfer_options = (labels[2], labels[3])
    pool = rng.sample(SOURCE_POOL, 2)
    sources = (pool[0], pool[1])
    # Six factors crossed on the index; 64 families close the design exactly.
    audit_targets = (index % 2, (index // 2) % 2)
    audit_position = ("before", "after")[(index // 4) % 2]
    transfer_reported_winner = transfer_options[(index // 8) % 2]
    pre_winner = main_options[(index // 16) % 2]
    transfer_reporter = sources[(index // 32) % 2]
    fid = hashlib.sha1(("spec|%s|%s|%s|%s|%s|%s|%s" % (
        main_options, transfer_options, sources, transfer_reporter,
        transfer_reported_winner, pre_winner, audit_targets)).encode()).hexdigest()[:10]

    episodes: List[Episode] = []
    for cell, calibrated, qkind in itertools.product(CELLS8, (False, True),
                                                     (MAIN_Q, TRANSFER_Q)):
        hidden = _hidden(main_options, transfer_options, pre_winner, cell, sources,
                         transfer_reporter, transfer_reported_winner)
        include_transfer = qkind == TRANSFER_Q
        trials = _trials(hidden, main_options, transfer_options, sources,
                         transfer_reporter, include_transfer)
        audits: Tuple[Audit, ...] = ()
        if calibrated:
            # One verified record per source. Independent calibration is what makes the
            # specificity claim clean: the verdict on each source rests on its own check.
            picked = []
            for src, tgt in zip(sources, audit_targets):
                rounds = [t for t in trials if t.phase == POST and t.source == src]
                picked.append(hidden.audit(rounds[tgt]))
            audits = tuple(sorted(picked, key=lambda a: a.round))
        question = Question(
            phase=TRANSFER if include_transfer else POST,
            options=transfer_options if include_transfer else main_options,
            source=transfer_reporter if include_transfer else sources[0])
        visible = VisibleEpisode(
            main_options=main_options,
            transfer_options=transfer_options if include_transfer else None,
            sources=sources, trials=trials, audits=audits,
            audit_position=audit_position, question=question,
            change_round=CHANGE_ROUND)
        episodes.append(Episode(
            episode_id="%s-%s-%s-%s" % (fid, cell, "cal" if calibrated else "nocal", qkind),
            family_id=fid, arm="specificity", cell=cell, calibrated=calibrated,
            question_kind=qkind, audit_target=None, visible=visible, hidden=hidden))

    return SpecFamily(fid, main_options, transfer_options, sources, transfer_reporter,
                      transfer_reported_winner, pre_winner, audit_targets,
                      audit_position, episodes)


def generate(n: int, seed: int = 0) -> List[SpecFamily]:
    """n should be a multiple of 64 for the counterbalance to close exactly."""
    rng = random.Random(seed)
    return [build_family(rng, index=i) for i in range(n)]


# What each rival account predicts on the calibrated transfer question. Written out here
# rather than left to the analysis script, because a prediction table produced after
# seeing results is not a prediction.
def predicted_answer(fam: SpecFamily, cell: str, account: str) -> str:
    """account: 'source_specific' (the hypothesis), 'blanket' (distrust everything once
    any source is caught), 'latest' (take the newest report at face value)."""
    reported = fam.transfer_reported_winner
    other = [o for o in fam.transfer_options if o != reported][0]
    reporter_inv = fam.state_of(cell, fam.transfer_reporter) == INVERTED
    any_inv = any(fam.state_of(cell, s) == INVERTED for s in fam.sources)
    if account == "source_specific":
        return other if reporter_inv else reported
    if account == "blanket":
        return other if any_inv else reported
    if account == "latest":
        return reported
    raise ValueError(account)


def cells_where(fam: SpecFamily, reporter_inverted: bool, other_inverted: bool):
    """Cells selected by the two source states, with the payout change left free. The
    payout change is a nuisance factor for the specificity question; it has to be in the
    design for the matched pairs to exist, and it has to be averaged over here."""
    reporter_is_main = fam.transfer_reporter == fam.sources[0]
    out = []
    for cell in CELLS8:
        _, inv_main, inv_alt = parse_cell(cell)
        rep, oth = ((inv_main, inv_alt) if reporter_is_main else (inv_alt, inv_main))
        if rep == reporter_inverted and oth == other_inverted:
            out.append(cell)
    return tuple(out)


def discriminating_cells(fam: SpecFamily):
    """THE primary comparison. Two conditions, both with a clean transfer reporter,
    differing only in whether the OTHER source was caught inverting.

    Source-specific learning predicts the same answer in both. Blanket distrust predicts
    the answer flips in the second and is therefore wrong there. Overall accuracy across
    all eight cells separates these accounts only 1.00 to 0.75, which is too thin a gap
    to hang the conclusion on; restricted to this contrast the gap is 1.00 to 0.00, and
    it is a within-model comparison, so a merely incompetent model cannot fake it."""
    return {"control": cells_where(fam, False, False),
            "other_source_caught": cells_where(fam, False, True)}
