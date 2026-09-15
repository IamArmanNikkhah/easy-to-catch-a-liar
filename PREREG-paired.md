# Pre-registration: the paired measure at 32B

Written 2026-09-11, before any model has seen a single item from
`data/paired_s9001_n64.jsonl`. The commit that adds this file is the timestamp; nothing
below may be edited after a run lands, only answered.

## What is being run

`Qwen2.5-32B-Instruct`, fp16, one-token forced choice, no chain of thought, four surface
forms, 64 families, DEV_SEED 9001, 8192 items. Each history is asked twice: **is the source
reporting accurately or the opposite** (source arm) and **which of the new pair pays**
(winner arm), byte-identical up to the question line for every variant except the floors.

Substrate is TRANSFER histories. On MAIN histories the audited action plus its verified
outcome names the winner 64/64 without the source entering, so a pair built there compares
a two-fact inference against a one-fact lookup.

## The honest statement of what is and is not new here

**A dissociation is close to already implied, and saying so now is the point of a
pre-registration.** Wednesday's ceiling data has the 32B at 0.234 in the inverted cells
*when the cause is printed in the prompt*, while it picks the reported winner 0.88 of the
time. That is already a model told the channel is corrupted and acting on the corrupted
reading. So "source arm beats winner arm" is not a 65/35 bet, which is what I called it
this morning. **Revised: 85%.** The morning number was wrong because I had not yet
decomposed the ceiling.

What this run actually buys, none of which exists at any size:

1. **P(winner right | source right) at the item level.** Every published number here is a
   marginal. A marginal gap is consistent with the model simply being wrong about the state
   in the cells where the winner arm fails. The conditional is not, and it is the only
   statistic that licenses the claim.
2. **Both arms on one prompt with no shortcut.** First clean measurement.
3. **The `world` cell on the source question, on a clean substrate.** This is the research
   question itself: the payouts moved, nobody lied, does the model say so.
4. **Whether 0.531 was a broken harness or a result.** The old main floor prints a payout
   table that the reports contradict. `reading_floor` strips the reports first. If the
   honest floor clears and the conflicted one does not, the gap is a finding about
   conflict, not a defect in the instrument.

## Registered predictions (32B, point estimates, per cell where it matters)

| quantity | predicted | scale |
|---|---|---|
| source arm, cal, overall | 0.78 | all four cells |
| source arm, cal, `world` cell | 0.60 | the honest-source-moved-payouts cell |
| source arm, cal, inverted cells | 0.92 | `reporter` + `both` |
| winner arm, cal, overall | 0.55 | all four cells |
| winner arm, cal, inverted cells | 0.20 | `reporter` + `both` |
| dissociation (DiD, source minus winner) | +0.30 | — |
| P(winner right \| source right), overall | 0.52 | byte-matched cal pairs |
| P(winner right \| source right), inverted cells | 0.20 | the claim-bearing number |
| honest reading floor, winner arm | ≥ 0.95 | all cells |

Every figure above is an **overall or per-cell accuracy**, never an inverted-cell number
dressed as an aggregate. That ambiguity is what made this morning's first plan wrong.

## Stopping rules, registered in advance

1. **A calibrated arm outscoring its own ceiling voids the run.** Inferring a source's state
   cannot beat being told it. `analyze_paired.py` checks this first and exits non-zero.
2. **The honest reading floor below 0.90 voids the winner arm.** The answer is printed and
   nothing contradicts it; a model that cannot read it is a harness fault, and no ceiling
   above it may be interpreted.
3. **The no-evidence items must not separate.** If `P(says flipped | reports flipped)` minus
   `P(says flipped | reports did not flip)` exceeds 0.25, the model is running the flip
   heuristic and every calibrated number must be reported against that, not against 0.5.
   The 72B already fails this at 1.00 versus 0.00, and the old marginal control passes it.

## What would make me abandon the claim

- P(winner right | source right) at or above the winner arm's marginal accuracy. That would
  mean getting the state right does not predict acting on it, and the dissociation is two
  unrelated failures rather than one.
- The honest reading floor failing. Then the winner arm is untestable at this size and
  nothing about action can be claimed.
- The dissociation vanishing once the round-5/6 strata are separated. It has to hold at
  both levels or it is a claim about round 5.

## Amendment, same day, before any run: a fifth condition

Added on Arman's explicit go, and declared here before a model has seen an item. Anything
added after a run is not an amendment, it is a story.

`cal_prior` is `cal` plus one line in the rules saying the two causes occur about equally
often, naming neither for this episode. The statement is true of the generator (cells are
crossed 2x2) and identical in every cell, so matched pairs stay byte-identical; verified.

**What it separates.** The `world` cell (payouts moved, source honest) is where everything
collapses, and the 72B is worse there than the 32B, 0.422 against 0.672, while both are
perfect where the source really did lie. Two explanations fit:

- **prior**: the model holds a blame-the-messenger prior, and evidence has to overcome it.
  Salience can move a prior, so `world` should lift.
- **absent**: the model never entertains world-change as a candidate, so it is not weighing
  two causes and losing, it is only ever seeing one. Salience alone should not help;
  only naming the alternative for the episode would.

| quantity | predicted | reading |
|---|---|---|
| `cal_prior` minus `cal`, `world` cell | +0.08 | ≥ +0.10 favours **prior**, ≤ +0.05 favours **absent** |
| `cal_prior` minus `cal`, other three cells | ±0.03 | a lift everywhere means the line added information, not salience, and the condition is void |
| odds on **prior** over **absent** | 55 / 45 | genuinely uncertain, and the 72B's zero-evidence policy is weak evidence for **absent** |

**Void condition.** If `cal_prior` lifts every cell roughly equally, the line is carrying
information rather than salience and the contrast says nothing about priors. That is the one
outcome that would make me discard the condition rather than interpret it.

## Estimator provenance

`analyze_paired.py` was written before this run and validated against four simulated
policies in `sim_policies.py`. The receipts: an oracle policy reports dissociation +0.031,
a knows-not-acts policy reports +0.500 with P(winner right | source right) at 0.000 in both
inverted cells, and a pure flip-heuristic policy passes the OLD leak control at exactly
0.500 while the new statistic reads 1.000 versus 0.000.

---

## Amendment 2, 2026-09-11 ~15:2x: the 72B as a kill-or-confirm test

Written before the 72B sees an item. Arman's go: "run the 72B, let's see if it holds."

**Why this run exists, stated honestly.** Three claims of mine died today, all the same way:
they rested on 16 worlds, and the 64-world measurement shrank or erased them. The asymmetry
does not grow with scale (four non-monotone points). The follow-the-visible-flip effect does
not exist (−0.004, interval straddles zero at 64 worlds). And the headline deficit is not
below chance: the best-measured "points moved, reporter honest" figure is 0.758 at 32B, not
the 0.344 the 16-world plain condition showed.

One dramatic number is left standing and it is standing on exactly the sample that ate the
other three: **the 72B's 0.422** on "points moved, reporter honest", 16 worlds, restated.

**Registered prediction.** I expect it to rise, because everything else did.

| quantity | predicted | reading |
|---|---|---|
| 72B, points moved / reporter honest, restated, 64 worlds | **0.65** | up from the 16-world 0.422 |
| 72B, lying-reporter situations, restated | 0.97 | it was 1.000 at 16 worlds; expect near-ceiling |
| 72B honest-minus-lying gap | **−0.25** | larger than the 32B's −0.135, but not the −0.46 of 16 worlds |
| 72B floor (answer printed, nothing contradicting) | ≥ 0.95 | else the row is uninterpretable |

**Decision rule, fixed now.**
- Below **0.55** with an interval excluding 0.60: the dramatic deficit is REAL at 72B, the
  scale question reopens on firm ground, and the paper has a headline.
- Above **0.70**: it was small-sample noise like the other three. What remains is a modest,
  well-bounded asymmetry plus a headline that Schmied et al. already published, and the
  honest next move is to say so rather than hunt for a fourth framing.
- Between 0.55 and 0.70: no call. Report the interval and stop claiming a deficit.

**What I am NOT allowed to do with this result.** Compare it to the 32B's 0.758 and call the
difference a scale effect. Two points are a pair. That is LESSON 6a30772f and it has already
fired once today.
