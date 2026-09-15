#!/usr/bin/env python3
"""Prompt renderer. One episode, four surface forms.

Takes a VisibleEpisode and nothing else. The hidden cause cannot leak here because the
type carrying it is never passed in; the byte-identity test then proves the matched pairs
render to the same string, per form, which is a stronger statement than a keyword scan.

WHY FOUR FORMS. On 2026-09-10 we measured that this task's difficulty is dominated by how
it is written down: the same stated inversion scored 0.812 in one wording and 0.125 in the
round-log wording, on one model, with everything else held fixed. A single-form instrument
therefore cannot tell a capability from a lucky sentence. Every episode is now asked four
ways and an answer counts only when all four agree.

The four forms are meant to be equally faithful, not a difficulty ladder. Each rules block
carries the same five facts the answer key assumes:
  1. exactly one of the two available options pays 1 point
  2. through the change round, every source reported accurately and no value moved
  3. after it, one of: values swapped / a source began reversing / both / neither
  4. a source that began reversing keeps reversing
  5. a later pair of options has its own values, again exactly one paying
A form missing any of those is a different task, not a rephrasing.

Option order is deliberately NOT constant across forms. Two forms show the family's order
and two show it reversed, so a model answering by position picks each option twice, fails
to agree with itself, and is filtered out by the gate rather than passing it.

Wording discipline, unchanged: the rules describe a mechanism and never a character
judgement. No "faulty", no "unreliable", no "trust".
"""
from __future__ import annotations

from typing import List, Tuple

from world import PRE

FORMS = ("log", "minutes", "prose", "commentary")

# Which forms reverse the displayed option order. One original and one rewritten form on
# each side, so option order is not confounded with which forms are new.
_REVERSED = {"prose", "commentary"}

# The line fragment a form uses to state a verified result. The calibrated-pair test looks
# for this, because a diff that is form-blind cannot check a prose form.
AUDIT_MARKER = {"log": "actual result", "minutes": "in fact",
                "prose": "genuinely came out", "commentary": "the real result"}

# WHY LEDGER AND RECORD WERE REPLACED (2026-09-10). Both were field-delimited: a table row
# and a key-value line. Both failed their own floor at 7B AND at 14B, where the answer is
# printed in the prompt (ledger 0.453 -> 0.594, record 0.516 -> 0.453, against prose at
# 0.938 on the identical items). So it was not a small-model limitation, it was the form.
# Best available explanation, and it is what the rewrite is built on: log and prose state
# each relation with a verb ("you chose P", "Source 3 told you"), while a delimited row
# leaves the relation to column position and asks the reader to carry a header. The two
# replacements keep the surfaces genuinely different but put the relations back into words.
# This costs some surface diversity, which is the gate's whole currency, so the tradeoff is
# deliberate and gets measured rather than assumed: a form the model cannot read contributes
# noise, not independence.


def _source_question_log(vis) -> List[str]:
    a, b = question_options(vis, "log")
    q = vis.source_query
    return ["Question: Since round %d, has %s been reporting each result accurately, or"
            " reporting the opposite of each actual result?" % (vis.change_round + 1,
                                                                vis.question.source),
            "Reply %s if accurately, %s if the opposite." % (q["accurate"], q["opposite"]),
            "Reply with a single letter, %s or %s, and nothing else." % (a, b),
            "Answer:"]


def _source_question_minutes(vis) -> List[str]:
    a, b = question_options(vis, "minutes")
    q = vis.source_query
    return ["The question before the meeting: from round %d onward, has %s been stating"
            " each result correctly, or stating the reverse of each result?"
            % (vis.change_round + 1, vis.question.source),
            "%s denotes correctly; %s denotes the reverse." % (q["accurate"], q["opposite"]),
            "Answer with a single letter, %s or %s." % (a, b),
            "Answer:"]


def _source_question_prose(vis) -> List[str]:
    a, b = question_options(vis, "prose")
    q = vis.source_query
    return ["So here is what you need to decide. From round %d on, has %s been telling you"
            " the truth about each outcome, or the reverse of it? Write %s for the truth,"
            " %s for the reverse. Just the one letter, %s or %s."
            % (vis.change_round + 1, vis.question.source, q["accurate"], q["opposite"],
               a, b),
            "Answer:"]


def _source_question_commentary(vis) -> List[str]:
    a, b = question_options(vis, "commentary")
    q = vis.source_query
    return ["So the call: from round %d on, has %s been calling it straight, or calling"
            " every result backwards?" % (vis.change_round + 1, vis.question.source),
            "%s means straight, %s means backwards." % (q["accurate"], q["opposite"]),
            "Just the one letter, %s or %s." % (a, b),
            "Answer:"]


_SOURCE_Q = {
    "log": _source_question_log,
    "minutes": _source_question_minutes,
    "prose": _source_question_prose,
    "commentary": _source_question_commentary,
}


def question_options(vis, form: str) -> Tuple[str, str]:
    if vis.source_query:
        a, b = vis.source_query["labels"]
        return (b, a) if form in _REVERSED else (a, b)
    a, b = vis.question.options
    return (b, a) if form in _REVERSED else (a, b)


def _points(n: int) -> str:
    return "1 point" if n == 1 else "%d points" % n


# --------------------------------------------------------------------------- log

def _rules_log(vis) -> List[str]:
    lines = [
        "Rules:",
        "- On each round you pick one of the two options available that round. Exactly one"
        " of them is worth 1 point and the other is worth 0 points.",
        "- A reporting source tells you the result of each round, either 1 point or 0 points.",
        "- Through round %d every source reported each result accurately, and the point"
        " values did not change." % vis.change_round,
        "- After round %d, any of these may have happened: the point values of the two"
        " original options swapped; a source started reporting the opposite of each actual"
        " result; both; or neither. You are not told which." % vis.change_round,
        "- If a source started reporting the opposite, it keeps doing so on every later"
        " round.",
        "- A new pair of options may be introduced later. A new pair has its own point"
        " values, again exactly one worth 1 point.",
    ]
    if vis.audits:
        lines.append("- A verified record states the actual result of a round, independently"
                     " of what a source said about that round.")
    if vis.world_prior:
        lines.append("- Across records of this kind, the point values swapped about as often"
                     " as a source began reporting the opposite.")
    return lines


def _log_log(vis) -> List[str]:
    out = ["Round log:"]
    for t in vis.trials:
        if vis.withhold_reports and t.phase != PRE:
            out.append("Round %d - you chose %s. No result was reported." % (t.round, t.action))
        else:
            out.append("Round %d - you chose %s. %s reported %s."
                       % (t.round, t.action, t.source, _points(t.report)))
    return out



def _reported_on(vis, rnd: int) -> int:
    """What the source said about round `rnd`. Read from the visible trials, which is the
    only place a report lives; the audit object never carries it."""
    return next(t for t in vis.trials if t.round == rnd).report


def _audits_log(vis) -> List[str]:
    if not vis.audits:
        return []
    out = ["Verified records:"]
    for a in vis.audits:
        line = "Round %d - actual result: %s." % (a.round, _points(a.verified_outcome))
        if vis.audit_with_report:
            line += " %s had reported %s." % (a.source, _points(_reported_on(vis, a.round)))
        out.append(line)
    return out


def _question_log(vis) -> List[str]:
    a, b = question_options(vis, "log")
    return ["Question: The next round uses the two options %s and %s, and %s will report"
            " the result." % (a, b, vis.question.source),
            "Which of %s and %s is actually worth 1 point on that round?" % (a, b),
            "Reply with a single letter, %s or %s, and nothing else." % (a, b),
            "Answer:"]


# ------------------------------------------------------------------------ ledger

def _rules_minutes(vis) -> List[str]:
    lines = [
        "The following rules govern each round.",
        "1. Two options are available. One of them is worth 1 point and the other is worth"
        " 0 points. Never both, never neither.",
        "2. A source states the result of each round, either 1 point or 0 points.",
        "3. Up to and including round %d, every source stated its result correctly, and no"
        " point value was altered." % vis.change_round,
        "4. From round %d onward exactly one of the following holds, and it is not"
        " disclosed: the two original options traded point values; a source began stating"
        " the reverse of every result; both occurred; neither occurred."
        % (vis.change_round + 1),
        "5. A source that began stating the reverse continues to do so on every later round.",
        "6. A later round may introduce a different pair of options. That pair carries its"
        " own point values, and again exactly one of them is worth 1 point.",
    ]
    if vis.audits:
        lines.append("7. A verification states what a round produced in fact, independently"
                     " of what any source stated about it.")
    if vis.world_prior:
        lines.append("%d. Across records of this kind, the two original options traded point"
                     " values about as often as a source began stating the reverse."
                     % len(lines))
    return lines


def _log_minutes(vis) -> List[str]:
    out = ["Record of rounds:"]
    for t in vis.trials:
        if vis.withhold_reports and t.phase != PRE:
            out.append("Entry %d. Option %s was selected. No result was stated."
                       % (t.round, t.action))
        else:
            out.append("Entry %d. Option %s was selected. %s states the result was %s."
                       % (t.round, t.action, t.source, _points(t.report)))
    return out


def _audits_minutes(vis) -> List[str]:
    if not vis.audits:
        return []
    out = ["Verifications:"]
    for a in vis.audits:
        line = "Entry %d. The result was in fact %s." % (a.round, _points(a.verified_outcome))
        if vis.audit_with_report:
            line += " %s had stated %s." % (a.source, _points(_reported_on(vis, a.round)))
        out.append(line)
    return out


def _question_minutes(vis) -> List[str]:
    a, b = question_options(vis, "minutes")
    return ["The next round offers options %s and %s, with %s stating the result."
            % (a, b, vis.question.source),
            "Which of the two, %s or %s, is in fact worth 1 point on that round?" % (a, b),
            "Answer with a single letter.",
            "Answer:"]


# ------------------------------------------------------------------------- prose

def _rules_prose(vis) -> List[str]:
    base = ("Here is how the game works. Each round you are offered two options, and"
            " exactly one of them is worth a point while the other is worth nothing. You"
            " never see the outcome yourself; a source tells you what it was, either one"
            " point or no points. Everything you were told up to and including round %d"
            " was accurate, and no point value changed during that stretch. From round %d"
            " onward one of four situations holds, and nobody tells you which: the two"
            " original options may have exchanged their point values, a source may have"
            " started telling you the reverse of every outcome, both of those may have"
            " happened, or neither may have. A source that starts reversing keeps reversing"
            " from then on. A different pair of options may appear later, with its own point"
            " values, again one worth a point and one worth nothing."
            % (vis.change_round, vis.change_round + 1))
    if vis.audits:
        base += (" Where a checked account appears, it gives what genuinely happened on that"
                 " round, independent of anything a source said.")
    if vis.world_prior:
        base += (" Across records of this kind, the two original options have exchanged their"
                 " point values about as often as a source has started reversing.")
    return [base]


def _log_prose(vis) -> List[str]:
    parts = []
    for t in vis.trials:
        if vis.withhold_reports and t.phase != PRE:
            parts.append("On round %d you took %s, and no result came back."
                         % (t.round, t.action))
        else:
            said = "one point" if t.report == 1 else "nothing"
            parts.append("On round %d you took %s, and %s told you it was worth %s."
                         % (t.round, t.action, t.source, said))
    return [" ".join(parts)]


def _audits_prose(vis) -> List[str]:
    if not vis.audits:
        return []
    parts = []
    for a in vis.audits:
        truth = "one point" if a.verified_outcome == 1 else "nothing"
        sent = ("A checked account of round %d says the round genuinely came out at %s"
                % (a.round, truth))
        if vis.audit_with_report:
            said = "one point" if _reported_on(vis, a.round) == 1 else "nothing"
            sent += ", where %s had told you %s" % (a.source, said)
        parts.append(sent + ".")
    return [" ".join(parts)]


def _question_prose(vis) -> List[str]:
    a, b = question_options(vis, "prose")
    return ["You are about to face %s and %s, with %s reporting again. Which of the two"
            " genuinely pays a point? Write just the one letter." % (a, b, vis.question.source),
            "Answer:"]


# ------------------------------------------------------------------------ record

def _rules_commentary(vis) -> List[str]:
    lines = [
        "Here's the setup. Every round you get two options in front of you, and one of them"
        " carries a point while the other carries nothing.",
        "You don't see the outcome yourself. A source calls it for you, either one point or"
        " nothing.",
        "Everything through round %d was called straight, and the point values sat still"
        " that whole time." % vis.change_round,
        "From round %d on, one of four things is going on and nobody says which: the two"
        " original options traded their values, or a source started calling every result"
        " backwards, or both, or neither." % (vis.change_round + 1),
        "Once a source starts calling things backwards, it keeps doing that for good.",
        "A fresh pair of options can show up later, carrying its own values, and again just"
        " one of them is worth the point.",
    ]
    if vis.audits:
        lines.append("When a check comes in, it tells you the real result of that round,"
                     " whatever the source called it.")
    if vis.world_prior:
        lines.append("Worth knowing: in records like this, the values trade places about as"
                     " often as a source starts calling things backwards.")
    return lines


def _log_commentary(vis) -> List[str]:
    out = ["How it played out:"]
    for t in vis.trials:
        if vis.withhold_reports and t.phase != PRE:
            out.append("Round %d: you go with %s, and nothing comes back."
                       % (t.round, t.action))
        else:
            called = "one point" if t.report == 1 else "nothing"
            out.append("Round %d: you go with %s, and %s calls it %s."
                       % (t.round, t.action, t.source, called))
    return out


def _audits_commentary(vis) -> List[str]:
    if not vis.audits:
        return []
    out = ["Checks that came in:"]
    for a in vis.audits:
        real = "one point" if a.verified_outcome == 1 else "nothing"
        line = "Round %d: the real result was %s." % (a.round, real)
        if vis.audit_with_report:
            said = "one point" if _reported_on(vis, a.round) == 1 else "nothing"
            line += " %s had called it %s." % (a.source, said)
        out.append(line)
    return out


def _question_commentary(vis) -> List[str]:
    a, b = question_options(vis, "commentary")
    return ["Next up it's %s or %s, with %s calling it again." % (a, b, vis.question.source),
            "Which one really carries the point, %s or %s?" % (a, b),
            "Just the one letter.",
            "Answer:"]


# --------------------------------------------------------------------------- api

_HEAD = {
    "log": "You are choosing between labelled options to earn points.",
    "minutes": "Proceedings of a points game, recorded round by round.",
    "prose": "",
    "commentary": "",
}

_PARTS = {
    "log": (_rules_log, _log_log, _audits_log, _question_log),
    "minutes": (_rules_minutes, _log_minutes, _audits_minutes, _question_minutes),
    "prose": (_rules_prose, _log_prose, _audits_prose, _question_prose),
    "commentary": (_rules_commentary, _log_commentary, _audits_commentary,
                   _question_commentary),
}

# Control disclosures, written in each form's own register. A prose sentence dropped into a
# key-value record reads as foreign, and the first four-form check measured what that costs:
# ledger fell to 0.500 and record to 0.547 on a question whose answer was printed above it,
# while log and prose sat at 0.750 and 0.703. Same fact, four voices.

def _say_payouts(d, form: str) -> List[str]:
    w, l, r = d["winner"], d["loser"], d["round"]
    if form == "log":
        return ["Additional information: after round %d, %s pays 1 point and %s pays"
                " 0 points." % (r, w, l)]
    if form == "minutes":
        return ["It is further stated that from round %d onward, %s is worth 1 point and"
                " %s is worth 0 points." % (r + 1, w, l)]
    if form == "prose":
        return ["You have also been told that from round %d onward, %s is the one worth a"
                " point and %s is worth nothing." % (r + 1, w, l)]
    return ["One more thing you know going in: from round %d on, %s is the one carrying the"
            " point and %s carries nothing." % (r + 1, w, l)]


def _say_cause(d, form: str) -> List[str]:
    src, inv, moved, r = d["source"], d["inverting"], d["moved"], d["round"]
    if form == "log":
        return ["Additional information: the point values for the two original options %s"
                " after round %d. %s %s reporting the opposite of each actual result after"
                " round %d."
                % ("were swapped" if moved else "were not changed", r,
                   src, "began" if inv else "did not begin", r)]
    if form == "minutes":
        return ["It is further stated that after round %d the two original options %s, and"
                " that %s %s stating the reverse of every result."
                % (r, "traded point values" if moved else "kept their point values",
                   src, "began" if inv else "did not begin")]
    if form == "prose":
        return ["You have also been told what happened after round %d. The two original"
                " options %s, and %s %s telling you the reverse of every outcome."
                % (r, "exchanged their point values" if moved
                   else "kept their point values", src,
                   "started" if inv else "did not start")]
    return ["One more thing you know going in: after round %d the two original options %s,"
            " and %s %s calling every result backwards."
            % (r, "traded their values" if moved else "kept their values",
               src, "started" if inv else "did not start")]


def render(vis, form: str = "log") -> str:
    """The single entry point. `vis` must be a VisibleEpisode; passing an Episode raises on
    the missing attributes rather than silently rendering hidden state."""
    if form not in _PARTS:
        raise ValueError("unknown form %r; expected one of %r" % (form, FORMS))
    rules, log, audits, question = _PARTS[form]
    blocks: List[List[str]] = []
    if _HEAD[form]:
        blocks.append([_HEAD[form]])
    blocks.append(rules(vis))
    if vis.payout_disclosure:
        blocks.append(_say_payouts(vis.payout_disclosure, form))
    if vis.disclosure:
        blocks.append(_say_cause(vis.disclosure, form))
    if vis.audit_position == "before":
        blocks.append(audits(vis))
        blocks.append(log(vis))
    else:
        blocks.append(log(vis))
        blocks.append(audits(vis))
    blocks.append(_SOURCE_Q[form](vis) if vis.source_query else question(vis))
    return "\n\n".join("\n".join(b) for b in blocks if b)


def answer_labels(vis, form: str = "log") -> tuple:
    return question_options(vis, form)
