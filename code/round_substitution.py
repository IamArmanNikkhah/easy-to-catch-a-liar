#!/usr/bin/env python3
"""Mechanism test for the round-6 collapse, written 2026-09-13 AFTER the collapse was measured
and BEFORE any prose. Hypothesis: when the verified record names round 6, the model checks the
record's actual value against round 5's REPORT instead of round 6's. In the honest cells the
record always agrees with its own round, so under the hypothesis the answer should track
whether the OTHER round's report happens to equal the record's actual value.
"""
import json, re, collections, random, sys

def val(seg):
    s = seg.lower()
    if re.search(r"\b1 point|one point|worth one", s): return 1
    if re.search(r"\b0 point|nothing|zero", s): return 0
    raise ValueError(seg)

def parse(p):
    rep, actual, ar = {}, None, None
    for line in p.split("\n"):
        m = re.search(r"(?:Round|Entry|On round|account of round) (\d)", line)
        if not m: continue
        r = int(m.group(1))
        if re.search(r"actual result|in fact|genuinely came out|the real result", line):
            ar, actual = r, val(line.split(". Source")[0].split(", where")[0])
        elif re.search(r"you chose|was selected|you took|you go with", line):
            rep[r] = val(line)
    for m in re.finditer(r"On round (\d) you took \w, and Source \d+ told you it was worth (one point|nothing)", p):
        rep[int(m.group(1))] = 1 if m.group(2) == "one point" else 0
    assert len(rep) == 8 and actual is not None, p[-400:]
    return rep, ar, actual

def main(runs):
    key = {k["episode_id"]: k for k in map(json.loads, open("../data/paired_s9001_n64_key.jsonl"))}
    prompts = {r["episode_id"]: r["prompt"] for r in map(json.loads, open("../data/paired_s9001_n64.jsonl"))}
    for name, f in runs.items():
        got = {r["episode_id"]: r for r in map(json.loads, open("../data/runs/%s.jsonl" % f))}
        print(name)
        for cell in ("stable", "world"):
            for rnd in (5, 6):
                other = 5 if rnd == 6 else 6
                by = collections.defaultdict(lambda: collections.defaultdict(list))
                for i, k in key.items():
                    if k["question"] != "source" or k["variant"] != "cal" or k["cell"] != cell or k["audit_round"] != rnd: continue
                    rep, ar, actual = parse(prompts[i]); assert ar == rnd and rep[rnd] == actual
                    by[rep[other] == actual][k["family_id"]].append(got[i]["choice"] == k["accurate_label"])
                line = "  %-7s audit r%d  P(says honest | r%d report == record):" % (cell, rnd, other)
                for ag in (True, False):
                    d = by[ag]; n = sum(len(v) for v in d.values())
                    if n == 0: line += "  %s n=0" % ("match" if ag else "clash"); continue
                    fams = sorted(d); rng = random.Random(0); pt = sum(sum(v) for v in d.values()) / n
                    bs = sorted(sum(sum(d[fm]) for fm in p) / sum(len(d[fm]) for fm in p)
                                for p in ([rng.choice(fams) for _ in fams] for _ in range(1500)))
                    line += "  %s %.3f [%.3f,%.3f] n=%d" % ("match" if ag else "clash", pt, bs[37], bs[1462], n)
                print(line)

if __name__ == "__main__":
    main({"Qwen-32B": "paired_32b", "Qwen-72B": "paired_72b", "Llama-70B": "paired_llama70b"})
