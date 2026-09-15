# Easy to Catch a Liar, Hard to Clear an Honest One

Code, prompts, answer keys, and every scored model output for the paper *Easy to Catch a
Liar, Hard to Clear an Honest One: Language Models Diagnosing a Corrupted Reward Channel
from a Verified Record* (Arman Nik Khah, The University of Texas at Dallas, 2026).

## The question in one paragraph

An agent that learns from rewards has to trust whatever reports those rewards. When the
reports suddenly change, either the world changed or the reporter broke, and from the
reports alone those two stories are the same. We build a two-option game in which a payout
swap and a lying reporter produce byte-identical histories, then add one verified record:
an independent check of one round's real result, printed beside what the reporter said
about that round. That one line settles the case. We ask three large language models,
from two families, one question with one letter: is the reporter honest or lying?

They catch a lying reporter almost perfectly. They clear an honest reporter far less
often, and how often depends on which round the record names (Qwen) or which letter
stands for "honest" (Llama). Every number in the paper is in `paper/NUMBERS.md`, and that
file regenerates byte for byte from the files here.

## Layout

| path | what it is |
|---|---|
| `PREREG-paired.md` | the pre-registration: predictions, stopping rules, and the decision rule, written before any model saw the 64-world prompts |
| `code/world.py`, `code/oracle.py`, `code/histories.py`, `code/prompts.py` | the game: payouts, sources, the exact answer key, counterbalancing, the renderer |
| `code/paired_attrib.py` | builds the 64-world paired item set and its key |
| `code/score_answers.py` | forward passes only; compares two answer tokens at the answer position |
| `code/paper_numbers.py` | every number the paper prints, regenerated from the run files |
| `code/analyze_paired.py`, `code/analyze_swap.py`, `code/round_substitution.py` | the estimators: per-situation accuracy, the round and letter margins, the paired action arm |
| `code/tests/` | the CPU battery: matched pairs are byte-identical, hidden causes never reach a prompt, the answer key never eliminates the true world |
| `data/paired_s9001_n64.jsonl` + `_key.jsonl` | the 10,240 prompts per model and their answer key (original worlds) |
| `data/paired_s9001_n64_swap.jsonl` + `_key.jsonl` | the same worlds with the letter pairing reversed (reversed worlds, Section 4.3 of the paper) |
| `data/runs/paired_*.jsonl` | scored outputs: Qwen2.5-32B, Qwen2.5-72B, Llama-3.1-70B, Llama-3.1-8B |
| `data/runs/swap_*.jsonl` | scored outputs on the reversed worlds, calibrated source prompts |
| `data/runs/*.log` | the run logs: model revision, precision, library versions, GPU, timing |
| `data/runs/attrib_*`, `big_*`, `ladder_*`, `devforms_*`, `dev*` | earlier prompt sets used during development; the paper's remark that a four-point Qwen ladder was not monotone comes from `attrib_7b`, `attrib_14b`, `big_attrib_32b`, `big_attrib_72b` |
| `data/runs/bridge_32b_blackwell.jsonl` | the 256-prompt bridge between two GPU types (256 of 256 choices reproduced) |
| `paper/NUMBERS.md` | the generated numbers file, the only source for every figure in the paper |

## Reproduce the numbers

```bash
cd code
python3 paper_numbers.py > /tmp/NUMBERS.md
diff /tmp/NUMBERS.md ../paper/NUMBERS.md && echo identical
```

Takes about a minute on a laptop (4,000 bootstrap resamples per interval). Needs only the
Python standard library.

## Run the design tests

```bash
python3 code/tests/test_design.py
python3 code/tests/test_specificity.py
```

## Score a model yourself

`code/score_answers.py` takes the prompt file and a Hugging Face model id, runs forward
passes in 16-bit floating point, and writes one line per prompt with the two answer-token
logits and the chosen letter. The paper's runs used transformers 5.16.1, batch 8, seed 0;
each run log records the exact versions. The answer key is a separate file on purpose, so
a scoring run never sees it.

## What the four situations are called

The paper names them by what happened after round 4. The code uses shorter keys:

| paper | code |
|---|---|
| nothing moved, honest | `stable` |
| points moved, honest | `world` |
| nothing moved, lying | `reporter` |
| points moved, lying | `both` |

The prompt variants are `cal` (with record), `nocal` (no record), `floor` (answer printed),
`ceiling` (answer printed plus record), and `cal_prior` (salience). NUMBERS.md uses the code
keys; the paper uses the names in parentheses.
