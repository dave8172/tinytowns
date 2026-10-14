"""Benchmark step 2: Jev (TypeSafe's decision model) on the same 300 names.

One request per name: the shared task description, plus one 28-option Choice per position
(bench/questions.py). Jev answers every question with a probability for each option, so we get
the same loss as our models: average -ln(probability given to the real next symbol).

Needs a TypeSafe key in the environment (TYPESAFE_API_KEY; get one at console.typesafe.ai).
Run from the repo root:
    python bench/step2_jev.py --n 1     # one name first: shows real tokens and the cost of 300
    python bench/step2_jev.py           # all 300
Jev charges per input token ($0.042 per million when this was written; output is free).
"""
import json
import math
import statistics
import sys
import time
from pathlib import Path

from typesafe_sdk import Choice, TypeSafeClient

from questions import STATE, questions

ROOT = Path(__file__).resolve().parent.parent
PRICE_PER_TOKEN = 0.042 / 1_000_000  # dollars per input token, from docs.typesafe.ai/models
# Jev rounds every probability to 2 decimals, so a 0 means "below 0.005", and -ln 0 is infinite.
# The main loss reads each 0 as 0.005, the most generous value it can stand for. The strict loss
# reads it as 0.000001 instead: the gap between the two shows how much the zeros cost.
FLOOR, STRICT_FLOOR = 0.005, 1e-6

names = [line for line in (ROOT / "bench/names.txt").read_text(encoding="utf-8").splitlines() if line]
n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else len(names)
names = names[:n]

client = TypeSafeClient(model="jev-latest")
rows, tokens, seconds, model_seen = [], 0, [], None
for i, name in enumerate(names):
    qs, answers = questions(name)
    start = time.perf_counter()
    response = client.system_one(
        state=STATE,
        questions={qid: Choice(**q) for qid, q in qs.items()},
    )
    seconds.append(time.perf_counter() - start)  # wall time, network included: what a caller waits
    tokens += response.usage.input_tokens
    model_seen = response.model  # the exact version behind "jev-latest"
    for qid, real in answers.items():
        probs = response.choices[qid].probabilities  # {"end": 0.01, "space": 0.0, "a": 0.2, ...}
        p = probs.get(real, 0.0)
        rows.append({"name": name, "pos": int(qid[1:]), "real": real, "p": p,
                     "top": max(probs, key=probs.get)})
    if (i + 1) % 25 == 0 or i == 0:
        print(f"  {i + 1}/{len(names)} names, {tokens:,} tokens so far")

result = {
    "model": model_seen,
    "n_names": len(names),
    "n_predictions": len(rows),
    # averaged over predictions, as for our models
    "loss": sum(-math.log(max(r["p"], FLOOR)) for r in rows) / len(rows),
    "loss_strict": sum(-math.log(max(r["p"], STRICT_FLOOR)) for r in rows) / len(rows),
    "zeros": sum(r["p"] == 0 for r in rows),  # real answer given probability 0
    "top_correct": sum(r["top"] == r["real"] for r in rows) / len(rows),  # Jev's pick was right
    "ms_per_name": statistics.median(seconds) * 1000,
    "input_tokens": tokens,
    "tokens_per_name": tokens / len(names),
    "cost_usd": tokens * PRICE_PER_TOKEN,
    "cost_per_1000_names_usd": tokens * PRICE_PER_TOKEN / len(names) * 1000,
}
print(json.dumps(result, indent=2))
if len(names) < 300:
    print(f"\nestimate for 300 names: {result['tokens_per_name'] * 300:,.0f} tokens, "
          f"${result['tokens_per_name'] * 300 * PRICE_PER_TOKEN:.4f}")
    for r in rows:  # the probe is small enough to read prediction by prediction
        print(f"  {r['name'][:r['pos']]!r:<24} real {r['real']:<6} p {r['p']:.4f}  top {r['top']}")
else:
    out = ROOT / "bench/results/jev.json"
    out.write_text(json.dumps({"result": result, "predictions": rows}, indent=1) + "\n")
    print(f"saved to {out.relative_to(ROOT)}")
