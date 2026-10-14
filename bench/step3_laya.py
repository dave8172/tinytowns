"""Benchmark step 3: Laya (an open decision model, 421M weights) on the same 300 names.

Laya takes the same kind of request as Jev, a state plus typed questions, so it gets exactly the
same wording (bench/questions.py). It runs on your own machine: no key, no cost per name. The
English checkpoint is a ModernBERT-large encoder, about 1.7 GB of weights, so it fits a 4 GB GPU.

One call per name, one forward pass per call: the state and all of that name's questions are
read together, and every question gets 28 probabilities back.
    pip install laya                      # once; keeps your CUDA torch (Laya needs torch>=2.0)
    python bench/step3_laya.py --n 1      # one name: check it loads, read every prediction
    python bench/step3_laya.py            # all 300 -> bench/results/laya-<device>.json
The first run downloads the checkpoint from Hugging Face (convaiinnovations/laya).
"""
import json
import math
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path

import laya
import torch
from huggingface_hub import model_info

from questions import STATE, questions

ROOT = Path(__file__).resolve().parent.parent
REPO = "convaiinnovations/laya"  # the English checkpoint, at the repo root
# Laya rounds probabilities to 4 decimals, so a 0 means "below 0.00005". As with Jev, the main
# loss reads a 0 as half the rounding step (the most generous value it can stand for); the
# strict loss reads it as 0.000001.
FLOOR, STRICT_FLOOR = 0.00005, 1e-6

names = [line for line in (ROOT / "bench/names.txt").read_text(encoding="utf-8").splitlines() if line]
n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else len(names)
names = names[:n]

revision = model_info(REPO).sha  # the exact checkpoint version, so the run can be repeated
agent = laya.load(REPO, revision=revision)  # picks the GPU when torch can see one
device = str(agent.device)
print(f"laya {version('laya')}, {REPO}@{revision[:8]}, device {device}")


def ask(name):
    qs, answers = questions(name)
    # Laya's questions are plain dicts; "type" says which kind of answer to give
    return agent.predict(STATE, {qid: {"type": "choice", **q} for qid, q in qs.items()}), answers


for name in names[:5]:  # warm-up: the first calls pay one-off GPU setup costs
    ask(name)

rows, seconds, cut_short = [], [], 0
for i, name in enumerate(names):
    start = time.perf_counter()
    response, answers = ask(name)
    if device.startswith("cuda"):
        torch.cuda.synchronize()  # the GPU runs ahead of Python; wait for it before timing
    seconds.append(time.perf_counter() - start)
    usage = response["usage"]
    # Laya reads at most a fixed number of tokens per question. If a name's questions didn't fit,
    # some options were merged or the state was cut, and the answer is not a fair one: count it.
    if usage.get("truncated") or usage.get("truncated_questions") or usage.get("options"):
        cut_short += 1
    for qid, real in answers.items():
        probs = response["answers"][qid]["probabilities"]
        p = probs.get(real, 0.0)
        rows.append({"name": name, "pos": int(qid[1:]), "real": real, "p": p,
                     "top": max(probs, key=probs.get)})
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(names)} names")

result = {
    "model": f"{REPO}@{revision}",
    "laya_version": version("laya"),
    "device": torch.cuda.get_device_name() if device.startswith("cuda") else device,
    "n_names": len(names),
    "n_predictions": len(rows),
    "loss": sum(-math.log(max(r["p"], FLOOR)) for r in rows) / len(rows),
    "loss_strict": sum(-math.log(max(r["p"], STRICT_FLOOR)) for r in rows) / len(rows),
    "zeros": sum(r["p"] == 0 for r in rows),
    "top_correct": sum(r["top"] == r["real"] for r in rows) / len(rows),
    "ms_per_name": statistics.median(seconds) * 1000,
    "names_cut_short": cut_short,  # should be 0; if not, the answers above are not trustworthy
}
print(json.dumps(result, indent=2))
if len(names) < 300:
    for r in rows:  # the probe is small enough to read prediction by prediction
        print(f"  {r['name'][:r['pos']]!r:<24} real {r['real']:<6} p {r['p']:.4f}  top {r['top']}")
else:
    out = ROOT / f"bench/results/laya-{device.split(':')[0]}.json"
    out.write_text(json.dumps({"result": result, "predictions": rows}, indent=1) + "\n")
    print(f"saved to {out.relative_to(ROOT)}")
