"""Benchmark step 2: open general language models on the same 300 names, both jobs.

A general language model also predicts the next piece of text, but its pieces are tokens (whole
words or parts of words), not letters. So we score the whole name at once:
    surprise of a name = -ln(probability the model gives to "name" + "end of name")
    loss = total surprise over all 300 names / number of symbols (letters, spaces, ends)
That is the same unit as our models' loss: average surprise per symbol. The model first reads 20
example names from the training list, one per line, so it knows the format; a new line is its
"end of name".

Job 2 is the same as step 3: after the examples, the first 3 letters; the model writes the rest
of the line, always taking its most likely next token (greedy, like our "best guess").

    pip install transformers accelerate                  # once
    python bench/step2_llms.py Qwen/Qwen2.5-1.5B --n 5   # 5 names: check it loads and fits
    python bench/step2_llms.py Qwen/Qwen2.5-1.5B         # all 300 -> bench/results/llm-<name>.json
    python bench/step2_llms.py HuggingFaceTB/SmolLM2-1.7B
"""
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.prepare import clean  # noqa: E402  the same cleaning as cities.txt

SEED, N_EXAMPLES = 42, 20
model_id = sys.argv[1]
n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 300
names = [line for line in (ROOT / "bench/names.txt").read_text(encoding="utf-8").splitlines() if line][:n]

# --- The same split as train.py; examples come from the training names only ------------------
with open(ROOT / "data/cities.txt", encoding="utf-8") as f:
    everything = [line.strip() for line in f if line.strip()]
random.Random(SEED).shuffle(everything)
cut = int(0.9 * len(everything))
train, val = everything[:cut], everything[cut:]
train_set, val_set = set(train), set(val)
context = "\n".join(random.Random(SEED).sample(train, N_EXAMPLES)) + "\n"


def kind(answer):
    if answer in train_set:
        return "real-train"
    return "real-new" if answer in val_set else "invented"


# --- Load: on the GPU in 16-bit numbers if there is one; layers that don't fit go to normal memory
cuda = torch.cuda.is_available()
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.float16 if cuda else torch.float32,
    device_map="auto" if cuda else None).eval()
first = next(model.parameters()).device
where = sorted({str(d) for d in getattr(model, "hf_device_map", {"": str(first)}).values()})
print(f"{model_id}: {sum(p.numel() for p in model.parameters()):,} weights, on {where}")
ctx_ids = tok(context, return_tensors="pt").input_ids[0]


@torch.no_grad()
def surprise(name):
    """-ln P(name + newline | examples), summed over the name's tokens."""
    ids = tok(context + name + "\n", return_tensors="pt").input_ids[0]
    # The examples must be split into the same tokens on their own and with the name after them;
    # otherwise the probabilities would belong to a slightly different text.
    assert torch.equal(ids[: len(ctx_ids)], ctx_ids), f"token boundary moved for {name!r}"
    scores = model(ids[None].to(first)).logits[0].float()        # [tokens, vocabulary]
    logp = F.log_softmax(scores[:-1], dim=-1)                     # position t predicts token t+1
    targets = ids[1:].to(first)
    per_token = -logp[torch.arange(len(targets)), targets]
    return per_token[len(ctx_ids) - 1:].sum().item()              # only the name's tokens


@torch.no_grad()
def finish(prefix):
    ids = tok(context + prefix, return_tensors="pt").input_ids.to(first)
    out = model.generate(ids, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)
    text = prefix + tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return clean(text.split("\n")[0]) or ""                       # one line, cleaned like the data


def timed(fn, arg):
    start = time.perf_counter()
    value = fn(arg)
    if cuda:
        torch.cuda.synchronize()  # the GPU runs ahead of Python; wait for it before timing
    return value, time.perf_counter() - start


for name in names[:3]:  # warm-up: the first calls pay one-off setup costs
    surprise(name)

rows, t_score, t_finish = [], [], []
for i, name in enumerate(names):
    s, sec = timed(surprise, name)
    answer, sec2 = timed(finish, name[:3])
    t_score.append(sec)
    t_finish.append(sec2)
    rows.append({"name": name, "surprise": s, "answer": answer, "kind": kind(answer)})
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(names)} names")

symbols = sum(len(r["name"]) + 1 for r in rows)  # letters and spaces, plus one "end" per name
result = {
    "model": model_id,
    "weights": sum(p.numel() for p in model.parameters()),
    "device": torch.cuda.get_device_name() if cuda else "cpu",
    "placed_on": where,  # more than one entry: part of the model did not fit on the GPU
    "n_names": len(rows),
    "loss": sum(r["surprise"] for r in rows) / symbols,
    "ms_per_name": statistics.median(t_score) * 1000,
    "real": sum(r["kind"] != "invented" for r in rows) / len(rows),
    "exact": sum(r["answer"] == r["name"] for r in rows) / len(rows),
    "finish_ms_per_name": statistics.median(t_finish) * 1000,
}
print(json.dumps(result, indent=2))
if n < 300:
    for r in rows:
        print(f"  {r['name']:<22} surprise {r['surprise']:6.2f}   {r['name'][:3]} -> {r['answer']} ({r['kind']})")
else:
    out = ROOT / f"bench/results/llm-{model_id.split('/')[-1].lower()}.json"
    out.write_text(json.dumps({"result": result, "names": rows}, indent=1) + "\n")
    print(f"saved to {out.relative_to(ROOT)}")
