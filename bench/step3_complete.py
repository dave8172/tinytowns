"""Benchmark step 3: finish a name from its first 3 letters. Claude Haiku 5.5 against our models.

Claude gives no probabilities, so it can't have a loss. Instead every model gets the same job:
the first 3 letters of each of the 300 test names, finish the name. Each answer is checked
against data/cities.txt, as in play.py:
    real-train  a real place our models trained on (they may have copied it)
    real-new    a real place only in validation: never seen by our models
    invented    not in the list (made up, or a place too small or spelled differently)
    exact       the share that is the very name the 3 letters came from
Our models answer twice: greedy (always the most likely next symbol, like a best guess) and
sampled at temperature 1 (their natural spread). Claude answers once with default settings.

    python bench/step3_complete.py --ours          # free: our three models only
    python bench/step3_complete.py --n 1           # Claude on one prefix: real tokens and cost
    python bench/step3_complete.py                 # our models and Claude, all 300
Claude needs ANTHROPIC_API_KEY in the environment.
"""
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.prepare import clean  # noqa: E402  the same cleaning as cities.txt
from model import CONTEXT, itos, load_weights, stoi  # noqa: E402

SEED = 42
CLAUDE = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.50 / 1e6  # $ per token, prompts under 100K (claude.com pricing)

names = [line for line in (ROOT / "bench/names.txt").read_text(encoding="utf-8").splitlines() if line]
n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else len(names)
names = names[:n]

with open(ROOT / "data/cities.txt", encoding="utf-8") as f:
    everything = [line.strip() for line in f if line.strip()]
random.Random(SEED).shuffle(everything)
cut = int(0.9 * len(everything))
train_set, val_set = set(everything[:cut]), set(everything[cut:])


def weights_file(size):
    """runs/ if you trained them yourself; otherwise the released copy from Hugging Face.
    (Copied from step1_ours.py: importing that file would run the whole of step 1.)"""
    local = ROOT / f"runs/{size}.safetensors"
    if local.exists():
        return str(local)
    from huggingface_hub import hf_hub_download
    return hf_hub_download("dave8172/tinytowns", f"{size}.safetensors")


def kind(answer):
    if answer in train_set:
        return "real-train"
    return "real-new" if answer in val_set else "invented"


def summary(rows, seconds):
    out = {k: sum(r["kind"] == k for r in rows) / len(rows)
           for k in ("real-train", "real-new", "invented")}
    out["exact"] = sum(r["answer"] == r["name"] for r in rows) / len(rows)
    out["ms_per_name"] = statistics.median(seconds) * 1000
    return out


# --- Our models: predict, pick, feed back in, until '.' ----------------------------------------
@torch.no_grad()
def finish(model, prefix, greedy, generator):
    ids = [0] + [stoi[ch] for ch in prefix]
    while len(ids) < CONTEXT:
        probs = F.softmax(model(torch.tensor([ids]))[0][0, -1], dim=-1)  # last position only
        nxt = probs.argmax().item() if greedy else torch.multinomial(probs, 1, generator=generator).item()
        if nxt == 0:  # '.' = end of name
            break
        ids.append(nxt)
    return "".join(itos[i] for i in ids[1:])


results = {}
if "--ours" in sys.argv or n == 300:
    for size in ["tinytowns-nano", "tinytowns-small", "tinytowns"]:
        model, _ = load_weights(weights_file(size))
        for greedy in (True, False):
            g = torch.Generator().manual_seed(SEED)
            rows, seconds = [], []
            for name in names:
                start = time.perf_counter()
                answer = finish(model, name[:3], greedy, g)
                seconds.append(time.perf_counter() - start)
                rows.append({"name": name, "answer": answer, "kind": kind(answer)})
            label = f"{size} ({'greedy' if greedy else 'temperature 1'})"
            results[label] = {"summary": summary(rows, seconds), "answers": rows}
            print(f"  {label:34} {json.dumps(results[label]['summary'])}")

# --- Claude Haiku 5.5: one request per prefix --------------------------------------------------
if "--ours" not in sys.argv:
    import anthropic
    client = anthropic.Anthropic()
    # The same facts our models learned from the data: real place names, lowercase a-z and spaces.
    system = ("A real place name is being written one symbol at a time. Guess what comes next. "
              "Cities, towns and villages anywhere in the world with at least 1,000 people "
              "(the GeoNames list). Written in lowercase, letters a-z and spaces only, "
              "accents removed, at most 20 symbols long. You are given the first letters of one "
              "such name. Reply with the whole name only: no other words, no punctuation.")
    rows, seconds, tokens_in, tokens_out = [], [], 0, 0
    for name in names:
        start = time.perf_counter()
        msg = client.messages.create(model=CLAUDE, max_tokens=4000, system=system,
                                     messages=[{"role": "user", "content": f"It starts with: {name[:3]}"}])
        seconds.append(time.perf_counter() - start)
        tokens_in += msg.usage.input_tokens
        tokens_out += msg.usage.output_tokens  # includes any thinking tokens: those are billed too
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        answer = clean(text) or ""  # same cleaning as the data; "" if it can't be a name here
        rows.append({"name": name, "answer": answer, "raw": text, "kind": kind(answer)})
    cost = tokens_in * PRICE_IN + tokens_out * PRICE_OUT
    s = summary(rows, seconds)
    s.update({"model": msg.model, "input_tokens": tokens_in, "output_tokens": tokens_out,
              "cost_usd": cost, "cost_per_1000_names_usd": cost / len(names) * 1000})
    results[CLAUDE] = {"summary": s, "answers": rows}
    print(f"  {CLAUDE:34} {json.dumps(s)}")
    if n < 300:
        print(f"estimate for 300 names: ${cost / n * 300:.4f}")
        for r in rows:
            print(f"  {r['name'][:3]!r} -> {r['raw']!r} ({r['kind']}; real name {r['name']!r})")

if n == 300:
    out = ROOT / "bench/results/complete.json"
    out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"saved to {out.relative_to(ROOT)}")
