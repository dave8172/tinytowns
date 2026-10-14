"""Benchmark step 1: the 300 test names, and how our own models score on them.

Every model in the benchmark (ours, Jev, Laya, Claude) is graded on the same 300 validation
names, so this step picks them once and saves them to bench/names.txt. Then it measures the
counting baseline and the three tinytowns sizes on those names:
    loss      average -ln(probability given to the real next symbol), as in training
    full      the same loss on all 14,271 validation names: shows how far 300 names are from it
    top       share of predictions where the model's most likely symbol was the real one
    ms/name   time to score one name: one forward pass gives every position's prediction at once
Run from the repo root:  python bench/step1_ours.py
Results go to bench/results/ours-cpu.json (or ours-cuda.json on a GPU) for the final table.
"""
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `import model` finds model.py one folder up
from model import encode, load_weights, stoi  # noqa: E402

SEED, N_TEST = 42, 300
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- The same split as train.py, then 300 names drawn from validation ----------------------
with open(ROOT / "data/cities.txt", encoding="utf-8") as f:
    names = [line.strip() for line in f if line.strip()]
random.Random(SEED).shuffle(names)
cut = int(0.9 * len(names))
train, val = names[:cut], names[cut:]
# A fixed random draw, not the first 300: the order already is random, but drawing makes the
# choice explicit and lets anyone re-draw a different 300 by changing one number.
test = random.Random(SEED).sample(val, N_TEST)
(ROOT / "bench/names.txt").write_text("\n".join(test) + "\n", encoding="utf-8")
n_predictions = sum(len(x) + 1 for x in test)  # each letter, plus "end of name"
print(f"{N_TEST} test names, {n_predictions} predictions ({n_predictions / N_TEST:.1f} per name)")
print(f"saved to bench/names.txt; e.g. {', '.join(test[:5])}")


# --- Counting baseline (step 1's bigram), rebuilt from the training names -------------------
counts = torch.ones((28, 28))  # add-one smoothing, as in step1_bigram.py
for name in train:
    ids = [0] + [stoi[ch] for ch in name] + [0]
    for a, b in zip(ids, ids[1:]):
        counts[a, b] += 1
bigram = counts / counts.sum(dim=1, keepdim=True)


def bigram_score(name_list):
    """(loss, top): loss averaged over predictions, not names (long names count more, as in
    training); top = share of predictions where the most likely symbol was the real one."""
    total, top, n = 0.0, 0, 0
    for name in name_list:
        ids = [0] + [stoi[ch] for ch in name] + [0]
        for a, b in zip(ids, ids[1:]):
            total += -math.log(bigram[a, b].item())
            top += bigram[a].argmax().item() == b
            n += 1
    return total / n, top / n


# --- Our three models ----------------------------------------------------------------------
def weights_file(size):
    """runs/ if you trained them yourself; otherwise the released copy from Hugging Face."""
    local = ROOT / f"runs/{size}.safetensors"
    if local.exists():
        return str(local)
    from huggingface_hub import hf_hub_download
    return hf_hub_download("dave8172/tinytowns", f"{size}.safetensors")


@torch.no_grad()
def model_score(model, name_list, chunk=1000):
    """(loss, top) over every prediction in name_list, in chunks so 14,271 names fit in memory."""
    total, top, n = 0.0, 0, 0
    for i in range(0, len(name_list), chunk):
        x, y = encode(name_list[i : i + chunk])
        scores, _ = model(x.to(device))
        # reduction="sum" adds instead of averaging, so chunks of different sizes combine fairly
        total += F.cross_entropy(scores.view(-1, 28), y.to(device).view(-1),
                                 ignore_index=-1, reduction="sum").item()
        top += (scores.argmax(dim=-1).cpu() == y).sum().item()  # padding (-1) never matches
        n += (y != -1).sum().item()
    return total / n, top / n


@torch.no_grad()
def ms_per_name(model, name_list):
    """Median time to score ONE name: '.' + name in, 28 probabilities for every position out.

    One name at a time, because that is how the API models will be called (one request per
    name). The median ignores the odd slow run caused by something else on the machine.
    """
    for name in name_list[:20]:  # warm-up: the first calls pay one-off setup costs
        model(encode([name])[0].to(device))
    times = []
    for name in name_list:
        x = encode([name])[0][:, : len(name) + 1].to(device)  # trim the padding: real length only
        start = time.perf_counter()
        F.softmax(model(x)[0], dim=-1)
        if device.type == "cuda":
            torch.cuda.synchronize()  # the GPU runs ahead of Python; wait for it before timing
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


results = {"device": str(device), "threads": torch.get_num_threads(), "n_test": N_TEST,
           "n_predictions": n_predictions, "models": {}}
loss, top = bigram_score(test)
results["models"]["bigram"] = {"weights": 28 * 28, "loss": loss, "full": bigram_score(val)[0],
                               "top": top}
for size in ["tinytowns-nano", "tinytowns-small", "tinytowns"]:
    model, _ = load_weights(weights_file(size), device)
    loss, top = model_score(model, test)
    results["models"][size] = {
        "weights": sum(w.numel() for w in model.parameters()),
        "loss": loss,
        "full": model_score(model, val)[0],
        "top": top,
        "ms_per_name": ms_per_name(model, test),
    }

print(f"\ndevice: {device} ({results['threads']} CPU threads)")
print(f"  {'model':16} {'weights':>10} {'loss (300)':>11} {'full val':>9} {'top':>6} {'ms/name':>8}")
print(f"  {'uniform guess':16} {'':>10} {math.log(28):>11.4f} {math.log(28):>9.4f} {1 / 28:>6.1%}")
for name, r in results["models"].items():
    ms = f"{r['ms_per_name']:.2f}" if "ms_per_name" in r else ""
    print(f"  {name:16} {r['weights']:>10,} {r['loss']:>11.4f} {r['full']:>9.4f} {r['top']:>6.1%} {ms:>8}")

out = ROOT / f"bench/results/ours-{device.type}.json"  # one file per machine type
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results, indent=2) + "\n")
print(f"\nsaved to {out.relative_to(ROOT)}")
