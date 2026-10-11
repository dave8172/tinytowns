"""Step 1: letters <-> numbers, and a counting baseline (bigram model).

Counts which symbol follows which in the training names, turns the counts into
probabilities, measures the loss on train and validation, and samples 10 names.
"""
import math
import random

import torch

SEED = 42

# --- Letters <-> numbers ---------------------------------------------------
# Index 0 is '.', which marks both the start and the end of a name.
ALPHABET = ". abcdefghijklmnopqrstuvwxyz"
stoi = {ch: i for i, ch in enumerate(ALPHABET)}  # symbol -> number
itos = {i: ch for i, ch in enumerate(ALPHABET)}  # number -> symbol
V = len(ALPHABET)  # 28

# --- Load and split --------------------------------------------------------
with open("data/cities.txt", encoding="utf-8") as f:
    names = [line.strip() for line in f if line.strip()]

all_names = set(names)  # for "real or invented?" checks
random.Random(SEED).shuffle(names)  # shuffle first: the file is alphabetical
cut = int(0.9 * len(names))
train, val = names[:cut], names[cut:]
print(f"names: {len(names)} total, {len(train)} train, {len(val)} validation")


def pairs(name):
    """Every (current, next) pair in '.' + name + '.', as numbers."""
    ids = [0] + [stoi[ch] for ch in name] + [0]
    return zip(ids, ids[1:])


# --- Count -----------------------------------------------------------------
# Start every cell at 1 (add-one smoothing): a pair never seen in training
# gets a small probability instead of 0, so the loss can't become infinite.
counts = torch.ones((V, V))
for name in train:  # only training names: validation stays unseen
    for a, b in pairs(name):
        counts[a, b] += 1

# Each row becomes "probabilities of the next symbol, given this one".
probs = counts / counts.sum(dim=1, keepdim=True)


# --- Loss ------------------------------------------------------------------
def loss(name_list):
    """Average of -ln(probability given to the symbol that really came next)."""
    total, n = 0.0, 0
    for name in name_list:
        for a, b in pairs(name):
            total += -math.log(probs[a, b].item())
            n += 1
    return total / n


print(f"\nuniform guess (1/28 each): loss {math.log(V):.4f}")
print(f"bigram, train:              loss {loss(train):.4f}")
print(f"bigram, validation:         loss {loss(val):.4f}")

# One name, pair by pair, to show where the number comes from.
example = val[0]
print(f"\nworked example, validation name '{example}':")
scores = []
for a, b in pairs(example):
    p = probs[a, b].item()
    scores.append(-math.log(p))
    print(f"  after '{itos[a]}' -> '{itos[b]}'  p = {p:.3f}   -ln p = {scores[-1]:.3f}")
print(f"  average = {sum(scores) / len(scores):.4f}  (this name's loss)")

# --- Sample ----------------------------------------------------------------
# Inventing = predicting in a loop: pick a next symbol by its probability,
# feed it back in, stop when '.' (end of name) is picked.
g = torch.Generator().manual_seed(SEED)
print("\n10 sampled names:")
for _ in range(10):
    i, out = 0, []
    while True:
        i = torch.multinomial(probs[i], num_samples=1, generator=g).item()
        if i == 0:
            break
        out.append(itos[i])
    name = "".join(out)
    print(f"  {name:<24} {'real' if name in all_names else 'invented'}")
