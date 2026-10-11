"""Step 2b: train the transformer by nudging its weights to lower the loss.

    python train.py                     # tinytowns-nano (default)
    python train.py tinytowns           # the bigger model
    python train.py --device cpu        # force the CPU even if a GPU is present

Prints train and validation loss as it learns, saves the weights to runs/<size>.pt,
and invents 10 names at the end.
"""
import argparse
import math
import random
import time

import torch

from model import CONFIGS, TinyTowns, encode, sample

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("size", nargs="?", default="tinytowns-nano", choices=CONFIGS)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--steps", type=int, default=6000)  # how many nudges
parser.add_argument("--batch", type=int, default=64)    # names per nudge
parser.add_argument("--lr", type=float, default=3e-3)   # learning rate = nudge size
args = parser.parse_args()

SEED = 42
EVAL_EVERY = 500  # how often (in steps) to print train/validation loss

torch.manual_seed(SEED)  # same random starting weights every run
cfg = CONFIGS[args.size]
device = torch.device(args.device)

# ---------------------------------------------------------------------------
# Data: exactly the same split as step 1, so the losses are comparable
# ---------------------------------------------------------------------------
with open("data/cities.txt", encoding="utf-8") as f:
    names = [line.strip() for line in f if line.strip()]
random.Random(SEED).shuffle(names)
cut = int(0.9 * len(names))
train_names, val_names = names[:cut], names[cut:]

# Encode every name once, up front: [28122, 21] and [3125, 21] grids of numbers.
x_train, y_train = (t.to(device) for t in encode(train_names))
x_val, y_val = (t.to(device) for t in encode(val_names))
print(f"{args.size} on {device}: {len(train_names)} train names, {len(val_names)} validation")

# ---------------------------------------------------------------------------
# Model and optimiser
# ---------------------------------------------------------------------------
model = TinyTowns(cfg).to(device)
print(f"weights: {sum(w.numel() for w in model.parameters()):,}")

# AdamW: a smarter nudger than plain "weight -= lr x gradient". For each weight it keeps
# a running average of recent gradients (momentum: consistent pushes build up speed) and
# of their size (a weight with jumpy gradients takes smaller steps). weight_decay gently
# pulls every weight towards 0, which discourages relying on a few huge weights.
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)


@torch.no_grad()  # grading only: no gradients needed, which saves time and memory
def full_loss(x, y, chunk=4096):
    """Loss over ALL the given names (not just one batch), computed in chunks."""
    model.eval()  # "grading mode" (no effect in this model yet, but the right habit)
    total, count = 0.0, 0
    for i in range(0, len(x), chunk):
        xb, yb = x[i : i + chunk], y[i : i + chunk]
        _, loss = model(xb, yb)
        n = (yb != -1).sum().item()  # how many real (not padding) predictions in this chunk
        total += loss.item() * n     # loss is an average, so x n gives this chunk's total
        count += n
    model.train()  # back to "learning mode"
    return total / count


# ---------------------------------------------------------------------------
# The training loop: predict, score, find gradients, nudge. Repeat.
# ---------------------------------------------------------------------------
g = torch.Generator().manual_seed(SEED)  # makes the random batch choices repeatable
print(f"\n{'step':>6} {'train loss':>11} {'val loss':>9} {'seconds':>8}")
print(f"{0:>6} {full_loss(x_train, y_train):>11.4f} {full_loss(x_val, y_val):>9.4f} {0:>8.1f}")
start = time.time()

for step in range(1, args.steps + 1):
    # 1) Pick a batch: 64 random training names (their row numbers).
    rows = torch.randint(0, len(x_train), (args.batch,), generator=g).to(device)
    xb, yb = x_train[rows], y_train[rows]

    # 2) Forward pass: predict every next symbol in the batch and score with the loss.
    _, loss = model(xb, yb)

    # 3) Backward pass (backpropagation): for every one of the weights, work out
    #    "if this weight went up a tiny bit, how much would the loss change?" (its gradient).
    #    zero_grad first, because PyTorch adds new gradients onto old ones otherwise.
    optimizer.zero_grad()
    loss.backward()

    # 4) Nudge: move every weight a small step in the direction that lowers the loss.
    optimizer.step()

    if step % EVAL_EVERY == 0 or step == args.steps:
        tl, vl = full_loss(x_train, y_train), full_loss(x_val, y_val)
        print(f"{step:>6} {tl:>11.4f} {vl:>9.4f} {time.time() - start:>8.1f}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print(f"\nbars to beat (validation): even guess {math.log(28):.4f}, counting table (step 1) 2.6616")

# Save the learned weights (state_dict = every weight grid, by name) plus the config.
# Step 6 will switch this to the safetensors format.
import os
os.makedirs("runs", exist_ok=True)
path = f"runs/{args.size}.pt"
torch.save({"config": cfg.__dict__, "weights": model.state_dict()}, path)
print(f"saved {path}")

all_names = set(names)
print("\n10 invented names (temperature 1.0):")
for name in sample(model, 10, generator=torch.Generator().manual_seed(SEED)):
    print(f"  {name:<24} {'real' if name in all_names else 'invented'}")
