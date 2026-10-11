"""Step 2b: train the transformer by nudging its weights to lower the loss.

    python train.py                     # tinytowns-nano (default)
    python train.py tinytowns-small     # ~105K weights
    python train.py tinytowns           # ~1M weights
    python train.py --device cpu        # force the CPU even if a GPU is present

Prints train and validation loss as it learns. Every time the validation loss reaches a new
best, saves the weights to runs/<size>.safetensors, so the file always holds the model at its best moment
on unseen names (a big model may start memorising later). Invents 10 names at the end.

Careful: even a short test run (--steps 2) overwrites runs/<size>.safetensors with a barely trained model.
"""
import argparse
import math
import os
import random
import time

import torch

from model import CONFIGS, TinyTowns, encode, load_weights, sample, save_weights

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("size", nargs="?", default="tinytowns-nano", choices=CONFIGS)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--steps", type=int, default=6000)  # how many nudges
parser.add_argument("--batch", type=int)  # names per nudge (default: see below)
parser.add_argument("--lr", type=float)   # learning rate = nudge size (default: see below)
args = parser.parse_args()
# Batch: on a GPU, 256 names cost about the same time per step as 64 (measured: the GPU is mostly
# waiting on tiny jobs), so each nudge sees 4x more names for free. On a CPU, 64 is faster.
if args.batch is None:
    args.batch = 256 if args.device == "cuda" else 64
# Nudge size: big, deep models tend to become unstable with large nudges, so the 1M model
# starts smaller. (3e-3 is what tinytowns-nano was trained with in step 2.)
if args.lr is None:
    args.lr = 1e-3 if args.size == "tinytowns" else 3e-3

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

# Encode every name once, up front: [128435, 21] and [14271, 21] grids of numbers.
x_train, y_train = (t.to(device) for t in encode(train_names))
x_val, y_val = (t.to(device) for t in encode(val_names))
print(f"{args.size} on {device}: {len(train_names)} train names, {len(val_names)} validation")
print(f"batch {args.batch}, learning rate {args.lr}, {args.steps} steps")

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
os.makedirs("runs", exist_ok=True)
path = f"runs/{args.size}.safetensors"
best = {"val": float("inf")}  # the lowest validation loss so far, and the step it happened at


def save_if_best(step, tl, vl):
    """Save the weights if this is the lowest validation loss yet. Returns a mark for the table."""
    if vl >= best["val"]:
        return ""
    best.update(step=step, train=tl, val=vl)
    # Every weight grid, by name, plus the config and when it was best (see save_weights).
    save_weights(model, path, best=dict(best))
    return "  <- best, saved"


g = torch.Generator().manual_seed(SEED)  # makes the random batch choices repeatable
print(f"\n{'step':>6} {'train loss':>11} {'val loss':>9} {'seconds':>8}")
tl, vl = full_loss(x_train, y_train), full_loss(x_val, y_val)
print(f"{0:>6} {tl:>11.4f} {vl:>9.4f} {0:>8.1f}{save_if_best(0, tl, vl)}")
start = time.time()

for step in range(1, args.steps + 1):
    # 1) Pick a batch: args.batch random training names (their row numbers).
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
        mark = save_if_best(step, tl, vl)
        print(f"{step:>6} {tl:>11.4f} {vl:>9.4f} {time.time() - start:>8.1f}{mark}")
seconds = time.time() - start

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print(f"\ntraining took {seconds:.1f} s")
print(f"best validation loss {best['val']:.4f} at step {best['step']} "
      f"(train loss there {best['train']:.4f}), saved in {path}")
print(f"bars to beat (validation): even guess {math.log(28):.4f}, counting table (step 1) 2.6732")

# From here on, use the best weights, not the last ones.
model, _ = load_weights(path, device)

# "Real" split in two: a real name from the training set may simply be copied;
# a real name that is only in the validation set was never shown to the model.
train_set, val_set = set(train_names), set(val_names)


def kind(name):
    if name in train_set:
        return "real (in training set)"
    return "real (new: validation only)" if name in val_set else "invented"


print("\n10 invented names (temperature 1.0, best weights):")
for name in sample(model, 10, generator=torch.Generator().manual_seed(SEED)):
    print(f"  {name:<24} {kind(name)}")
