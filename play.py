"""Step 5: type a few letters, pick a model, watch it predict the next.

Run after training the three models (`python train.py`, `... tinytowns-small`, `... tinytowns`):
    python play.py              # interactive
    python play.py --compare    # just print the side-by-side table and exit

Type a prefix (e.g. `san`) to see the next-symbol chances and 5 completions. Commands:
    :model nano | small | tinytowns   switch model        :all       same prefix, all three models
    :t 0.7                            set temperature     :compare   side-by-side table
    :q                                quit

Every invented name is marked:
    real-train  in data/cities.txt AND in the training set: the model may have copied it
    real-new    in data/cities.txt but only in validation: the model never saw it, yet made it
    invented    not a real place in our list
"""
import random
import sys

import torch
import torch.nn.functional as F

from model import CONTEXT, Config, TinyTowns, itos, stoi

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)

# --- The three models, loaded once ------------------------------------------------
models = {}  # short name -> (model, best validation loss from training)
for short, size in [("nano", "tinytowns-nano"), ("small", "tinytowns-small"), ("tinytowns", "tinytowns")]:
    try:
        saved = torch.load(f"runs/{size}.pt", map_location=device)
    except FileNotFoundError:
        print(f"(skipping {size}: run `python train.py {size}` first)")
        continue
    m = TinyTowns(Config(**saved["config"])).to(device)
    m.load_state_dict(saved["weights"])
    m.eval()
    models[short] = (m, saved.get("best", {}).get("val", float("nan")))
if not models:
    sys.exit("no trained models in runs/")

# --- The same split as train.py, so we know which real names the models trained on ---
with open("data/cities.txt", encoding="utf-8") as f:
    names = [line.strip() for line in f if line.strip()]
random.Random(SEED).shuffle(names)
cut = int(0.9 * len(names))
train_set, val_set = set(names[:cut]), set(names[cut:])


def kind(name):
    if name in train_set:
        return "real-train"
    return "real-new" if name in val_set else "invented"


# --- Inventing many names at once --------------------------------------------------
@torch.no_grad()
def complete(model, prefix, n, temperature, generator):
    """n names that start with `prefix`, all built side by side in one batch.

    ids is a [n, T] grid: n rows, one per name, all starting as '.' + prefix. Each loop turn
    runs ONE forward pass for all n rows, picks the next symbol for every row, and appends
    it as a new column. A row that has picked '.' (end) is done: we keep appending to it
    (the grid must stay a rectangle) but ignore everything after its first '.'.
    """
    start = [0] + [stoi[ch] for ch in prefix]
    ids = torch.tensor([start] * n, device=device)              # [n, T]
    while ids.shape[1] < CONTEXT:
        scores, _ = model(ids)                                  # [n, T, 28]
        probs = F.softmax(scores[:, -1] / temperature, dim=-1)  # [n, 28]: last position only
        nxt = torch.multinomial(probs.cpu(), 1, generator=generator).to(device)  # [n, 1]
        ids = torch.cat([ids, nxt], dim=1)
        if (ids[:, len(start):] == 0).any(dim=1).all():         # every row has ended
            break
    out = []
    for row in ids[:, 1:].tolist():                             # drop the starting '.'
        row = row + [0]                                         # a name that hit 20 letters
        out.append("".join(itos[i] for i in row[: row.index(0)]))
    return out


@torch.no_grad()
def next_chances(model, prefix, temperature):
    x = torch.tensor([[0] + [stoi[ch] for ch in prefix]], device=device)
    scores, _ = model(x)
    return F.softmax(scores[0, -1] / temperature, dim=-1).cpu()  # [28]


def show(short, prefix, temperature):
    model, _ = models[short]
    p = next_chances(model, prefix, temperature)
    top = p.topk(8)
    label = lambda i: "end" if i == 0 else "_" if i == 1 else itos[i]
    print(f"  [{short}] after '.{prefix}': " +
          "  ".join(f"{label(i.item())} {v:.0%}" for v, i in zip(top.values, top.indices)))
    g = torch.Generator().manual_seed(SEED)
    for name in complete(model, prefix, 5, temperature, g):
        print(f"      {name:<22} {kind(name)}")


def compare(n=10000, temperature=1.0):
    """Side-by-side: n names from each model (no prefix), counted by kind."""
    print(f"\n{n} names per model, temperature {temperature}, seed {SEED}:")
    print(f"  {'model':10} {'val loss':>8} {'real-train':>11} {'real-new':>9} {'invented':>9} "
          f"{'different':>10} {'avg length':>11}")
    for short, (model, val) in models.items():
        # In chunks of 1,000: 10,000 rows at once doesn't fit in 4 GB for the 1M model.
        g = torch.Generator().manual_seed(SEED)
        made = [x for _ in range(n // 1000) for x in complete(model, "", 1000, temperature, g)]
        count = {k: sum(kind(x) == k for x in made) for k in ("real-train", "real-new", "invented")}
        print(f"  {short:10} {val:>8.4f} {count['real-train']:>11} {count['real-new']:>9} "
              f"{count['invented']:>9} {len(set(made)):>10} {sum(map(len, made)) / n:>11.1f}")
    print(f"  (for scale: the training set is {len(train_set):,} names, validation {len(val_set):,})")


if "--compare" in sys.argv:
    compare()
    sys.exit()

# --- The interactive loop ---------------------------------------------------------
current, temperature = list(models)[-1], 1.0  # start with the biggest model available
print(f"models: {', '.join(models)}   current: {current}   device: {device}")
print("type a prefix (letters and spaces), or :model, :all, :t, :compare, :q")
while True:
    try:
        line = input(f"{current} T={temperature}> ").strip().lower()
    except EOFError:
        break
    if line in (":q", ":quit"):
        break
    elif line.startswith(":model"):
        choice = line.split()[-1]
        if choice in models:
            current = choice
        else:
            print(f"  choose one of: {', '.join(models)}")
    elif line.startswith(":t"):
        try:
            temperature = max(0.05, float(line.split()[-1]))  # 0 would mean dividing by zero
        except ValueError:
            print("  e.g. :t 0.7")
    elif line == ":compare":
        compare(temperature=temperature)
    elif line.startswith(":all"):
        prefix = line[4:].strip()
        for short in models:
            show(short, prefix, temperature)
    elif all(ch in stoi and ch != "." for ch in line) and len(line) < CONTEXT - 1:
        show(current, line, temperature)
    else:
        print("  letters a-z and spaces only, at most 19 of them (or a :command)")
