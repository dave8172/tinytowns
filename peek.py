"""Step 3a: look inside the trained tinytowns-nano. Every weight by name, then one name traced by hand.

Run after training (`python train.py`):
    python peek.py            # traces 'narita'
    python peek.py london     # traces any name you like

What it does:
  1) Lists every weight grid: its name, its shape, how many numbers, and what it is in our words.
  2) Runs the forward pass BY HAND for one name, one small stage at a time, printing each shape.
  3) Checks that the hand-made scores equal model(x). If they do, the hand trace IS the model.
  4) Draws the attention shares as a heat map and saves it to runs/attention_<name>.png.

Only tinytowns-nano for now (1 layer, 1 head). That keeps the trace short and the picture single.
"""
import math
import sys

import matplotlib
matplotlib.use("Agg")  # draw into a file, not a window: works everywhere, never blocks
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from model import ALPHABET, Config, TinyTowns, itos, stoi

# --- Load the trained model --------------------------------------------------
# train.py saved a dict: the config (sizes) and the weights (every grid, by name).
saved = torch.load("runs/tinytowns-nano.pt", map_location="cpu")
cfg = Config(**saved["config"])
model = TinyTowns(cfg)
model.load_state_dict(saved["weights"])
model.eval()

# --- 1) Every weight, by name ------------------------------------------------
# The names come from the attribute names in model.py: self.lookup -> "lookup",
# self.layers[0].attention.q -> "layers.0.attention.q". ".weight" is the grid, ".bias" the
# list of numbers added after it. Note: nn.Linear stores its grid as [out, in], so the
# Q grid shows as [16, 16] and the knowledge step's "up" grid (16 -> 64) shows as [64, 16].
words = {
    "lookup.weight": "lookup table: one 16-number row per symbol",
    "position.weight": "position table: one 16-number row per position",
    "layers.0.norm1.weight": "tidy-up before attention: scale",
    "layers.0.norm1.bias": "tidy-up before attention: shift",
    "layers.0.attention.q.weight": "Q grid",
    "layers.0.attention.q.bias": "Q grid: added after",
    "layers.0.attention.k.weight": "K grid",
    "layers.0.attention.k.bias": "K grid: added after",
    "layers.0.attention.v.weight": "V grid",
    "layers.0.attention.v.bias": "V grid: added after",
    "layers.0.attention.out.weight": "attention out grid: mix -> note update",
    "layers.0.attention.out.bias": "attention out grid: added after",
    "layers.0.norm2.weight": "tidy-up before knowledge step: scale",
    "layers.0.norm2.bias": "tidy-up before knowledge step: shift",
    "layers.0.knowledge.up.weight": "knowledge step: 16 -> 64 grid",
    "layers.0.knowledge.up.bias": "knowledge step: 16 -> 64, added after",
    "layers.0.knowledge.down.weight": "knowledge step: 64 -> 16 grid",
    "layers.0.knowledge.down.bias": "knowledge step: 64 -> 16, added after",
    "norm.weight": "last tidy-up before output: scale",
    "norm.bias": "last tidy-up before output: shift",
    "output.weight": "output table: one 16-number row per symbol",
    "output.bias": "output table: one number per symbol, added after",
}
print(f"{'name':32} {'shape':10} {'count':>6}  in our words")
total = 0
for name, w in model.named_parameters():  # weights only; the fixed mask is not a weight
    total += w.numel()
    print(f"{name:32} {str(list(w.shape)):10} {w.numel():>6}  {words.get(name, '?')}")
print(f"{'total':43} {total:>6}\n")

# --- 2) One name, by hand ----------------------------------------------------
name = sys.argv[1] if len(sys.argv) > 1 else "narita"
symbols = "." + name          # what goes in: '.', then the letters (we don't feed the final '.')
x = torch.tensor([[stoi[ch] for ch in symbols]])  # [1, T]: a batch of one name
T = x.shape[1]
L = model.layers[0]           # the one layer

def show(label, grid):
    print(f"  {label:44} {list(grid.shape)}")

print(f"Tracing '{symbols}' ({T} positions) by hand:")
with torch.no_grad():
    # Lookup + position: row x[t] of the lookup table, plus row t of the position table.
    note = model.lookup.weight[x] + model.position.weight[:T]
    show("note = lookup row + position row", note)

    # Tidy-up (LayerNorm) by hand: make each note average 0, spread 1, then scale and shift.
    # The 1e-5 only avoids dividing by 0; it is the same value nn.LayerNorm uses.
    def tidy(n, norm):
        mean = n.mean(-1, keepdim=True)
        var = n.var(-1, keepdim=True, unbiased=False)
        return (n - mean) / torch.sqrt(var + 1e-5) * norm.weight + norm.bias

    # Grid step by hand: note x grid + bias. ".T" flips the stored [out, in] grid to [in, out].
    def grid(n, lin):
        return n @ lin.weight.T + lin.bias

    a_in = tidy(note, L.norm1)
    show("tidied note", a_in)
    q, k, v = grid(a_in, L.attention.q), grid(a_in, L.attention.k), grid(a_in, L.attention.v)
    show("q, k, v (each)", q)

    # Score: every q against every k. [1, T, 16] @ [1, 16, T] -> [1, T, T]. Divide by sqrt(16) = 4.
    score = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    show("score = q . k / 4", score)
    # Mask: a position may not look at later positions.
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    score = score.masked_fill(future, float("-inf"))
    shares = F.softmax(score, dim=-1)
    show("shares (each row sums to 1)", shares)
    mix = shares @ v                                  # [1, T, T] @ [1, T, 16] -> [1, T, 16]
    show("mix = shares x v", mix)
    note = note + grid(mix, L.attention.out)          # first add onto the note
    show("note + attention update", note)

    # Knowledge step: 16 -> 64, bend (GELU), 64 -> 16. Each position on its own.
    middle = F.gelu(grid(tidy(note, L.norm2), L.knowledge.up))
    show("knowledge middle (after the bend)", middle)
    note = note + grid(middle, L.knowledge.down)      # second add onto the note
    show("note + knowledge update", note)

    scores = grid(tidy(note, model.norm), model.output)
    show("scores = output table (28 per position)", scores)

    # --- 3) Does the hand trace match the real model? ---
    real, _ = model(x)
    gap = (scores - real).abs().max().item()
    print(f"\nlargest difference from model(x): {gap:.2e}  "
          f"({'same model' if gap < 1e-4 else 'MISMATCH'})")

    # What each position predicts next: the top 3 symbols and their chances.
    probs = F.softmax(scores[0], dim=-1)  # [T, 28]
    print("\nAfter each prefix, the 3 likeliest next symbols:")
    for t in range(T):
        top = probs[t].topk(3)
        bets = "  ".join(f"{itos[i.item()]!r} {p:.2f}" for p, i in zip(top.values, top.indices))
        print(f"  {symbols[: t + 1]:22} {bets}")

# --- 4) The attention shares, as numbers and as a picture -------------------
s = shares[0]  # [T, T]
print("\nAttention shares (row = position looking, column = position looked at):")
print("      " + "".join(f"{ch:>6}" for ch in symbols))
for i, ch in enumerate(symbols):
    print(f"{ch:>6}" + "".join(f"{s[i, j]:6.2f}" if j <= i else "     -" for j in range(T)))

fig, ax = plt.subplots(figsize=(5, 4.5))
img = ax.imshow(s.masked_fill(future, float("nan")), cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(T), list(symbols))
ax.set_yticks(range(T), list(symbols))
ax.set_xlabel("looked at")
ax.set_ylabel("looking")
ax.set_title(f"tinytowns-nano attention shares: '{symbols}'")
for i in range(T):
    for j in range(i + 1):
        ax.text(j, i, f"{s[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if s[i, j] > 0.5 else "black")
fig.colorbar(img, ax=ax, fraction=0.046)
fig.tight_layout()
path = f"runs/attention_{name.replace(' ', '_')}.png"
fig.savefig(path, dpi=120)
print(f"\nsaved {path}")
