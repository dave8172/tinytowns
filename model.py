"""Step 2a: the transformer, written by hand. Symbols in, next-symbol probabilities out.

Run on its own (`python model.py`) to print every weight and check that an untrained
model's loss is near ln 28 = 3.33: with random weights it knows nothing yet, so it
should guess every symbol about evenly.

How to read this file: start at `TinyTowns.forward` near the bottom. It is the whole
model in ~10 lines. Then go up into `Layer`, `Attention` and `Knowledge` for the parts.

A note on shapes. Comments like [B, T, C] give the size of a grid of numbers:
  B = batch      how many names we process at once (e.g. 64)
  T = positions  how many symbols per name (at most 21)
  C = note size  how many numbers in each note (16 for nano, 64 for tinytowns)
So [B, T, C] means: for every name, for every position, a note of C numbers.
"""
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Letters <-> numbers (same as step 1)
# ---------------------------------------------------------------------------
# The model only understands numbers, so every symbol gets an index 0..27.
# Index 0 '.' means "start of name" when it goes in, and "end of name" when it comes out.
ALPHABET = ". abcdefghijklmnopqrstuvwxyz"
stoi = {ch: i for i, ch in enumerate(ALPHABET)}  # "string to int":  'a' -> 2
itos = {i: ch for i, ch in enumerate(ALPHABET)}  # "int to string":   2 -> 'a'
CONTEXT = 21  # longest input the model sees: one '.' plus at most 20 letters


# ---------------------------------------------------------------------------
# Config: the only thing that differs between the two sizes
# ---------------------------------------------------------------------------
# The code below is identical for both models; only these numbers change.
@dataclass
class Config:
    note: int      # numbers per note (C). Bigger = each symbol can carry more information.
    layers: int    # how many (attention + knowledge step) layers are stacked
    heads: int     # attention heads per layer; each head looks for its own pattern
    mlp: int       # width of the knowledge step's middle (16 -> 64 -> 16 for nano)
    vocab: int = len(ALPHABET)  # 28 symbols
    context: int = CONTEXT      # 21 positions


CONFIGS = {
    "tinytowns-nano": Config(note=16, layers=1, heads=1, mlp=64),   # 4,572 weights
    "tinytowns": Config(note=64, layers=2, heads=4, mlp=256),       # 105,052 weights
}


# ---------------------------------------------------------------------------
# Turning names into model input
# ---------------------------------------------------------------------------
def encode(names):
    """Names -> (inputs x, targets y), each a [batch, 21] grid of numbers.

    Example, 'paris':
        position:  0  1  2  3  4  5  6 ...
        x (input): .  p  a  r  i  s  .  .  .   <- '.' + name, padded with '.'
        y (answer):p  a  r  i  s  . -1 -1 -1   <- name + '.', padded with -1

    Read it column by column: at position t, the model sees x[0..t] and must predict y[t].
    So one name gives one prediction per letter, plus one for "end of name".

    Why padding: a batch must be a neat rectangle, but names have different lengths.
    The padded inputs are harmless: the mask (in Attention) stops real positions from
    looking ahead at them. The padded targets are -1, which the loss is told to skip.
    """
    x = torch.zeros((len(names), CONTEXT), dtype=torch.long)       # all '.' to start
    y = torch.full((len(names), CONTEXT), -1, dtype=torch.long)    # all "skip" to start
    for row, name in enumerate(names):
        ids = [stoi[ch] for ch in name]
        x[row, 1 : len(ids) + 1] = torch.tensor(ids, dtype=torch.long)    # after the first '.'
        y[row, : len(ids) + 1] = torch.tensor(ids + [0], dtype=torch.long)  # name, then '.'
    return x, y


# ---------------------------------------------------------------------------
# Attention: each note gathers information from the notes before it
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        # The Q, K, V grids. nn.Linear(16, 16) is a 16x16 grid of weights plus 16 biases
        # (one number added to each output). note x grid + bias = q (or k, or v).
        # With several heads, all heads' grids sit side by side in one big grid;
        # we split the result into per-head slices in forward().
        self.q = nn.Linear(cfg.note, cfg.note)  # Q grid: "what am I looking for?"
        self.k = nn.Linear(cfg.note, cfg.note)  # K grid: "what do I contain?"
        self.v = nn.Linear(cfg.note, cfg.note)  # V grid: "what do I hand over if chosen?"
        # After mixing, one more grid turns the heads' results back into one note-shaped
        # update. With one head it is still useful: it decides how the mix is added in.
        self.out = nn.Linear(cfg.note, cfg.note)
        # The mask: future[i, j] is True when position j comes after position i.
        # torch.triu(..., diagonal=1) keeps only the part above the diagonal, e.g. for 4:
        #     . T T T      row = "me", column = "the one I might look at"
        #     . . T T      True = in my future = forbidden
        #     . . . T
        #     . . . .
        future = torch.triu(torch.ones(cfg.context, cfg.context, dtype=torch.bool), diagonal=1)
        # register_buffer: store it with the model (moves to GPU with it) but it is a fixed
        # table, not a weight, so training never changes it.
        self.register_buffer("future", future)

    def forward(self, notes):
        B, T, C = notes.shape  # batch, positions, numbers per note
        hs = C // self.heads   # numbers per head ("head size"): 16 for nano, 64/4 = 16 for tinytowns

        # 1) q, k, v for every position at once: [B, T, C] x grid -> [B, T, C].
        #    .view splits the last C numbers into (heads, hs); .transpose moves "heads" forward
        #    so each head becomes its own little [T, hs] problem: shape [B, heads, T, hs].
        #    With 1 head these two calls change nothing but the shape.
        q = self.q(notes).view(B, T, self.heads, hs).transpose(1, 2)
        k = self.k(notes).view(B, T, self.heads, hs).transpose(1, 2)
        v = self.v(notes).view(B, T, self.heads, hs).transpose(1, 2)

        # 2) Score: every q against every k, all in one matrix multiply.
        #    score[i, j] = q_i . k_j = "how relevant is position j to position i?"  -> [B, heads, T, T]
        #    Divide by sqrt(hs): a dot product of hs numbers tends to grow with hs, and big
        #    scores make softmax put ~100% on one position, which makes learning slow.
        score = q @ k.transpose(-2, -1) / math.sqrt(hs)

        # 3) Mask: set every "future" score to minus infinity. After softmax, e^-inf = 0,
        #    so no position can take anything from a later one. This is what lets the model
        #    train on all positions at once without cheating by looking at the answer.
        score = score.masked_fill(self.future[:T, :T], float("-inf"))

        # 4) Shares: softmax along each row turns scores into percentages that sum to 1.
        shares = F.softmax(score, dim=-1)

        # 5) Mix: each position's new information = its shares x everyone's v.
        #    [B, heads, T, T] @ [B, heads, T, hs] -> [B, heads, T, hs]
        mix = shares @ v

        # 6) Put the heads back side by side ([B, T, C]) and pass through the out grid.
        mix = mix.transpose(1, 2).reshape(B, T, C)
        return self.out(mix)


# ---------------------------------------------------------------------------
# Knowledge step: each note is processed on its own, no looking at neighbours
# ---------------------------------------------------------------------------
class Knowledge(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.up = nn.Linear(cfg.note, cfg.mlp)    # 16 -> 64: spread out into more room
        self.down = nn.Linear(cfg.mlp, cfg.note)  # 64 -> 16: squeeze back to note size

    def forward(self, notes):
        # GELU is a bend: roughly keeps positive numbers and squashes negative ones to ~0.
        # Without it, "x grid, then x grid" is the same as one combined grid, so the extra
        # 64-wide middle would add no new ability. The bend is what lets it learn rules like
        # "IF the note says X AND Y, THEN push towards letter Z".
        return self.down(F.gelu(self.up(notes)))


# ---------------------------------------------------------------------------
# One layer = attention, then knowledge step, each added onto the note
# ---------------------------------------------------------------------------
class Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # LayerNorm rescales each note to average 0 and spread 1, then applies a learned
        # scale and shift (16 + 16 weights). It keeps numbers in a steady range so that
        # training doesn't blow up or fade out as notes pass through the layers.
        self.norm1 = nn.LayerNorm(cfg.note)
        self.attention = Attention(cfg)
        self.norm2 = nn.LayerNorm(cfg.note)
        self.knowledge = Knowledge(cfg)

    def forward(self, notes):
        # "notes + ..." not "notes = ...": each part ADDS an update to the note instead of
        # replacing it. The original information always flows through untouched, and each
        # part only has to learn a correction. (Standard name: residual connection.)
        notes = notes + self.attention(self.norm1(notes))
        notes = notes + self.knowledge(self.norm2(notes))
        return notes


# ---------------------------------------------------------------------------
# The whole model
# ---------------------------------------------------------------------------
class TinyTowns(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # nn.Embedding(28, 16) is a 28-row table: give it symbol 4, get row 4 (16 numbers).
        self.lookup = nn.Embedding(cfg.vocab, cfg.note)      # lookup table: symbol -> note
        self.position = nn.Embedding(cfg.context, cfg.note)  # position table: position -> note
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.note)                   # one last tidy-up before output
        self.output = nn.Linear(cfg.note, cfg.vocab)         # output table: note -> 28 scores
        self.apply(self._small_start)  # run _small_start on every part above

    @staticmethod
    def _small_start(m):
        # Every weight starts as a small random number (around +-0.02), every bias at 0.
        # Small => all 28 output scores start near 0 => softmax gives ~1/28 each =>
        # the untrained loss starts near ln 28 = 3.33. Random (not all equal) => each weight
        # starts slightly different, so each one gets a different nudge and they can learn
        # different jobs. If they all started equal they would all stay equal forever.
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)

    def forward(self, x, y=None):
        """x: [B, T] symbol numbers. y: [B, T] answers, or None when we only want predictions.

        Returns (scores, loss). scores is [B, T, 28]: for every name and every position,
        28 raw scores for the next symbol (softmax of them = the probabilities).
        """
        B, T = x.shape
        # 1) Lookup + position: each symbol becomes a note, plus "where am I" information.
        #    torch.arange(T) = [0, 1, ..., T-1], the position numbers.  -> [B, T, C]
        notes = self.lookup(x) + self.position(torch.arange(T, device=x.device))
        # 2) The layers: attention (look back) and knowledge step (think), repeated.
        for layer in self.layers:
            notes = layer(notes)
        # 3) Output table: note -> 28 scores, one per possible next symbol.  -> [B, T, 28]
        #    These raw scores are called "logits"; softmax turns them into probabilities.
        scores = self.output(self.norm(notes))
        if y is None:
            return scores, None
        # 4) Loss, exactly as in step 1: average of -ln(probability of the real next symbol).
        #    F.cross_entropy does softmax + -ln p + average in one go. .view(-1, 28) flattens
        #    [B, T, 28] into a plain list of predictions [B*T, 28], and y into [B*T] answers.
        #    ignore_index=-1 skips the padding positions.
        loss = F.cross_entropy(scores.view(-1, self.cfg.vocab), y.view(-1), ignore_index=-1)
        return scores, loss


# ---------------------------------------------------------------------------
# Inventing names: predict, pick, feed back in, repeat (as in step 1)
# ---------------------------------------------------------------------------
@torch.no_grad()  # we're only predicting, not learning, so skip the gradient bookkeeping
def sample(model, n, temperature=1.0, generator=None):
    """Invent n names. temperature < 1 = safer, more typical; > 1 = wilder."""
    device = next(model.parameters()).device
    names = []
    for _ in range(n):
        ids = [0]  # start with '.'
        while len(ids) < CONTEXT:
            scores, _ = model(torch.tensor([ids], device=device))
            # Only the last position matters: it's the prediction for the next symbol.
            probs = F.softmax(scores[0, -1] / temperature, dim=-1)
            nxt = torch.multinomial(probs.cpu(), 1, generator=generator).item()
            if nxt == 0:  # '.' = end of name
                break
            ids.append(nxt)
        names.append("".join(itos[i] for i in ids[1:]))
    return names


# ---------------------------------------------------------------------------
# Self-check: `python model.py`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)  # same random starting weights every run
    with open("data/cities.txt", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    batch = random.Random(42).sample(names, 256)
    x, y = encode(batch)
    print(f"example: '{batch[0]}'\n  input  {x[0].tolist()}\n  target {y[0].tolist()}")

    for size, cfg in CONFIGS.items():
        model = TinyTowns(cfg)
        print(f"\n{size}: {cfg}")
        if size == "tinytowns-nano":
            # named_parameters() lists every weight grid the model owns, by name.
            for name, w in model.named_parameters():
                print(f"  {name:<32} {str(list(w.shape)):<10} {w.numel():>6,}")
        total = sum(w.numel() for w in model.parameters())
        _, loss = model(x, y)
        print(f"  total weights: {total:,}")
        print(f"  untrained loss on 256 names: {loss.item():.4f}  (even guess: {math.log(28):.4f})")
