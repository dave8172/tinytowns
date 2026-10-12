---
license: mit
language:
  - en
tags:
  - pytorch
  - safetensors
  - character-level
  - text-generation
  - educational
  - from-scratch
pipeline_tag: text-generation
---

# tinytowns

**Three tiny letter-level GPTs that invent city names.** Trained on 128,435 real place names from
GeoNames. The smallest has 4,572 weights, the largest about a million.

This is a learning project: a transformer written by hand in plain PyTorch, small enough to read
every number of. Code, step-by-step build and all measurements:
[github.com/dave8172/tinytowns](https://github.com/dave8172/tinytowns).

## The three models

All three use one design and one code file (`model.py`); only the config differs.

| File | Note size | Layers × heads | MLP width | Weights | Validation loss |
|---|---|---|---|---|---|
| `tinytowns-nano.safetensors` | 16 | 1 × 1 | 64 | 4,572 | 2.4676 |
| `tinytowns-small.safetensors` | 64 | 2 × 4 | 256 | 105,052 | 2.2049 |
| `tinytowns.safetensors` | 128 | 5 × 4 | 512 | 1,001,500 | **2.1504** |

Loss is cross-entropy per symbol on 14,271 held-out names (lower is better). For reference:
guessing uniformly over 28 symbols scores 3.3322; a 28×28 bigram counting table scores 2.6732.

## How it works

- **Tokens are letters.** Alphabet of 28: index 0 = `.` (start / end of name), 1 = space, 2–27 = `a`–`z`.
- **Context:** 21 positions, one `.` plus at most 20 letters.
- **Architecture:** token + learned positional embeddings, then pre-norm transformer layers
  (causal self-attention + MLP), then a final LayerNorm and an unembedding to 28 scores.
- **Training:** predict each next symbol of `.` + name + `.`; AdamW, 6,000 steps. Batch 64 (nano,
  CPU) or 256 (small and tinytowns, GPU). The saved weights come from the step with the lowest
  validation loss (step 5,500 for tinytowns).

Each `.safetensors` file also stores its config as JSON text in the header metadata, so
`load_weights` can rebuild the right model from the file alone.

## Use

Needs `torch>=2.2`, `safetensors` and `huggingface_hub`.

```python
from huggingface_hub import hf_hub_download
import importlib.util, torch

repo = "DaveMeh94/tinytowns"
spec = importlib.util.spec_from_file_location("model", hf_hub_download(repo, "model.py"))
model = importlib.util.module_from_spec(spec); spec.loader.exec_module(model)

net, notes = model.load_weights(hf_hub_download(repo, "tinytowns.safetensors"))
print(model.sample(net, 10, temperature=1.0, generator=torch.Generator().manual_seed(42)))
```

`model.py` is plain PyTorch with no other code from this project. Read it before you run it, as with
any downloaded code.

## What to expect

Out of 10,000 names sampled at temperature 1 (seed 42), checked against the full name list:

| Model | Real, seen in training | Real, only in validation | Invented |
|---|---|---|---|
| tinytowns-nano | 194 | 14 | 9,792 |
| tinytowns-small | 242 | 20 | 9,738 |
| tinytowns | 386 | 30 | 9,584 |

## Limits

- It knows only lowercase `a`–`z` and space. Accents were removed, and non-Latin names are not in the data.
- A generated name is a sound-alike, not a place. Most outputs don't exist, and a "real" one may be
  copied from the training data.
- It is built for learning, not for any real-world use.

## Data and credit

Trained on [GeoNames](https://www.geonames.org) `cities1000` (every place with 1,000+ people,
243 countries), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changed:
names lowercased, hyphens turned into spaces, apostrophes removed, duplicates merged, names with
digits or brackets or over 20 letters dropped. 90% train / 10% validation, fixed seed.

## Licence

Code and weights: MIT. Training data: CC BY 4.0, GeoNames.
