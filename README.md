# tinytowns

**A transformer small enough to read every number of. It learns from 31,247 real city names and invents new ones.**

This is a learning project, built in the open, one step at a time. The goal is to see how a
transformer (the design behind GPT and Llama) works by training one so small that every
weight fits on a screen: about **4,500 numbers**, against the billions in a chat model. Same parts,
same maths, nothing hidden.

The task is the one every GPT-style model does: **predict the next token**. Here a token is one
letter. Type `ban` and the model gives a probability for every possible next letter; keep picking
letters until it predicts "end of name", and you have a city that may or may not exist. It has never
been to any of them.

## Status

| Step | What | Status |
|---|---|---|
| 0 | City data, cleaned (`data/cities.txt`) | Done |
| 1 | Letters to numbers, and a counting baseline to beat | Done |
| 2 | The tinytowns-nano transformer (~4,500 weights), trained on a CPU | Next |
| 3 | Look inside: every weight, the attention map, the letter table | Planned |
| 4 | Type a few letters, watch it predict the next | Planned |
| 5 | tinytowns, the bigger model (~105,000 weights), on a GPU | Planned |
| 6 | Save as safetensors and read the file by hand | Planned |
| 7 | Release on Hugging Face | Planned |

Results land in this README as each step is measured, not before.

## Results so far

Loss is the average of −ln(probability the model gave to the letter that really came next).
Lower is better. Validation names (10%, 3,125) are never used for learning.

| Model | Train loss | Validation loss | Run with |
|---|---|---|---|
| Uniform guess (1/28 for every symbol) | 3.3322 | 3.3322 | — |
| Counting baseline (bigram, 28×28 table) | 2.6559 | 2.6616 | `python step1_bigram.py` |

Names sampled from the counting baseline (seed 42): `barchirn`, `famoou`, `dol`, `co`, `e`,
`tareilli da`, `saves`, `buvg`, `unde`, `ncode`. None are real cities. One letter of memory
isn't enough to make a name sound right; that's the transformer's job.

## Two sizes, one design

| | tinytowns-nano | tinytowns |
|---|---|---|
| Alphabet | 28 symbols: a–z, space, end of name | same |
| Note size per letter | 16 numbers | 64 numbers |
| Layers × attention heads | 1 × 1 | 2 × 4 |
| Weights | ≈ 4,500 | ≈ 105,000 |
| File size | ≈ 18 KB | ≈ 420 KB |
| Runs on | any CPU, in seconds | a laptop GPU, in minutes |

For scale, Llama 3 8B has 8,000,000,000 weights and is a 16 GB download.

## What you'll be able to see

- **The next-letter probabilities** as you type, so the model's output is a list of percentages, not a mystery.
- **Real or invented:** every generated name is checked against the training list.
- **The attention map:** for a name like `bengaluru`, which earlier letters each letter looked at.
- **The letter table:** each letter's 16 numbers, flattened to a 2D picture.
- **Training:** the loss falling as random numbers turn into useful ones.
- **The weights file:** a `.safetensors` file whose header you can open in a text editor. The same format most open models are downloaded in, just tiny.

## Quick start

You need Python 3.10 or newer. A GPU is optional.

```bash
git clone https://github.com/dave8172/tinytowns && cd tinytowns
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\activate
# macOS / Linux:         source .venv/bin/activate
pip install -r requirements.txt
```

For an NVIDIA GPU, install PyTorch's CUDA build first, using the command from the selector at
[pytorch.org/get-started](https://pytorch.org/get-started/locally/), then the line above.

The city list is already in the repo. To rebuild it from the source: `python data/prepare.py`.

## The data

`data/cities.txt` comes from [GeoNames](https://www.geonames.org) ("cities15000": every city with
15,000+ people), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changed by us:
names lowercased, hyphens turned into spaces, apostrophes removed, duplicates merged, and names
with digits or brackets or over 20 letters dropped. `data/prepare.py` does all of it.

## Licence

Code: MIT. City data: CC BY 4.0, GeoNames.
