# tinytowns

**A transformer small enough to read every number of. It learns from 142,706 real place names and invents new ones.**

This is a learning project, built in the open, one step at a time. The goal is to see how a
transformer (the design behind GPT and Llama) works by training one so small that every
weight fits on a screen: the smallest has **4,572 numbers**, the largest about a million, against the billions in a chat model. Same parts,
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
| 2 | The tinytowns-nano transformer (~4,500 weights), trained on a CPU | Done |
| 3 | Look inside: every weight, the attention map, the letter table | Done |
| 4 | Bigger models on a GPU: tinytowns-small (~105,000 weights) and tinytowns (~1,000,000) | Next |
| 5 | Type a few letters, pick a model, watch it predict the next | Planned |
| 6 | Save as safetensors and read the file by hand | Planned |
| 7 | Release on Hugging Face | Planned |

Results land in this README as each step is measured, not before.

## Results so far

Loss is the average of −ln(probability the model gave to the letter that really came next).
Lower is better. Validation names (10%) are never used for learning.

The data grew in step 4, from GeoNames `cities15000` (31,247 names) to `cities1000` (142,706).
A different validation set can be easier or harder, so compare losses only within one table.

**Current data: `cities1000`** (128,435 train names, 14,271 validation)

| Model | Train loss | Validation loss | Run with |
|---|---|---|---|
| Uniform guess (1/28 for every symbol) | 3.3322 | 3.3322 | — |
| Counting baseline (bigram, 28×28 table) | 2.6749 | 2.6732 | `python step1_bigram.py` |
| tinytowns-nano (4,572 weights, 6,000 steps, 64 s on CPU) | 2.4716 | **2.4676** | `python train.py --device cpu` |

**Steps 1–3: `cities15000`** (28,122 train names, 3,125 validation)

| Model | Train loss | Validation loss |
|---|---|---|
| Uniform guess | 3.3322 | 3.3322 |
| Counting baseline | 2.6559 | 2.6616 |
| tinytowns-nano (6,000 steps, 41 s on CPU) | 2.4555 | **2.4722** |

The rest of this section and the next were measured on `cities15000`.

Names sampled from the counting baseline (seed 42): `barchirn`, `famoou`, `dol`, `co`, `e`,
`tareilli da`, `saves`, `buvg`, `unde`, `ncode`. None are real cities. One letter of memory
isn't enough to make a name sound right; that's the transformer's job.

Names invented by tinytowns-nano after training (seed 42): `barchirno`, `paoou`, `dalico`,
`el modimar da sales`, `bulgaun`, `el code`, `mista`, `ramazom`, `nigaral landesc`, `patjivskkin`.
Multi-word names like `el ...` and `... da ...` appear: attention lets it see more than one letter back.

## Looking inside tinytowns-nano

`python peek.py` (or `python peek.py <name>`) prints every weight by name and shape, runs the
forward pass by hand for one name and checks it against the model (largest difference 9.5e-07),
then saves three pictures to `runs/`. What it showed:

- **Attention looks one letter back.** For `.narita`, at `r` the head gives 0.52 of its attention
  to `a` and 0.28 to `r` itself. The note already carries its own letter, so attention is spent
  fetching the one before.
- **It found vowels on its own.** Flattened to 2D with PCA (keeping 33% of the spread), the lookup
  table puts all five vowels on one side. Measured over all 16 numbers, vowels sit 27% closer to
  each other than to consonants; in the untrained model there is no such gap. The output table
  groups the vowels even more tightly and puts "end of name" and space apart from every letter.
- **Temperature trades safe for new.** Of 500 names at each temperature (seed 42):

| Temperature | Real cities | Different names | Average length |
|---|---|---|---|
| 0.5 | 28 (5.6%) | 488 | 7.4 letters |
| 1.0 | 3 (0.6%) | 500 | 8.7 letters |
| 2.0 | 1 (0.2%) | 500 | 11.8 letters |

## Three sizes, one design

| | tinytowns-nano | tinytowns-small | tinytowns |
|---|---|---|---|
| Alphabet | 28 symbols: a–z, space, end of name | same | same |
| Note size per letter | 16 numbers | 64 numbers | 128 numbers |
| Layers × attention heads | 1 × 1 | 2 × 4 | 5 × 4 |
| Knowledge step width | 64 | 256 | 512 |
| Weights | 4,572 | 105,052 | 1,001,500 |
| File size (4 bytes per weight) | ≈ 18 KB | ≈ 420 KB | ≈ 4 MB |

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

`data/cities.txt` comes from [GeoNames](https://www.geonames.org) ("cities1000": every place with
1,000+ people, 243 countries), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changed by us:
names lowercased, hyphens turned into spaces, apostrophes removed, duplicates merged, and names
with digits or brackets or over 20 letters dropped. `data/prepare.py` does all of it.

## Licence

Code: MIT. City data: CC BY 4.0, GeoNames.
