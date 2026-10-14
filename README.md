# tinytowns

**A transformer small enough to read every number of. It learns from 142,706 real place names and invents new ones.**

**What it is for:** tinytowns makes up new town names in the style of real ones. It learned the
letter patterns of 128,435 real place names: it was trained to guess the next letter of a place
name as well as it can. It does not know which places exist, so most names it writes are new. To
find real places, look them up in a list of places, or ask a large AI model such as Claude.

This is a learning project, built in the open, one step at a time. The goal is to see how a
transformer (the design behind GPT and Llama) works by training one so small that every
weight fits on a screen: the smallest has **4,572 numbers**, the largest about a million, against the billions in a chat model. Same parts,
same maths, nothing hidden.

The task is the one every GPT-style model does: **predict the next token**. Here a token is one
letter. Type `ban` and the model gives a probability for every possible next letter; keep picking
letters until it predicts "end of name", and you have a new name. It may match a real place by
chance; it has never been to any of them.

**Try it in your browser:** [quirkyagents.com/wizard/projects/tinytowns](https://quirkyagents.com/wizard/projects/tinytowns/).
Type a few letters, watch the model's next-letter chances, and let it finish a name. The page
checks the name against the list of real places; most are new, as intended.

## Status

| Step | What | Status |
|---|---|---|
| 0 | City data, cleaned (`data/cities.txt`) | Done |
| 1 | Letters to numbers, and a counting baseline to beat | Done |
| 2 | The tinytowns-nano transformer (~4,500 weights), trained on a CPU | Done |
| 3 | Look inside: every weight, the attention map, the letter table | Done |
| 4 | Bigger models on a GPU: tinytowns-small (~105,000 weights) and tinytowns (~1,000,000) | Done |
| 5 | Type a few letters, pick a model, watch it predict the next | Done |
| 6 | Save as safetensors and read the file by hand | Done |
| 7 | Release on Hugging Face | Done |

Results land in this README as each step is measured, not before.

## Results

Loss is the average of −ln(probability the model gave to the letter that really came next).
Lower is better. Validation names (10%) are never used for learning.

The data grew in step 4, from GeoNames `cities15000` (31,247 names) to `cities1000` (142,706).
A different validation set can be easier or harder, so compare losses only within one table.

**Current data: `cities1000`** (128,435 train names, 14,271 validation)

| Model | Train loss | Validation loss | Run with |
|---|---|---|---|
| Uniform guess (1/28 for every symbol) | 3.3322 | 3.3322 | — |
| Counting baseline (bigram, 28×28 table) | 2.6749 | 2.6732 | `python step1_bigram.py` |
| tinytowns-nano (4,572 weights, 6,000 steps × 64 names, 53 s on CPU) | 2.4716 | 2.4676 | `python train.py --device cpu` |
| tinytowns-small (105,052 weights, 6,000 steps × 256 names, 135 s on GPU) | 2.1635 | 2.2049 | `python train.py tinytowns-small` |
| tinytowns (1,001,500 weights, best at step 5,500 of 6,000 × 256 names, 550 s on GPU) | 2.0085 | **2.1504** | `python train.py tinytowns` |

GPU: an RTX 3050 4 GB laptop card. Times include grading all names every 500 steps.
`train.py` keeps the weights from the step with the lowest validation loss.

Ten times the weights buys less each time: nano → small lowers validation loss by 0.263,
small → tinytowns by only 0.055. The 1M model also starts to memorise: its gap between
validation and train loss grows to 0.142 by step 5,500, and at step 6,000 validation loss
rises (2.1543) while train loss keeps falling (1.9911).

Invented by tinytowns (seed 42, temperature 1): `brech` (a real name from the training set),
`gnolpanou`, `dolna atherbergtown`, `araves`, `gulgaun`, `el corregos`, `acro`, `volmonmaekil`,
`landescurtejovs`, `kingstonville`.

### Do bigger models invent more real places?

`python play.py --compare` invents 10,000 names per model (temperature 1, seed 42) and checks each
against the list. *real-train*: a real name from the training set (may be copied). *real-new*: a
real name only in validation, which the model never saw.

| Model | Val loss | real-train | real-new | invented | different names |
|---|---|---|---|---|---|
| tinytowns-nano | 2.4676 | 194 | 14 | 9,792 | 9,977 |
| tinytowns-small | 2.2049 | 242 | 20 | 9,738 | 9,974 |
| tinytowns | 2.1504 | 386 | 30 | 9,584 | 9,971 |

The 1M model makes about twice as many real names as nano (4.2% against 2.1%), but still invents
over 95%. The training set is 9 times the validation set, so a model that never copies would hit
them about 9 to 1; the measured ratios are 13.9, 12.1 and 12.9 to 1. With only 14–30 real-new
names per model, that's too few to say one model copies more than another.

`python play.py` is interactive: type a prefix like `san` to see the next-letter chances and five
completions; `:all san` shows all three models, `:model small` switches, `:t 0.7` sets the temperature.

### Inside the weights file

`train.py` saves each model as `runs/<size>.safetensors`: 8 bytes giving the header length, a JSON
header (each grid's name, number type, shape and byte position, plus the config as text), then
every weight as a 4-byte number, back to back. `python read_weights.py [size]` opens it with only
`struct` and `json`, prints that table, and rebuilds the letter table from raw bytes; it matches
the model's own grid bit for bit.

| File | Header | Data | Whole file |
|---|---|---|---|
| tinytowns-nano | 2,024 bytes | 18,288 (4,572 × 4) | 20,320 bytes |
| tinytowns-small | 3,504 bytes | 420,208 (105,052 × 4) | 423,720 bytes |
| tinytowns | 8,040 bytes | 4,006,000 (1,001,500 × 4) | 4,014,048 bytes |

### On Hugging Face

All three models are at [huggingface.co/dave8172/tinytowns](https://huggingface.co/dave8172/tinytowns):
the three `.safetensors` files, `model.py`, `config.json` and a model card (`hf/README.md` here).
`python release_hf.py` lists what would be sent; `--upload` sends it in one commit; `--check`
downloads the repo into an empty folder and re-grades each model with the downloaded `model.py`.
Measured after upload, without a login: validation loss 2.4676 / 2.2049 / 2.1504, the same as
before upload to 4 decimals, and the model card's example prints the same ten names as above.

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

## What you can see

- **The next-letter probabilities** as you type, so the model's output is a list of percentages, not a mystery.
- **Real or invented:** every generated name is checked against the full name list.
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

## In JavaScript

`tinytowns.js` is the same model for a browser or Node: one file, no libraries. It reads a
`.safetensors` file and runs the forward pass and sampling by hand (no training). Measured: on all
14,271 validation names it gives the same loss as the PyTorch model to six decimal places, for all
three sizes (2.467599, 2.204930, 2.150443).

```html
<script src="tinytowns.js"></script>
<script>
  TinyTowns.load(TinyTowns.url("tinytowns-small")).then((model) => {
    console.log(model.next("ban"));                          // 28 probabilities, in ". abc...z" order
    console.log(model.sample("ban", { temperature: 0.8 }));  // a name starting with "ban"
  });
</script>
```

`TinyTowns.url(size)` points at the weights on Hugging Face; `TinyTowns.load` also takes an
`ArrayBuffer` if you host the file yourself. In Node, `require("./tinytowns.js")`.

## Ideas for using it

Not built or tested yet, just directions:

- **Games and maps:** town names made on the player's device. nano is a 20 KB file.
- **Test data that names no real place:** generate names, keep only the invented ones.
- **A "does this look like a place name?" score:** the model's loss on a name says how surprising
  each letter was, which could flag `asdfgh` typed into a city field.

## The data

`data/cities.txt` comes from [GeoNames](https://www.geonames.org) ("cities1000": every place with
1,000+ people, 243 countries), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changed by us:
names lowercased, hyphens turned into spaces, apostrophes removed, duplicates merged, and names
with digits or brackets or over 20 letters dropped. `data/prepare.py` does all of it.

## Licence

Code: MIT. City data: CC BY 4.0, GeoNames.
