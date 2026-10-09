# tinytowns — notes for an AI assistant working in this repo

This repo is public. Everything here, including this file, is read by strangers.

## What this is

A learning project: the owner is learning how transformers work by building a tiny one from
scratch, a letter-level GPT that learns city names (`data/cities.txt`, 31,247 names) and invents
new ones. **Understanding is the product.** A working model the owner can't explain line by line
is a failed step.

## How to work with the owner

- **Teach, then build.** Before writing code for a step, explain in plain words what it does and
  why. Build one step at a time; stop after each so the owner can run it, read it, and ask.
- **Plain English, short sentences, step by step.** Define every term the first time. Use the
  owner's vocabulary below; give the standard name alongside so they can search for it.
- **Check understanding.** When the owner summarises something back, say what's right, then
  correct what's wrong, specifically. Don't just agree.
- **Numbers over adjectives.** Report losses, sizes and timings as measured numbers. Say "measured"
  or "expected" so the two never blur.
- **Don't write big chunks unasked.** If a step needs more than ~150 lines, split it.

## Vocabulary the owner already knows

| Owner's word | Standard term |
|---|---|
| lookup table | token embedding table |
| position table | positional embedding |
| note | hidden state, residual stream |
| Q, K, V grids | W_Q, W_K, W_V (fixed weights, one set per layer) |
| q, k, v | the per-token results of note × grid |
| attention: score, shares, mix | Q·K, softmax, weighted sum of V |
| knowledge step | feed-forward network, MLP |
| output table | unembedding, language-model head |
| temperature | sampling temperature |
| loss | cross-entropy loss |
| nudging the weights | gradient descent, backpropagation |

The owner understands, from reading: the token loop, attention with the mask, Q/K/V, layers and
heads, the KV cache, why a transformer trains in parallel on a GPU. New here: how training actually
changes the numbers (loss, gradients, optimiser), and seeing all of it in real code.

## Design decisions (made; change only with the owner)

- **Letter-level.** Alphabet of 28: index 0 = `.` meaning "start/end of name", then space, a–z.
- **Context:** 21 positions (one `.` plus at most 20 letters). Train on `.` + name + `.`; predict
  each next symbol. Pad short names and ignore padding in the loss.
- **Split:** 90% train, 10% validation, fixed seed. Report validation loss.
- **Reference point:** guessing uniformly over 28 symbols is a loss of ln 28 ≈ 3.33. Step 1's
  counting baseline sets the real bar; the transformer must beat it.
- **Two sizes, one code path, chosen by config:** pico (16-number note, 1 layer, 1 head, MLP 64,
  ≈ 4,500 weights) and small (64, 2 layers, 4 heads, MLP 256, ≈ 105,000).
- **Real vs invented:** a generated name is "real" if it's in `data/cities.txt`.

## Build plan

Each step: one short file, runnable on its own, committed when it works.

1. **Letters ↔ numbers + counting baseline** (`step1_bigram.py`). A 28×28 table of "which letter
   follows which", sample names from it, compute its loss. Done when: loss printed, 10 sample
   names printed. Teaches what the loss number means.
2. **Pico transformer** (`model.py`, `train.py`). Plain PyTorch, written by hand: lookup and
   position tables, one attention head with the mask, knowledge step, output table. Train on CPU.
   Done when: validation loss beats step 1, with numbers in the README.
3. **Look inside** (`peek.py`). Print every weight by name and shape, mapped to the vocabulary
   above; plot attention for one name; plot the letter table in 2D (PCA); bar chart of next-letter
   probabilities; show the effect of temperature.
4. **Type and predict** (`play.py`). Type a prefix; show next-letter percentages and 5 completions
   marked real or invented.
5. **Small model on GPU.** Same code, small config, `device="cuda"` when available. Compare loss,
   samples, and time against pico.
6. **Weights file.** Save with `safetensors`; read the header by hand (8-byte length, then JSON
   with each tensor's name, dtype, shape, byte offsets).
7. **Hugging Face release.** Weights, config, model card (data credit included), maybe a demo.

## Rules

- **Plain PyTorch for the model.** No `transformers` library model classes: writing it by hand is
  the point. Libraries for plotting and saving are fine.
- **CPU must always work; GPU is optional.** Pick the device automatically.
- **Fixed random seeds** so runs are repeatable.
- **Comments explain why**, not what the line already says.
- **Dependencies as ranges, never pinned** (`torch>=2.2`, not `==2.4.1`).
- **The README claims only what has been run and measured.** Update its Status table and results
  as each step lands.
- **Keep the GeoNames credit** (CC BY 4.0) in the README and any model card.
- **Nothing private in commits:** no personal names, emails, paths or details about the owner.

## Running on Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
# For the GPU: install PyTorch's CUDA build from https://pytorch.org/get-started/locally/ first
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```
