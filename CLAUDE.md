# tinytowns — notes for an AI assistant working in this repo

This repo is public. Everything here, including this file, is read by strangers.

## What this is

A learning project: the owner is learning how transformers work by building a tiny one from
scratch, a letter-level GPT that learns city names (`data/cities.txt`, 142,706 names) and invents
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
heads, the KV cache, why a transformer trains in parallel on a GPU.

Learned while building this repo (build on it, don't re-teach unless asked):
- **Loss** is the average −ln(probability given to the real next symbol), not right/wrong
  (that would be accuracy). Validation loss is the real score.
- **Batch:** a grid of names in, the same grid shifted left as answers; padding (−1) is skipped.
- **Training loop:** batch → forward → backward → nudge → erase gradients → repeat. One backward
  pass gives a slope for every weight at once (chain rule: multiply local slopes along each
  path, add over paths). "new = old − step × slope" moves each weight downhill; too big a step
  overshoots. Weights are what gets tuned; gradients are measured, then erased.
- **Overfitting:** validation loss stops falling or rises while train loss keeps falling.
  Keeping the best-validation weights rescues an earlier point; it doesn't prevent overfitting.
- **Attention map:** the mask blocks the future by position, not by letter, and a position may
  look at itself. Its own letter is already in the note, so nano's head mostly looks one back.
- **Temperature** divides the scores before softmax; it changes sampling only, not training.
- **Weights vs code:** `model.py` is the recipe with blanks; a `.safetensors` file holds the
  numbers for the blanks. Loading matches each grid by name and shape (`load_state_dict`).
  The config sits in the file's header metadata so the right-sized empty model can be built first.

Misconceptions already corrected once (watch for them): loss "picks the top letter"; nudging is
trial and error per weight; the mask blocks a position from itself; temperature 0.5 "doubles"
top-letter odds (it squares the ratios); best-val saving "prevents" overfitting; float rounding
differences are "noise" (they come from the order of additions). The owner sometimes answers a
"why" question with just "yes": ask for the reason.

Not yet taught: AdamW internals beyond momentum, size-scaling and decay; LayerNorm maths; GELU
shape; why exactly √(head size); dropout; learning-rate schedules.

## Design decisions (made; change only with the owner)

- **Letter-level.** Alphabet of 28: index 0 = `.` meaning "start/end of name", then space, a–z.
- **Context:** 21 positions (one `.` plus at most 20 letters). Train on `.` + name + `.`; predict
  each next symbol. Pad short names and ignore padding in the loss.
- **Split:** 90% train, 10% validation, fixed seed. Report validation loss.
- **Reference point:** guessing uniformly over 28 symbols is a loss of ln 28 ≈ 3.33. Step 1's
  counting baseline sets the real bar; the transformer must beat it.
- **Data:** GeoNames `cities1000` (1,000+ people, 243 countries), 142,706 names. Steps 1–3 were
  first measured on `cities15000` (31,247 names); losses on different data aren't comparable, so
  the step 1 baseline and tinytowns-nano were re-run on `cities1000`. Old numbers stay in the
  README, labelled with their dataset.
- **Three sizes, one code path, chosen by config:** tinytowns-nano (16-number note, 1 layer, 1 head,
  MLP 64, 4,572 weights), tinytowns-small (64, 2 layers, 4 heads, MLP 256, 105,052) and tinytowns,
  the default (128, 5 layers, 4 heads, MLP 512, 1,001,500).
- **Real vs invented:** a generated name is "real" if it's in `data/cities.txt`. Split it: real and
  in the training set (may be copied), real and only in validation (truly invented a real place),
  invented. Validation loss stays the main score; a model that memorises makes many "real" names.

## Build plan

Each step: one short file, runnable on its own, committed when it works.

1. **Letters ↔ numbers + counting baseline** (`step1_bigram.py`). A 28×28 table of "which letter
   follows which", sample names from it, compute its loss. Done when: loss printed, 10 sample
   names printed. Teaches what the loss number means.
2. **tinytowns-nano transformer** (`model.py`, `train.py`). Plain PyTorch, written by hand: lookup and
   position tables, one attention head with the mask, knowledge step, output table. Train on CPU.
   Done when: validation loss beats step 1, with numbers in the README.
3. **Look inside** (`peek.py`). Print every weight by name and shape, mapped to the vocabulary
   above; plot attention for one name; plot the letter table in 2D (PCA); bar chart of next-letter
   probabilities; show the effect of temperature.
4. **Bigger models on GPU** (tinytowns-small, tinytowns). Same code, bigger configs, `device="cuda"`
   when available, bigger batch on GPU, keep the weights from the step with the best validation
   loss (bigger models can overfit). Compare loss, time and samples across all three sizes.
5. **Type and predict** (`play.py`). Pick a model; type a prefix; show next-letter percentages and
   5 completions marked real-in-train, real-new or invented. Plus a side-by-side table of the three
   models: does a small model do as well on this small task?
6. **Weights file.** Save with `safetensors`; read the header by hand (8-byte length, then JSON
   with each tensor's name, dtype, shape, byte offsets).
7. **Hugging Face release.** Weights, config, model card (data credit included), maybe a demo.

**All seven steps are done** (step 7 released at huggingface.co/DaveMeh94/tinytowns: one repo,
three `.safetensors`, `model.py`, `config.json`, card from `hf/README.md`; `release_hf.py`
lists, uploads in one commit, and `--check` re-grades a fresh anonymous download).

Measured validation loss on `cities1000`: uniform 3.3322, bigram 2.6732, nano 2.4676 (53 s CPU),
small 2.2049 (135 s GPU), tinytowns 2.1504 (best at step 5,500 of 6,000; 550 s GPU, RTX 3050 4 GB).
Full tables are in the README.

Ideas for later, not started (only with the owner): a browser demo (the forward pass written in
JavaScript, reading the weights from Hugging Face); an MCP tool to invent names, score a name,
and say whether it is real or invented.

## Lessons and gotchas

- **Use the venv's Python** (`.venv\Scripts\python.exe`); it has the CUDA build of torch.
- **`train.py` overwrites `runs/<size>.safetensors`**, even with `--steps 2`. Never test-run it on
  a real size without retraining afterwards. `runs/` is git-ignored, so the Hugging Face copy is
  the backup (`python release_hf.py --check` downloads it).
- **Time long runs before estimating them.** The first estimate for the 1M model was 2× too low:
  grading every 500 steps costs ~8 s each, and sustained steps ran ~74 ms against a 42 ms benchmark.
- **Commands typed with `!` in Claude Code run in bash**, even on Windows: use forward slashes
  (`.venv/Scripts/pip`), because bash eats backslashes.
- **Tokens:** the owner logs in with `hf auth login` in their own terminal; never paste a token
  into chat or write one to a file. Show the exact file list and get a yes before any upload.

## Rules

- **Plain PyTorch for the model.** No `transformers` library model classes: writing it by hand is
  the point. Libraries for plotting and saving are fine.
- **CPU must always work; GPU is optional.** Pick the device automatically.
- **Fixed random seeds** so runs are repeatable.
- **Comments explain why, and walk through what each block does** (with grid shapes): the owner
  asked for detailed comments to learn from. Still skip comments that only repeat the line.
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
