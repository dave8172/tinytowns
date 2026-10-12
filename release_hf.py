"""Step 7: release the three models on Hugging Face, then download them back and check them.

    python release_hf.py            # 1) look: write hf/config.json, list what would be uploaded
    python release_hf.py --upload   # 2) send: one commit with all files to the Hugging Face repo
    python release_hf.py --check    # 3) prove: download into an empty folder, re-grade every model

A Hugging Face "model repo" is a git repo for weights. Uploading needs a token with the Write
role. huggingface_hub finds it by itself: the HF_TOKEN setting, or a saved `hf auth login`.
The token never goes into any file here.
"""
import argparse
import importlib.util
import json
import os
import random
import sys
import tempfile

import torch
from safetensors import safe_open

from model import CONFIGS

REPO = "DaveMeh94/tinytowns"
SEED = 42  # same as train.py, so --check grades on the same 14,271 validation names

# Every file in the release: name on Hugging Face -> where it is on this computer.
FILES = {
    "README.md": "hf/README.md",       # the model card (Hugging Face shows it as the repo's front page)
    "config.json": "hf/config.json",   # the three configs + measured losses, written below
    "model.py": "model.py",            # the code that turns the numbers back into a model
}
for size in CONFIGS:
    FILES[f"{size}.safetensors"] = f"runs/{size}.safetensors"


def write_config():
    """Build config.json from the weight files themselves, so no number is typed in twice."""
    out = {"alphabet": ". abcdefghijklmnopqrstuvwxyz", "context": 21, "models": {}}
    for size in CONFIGS:
        # Read only the header's text metadata (config + best step), not the weights.
        with safe_open(f"runs/{size}.safetensors", framework="pt") as f:
            meta = {k: json.loads(v) for k, v in f.metadata().items()}
        out["models"][size] = {"file": f"{size}.safetensors", "config": meta["config"], "best": meta["best"]}
    with open("hf/config.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def check():
    """Download the release into an empty folder and re-grade each model on validation names."""
    from huggingface_hub import snapshot_download
    folder = snapshot_download(REPO, local_dir=tempfile.mkdtemp(prefix="tinytowns_check_"))
    print(f"downloaded to {folder}: {sorted(os.listdir(folder))}")

    # Import the DOWNLOADED model.py, not ours, so we test exactly what a stranger gets.
    spec = importlib.util.spec_from_file_location("hf_model", os.path.join(folder, "model.py"))
    hf_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hf_model)

    # The same validation split as train.py: shuffle with seed 42, last 10%.
    with open("data/cities.txt", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    random.Random(SEED).shuffle(names)
    x, y = hf_model.encode(names[int(0.9 * len(names)):])  # [14271, 21] each

    for size in CONFIGS:
        net, notes = hf_model.load_weights(os.path.join(folder, f"{size}.safetensors"))
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(x), 4096):  # in chunks, weighting each by its real (non-padding) letters
                _, loss = net(x[i:i + 4096], y[i:i + 4096])
                n = (y[i:i + 4096] != -1).sum().item()
                total, count = total + loss.item() * n, count + n
        print(f"{size:<16} validation loss {total / count:.4f}   (saved in file: {notes['best']['val']:.4f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--upload", action="store_true")
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    if args.check:
        check()
        sys.exit()

    write_config()
    print(f"Repo: https://huggingface.co/{REPO}  (public)\n")
    for hf_name, local in FILES.items():
        print(f"  {hf_name:<30} {os.path.getsize(local):>10,} bytes   from {local}")

    if args.upload:
        from huggingface_hub import CommitOperationAdd, HfApi
        api = HfApi()
        api.create_repo(REPO, repo_type="model", exist_ok=True)  # make it if it isn't there yet
        # One commit with every file, so the repo is never half uploaded.
        ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=v) for k, v in FILES.items()]
        info = api.create_commit(REPO, operations=ops, commit_message="tinytowns: three sizes, safetensors, model card")
        print(f"\nuploaded: {info.commit_url}")
    else:
        print("\nNothing sent. Run with --upload to send these files.")
