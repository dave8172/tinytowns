"""Step 6: open a .safetensors weights file by hand, without the safetensors library.

    python read_weights.py                   # runs/tinytowns-nano.safetensors
    python read_weights.py tinytowns-small   # or tinytowns-small / tinytowns
    python read_weights.py tinytowns-nano output.weight   # rebuild a different grid

The file has three parts, one after the other:
    [ 8 bytes: header length N ][ N bytes: JSON header ][ every number, back to back ]

The reading below uses only the Python standard library (struct, json). PyTorch is used once,
at the end, to check that our hand-made grid equals the one the model actually loads.
"""
import json
import struct
import sys

size = sys.argv[1] if len(sys.argv) > 1 else "tinytowns-nano"
grid_name = sys.argv[2] if len(sys.argv) > 2 else "lookup.weight"
path = f"runs/{size}.safetensors"

with open(path, "rb") as f:
    raw = f.read()  # the whole file as bytes (4 MB at most here, so reading it all is fine)
print(f"{path}: {len(raw):,} bytes\n")

# --- Part 1: the first 8 bytes = how long the header is ------------------------------
# "<Q" = one unsigned 64-bit whole number, little-endian (smallest byte first).
# e.g. bytes 18 07 00 00 00 00 00 00 -> 0x0718 = 7 x 256 + 24 = 1,816.
n = struct.unpack("<Q", raw[:8])[0]
print(f"first 8 bytes: {raw[:8].hex(' ')}  ->  header is {n:,} bytes long")

# --- Part 2: the header, plain JSON text ------------------------------------------
header = json.loads(raw[8 : 8 + n].decode("utf-8"))
meta = header.pop("__metadata__", {})  # our notes (text only): config, best step
print("\nnotes in the header (__metadata__):")
for key, value in meta.items():
    print(f"  {key}: {value}")

# --- Part 3: where each grid lives in the data part --------------------------------
# data_offsets [start, end] count from the START OF THE DATA PART, i.e. from byte 8 + n of the file.
data = raw[8 + n :]
BYTES_PER = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8}  # bytes per number for each dtype
print(f"\n{'name':32} {'dtype':5} {'shape':10} {'data_offsets':>18} {'bytes':>9}")
total_numbers, end_of_last = 0, 0
for name, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):  # in file order
    start, end = info["data_offsets"]
    count = 1
    for d in info["shape"]:
        count *= d  # [28, 16] -> 448 numbers
    assert end - start == count * BYTES_PER[info["dtype"]], name  # size matches shape x dtype
    assert start == end_of_last, name                            # no gaps between grids
    end_of_last = end
    total_numbers += count
    print(f"{name:32} {info['dtype']:5} {str(info['shape']):10} {str([start, end]):>18} {end - start:>9,}")
print(f"\n{len(header)} grids, {total_numbers:,} numbers, data part {len(data):,} bytes "
      f"(= {total_numbers:,} x 4: {len(data) == total_numbers * 4})")
print(f"whole file = 8 + {n:,} + {len(data):,} = {8 + n + len(data):,} bytes")

# --- Rebuild one grid from raw bytes ------------------------------------------------
# Cut out its bytes, read them as little-endian 32-bit decimals ("<448f"), then fold the flat
# list into rows: the file stores [28, 16] row after row (row 0's 16 numbers, then row 1's, ...).
info = header[grid_name]
start, end = info["data_offsets"]
rows, cols = info["shape"] if len(info["shape"]) == 2 else (1, info["shape"][0])
flat = struct.unpack(f"<{rows * cols}f", data[start:end])
grid = [list(flat[r * cols : (r + 1) * cols]) for r in range(rows)]
print(f"\n{grid_name} rebuilt by hand: {rows} x {cols}")
if grid_name in ("lookup.weight", "output.weight"):
    from model import itos  # row r belongs to symbol r: '.', ' ', 'a', ...
    for r in (0, 2, 3):
        print(f"  row {r} ({itos[r]!r}): " + ", ".join(f"{v:+.4f}" for v in grid[r][:6]) + ", ...")

# --- Check against the model -------------------------------------------------------
import torch  # noqa: E402

from model import load_weights  # noqa: E402

model, _ = load_weights(path)
real = model.state_dict()[grid_name].reshape(rows, cols)
same = torch.equal(torch.tensor(grid, dtype=torch.float32), real)
print(f"\nsame as the grid the model loads: {same}  (exact, bit for bit)")
