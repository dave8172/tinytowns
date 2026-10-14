"""Benchmark step 4: no model at all. Finish a name by looking its first 3 letters up in the list.

If the job is "give me a real place that starts with these letters", the list itself answers it.
data/cities.txt holds every name, so a sorted copy plus a binary search (bisect) finds the first
real name with that start. This is the bar any model should be compared with for real names.
It also counts how many real places share each 3-letter start: the more there are, the less a
model can know which one was meant.
    python bench/step4_lookup.py   -> bench/results/lookup.json
"""
import bisect
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "data/cities.txt", encoding="utf-8") as f:
    names = sorted({line.strip() for line in f if line.strip()})
test = [line for line in (ROOT / "bench/names.txt").read_text(encoding="utf-8").splitlines() if line]

found, seconds, sharing = 0, [], []
for name in test:
    prefix = name[:3]
    start = time.perf_counter()
    i = bisect.bisect_left(names, prefix)  # where the prefix would go in the sorted list
    found += i < len(names) and names[i].startswith(prefix)
    seconds.append(time.perf_counter() - start)
    # every name starting with the prefix sits between prefix and prefix + a character after 'z'
    sharing.append(bisect.bisect_left(names, prefix + "~") - i)

result = {
    "n_names": len(test),
    "real": found / len(test),
    "us_per_name": statistics.median(seconds) * 1e6,  # microseconds: millionths of a second
    "real_places_per_prefix": {"median": statistics.median(sharing), "min": min(sharing),
                               "max": max(sharing)},
}
print(json.dumps(result, indent=2))
(ROOT / "bench/results/lookup.json").write_text(json.dumps(result, indent=2) + "\n")
