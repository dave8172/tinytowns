"""Build data/cities.txt: one lowercase city name per line, from GeoNames.

Run it once (it's already been run; cities.txt is in the repo):

    python data/prepare.py

Source: GeoNames "cities1000", every place with 1,000+ people, about 171,000 rows from
243 countries. (We started with "cities15000", 15,000+ people, 31,247 names; the bigger
list gives ~4.6x more names, so the larger models have more to learn from.)
https://www.geonames.org, licensed CC BY 4.0. We keep only the name column
and clean it as described below.

Standard library only, so it runs before anything is installed.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

LIST = "cities1000"  # GeoNames also has cities500, cities5000, cities15000
URL = f"https://download.geonames.org/export/dump/{LIST}.zip"
OUT = Path(__file__).parent / "cities.txt"

# The model's whole alphabet: 26 letters and a space. A 28th symbol, "end of
# name", is added by the model code, not stored here. A small alphabet keeps
# the lookup table and the output table tiny enough to print.
ALLOWED = set("abcdefghijklmnopqrstuvwxyz ")

# Long names are rare (about 2% are over 20 letters) and would make every
# training example longer. Dropping them keeps the context short.
MAX_LEN = 20


def clean(name: str) -> str | None:
    """Return the cleaned name, or None to skip it."""
    name = name.lower()
    name = name.replace("-", " ")  # "saint-denis" -> "saint denis"
    name = name.replace("'", "").replace("`", "")  # "val-d'or" -> "val dor"
    name = " ".join(name.split())  # collapse double spaces, trim the ends
    if not name or len(name) > MAX_LEN:
        return None
    if not set(name) <= ALLOWED:  # digits, brackets, dots: a few hundred names
        return None
    return name


def main() -> None:
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))

    names = set()  # a set, because many countries share city names
    with archive.open(f"{LIST}.txt") as f:
        for line in io.TextIOWrapper(f, encoding="utf-8"):
            # Tab-separated. Column 3 is "asciiname": the name in plain
            # Latin letters, so "São Paulo" arrives as "Sao Paulo".
            ascii_name = line.split("\t")[2]
            cleaned = clean(ascii_name)
            if cleaned:
                names.add(cleaned)

    OUT.write_text("\n".join(sorted(names)) + "\n", encoding="utf-8")
    print(f"Wrote {len(names):,} names to {OUT}")


if __name__ == "__main__":
    main()
