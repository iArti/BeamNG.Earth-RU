#!/usr/bin/env python3
"""Build server/arcticdem_index.json.gz: the ArcticDEM v4.1 2 m mosaic files that exist (arcticdem.py).

PGC publishes the mosaic as 50 x 50 km Cloud-Optimised GeoTIFFs (25100 px of 2 m: 100 m of overlap each side) in
EPSG:3413, named <row>_<col>_<subrow>_<subcol>; 2m_dem_tiles.vrt lists every one with its place in the global grid.
This reads that list, checks each file's name against the place arcticdem.file_of computes for its centre (so the
formula and PGC's layout agree), and writes the names (sorted) as gzipped JSON.

    python3 tools/build_arcticdem_index.py [--vrt PATH]     # default: downloads the VRT (~4.6 MB)
"""
import argparse
import gzip
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
import arcticdem  # noqa: E402

VRT_URL = arcticdem.BASE + "2m_dem_tiles.vrt"
OUT = os.path.join(HERE, "..", "server", "arcticdem_index.json.gz")
ENTRY = re.compile(r"/2m/(\d\d_\d\d)/(\d\d_\d\d_\d_\d)_2m_v4\.1_dem\.tif</SourceFilename>.*?"
                   r"<DstRect xOff=\"(\d+)\" yOff=\"(\d+)\" xSize=\"(\d+)\" ySize=\"(\d+)\"", re.S)
GRID_X0, GRID_Y0, PX = -4000100.0, 4100100.0, 2.0     # the VRT's geotransform


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vrt", help="a local copy of 2m_dem_tiles.vrt")
    args = ap.parse_args(argv)
    if args.vrt:
        with open(args.vrt, encoding="utf-8") as f:
            text = f.read()
    else:
        with urllib.request.urlopen(VRT_URL, timeout=120) as r:
            text = r.read().decode("utf-8")
    names, bad = [], 0
    for m in ENTRY.finditer(text):
        folder, name = m.group(1), m.group(2)
        xo, yo, xs, ys = (int(m.group(i)) for i in range(3, 7))
        cx = GRID_X0 + (xo + xs / 2) * PX
        cy = GRID_Y0 - (yo + ys / 2) * PX
        if not name.startswith(folder) or arcticdem.file_of(cx, cy) != name:
            bad += 1
            print(f"layout mismatch: {name} at ({cx:.0f}, {cy:.0f}) -> {arcticdem.file_of(cx, cy)}", file=sys.stderr)
            continue
        names.append(name)
    if not names or bad:
        raise SystemExit(f"{len(names)} files, {bad} mismatches: not written")
    names = sorted(set(names))
    with gzip.open(OUT, "wt", encoding="ascii") as f:
        json.dump({"release": arcticdem.RELEASE, "files": names}, f, separators=(",", ":"))
    print(f"{len(names)} files -> {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
