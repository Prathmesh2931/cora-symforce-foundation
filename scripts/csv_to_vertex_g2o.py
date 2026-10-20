#!/usr/bin/env python3

import csv
import sys

if len(sys.argv) != 3:
    raise SystemExit(
        "usage: csv_to_vertex_g2o.py input.csv output.g2o"
    )

with open(sys.argv[1], newline="") as f:
    rows = list(csv.DictReader(f))

with open(sys.argv[2], "w") as f:
    for row in rows:
        f.write(
            "VERTEX_SE2 "
            f'{int(row["id"])} '
            f'{float(row["x"]):.17g} '
            f'{float(row["y"]):.17g} '
            f'{float(row["theta"]):.17g}\n'
        )

print(f"Wrote {len(rows)} poses.")
