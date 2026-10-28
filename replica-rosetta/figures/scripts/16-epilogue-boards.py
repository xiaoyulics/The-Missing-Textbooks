"""Epilogue: read the status boards of the prologue and the fifteen chapters, and count them.

Parses every \\statusrow{prediction}{\\status}{source} inside a statusboard environment of
chapters/00-prologue.tex ... chapters/15-sgd.tex, writes the counts per chapter to
figures/data/16-epilogue-boards.dat, and prints every row (and the totals) for the epilogue.
Run from the volume root: python figures/scripts/16-epilogue-boards.py   (under a second)
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATUSES = ["proved", "provedunder", "provedpreprint", "conjectured", "heuristicstatus", "knownfalse", "openq"]
SHORT = {"proved": "proved", "provedunder": "under", "provedpreprint": "preprint",
         "conjectured": "conj", "heuristicstatus": "heur", "knownfalse": "false", "openq": "open"}


def braced(text, i):
    """Return (content, index after the closing brace) for the group that opens at text[i] == '{'."""
    assert text[i] == "{", text[i:i + 30]
    depth, j = 0, i
    while True:
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j + 1
        j += 1


def rows_of(path):
    text = open(path, encoding="utf-8").read()
    text = re.sub(r"(?<!\\)%.*", "", text)          # drop comments
    rows = []
    for block in re.findall(r"\\begin\{statusboard\}(.*?)\\end\{statusboard\}", text, re.S):
        i = 0
        while True:
            k = block.find("\\statusrow", i)
            if k < 0:
                break
            j = k + len("\\statusrow")
            while block[j].isspace():
                j += 1
            pred, j = braced(block, j)
            while block[j].isspace():
                j += 1
            status, j = braced(block, j)
            while block[j].isspace():
                j += 1
            source, j = braced(block, j)
            s = status.strip().lstrip("\\")
            if s not in STATUSES:
                raise ValueError(f"{path}: unknown status {status!r}")
            rows.append((pred.strip(), s, source.strip()))
            i = j
    return rows


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "chapters", "[01][0-9]-*.tex")))
    files = [f for f in files if int(os.path.basename(f)[:2]) <= 15]
    table, totals = [], {s: 0 for s in STATUSES}
    for f in files:
        base = os.path.basename(f)[:-4]
        num, key = int(base[:2]), base[3:]
        rows = rows_of(f)
        counts = {s: sum(1 for r in rows if r[1] == s) for s in STATUSES}
        for s in STATUSES:
            totals[s] += counts[s]
        label = "P" if num == 0 else str(num)
        table.append([num, label, key] + [counts[s] for s in STATUSES] + [len(rows)])
        print(f"== {base}: {len(rows)} rows  " + "  ".join(f"{SHORT[s]} {counts[s]}" for s in STATUSES if counts[s]))
        for pred, s, src in rows:
            print(f"   [{SHORT[s]:>8}] {pred}  ||  {src}")
    rk.write_dat(os.path.join(ROOT, "figures", "data", "16-epilogue-boards.dat"),
                 ["num", "label", "key"] + [SHORT[s] for s in STATUSES] + ["total"], table)
    n = sum(totals.values())
    print(f"\ntotal rows {n}: " + ", ".join(f"{SHORT[s]} {totals[s]}" for s in STATUSES))
    print(f"proved or proved under: {totals['proved'] + totals['provedunder']} ({(totals['proved'] + totals['provedunder']) / n:.0%})")


if __name__ == "__main__":
    main()
