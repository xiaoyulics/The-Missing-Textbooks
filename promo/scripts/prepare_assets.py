#!/usr/bin/env python3
"""
Rasterize the real covers and a sample of interior pages from the five volume PDFs into public/.
Needs `pdftoppm` (poppler) on PATH.   Run from the promo/ folder:  python3 scripts/prepare_assets.py
"""
import os, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "public"))
BOOKS = {
    1: ("dissolving-problems/Dissolving-Problems.pdf", 333),
    2: ("hydrogen-atoms/Hydrogen-Atoms-of-Deep-Learning.pdf", 394),
    3: ("optimizers-in-the-wild/Optimizers-in-the-Wild.pdf", 369),
    4: ("experimental-science/Deep-Learning-as-an-Experimental-Science.pdf", 365),
    5: ("verifiable-intelligence/Verifiable-Intelligence.pdf", 357),
}
os.makedirs(f"{OUT}/covers", exist_ok=True)
os.makedirs(f"{OUT}/pages", exist_ok=True)
for k, (pdf, n) in BOOKS.items():
    pdf = os.path.join(ROOT, pdf)
    if not os.path.exists(pdf):
        sys.exit(f"missing {pdf}")
    # cover = title page, 150 dpi
    subprocess.run(["pdftoppm", "-r", "150", "-f", "1", "-l", "1", "-jpeg", "-jpegopt", "quality=92", "-singlefile", pdf, f"{OUT}/covers/v{k}"], check=True)
    # 36 evenly spaced interior pages, 80 dpi (they become the page storm in the finale)
    lo, hi = 18, n - 12
    for i in range(36):
        p = round(lo + (hi - lo) * i / 35)
        subprocess.run(["pdftoppm", "-r", "80", "-f", str(p), "-l", str(p), "-jpeg", "-jpegopt", "quality=82", "-singlefile", pdf, f"{OUT}/pages/v{k}-{i:02d}"], check=True)
print("assets written to", OUT)
