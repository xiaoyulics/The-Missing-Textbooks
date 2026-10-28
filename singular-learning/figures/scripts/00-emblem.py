"""The title page's emblem: posterior draws of y = abx at its singular truth.

n = 400 observations of y = 0 * x + N(0, 1) with x ~ N(0, 1); the prior is uniform on [-1, 1]^2.
Among the data seeds 0..99 the script takes the one whose least-squares coefficient c_hat = Sxy/Sxx
is closest to 0, so that the picture shows the posterior of a data set whose best fit is the truth
itself. It draws 2400 points from the gridded posterior (sltkit.sample_grid, 1200 x 1200 cells)
and writes figures/data/00-emblem.tex, which defines \\sltemblempoints for singular.sty.

Run from the volume root:  python figures/scripts/00-emblem.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import sltkit as sk  # noqa: E402

N, DRAWS, CELLS = 400, 2400, 1200


def main() -> None:
    best = None
    for seed in range(100):
        x, y = sk.PRODUCT.sample([0.0, 0.0], N, seed=seed)
        c_hat = float((x[:, 0] @ y) / (x[:, 0] @ x[:, 0]))
        if best is None or abs(c_hat) < abs(best[1]):
            best = (seed, c_hat)
    seed, c_hat = best
    x, y = sk.PRODUCT.sample([0.0, 0.0], N, seed=seed)
    sk.PRODUCT.fit(x, y)
    pts = sk.sample_grid(sk.PRODUCT.energy, [(-1, 1), (-1, 1)], CELLS, DRAWS, seed=1985)
    near = float(np.mean(np.abs(pts[:, 0] * pts[:, 1]) < 0.1))
    print(f"data seed {seed}: c_hat = {c_hat:.5f}; {DRAWS} draws; fraction with |ab| < 0.1: {near:.3f}")
    out = os.path.join("figures", "data", "00-emblem.tex")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="\n") as fh:
        fh.write("% posterior draws of y = abx at the singular truth (figures/scripts/00-emblem.py)\n")
        fh.write("\\def\\sltemblempoints{%\n")
        items = [f"{a:.3f}/{b:.3f}" for a, b in pts]
        for i in range(0, len(items), 8):
            fh.write(",".join(items[i:i + 8]) + ("," if i + 8 < len(items) else "") + "%\n")
        fh.write("}\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
