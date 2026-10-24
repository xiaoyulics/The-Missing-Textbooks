"""The title page's emblem: Gardner's capacity curve alpha_c(kappa) of the spherical perceptron.

The title page carries one fact in three registers (rosetta.sty): Gardner's replica formula
1/alpha_c(kappa) = E[(Z + kappa)_+^2], its curve, and the theorem that proves it for kappa >= 0
(Shcherbina--Tirozzi; Stojnic; Talagrand). For kappa < 0 the same formula is only an upper bound
(Stojnic), and physics predicts that full replica symmetry breaking lowers it; the emblem draws that
part dotted. Writes figures/data/00-emblem.tex, which defines \\rstemblemcurve (kappa >= 0) and
\\rstemblemcurveneg (kappa < 0) as pgf coordinate lists.

Run from the volume root:  python figures/scripts/00-emblem.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import rosettakit as rk  # noqa: E402


def coords(ks):
    return " ".join(f"({k:.4f},{rk.gardner_capacity(k):.4f})" for k in ks)


def main():
    pos = np.linspace(0.0, 3.3, 34)
    neg = np.linspace(-0.62, 0.0, 16)
    path = os.path.join("figures", "data", "00-emblem.tex")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("% written by figures/scripts/00-emblem.py: Gardner's capacity alpha_c(kappa)\n")
        fh.write("\\def\\rstemblemcurve{" + coords(pos) + "}\n")
        fh.write("\\def\\rstemblemcurveneg{" + coords(neg) + "}\n")
    print(f"alpha_c(0) = {rk.gardner_capacity(0.0):.6f}; alpha_c(1) = {rk.gardner_capacity(1.0):.6f}; "
          f"alpha_c(-0.62) = {rk.gardner_capacity(-0.62):.6f}; alpha_c(3.3) = {rk.gardner_capacity(3.3):.6f}")
    print("wrote", path)


if __name__ == "__main__":
    main()
