"""Chapter 5 (Blow-ups), "Measuring it": the posterior of the population loss of the quartic model
y = a x1 + b^2 x2, K = (a^2 + b^4)/2, uniform prior on [-1, 1]^2, in the original coordinates and
in the coordinates of its resolution.

The Gibbs measure pi_n(dw) proportional to exp(-n K(w)) phi(w) dw is the posterior with the data's
fluctuation removed (its normalizer is the partition function Z(n) of Chapter 4). The resolution
a = s t^2, b = t (Section 5.2) predicts that, as long as nK < n/2,
  * the energy E = nK and the point s = a / b^2 of the exceptional curve are independent;
  * E ~ Gamma(3/4, 1): density proportional to E^(lambda - 1) e^(-E) with lambda = 3/4;
  * s has density proportional to (1 + s^2)^(-3/4), i.e. theta = arctan s has density
    proportional to |cos theta|^(-1/2) on (-pi/2, pi/2).
The regular theory (a nondegenerate minimum, Laplace's method) would give E ~ Gamma(d/2) = Gamma(1).

The script computes pi_n by midpoint quadrature on a 1500 x 1500 grid in the ORIGINAL coordinates
(a, b), on a box that holds all but e^(-60) of the mass, for n = 10^2, 10^4, 10^6; reads off the
distribution function of E = nK from the grid weights; and draws 2000 points (sk.sample_grid) at
n = 10^4 to plot in both coordinate systems.
Writes figures/data/05-blowups-post-cdf.dat and 05-blowups-post-draws.dat.
Run from the volume root: python figures/scripts/05-blowups-posterior.py  (about 10 seconds).
"""
import math
import os
import sys

import numpy as np
from scipy import integrate, optimize, special, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sltkit as sk  # noqa: E402

OUT = "figures/data"
LAM = 0.75
GRID = 1500


def K(w):
    return 0.5 * (w[:, 0] ** 2 + w[:, 1] ** 4)


def box_for(n):
    """A box inside [-1,1]^2 outside which nK > 60."""
    A = min(1.0, math.sqrt(120.0 / n))
    Bb = min(1.0, (120.0 / n) ** 0.25)
    return [(-A, A), (-Bb, Bb)]


def main():
    xs = np.concatenate([10.0 ** np.linspace(-3, 0, 31)[:-1], np.linspace(1, 8, 36)])
    cols = {}
    means = {}
    for n in (10 ** 2, 10 ** 4, 10 ** 6):
        box = box_for(n)
        pts, cell, _ = sk.grid(box, GRID)
        E = n * K(pts)
        wts = np.exp(-(E - E.min()))
        wts /= wts.sum()
        order = np.argsort(E)
        cdf_sorted = np.cumsum(wts[order])
        idx = np.searchsorted(E[order], xs, side="left")
        cols[n] = np.where(idx > 0, cdf_sorted[np.maximum(idx - 1, 0)], 0.0)
        means[n] = float(wts @ E)
        ks = float(np.max(np.abs(cols[n] - special.gammainc(LAM, xs))))
        print(f"n = {n:>7d}: box {box[0][1]:.3g} x {box[1][1]:.3g}; posterior mean of nK = {means[n]:.4f}"
              f" (singular: {LAM}, regular: 1); max |CDF - Gamma(3/4) CDF| on the x-grid = {ks:.4f}")
    rows = [[x, cols[100][i], cols[10 ** 4][i], cols[10 ** 6][i], special.gammainc(LAM, x), 1 - math.exp(-x)]
            for i, x in enumerate(xs)]
    sk.write_dat(os.path.join(OUT, "05-blowups-post-cdf.dat"), ["x", "n2", "n4", "n6", "pred", "reg"], rows)

    # draws at n = 10^4 in both coordinate systems
    n = 10 ** 4
    energy = lambda w: n * K(w)
    draws = sk.sample_grid(energy, box_for(n), GRID, 2000, seed=5004)
    a, b = draws[:, 0], draws[:, 1]
    E = n * K(draws)
    theta = np.arctan(a / (b * b))
    sk.write_dat(os.path.join(OUT, "05-blowups-post-draws.dat"), ["a", "b", "theta", "E", "logE"],
                 [[a[i], b[i], theta[i], E[i], math.log10(E[i])] for i in range(len(a))])
    rho, p = stats.spearmanr(np.abs(theta), E)
    print(f"draws at n = 10^4: Spearman correlation of |theta| and nK = {rho:+.3f} (p = {p:.2f});"
          f" mean nK = {E.mean():.3f}")
    # the law of theta against |cos theta|^(-1/2) / Z: compare quartiles (the density is symmetric,
    # so the upper quartile solves int_0^q cos^(-1/2) = Z/4 with Z = B(1/2,1/4))
    Z = special.beta(0.5, 0.25)
    F = lambda q: integrate.quad(lambda x: math.cos(x) ** -0.5, 0.0, q)[0] - Z / 4
    q3 = optimize.brentq(F, 0.0, math.pi / 2 - 1e-12)
    q_pred = np.array([-q3, 0.0, q3])
    q_meas = np.quantile(theta, [0.25, 0.5, 0.75])
    print("quartiles of theta = arctan(a/b^2): measured", np.round(q_meas, 3), " predicted", np.round(q_pred, 3))
    print("normalizing constant of the s-density: int (1+s^2)^(-3/4) ds = B(1/2,1/4) =", special.beta(0.5, 0.25))


if __name__ == "__main__":
    main()
