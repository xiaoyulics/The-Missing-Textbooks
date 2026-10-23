"""Chapter 2 (Singular Models), "Measuring it": four singular models whose true sets are crosses,
and whose posteriors at n = 1000 look alike.

  product   y = abx + N(0,1),           x ~ N(0,1),     prior uniform on [-1,1]^2
  tanh      y = a tanh(bx) + N(0,1),    x ~ U[-1,1],    prior uniform on [-2,2]^2
  relu      y = a relu(bx) + N(0,1),    x ~ N(0,1),     prior uniform on [-1,1]^2
  mixture   (1-a) N(0,1) + a N(b,1),    data from N(0,1), prior uniform on [0,1] x [-2,2]
All at the truth "y is pure noise" (for the mixture, N(0,1)), one data set of size n = 1000 each.
The posterior is computed by midpoint quadrature on a grid (sk.free_energy_grid / sk.sample_grid),
1500 draws are written for the scatter figure, and grid quantiles of log|a| and log|b| are
printed. For the product model the grid quantiles of log|a| are checked against the exact
formula of Theorem thm:singular:scales (i).

Outputs: figures/data/02-singular-zoo-{product,tanh,relu,mixture}.dat (columns a b).
Run from the volume root: python figures/scripts/02-singular-zoo.py   (about 40 s)
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import special

import sltkit as sk

OUT = os.path.join("figures", "data")
N = 1000
PS = (0.25, 0.5, 0.75)


def grid_quantiles(E, box, sizes, coord):
    """Quantiles of log|w_coord| under the gridded posterior (cell midpoints); E = energies on the grid."""
    pts, cell, shape = sk.grid(box, sizes)
    logw = -(E - E.min())
    p = np.exp(logw - special.logsumexp(logw))
    v = np.log(np.abs(pts[:, coord]))
    order = np.argsort(v)
    cdf = np.cumsum(p[order])
    return [float(v[order][np.searchsorted(cdf, q)]) for q in PS]


def exact_product_quantiles(Sxx, Sxy):
    w = np.arange(-50.0, 30.0, 0.001)
    z = Sxy / math.sqrt(Sxx)
    g = special.ndtr(np.exp(w) - z) - special.ndtr(-np.exp(w) - z)
    H = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(w))])
    ell = 0.5 * math.log(Sxx)
    D = np.interp(ell, w, H)
    return [float(np.interp(q * D, H, w) - ell) for q in PS]


def main():
    t0 = time.time()
    gen = sk.rng(1000)
    models = {}

    x = gen.standard_normal(N); y = gen.standard_normal(N)
    prod = sk.LinearFeatureModel(lambda w: (w[..., 0] * w[..., 1])[..., None], d=2, k=1).fit(x, y)
    models["product"] = (prod.energy, [(-1, 1), (-1, 1)], (600, 600))
    Sxx, Sxy = float(prod.Sxx[0, 0]), float(prod.Sxy[0])

    xt = gen.uniform(-1, 1, N); yt = gen.standard_normal(N)
    models["tanh"] = (lambda w: sk.regression_energy(sk.mean_tanh, w, xt, yt), [(-2, 2), (-2, 2)], (600, 600))

    xr = gen.standard_normal(N); yr = gen.standard_normal(N)
    models["relu"] = (lambda w: sk.regression_energy(sk.mean_relu, w, xr, yr), [(-1, 1), (-1, 1)], (600, 600))

    xm = gen.standard_normal(N)
    models["mixture"] = (lambda w: sk.mixture_energy(w, xm), [(0, 1), (-2, 2)], (300, 1200))

    for k, (name, (energy, box, sizes)) in enumerate(models.items()):
        E = np.asarray(energy(sk.grid(box, sizes)[0]), dtype=float)      # the energy n L_n, once per grid
        d = sk.sample_grid(lambda pts: E, box, sizes, 1500, seed=2000 + k)
        sk.write_dat(os.path.join(OUT, f"02-singular-zoo-{name}.dat"), ["a", "b"], d)
        qa = grid_quantiles(E, box, sizes, 0)
        qb = grid_quantiles(E, box, sizes, 1)
        near = np.mean(np.minimum(np.abs(d[:, 0]) / (box[0][1] - box[0][0]), np.abs(d[:, 1]) / (box[1][1] - box[1][0])) < 0.05)
        print(f"{name:8s}: quartiles of log|a|/log n {np.round(np.array(qa) / math.log(N), 3)}, "
              f"of log|b|/log n {np.round(np.array(qb) / math.log(N), 3)}; "
              f"draws within 5% of the box width from the cross: {near:.3f}")
    ex = exact_product_quantiles(Sxx, Sxy)
    print(f"product, exact formula: quartiles of log|a|/log n {np.round(np.array(ex) / math.log(N), 3)} "
          f"(Z = {Sxy / math.sqrt(Sxx):+.3f})")
    print(f"runtime {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
