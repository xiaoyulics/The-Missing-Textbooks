"""sltkit -- the shared kit for the "Measuring it" sections of Singular Learning Theory for the
Rest of Us.

Every chapter measures what it computes by hand: a free energy, a learning coefficient, a
generalization error, a WAIC or WBIC value. The measurements run on the book's running toy models,
where the answer is known, so that every measured number can be checked against a theorem.

Conventions (the book's notation; see frontmatter/notation.tex)
  * A regression model is y = f(w, x) + N(0, sigma^2) noise with x ~ N(0, I); a density model is
    p(x | w). Parameters w have d coordinates; batches of parameters have shape (K, d).
  * L_n(w) = -(1/n) sum_i log p(y_i | x_i, w) is the empirical loss (negative log-likelihood per
    sample, Gaussian normalizing constant included). The kit works with the energy n L_n(w).
  * The prior phi is uniform on a box [(lo_1, hi_1), ..., (lo_d, hi_d)] unless stated.
  * The tempered posterior at inverse temperature beta is prop. to exp(-beta n L_n(w)) phi(w), and
    F_n(beta) = -log int exp(-beta n L_n(w)) phi(w) dw; the free energy is F_n = F_n(1). Natural logs.

Contents
  rng(seed)                                a NumPy generator (always pass an explicit seed)
  -- models
  LinearFeatureModel(g, d, k, sigma)       mean g(w) . x with x ~ N(0, I_k): n L_n(w) from sufficient
                                           statistics, fast on large grids of w
  PRODUCT, CUSP, QUARTIC, REGULAR1, REGULAR2, chain(L)
                                           the running toy models y = abx, y = ab^2 x,
                                           y = a x1 + b^2 x2, y = ax, y = a x1 + b x2, y = a1...aL x
  regression_energy(mean_fn, w, x, y)      n L_n(w) for a general mean function (slow, chunked)
  mean_tanh, mean_relu                     y = a tanh(bx) and y = a relu(bx) for regression_energy
  mixture_energy(w, x)                     n L_n for (1-a) N(0,1) + a N(b,1), w = (a, b)
  -- quadrature (d <= 2)
  grid(box, sizes)                         midpoint grid: points (K, d), cell volume, grid shape
  free_energy_grid(energy, box, sizes, beta=1)   F_n(beta) by midpoint quadrature (log-sum-exp)
  tempered_mean_grid(energy, box, sizes, beta, fn=None)   E^beta[fn(w)] (default fn: the energy)
  sample_grid(energy, box, sizes, m, seed, beta=1)        m draws from the gridded posterior
  coefficient_free_energy(E_hat, curvature, rho, lo, hi, c_hat, beta=1)
                                           F_n when n L_n depends on w only through a scalar c = g(w):
                                           -log int exp(-beta E(c)) rho(c) dc, E quadratic in c
  product_density(c, L=2)                  density of a1...aL for independent uniforms on [-1, 1]
  cusp_density(c)                          density of a b^2 for independent uniforms on [-1, 1]
  free_energy_chain(model, L=2)            exact F_n of y = a1...aL x on [-1, 1]^L (via product_density)
  sample_product_posterior(model, m, seed) exact posterior draws of y = abx at every n (via c = ab)
  -- sampling
  metropolis(logp, x0, step, n_steps, seed, adapt=0, thin=1)  random-walk Metropolis, many chains
  -- estimators
  waic(loglik)                             (WAIC, T_n, V_n) from an (S, n) matrix of log p(X_i | w_s)
  wbic(energies)                           the mean of n L_n over draws at beta = 1/log n
  lambda_two_temperatures(E1, E2, b1, b2)  (E1 - E2) / (1/b1 - 1/b2)
  thermodynamic_integration(betas, means)  int_0^1 E^beta[n L_n] dbeta, trapezoid rule
  power_ladder(K, power=5)                 betas (k/K)^power, k = 0..K, for thermodynamic integration
  volume_fraction(K_fn, box, eps, m, seed) Monte Carlo V(eps) = phi{K < eps} with 95% intervals
  loglog_slope(x, y)                       least-squares slope and intercept of log y on log x
  mean_ci(v) / clopper_pearson(k, n)       Student-t and exact binomial 95% intervals
  write_dat(path, header, rows)            whitespace-separated columns with a header row (pgfplots)

Scripts live in figures/scripts/NN-key-name.py, import this module with
    import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import sltkit as sk
run from the volume root, write figures/data/NN-key-name*.dat, print every number the chapter
quotes, and finish in about two minutes on a laptop CPU.

Self-test:  python figures/scripts/sltkit.py
"""
from __future__ import annotations

import math
import os
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy import integrate, special, stats


def rng(seed: int) -> np.random.Generator:
    """A NumPy generator with an explicit seed (never use global random state)."""
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------
class LinearFeatureModel:
    """y = g(w) . x + N(0, sigma^2) with x ~ N(0, I_k): the mean is linear in the input, with
    coefficients g(w). The energy n L_n(w) depends on the data only through
    Sxx = sum x x^T, Sxy = sum x y, Syy = sum y^2, so it can be evaluated on millions of w at once.

    g maps an array of shape (..., d) to shape (..., k)."""

    def __init__(self, g: Callable[[np.ndarray], np.ndarray], d: int, k: int, sigma: float = 1.0,
                 name: str = ""):
        self.g, self.d, self.k, self.sigma, self.name = g, d, k, float(sigma), name
        self.n = 0

    def sample(self, w0: Sequence[float], n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Draw n pairs (x, y) from the model at the true parameter w0; returns x of shape (n, k)."""
        gen = rng(seed)
        x = gen.standard_normal((n, self.k))
        c0 = np.asarray(self.g(np.asarray(w0, dtype=float)), dtype=float).reshape(self.k)
        y = x @ c0 + self.sigma * gen.standard_normal(n)
        return x, y

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearFeatureModel":
        """Store the sufficient statistics of the data (x of shape (n, k) or (n,), y of shape (n,))."""
        x = np.asarray(x, dtype=float).reshape(len(y), self.k)
        y = np.asarray(y, dtype=float)
        self.n = len(y)
        self.Sxx, self.Sxy, self.Syy = x.T @ x, x.T @ y, float(y @ y)
        return self

    def energy_of_coefficients(self, c: np.ndarray) -> np.ndarray:
        """n L_n as a function of the coefficient vector c (shape (..., k))."""
        c = np.asarray(c, dtype=float)
        quad = np.einsum("...i,ij,...j->...", c, self.Sxx, c)
        lin = np.einsum("...i,i->...", c, self.Sxy)
        s2 = self.sigma ** 2
        return (self.Syy - 2.0 * lin + quad) / (2.0 * s2) + 0.5 * self.n * math.log(2 * math.pi * s2)

    def energy(self, w: np.ndarray) -> np.ndarray:
        """n L_n(w) for w of shape (..., d)."""
        c = np.asarray(self.g(np.asarray(w, dtype=float)), dtype=float)
        if self.k == 1 and c.shape[-1:] != (1,):
            c = c[..., None]
        return self.energy_of_coefficients(c)

    def least_squares(self) -> np.ndarray:
        """The coefficient vector c_hat minimizing n L_n over all of R^k."""
        return np.linalg.solve(self.Sxx, self.Sxy)

    def min_energy(self) -> float:
        """min over c in R^k of n L_n (a lower bound for min over w; equal when g is onto R^k)."""
        return float(self.energy_of_coefficients(self.least_squares()))


def _col(w: np.ndarray, j: int) -> np.ndarray:
    return np.asarray(w, dtype=float)[..., j]


PRODUCT = LinearFeatureModel(lambda w: (_col(w, 0) * _col(w, 1))[..., None], d=2, k=1, name="y = abx")
CUSP = LinearFeatureModel(lambda w: (_col(w, 0) * _col(w, 1) ** 2)[..., None], d=2, k=1, name="y = ab^2x")
QUARTIC = LinearFeatureModel(lambda w: np.stack([_col(w, 0), _col(w, 1) ** 2], axis=-1), d=2, k=2,
                             name="y = a x1 + b^2 x2")
REGULAR1 = LinearFeatureModel(lambda w: _col(w, 0)[..., None], d=1, k=1, name="y = ax")
REGULAR2 = LinearFeatureModel(lambda w: np.stack([_col(w, 0), _col(w, 1)], axis=-1), d=2, k=2,
                              name="y = a x1 + b x2")


def chain(L: int) -> LinearFeatureModel:
    """The deep linear chain y = a1 a2 ... aL x (d = L)."""
    return LinearFeatureModel(lambda w: np.prod(np.asarray(w, dtype=float), axis=-1)[..., None],
                              d=L, k=1, name=f"y = a1...a{L} x")


def regression_energy(mean_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], w: np.ndarray,
                      x: np.ndarray, y: np.ndarray, sigma: float = 1.0, chunk: int = 2048) -> np.ndarray:
    """n L_n(w) for y = mean_fn(w, x) + N(0, sigma^2), for a batch w of shape (K, d).
    mean_fn(w, x) takes w of shape (B, d) and x of shape (n, ...) and returns shape (B, n)."""
    w = np.atleast_2d(np.asarray(w, dtype=float))
    y = np.asarray(y, dtype=float)
    n = len(y)
    out = np.empty(len(w))
    const = 0.5 * n * math.log(2 * math.pi * sigma ** 2)
    for s in range(0, len(w), chunk):
        r = y[None, :] - mean_fn(w[s:s + chunk], x)
        out[s:s + chunk] = 0.5 * np.einsum("bn,bn->b", r, r) / sigma ** 2 + const
    return out


def mean_tanh(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """y = a tanh(b x) for w = (a, b)."""
    return w[:, 0:1] * np.tanh(w[:, 1:2] * np.asarray(x, dtype=float).reshape(1, -1))


def mean_relu(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """y = a relu(b x) for w = (a, b)."""
    return w[:, 0:1] * np.maximum(w[:, 1:2] * np.asarray(x, dtype=float).reshape(1, -1), 0.0)


def mixture_energy(w: np.ndarray, x: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """n L_n(w) for the mixture (1 - a) N(0, 1) + a N(b, 1), w = (a, b) with a in [0, 1]."""
    w = np.atleast_2d(np.asarray(w, dtype=float))
    x = np.asarray(x, dtype=float).reshape(1, -1)
    out = np.empty(len(w))
    log_norm = -0.5 * math.log(2 * math.pi)
    for s in range(0, len(w), chunk):
        a, b = w[s:s + chunk, 0:1], w[s:s + chunk, 1:2]
        # log((1-a) e^{-x^2/2} + a e^{-(x-b)^2/2}) = -x^2/2 + log(1 - a + a e^{bx - b^2/2})
        t = np.log1p(a * np.expm1(b * x - 0.5 * b * b))
        out[s:s + chunk] = -np.sum(log_norm - 0.5 * x * x + t, axis=1)
    return out


# ---------------------------------------------------------------------------------------------
# Quadrature on a grid (d <= 2)
# ---------------------------------------------------------------------------------------------
def _box(box) -> np.ndarray:
    b = np.asarray(box, dtype=float).reshape(-1, 2)
    if np.any(b[:, 1] <= b[:, 0]):
        raise ValueError("each box interval must have lo < hi")
    return b


def grid(box, sizes) -> tuple[np.ndarray, float, tuple]:
    """Midpoints of a regular grid on the box: points of shape (K, d), the volume of one cell,
    and the grid shape (for reshaping values to an image)."""
    b = _box(box)
    sizes = [int(s) for s in np.atleast_1d(sizes)]
    if len(sizes) == 1 and len(b) > 1:
        sizes = sizes * len(b)
    axes, cell = [], 1.0
    for (lo, hi), m in zip(b, sizes):
        h = (hi - lo) / m
        axes.append(lo + h * (np.arange(m) + 0.5))
        cell *= h
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([g.ravel() for g in mesh], axis=-1)
    return pts, cell, tuple(sizes)


def _log_weights(energy, box, sizes, beta):
    pts, cell, shape = grid(box, sizes)
    E = np.asarray(energy(pts), dtype=float).reshape(-1)
    b = _box(box)
    log_prior = -float(np.sum(np.log(b[:, 1] - b[:, 0])))
    return pts, E, -beta * E + math.log(cell) + log_prior, shape


def free_energy_grid(energy: Callable[[np.ndarray], np.ndarray], box, sizes, beta: float = 1.0) -> float:
    """F_n(beta) = -log int exp(-beta n L_n(w)) phi(w) dw with phi uniform on the box, by the
    midpoint rule. energy maps points (K, d) to n L_n (K,). Check convergence by doubling sizes."""
    _, _, logw, _ = _log_weights(energy, box, sizes, beta)
    return float(-special.logsumexp(logw))


def tempered_mean_grid(energy, box, sizes, beta: float = 1.0, fn=None) -> float:
    """E^beta[fn(w)] under the tempered posterior on the grid; fn defaults to the energy n L_n."""
    pts, E, logw, _ = _log_weights(energy, box, sizes, beta)
    p = np.exp(logw - special.logsumexp(logw))
    vals = E if fn is None else np.asarray(fn(pts), dtype=float).reshape(-1)
    return float(p @ vals)


def sample_grid(energy, box, sizes, m: int, seed: int, beta: float = 1.0) -> np.ndarray:
    """m approximate draws from the tempered posterior: a grid cell with probability proportional
    to its weight, then a uniform point inside the cell. Returns shape (m, d)."""
    pts, _, logw, shape = _log_weights(energy, box, sizes, beta)
    p = np.exp(logw - special.logsumexp(logw))
    gen = rng(seed)
    idx = gen.choice(len(p), size=m, p=p / p.sum())
    b = _box(box)
    h = (b[:, 1] - b[:, 0]) / np.asarray(shape)
    return pts[idx] + (gen.random((m, len(b))) - 0.5) * h


# ---------------------------------------------------------------------------------------------
# Exact free energies through the density of a scalar coefficient
# ---------------------------------------------------------------------------------------------
def product_density(c, L: int = 2) -> np.ndarray:
    """Density of c = a1 a2 ... aL for independent uniforms on [-1, 1]:
    (-log|c|)^(L-1) / (2 (L-1)!) on [-1, 1] (L = 1: the uniform density 1/2)."""
    c = np.abs(np.asarray(c, dtype=float))
    with np.errstate(divide="ignore"):
        val = np.where(c < 1.0, (-np.log(c)) ** (L - 1) / (2.0 * math.factorial(L - 1)), 0.0)
    return np.where(c == 0.0, np.inf if L > 1 else 0.5, val)


def cusp_density(c) -> np.ndarray:
    """Density of c = a b^2 for independent uniforms a, b on [-1, 1]: (|c|^(-1/2) - 1) / 2 on [-1, 1]."""
    c = np.abs(np.asarray(c, dtype=float))
    with np.errstate(divide="ignore"):
        return np.where(c < 1.0, 0.5 * (c ** -0.5 - 1.0), 0.0)


def coefficient_free_energy(E_hat: float, curvature: float, rho: Callable, lo: float, hi: float,
                            c_hat: float, beta: float = 1.0, singular_points: Sequence[float] = (0.0,)) -> float:
    """-log int_lo^hi exp(-beta E(c)) rho(c) dc for E(c) = E_hat + curvature (c - c_hat)^2 / 2.

    This is the free energy of every model whose energy depends on w only through one scalar
    c = g(w) (y = abx, y = ab^2 x, the chain y = a1...aL x): rho is the density of g(w) under the
    prior. Adaptive quadrature over c_hat +- 40 standard deviations, split at the singular points
    of rho (where it may be infinite but integrable)."""
    s = 1.0 / math.sqrt(beta * curvature)
    a, b = max(lo, c_hat - 40 * s), min(hi, c_hat + 40 * s)
    if a >= b:
        return math.inf
    cuts = sorted({a, b, *[p for p in singular_points if a < p < b]})
    total = 0.0
    for u, v in zip(cuts[:-1], cuts[1:]):
        val, _ = integrate.quad(lambda c: math.exp(-0.5 * ((c - c_hat) / s) ** 2) * float(rho(c)),
                                u, v, limit=400, epsabs=0.0, epsrel=1e-10)
        total += val
    return beta * E_hat - math.log(total)


def sample_product_posterior(model: LinearFeatureModel, m: int, seed: int, beta: float = 1.0) -> np.ndarray:
    """Exact draws from the tempered posterior of y = abx with the uniform prior on [-1, 1]^2
    (fit PRODUCT first). The coefficient c = ab is drawn from its one-dimensional posterior
    (inverse CDF on a fine grid), then, given c, log|a| is uniform on [log|c|, 0] with a random
    sign and b = c / a: under the uniform prior the conditional density of a given ab = c is
    proportional to 1/|a| on |c| <= |a| <= 1. Returns shape (m, 2)."""
    if model.k != 1 or model.n == 0:
        raise ValueError("fit the product model first")
    curv = beta * float(model.Sxx[0, 0]) / model.sigma ** 2
    c_hat = float(model.Sxy[0] / model.Sxx[0, 0])
    s = 1.0 / math.sqrt(curv)
    lo, hi = max(-1.0, c_hat - 14 * s), min(1.0, c_hat + 14 * s)
    u = np.linspace(-1.0, 1.0, 40001)
    near0 = np.sign(u) * np.abs(u) ** 8 * max(abs(lo), abs(hi))      # dense near 0
    grid_c = np.unique(np.concatenate([np.linspace(lo, hi, 40001), near0]))
    grid_c = grid_c[(grid_c >= lo) & (grid_c <= hi) & (grid_c != 0.0) & (np.abs(grid_c) < 1.0)]
    logp = -0.5 * ((grid_c - c_hat) / s) ** 2 + np.log(product_density(grid_c, 2))
    p = np.exp(logp - logp.max())
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(grid_c))])
    cdf /= cdf[-1]
    gen = rng(seed)
    c = np.interp(gen.random(m), cdf, grid_c)
    la = np.log(np.abs(c)) * gen.random(m)                             # uniform on [log|c|, 0]
    a = np.where(gen.random(m) < 0.5, -1.0, 1.0) * np.exp(la)
    return np.stack([a, c / a], axis=1)


def free_energy_chain(model: LinearFeatureModel, L: int = 2, beta: float = 1.0) -> float:
    """Exact F_n(beta) for y = a1...aL x with the uniform prior on [-1, 1]^L (L = 2: y = abx).
    The model must be fitted (model.fit(x, y)) and have k = 1."""
    if model.k != 1 or model.n == 0:
        raise ValueError("fit a one-coefficient model first")
    curv = float(model.Sxx[0, 0]) / model.sigma ** 2
    c_hat = float(model.Sxy[0] / model.Sxx[0, 0])
    E_hat = float(model.energy_of_coefficients(np.array([c_hat])))
    return coefficient_free_energy(E_hat, curv, lambda c: product_density(c, L), -1.0, 1.0, c_hat, beta)


# ---------------------------------------------------------------------------------------------
# Random-walk Metropolis over many chains
# ---------------------------------------------------------------------------------------------
def metropolis(logp: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, step, n_steps: int, seed: int,
               adapt: int = 0, target: float = 0.3, thin: int = 1) -> tuple[np.ndarray, float]:
    """Random-walk Metropolis with Gaussian proposals, run on C chains at once.

    x0 has shape (C, d); logp maps (C, d) to (C,) and may return -inf outside the support. During
    the first `adapt` steps each chain's step size is tuned toward the target acceptance rate; those
    steps are discarded. Returns the kept draws, shape (n_kept, C, d), and the acceptance rate
    after adaptation."""
    gen = rng(seed)
    x = np.array(x0, dtype=float, copy=True)
    C, d = x.shape
    lp = np.asarray(logp(x), dtype=float)
    h = np.full(C, float(step)) if np.ndim(step) == 0 else np.asarray(step, dtype=float).copy()
    kept, acc, tot = [], 0, 0
    for t in range(adapt + n_steps):
        prop = x + h[:, None] * gen.standard_normal((C, d))
        lq = np.asarray(logp(prop), dtype=float)
        ok = np.log(gen.random(C)) < (lq - lp)
        x[ok], lp[ok] = prop[ok], lq[ok]
        if t < adapt:
            h *= np.exp(0.05 * (ok.astype(float) - target))
        else:
            acc += int(ok.sum()); tot += C
            if (t - adapt) % thin == 0:
                kept.append(x.copy())
    return np.array(kept), (acc / tot if tot else float("nan"))


# ---------------------------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------------------------
def waic(loglik: np.ndarray) -> tuple[float, float, float]:
    """WAIC on the per-sample loss scale from loglik[s, i] = log p(X_i | w_s) for S posterior draws:
    T_n = -(1/n) sum_i log E_w[p(X_i | w)] (Bayes training loss), V_n = sum_i Var_w[log p(X_i | w)]
    (functional variance), WAIC = T_n + V_n / n. Returns (WAIC, T_n, V_n)."""
    ll = np.asarray(loglik, dtype=float)
    S, n = ll.shape
    lppd = special.logsumexp(ll, axis=0) - math.log(S)
    T = -float(np.mean(lppd))
    V = float(np.sum(np.var(ll, axis=0, ddof=1)))
    return T + V / n, T, V


def wbic(energies: np.ndarray) -> float:
    """WBIC: the mean of n L_n(w) over draws from the tempered posterior at beta = 1 / log n."""
    return float(np.mean(energies))


def lambda_two_temperatures(E1: float, E2: float, beta1: float, beta2: float) -> float:
    """Estimate of the learning coefficient from tempered means E_j = E^{beta_j}[n L_n]:
    (E1 - E2) / (1/beta1 - 1/beta2)."""
    return (E1 - E2) / (1.0 / beta1 - 1.0 / beta2)


def power_ladder(K: int, power: float = 5.0) -> np.ndarray:
    """Inverse temperatures (k/K)^power, k = 0..K: dense near 0, where E^beta[n L_n] changes fast."""
    return (np.arange(K + 1) / K) ** power


def thermodynamic_integration(betas: Sequence[float], means: Sequence[float]) -> float:
    """F_n = int_0^1 E^beta[n L_n] dbeta (prior normalized), by the trapezoid rule; betas must run
    from 0 to 1."""
    b, m = np.asarray(betas, dtype=float), np.asarray(means, dtype=float)
    if abs(b[0]) > 1e-12 or abs(b[-1] - 1.0) > 1e-12:
        raise ValueError("betas must start at 0 and end at 1")
    return float(np.sum(0.5 * (m[1:] + m[:-1]) * np.diff(b)))


def volume_fraction(K_fn: Callable[[np.ndarray], np.ndarray], box, eps: Sequence[float], m: int, seed: int,
                    chunk: int = 1_000_000) -> np.ndarray:
    """Monte Carlo estimate of V(eps) = phi{w : K(w) < eps} for the uniform prior on the box.
    Returns rows (eps, V, lo, hi, count) with Clopper-Pearson 95% intervals."""
    b = _box(box)
    gen = rng(seed)
    eps = np.asarray(eps, dtype=float)
    counts = np.zeros(len(eps), dtype=np.int64)
    done = 0
    while done < m:
        k = min(chunk, m - done)
        w = b[:, 0] + (b[:, 1] - b[:, 0]) * gen.random((k, len(b)))
        Kw = np.asarray(K_fn(w), dtype=float).reshape(-1)
        counts += np.searchsorted(np.sort(Kw), eps, side="left")
        done += k
    rows = []
    for e, c in zip(eps, counts):
        lo, hi = clopper_pearson(int(c), m)
        rows.append((e, c / m, lo, hi, int(c)))
    return np.array(rows)


def loglog_slope(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Least-squares slope and intercept of log y against log x (positive data only)."""
    lx, ly = np.log(np.asarray(x, dtype=float)), np.log(np.asarray(y, dtype=float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, icept = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(slope), float(icept)


def mean_ci(v: Sequence[float], conf: float = 0.95) -> tuple[float, float, float]:
    """Mean and Student-t interval: (mean, lo, hi)."""
    v = np.asarray(v, dtype=float)
    mu = float(np.mean(v))
    if len(v) < 2:
        return mu, float("nan"), float("nan")
    half = float(stats.t.ppf(0.5 + conf / 2, len(v) - 1) * np.std(v, ddof=1) / math.sqrt(len(v)))
    return mu, mu - half, mu + half


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) interval for a binomial proportion with k successes in n trials."""
    a = 1.0 - conf
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def write_dat(path: str, header: Iterable[str], rows: Iterable[Iterable]) -> None:
    """Write whitespace-separated columns with a header row (pgfplots reads it by column name)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = list(header)
    with open(path, "w", newline="\n") as fh:
        fh.write(" ".join(header) + "\n")
        for row in rows:
            row = list(row)
            if len(row) != len(header):
                raise ValueError("row length does not match header")
            fh.write(" ".join(_fmt(v) for v in row) + "\n")


def _fmt(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    return str(v)


# ---------------------------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------------------------
def _self_test() -> None:
    # densities integrate to one
    for L in (1, 2, 3):
        tot = 2 * integrate.quad(lambda c: float(product_density(c, L)), 0, 1, limit=200)[0]
        assert abs(tot - 1) < 1e-6, (L, tot)
    assert abs(2 * integrate.quad(lambda c: float(cusp_density(c)), 0, 1, limit=200)[0] - 1) < 1e-6

    # y = abx at the singular truth: exact (1-D) and grid (2-D) free energies agree
    x, y = PRODUCT.sample([0.0, 0.0], 400, seed=1)
    PRODUCT.fit(x, y)
    F_exact = free_energy_chain(PRODUCT, 2)
    F_grid = free_energy_grid(PRODUCT.energy, [(-1, 1), (-1, 1)], 1600)
    assert abs(F_exact - F_grid) < 2e-3, (F_exact, F_grid)
    # exact posterior draws of y = abx agree with gridded draws (quartiles of log|a|)
    d_exact = sample_product_posterior(PRODUCT, 20000, seed=11)
    d_grid = sample_grid(PRODUCT.energy, [(-1, 1), (-1, 1)], 1600, 20000, seed=12)
    q1 = np.quantile(np.log(np.abs(d_exact[:, 0])), [0.25, 0.5, 0.75])
    q2 = np.quantile(np.log(np.abs(d_grid[:, 0])), [0.25, 0.5, 0.75])
    assert np.all(np.abs(q1 - q2) < 0.08), (q1, q2)

    # regular y = ax: Laplace's approximation F = E_hat + (1/2) log(curv / 2 pi) - log(prior density)
    x, y = REGULAR1.sample([0.3], 20000, seed=2)
    REGULAR1.fit(x, y)
    F = free_energy_grid(REGULAR1.energy, [(-1, 1)], 200000)
    lap = REGULAR1.min_energy() + 0.5 * math.log(REGULAR1.Sxx[0, 0] / (2 * math.pi)) - math.log(0.5)
    assert abs(F - lap) < 1e-3, (F, lap)

    # tempered means of a regular model: E^beta[n L_n] - min = d / (2 beta) up to O(1/sqrt(n))
    E1 = tempered_mean_grid(REGULAR1.energy, [(-1, 1)], 200000, beta=1.0)
    E2 = tempered_mean_grid(REGULAR1.energy, [(-1, 1)], 200000, beta=0.5)
    lam = lambda_two_temperatures(E1, E2, 1.0, 0.5)
    assert abs(lam - 0.5) < 1e-3, lam

    # thermodynamic integration reproduces the free energy of a 1-D model
    betas = power_ladder(400)
    means = [tempered_mean_grid(REGULAR1.energy, [(-1, 1)], 200000, beta=b) for b in betas]
    assert abs(thermodynamic_integration(betas, means) - F) < 0.05

    # Metropolis samples a standard normal
    draws, acc = metropolis(lambda z: -0.5 * np.sum(z * z, axis=1), np.zeros((64, 1)), 1.0, 2000,
                            seed=3, adapt=500)
    assert abs(draws.mean()) < 0.05 and abs(draws.var() - 1) < 0.05 and 0.15 < acc < 0.5

    # WAIC of a regular model: n (WAIC - T_n) = V_n is close to d = 1
    x, y = REGULAR1.sample([0.3], 2000, seed=4)
    REGULAR1.fit(x, y)
    post = sample_grid(REGULAR1.energy, [(-1, 1)], 100000, 4000, seed=4)
    ll = -0.5 * (y[None, :] - post[:, 0:1] * x[None, :, 0]) ** 2 - 0.5 * math.log(2 * math.pi)
    _, _, V = waic(ll)
    assert abs(V - 1.0) < 0.2, V

    # volume of {(ab)^2 < eps}: the exact value is t (1 + log(1/t)) with t = sqrt(eps)
    rows = volume_fraction(lambda w: (w[:, 0] * w[:, 1]) ** 2, [(-1, 1), (-1, 1)], [1e-4], 2_000_000, seed=5)
    t = 1e-2
    assert rows[0, 2] < t * (1 - math.log(t)) < rows[0, 3]

    # mixture energy at a = 0 equals the N(0, 1) energy
    xs = rng(6).standard_normal(50)
    e0 = mixture_energy(np.array([[0.0, 1.3]]), xs)[0]
    assert abs(e0 - (0.5 * np.sum(xs ** 2) + 25 * math.log(2 * math.pi))) < 1e-9
    print("sltkit self-test passed")


if __name__ == "__main__":
    _self_test()
