"""rosettakit -- the shared kit of *The Replica Rosetta Stone* (The Missing Textbooks, volume 7).

Every chapter script imports it:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rosettakit as rk

Conventions (OUTLINE.md, "Global notation and conventions"):
- Z ~ N(0,1); E f(Z) by Gauss--Hermite quadrature (`gauss`); H(x) = P(Z > x) (`H`).
- Free entropy phi = (1/N) E log Z_N (not the physicist's free energy f = -phi/beta).
- SK: H_N(s) = (beta/sqrt N) sum_{i<j} g_ij s_i s_j + h sum_i s_i, so beta_c = 1.
- REM: Z_N = sum_s exp(beta sqrt(N) X_s), X_s iid N(0,1), beta_c = sqrt(2 log 2).
- Spiked Wigner: A = (lam/N) x x^T + W, W ~ GOE(N) (off-diagonal variance 1/N); BBP at lam = 1.
  Scalar channel Y = sqrt(gamma) X + Z; replica-symmetric fixed point q = 1 - mmse(lam^2 q).
- Perceptron: patterns g ~ N(0, I_d), n = alpha d, constraints <g, w>/sqrt d >= kappa;
  Gardner's capacity alpha_c(kappa) = 1 / E[(Z + kappa)_+^2].
- Ridge (per-sample normalized): beta_hat = argmin (1/2n)|y - X b|^2 + (lam/2)|b|^2, x ~ N(0, I_d),
  y = <x, beta*> + sigma eps, r^2 = |beta*|^2, excess risk R; alpha = n/d, gamma = d/n.

Run `python figures/scripts/rosettakit.py` for the self-test. numpy and scipy only.
"""
import math
import os

import numpy as np
from scipy import optimize, special

# ---------------------------------------------------------------------------------------------
# Random numbers, quadrature, fixed points
# ---------------------------------------------------------------------------------------------


def rng(seed):
    """A numpy Generator with a fixed seed (every script must be deterministic)."""
    return np.random.default_rng(seed)


_GH = {}


def _gh(n):
    if n not in _GH:
        x, w = np.polynomial.hermite_e.hermegauss(n)   # weight exp(-x^2/2)
        _GH[n] = (x, w / math.sqrt(2.0 * math.pi))
    return _GH[n]


def gauss(f, n=121):
    """E f(Z) for Z ~ N(0,1), by Gauss--Hermite quadrature with n nodes. f must accept arrays.
    Accurate for smooth f; for f with kinks (relu, sign) use `gauss_split` or a larger n."""
    x, w = _gh(n)
    return float(np.sum(w * f(x)))


def gauss_split(f, a=0.0, n=200):
    """E f(Z) for f smooth on each side of the point a: Gauss--Legendre on (-12, a) and (a, 12)
    against the normal density (error below 1e-12 for polynomially bounded f)."""
    x, w = np.polynomial.legendre.leggauss(n)
    total = 0.0
    for lo, hi in ((-12.0, a), (a, 12.0)):
        if hi <= lo:
            continue
        t = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
        total += 0.5 * (hi - lo) * float(np.sum(w * f(t) * np.exp(-0.5 * t * t))) / math.sqrt(2 * math.pi)
    return total


def gauss2(f, n=61):
    """E f(Z1, Z2) for independent standard normals (tensor Gauss--Hermite)."""
    x, w = _gh(n)
    X1, X2 = np.meshgrid(x, x, indexing="ij")
    W = np.outer(w, w)
    return float(np.sum(W * f(X1, X2)))


def H(x):
    """Gaussian tail P(Z > x) (Gardner's H)."""
    return special.ndtr(-np.asarray(x, dtype=float))


def logH(x):
    """log P(Z > x), stable for large x."""
    return special.log_ndtr(-np.asarray(x, dtype=float))


def fixed_point(F, x0, damping=0.5, tol=1e-12, maxit=100000):
    """Damped iteration x <- (1 - damping) x + damping F(x) until |change| < tol.
    Returns (x, iterations, converged). Works for scalars and numpy arrays."""
    x = np.asarray(x0, dtype=float)
    for it in range(1, maxit + 1):
        x_new = (1.0 - damping) * x + damping * np.asarray(F(x), dtype=float)
        if np.max(np.abs(x_new - x)) < tol:
            return (float(x_new) if x_new.ndim == 0 else x_new), it, True
        x = x_new
    return (float(x) if x.ndim == 0 else x), maxit, False


def root(f, a, b, tol=1e-14):
    """A root of f in [a, b] (f(a) and f(b) of opposite signs), by Brent's method."""
    return optimize.brentq(f, a, b, xtol=tol, rtol=4 * np.finfo(float).eps, maxiter=500)


# ---------------------------------------------------------------------------------------------
# Scalar Gaussian channels Y = sqrt(gamma) X + Z
# ---------------------------------------------------------------------------------------------


def mmse_gaussian(gamma):
    """MMSE of X ~ N(0,1) from Y = sqrt(gamma) X + Z."""
    return 1.0 / (1.0 + gamma)


def mmse_rademacher(gamma, n=161):
    """MMSE of X uniform on {-1, +1}: 1 - E tanh(gamma + sqrt(gamma) Z)."""
    if gamma <= 0:
        return 1.0
    return 1.0 - gauss(lambda z: np.tanh(gamma + math.sqrt(gamma) * z), n)


def sparse_rademacher_posterior_mean(y, gamma, rho):
    """E[X | sqrt(gamma) X + Z = y] for X = +-1/sqrt(rho) w.p. rho/2 each, 0 w.p. 1 - rho
    (so E X^2 = 1)."""
    a = 1.0 / math.sqrt(rho)
    sg = math.sqrt(gamma)
    # log weights of +a, -a, 0 (the common factor exp(-y^2/2) cancels)
    lp = np.log(rho / 2) + sg * a * y - gamma * a * a / 2
    lm = np.log(rho / 2) - sg * a * y - gamma * a * a / 2
    l0 = (np.log(1 - rho) if rho < 1 else -np.inf) + 0.0 * y
    m = np.maximum(np.maximum(lp, lm), l0)
    wp, wm, w0 = np.exp(lp - m), np.exp(lm - m), np.exp(l0 - m)
    return a * (wp - wm) / (wp + wm + w0)


def mmse_sparse_rademacher(gamma, rho, n=201):
    """MMSE of the sparse Rademacher prior (E X^2 = 1) at SNR gamma, by quadrature over the three
    values of X."""
    if gamma <= 0:
        return 1.0
    a = 1.0 / math.sqrt(rho)
    sg = math.sqrt(gamma)
    err = 0.0
    for xv, p in ((a, rho / 2), (-a, rho / 2), (0.0, 1 - rho)):
        err += p * gauss(lambda z: (sparse_rademacher_posterior_mean(sg * xv + z, gamma, rho) - xv) ** 2, n)
    return err


# ---------------------------------------------------------------------------------------------
# Spiked Wigner, AMP and state evolution
# ---------------------------------------------------------------------------------------------


def goe(N, gen):
    """W ~ GOE(N): symmetric, off-diagonal N(0, 1/N), diagonal N(0, 2/N); semicircle on [-2, 2]."""
    G = gen.standard_normal((N, N)) / math.sqrt(N)
    return (G + G.T) / math.sqrt(2.0)


def spiked_wigner(x, lam, gen):
    """A = (lam/N) x x^T + W for a signal x in R^N (E x_i^2 = 1)."""
    N = x.shape[0]
    return (lam / N) * np.outer(x, x) + goe(N, gen)


def bbp_eigenvalue(lam):
    """Limit of the top eigenvalue of the spiked Wigner matrix: lam + 1/lam for lam > 1, else 2."""
    return lam + 1.0 / lam if lam > 1 else 2.0


def bbp_overlap(lam):
    """Limit of <v1, x/|x|>^2 for the top eigenvector v1: 1 - 1/lam^2 for lam > 1, else 0."""
    return 1.0 - 1.0 / lam ** 2 if lam > 1 else 0.0


def rs_overlap(lam, mmse=mmse_rademacher, q0=1.0, damping=1.0, tol=1e-13):
    """Replica-symmetric fixed point q = 1 - mmse(lam^2 q) of spiked Wigner, iterated from q0
    (q0 = 1 finds the informative branch, q0 small the uninformative one). Returns (q, iterations)."""
    q, it, ok = fixed_point(lambda q: 1.0 - mmse(lam ** 2 * max(q, 0.0)), q0, damping=damping, tol=tol)
    if not ok:
        raise RuntimeError("rs_overlap did not converge")
    return q, it


def se_z2(lam, m0, iters):
    """Bayes-optimal state evolution for Z2 synchronization (Rademacher prior), started from an
    estimate with overlap m0 = E[X xhat0] and E xhat0^2 = m0 (Nishimori): returns the overlaps
    m_t = E[X xhat_t], t = 0..iters, with m_{t+1} = 1 - mmse_rademacher(lam^2 m_t)."""
    ms = [m0]
    for _ in range(iters):
        ms.append(1.0 - mmse_rademacher(lam ** 2 * ms[-1]))
    return np.array(ms)


def amp_z2(A, lam, xhat0, iters):
    """Bayes-AMP for A = (lam/N) x x^T + W with a Rademacher prior:
        y^{t+1} = A f_t(y^t) - b_t f_{t-1}(y^{t-1}),  f_t(y) = tanh(lam y),  b_t = mean f_t'(y^t),
    the denoiser tanh(mu y / sigma^2) with mu = lam m, sigma^2 = m (Nishimori), which is tanh(lam y).
    xhat0 is the initial estimate f_0 (entries in [-1, 1]); the first step has no Onsager term.
    Returns the list of estimates xhat_t = f_t(y^t), t = 0..iters."""
    xh_prev = np.zeros_like(xhat0)
    xh = np.array(xhat0, dtype=float)
    b = 0.0
    out = [xh.copy()]
    for _ in range(iters):
        y = A @ xh - b * xh_prev
        xh_new = np.tanh(lam * y)
        b = lam * float(np.mean(1.0 - xh_new ** 2))   # (1/N) sum f'(y) for f = tanh(lam y)
        xh_prev, xh = xh, xh_new
        out.append(xh.copy())
    return out


# ---------------------------------------------------------------------------------------------
# Spin glasses: SK and the REM
# ---------------------------------------------------------------------------------------------


def sk_rs_q(beta, h, q0=0.5):
    """Replica-symmetric overlap of SK: q = E tanh^2(beta sqrt(q) Z + h)."""
    F = lambda q: gauss(lambda z: np.tanh(beta * math.sqrt(max(q, 0.0)) * z + h) ** 2)
    q, it, ok = fixed_point(F, q0, damping=0.5)
    if not ok:
        raise RuntimeError("sk_rs_q did not converge")
    return q


def sk_rs_free_entropy(beta, h, q=None):
    """phi_RS = log 2 + E log cosh(beta sqrt(q) Z + h) + beta^2 (1 - q)^2 / 4 at the RS overlap."""
    if q is None:
        q = sk_rs_q(beta, h)
    return math.log(2.0) + gauss(lambda z: np.log(np.cosh(beta * math.sqrt(q) * z + h))) + beta ** 2 * (1 - q) ** 2 / 4


def sk_at(beta, h, q=None):
    """The de Almeida--Thouless quantity beta^2 E sech^4(beta sqrt(q) Z + h); RS is stable iff <= 1."""
    if q is None:
        q = sk_rs_q(beta, h)
    return beta ** 2 * gauss(lambda z: 1.0 / np.cosh(beta * math.sqrt(q) * z + h) ** 4)


def all_spins(N):
    """All 2^N configurations of N spins as a (2^N, N) array of +-1 (N <= 20)."""
    idx = np.arange(2 ** N)[:, None]
    return 1 - 2 * ((idx >> np.arange(N)) & 1)


def sk_log_partition(g, beta, h, spins=None):
    """log Z_N for one SK sample with couplings g (an N x N array; only i < j is used), by exact
    enumeration (N <= 18)."""
    N = g.shape[0]
    S = all_spins(N) if spins is None else spins
    J = np.triu(g, 1)
    energy = (beta / math.sqrt(N)) * np.einsum("ki,ij,kj->k", S, J, S, optimize=True) + h * S.sum(axis=1)
    m = energy.max()
    return float(m + np.log(np.exp(energy - m).sum()))


def rem_free_entropy(beta):
    """Free entropy of the REM: log 2 + beta^2/2 below beta_c = sqrt(2 log 2), beta sqrt(2 log 2) above."""
    bc = math.sqrt(2 * math.log(2))
    return math.log(2) + beta ** 2 / 2 if beta <= bc else beta * bc


# ---------------------------------------------------------------------------------------------
# Perceptrons
# ---------------------------------------------------------------------------------------------


def gardner_capacity(kappa):
    """Gardner's replica-symmetric capacity 1 / E[(Z + kappa)_+^2] = 1/[(1 + k^2) Phi(k) + k phi(k)].
    Proved for kappa >= 0; for kappa < 0 an upper bound only (Chapter 8)."""
    k = float(kappa)
    Phi = special.ndtr(k)
    phi = math.exp(-0.5 * k * k) / math.sqrt(2 * math.pi)
    return 1.0 / ((1 + k * k) * Phi + k * phi)


def cover_probability(n, d):
    """Cover (1965): probability that n points in general position in R^d, with uniformly random
    labels, are separable by a hyperplane through the origin: 2^(1-n) sum_{k<d} C(n-1, k)."""
    if n <= d:
        return 1.0
    k = np.arange(d)
    logs = special.gammaln(n) - special.gammaln(k + 1) - special.gammaln(n - k) + (1 - n) * math.log(2)
    return float(min(1.0, np.exp(special.logsumexp(logs))))


def separable(G):
    """True if there is w with G w > 0 componentwise (rows of G are the labeled patterns y_mu x_mu),
    decided by the linear program: find w with G w >= 1."""
    n, d = G.shape
    res = optimize.linprog(np.zeros(d), A_ub=-G, b_ub=-np.ones(n), bounds=[(None, None)] * d,
                           method="highs")
    return res.status == 0


# ---------------------------------------------------------------------------------------------
# Ridge regression (isotropic Gaussian design)
# ---------------------------------------------------------------------------------------------


def mp_stieltjes(z, gamma):
    """Stieltjes transform m(z) = int mu(ds)/(s - z) of the Marchenko--Pastur law of X^T X / n for
    x ~ N(0, I_d), gamma = d/n, at real z < 0. It solves gamma z m^2 + (z - 1 + gamma) m + 1 = 0."""
    a, b, c = gamma * z, z - 1 + gamma, 1.0
    disc = math.sqrt(b * b - 4 * a * c)
    roots = [(-b + disc) / (2 * a), (-b - disc) / (2 * a)]
    pos = [r for r in roots if r > 0]
    return min(pos)


def mp_stieltjes_prime(z, gamma):
    """dm/dz, from implicit differentiation of gamma z m^2 + (z - 1 + gamma) m + 1 = 0."""
    m = mp_stieltjes(z, gamma)
    return -(gamma * m * m + m) / (2 * gamma * z * m + z - 1 + gamma)


def ridge_risk(alpha, lam, r2=1.0, sigma2=0.0):
    """Limiting excess risk of ridge (per-sample normalized penalty lam > 0), isotropic design,
    load alpha = n/d: R = r^2 lam^2 m'(-lam) + sigma^2 gamma (m(-lam) - lam m'(-lam)), gamma = 1/alpha
    (Hastie--Montanari--Rosset--Tibshirani, Cor. 5; Dobriban--Wager)."""
    gamma = 1.0 / alpha
    m = mp_stieltjes(-lam, gamma)
    mp = mp_stieltjes_prime(-lam, gamma)
    return r2 * lam ** 2 * mp + sigma2 * gamma * (m - lam * mp)


def ridgeless_risk(alpha, r2=1.0, sigma2=0.0):
    """Limit lam -> 0+: sigma^2/(alpha - 1) for alpha > 1; r^2 (1 - alpha) + sigma^2 alpha/(1 - alpha)
    for alpha < 1."""
    if alpha > 1:
        return sigma2 / (alpha - 1)
    return r2 * (1 - alpha) + sigma2 * alpha / (1 - alpha)


def ridge_simulate(n, d, lam, r2, sigma2, gen):
    """One draw: excess risk |beta_hat - beta*|^2 of ridge with beta* uniform on the sphere of
    radius r (x ~ N(0, I_d), so the excess risk is the squared distance)."""
    beta = gen.standard_normal(d)
    beta *= math.sqrt(r2) / np.linalg.norm(beta)
    X = gen.standard_normal((n, d))
    y = X @ beta + math.sqrt(sigma2) * gen.standard_normal(n)
    if lam > 0:
        bh = np.linalg.solve(X.T @ X / n + lam * np.eye(d), X.T @ y / n)
    else:
        bh = np.linalg.pinv(X) @ y
    return float(np.sum((bh - beta) ** 2))


# ---------------------------------------------------------------------------------------------
# Statistics and output
# ---------------------------------------------------------------------------------------------


def mean_ci(x, level=0.95):
    """Mean and Student-t interval half-width of a sample."""
    x = np.asarray(x, dtype=float)
    n = x.size
    m = float(x.mean())
    if n < 2:
        return m, float("nan")
    from scipy import stats
    half = float(stats.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / math.sqrt(n))
    return m, half


def clopper_pearson(k, n, level=0.95):
    """Exact binomial interval for k successes in n trials."""
    from scipy import stats
    a = 1 - level
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def write_dat(path, header, rows):
    """Write whitespace-separated columns with a header row (for pgfplots `table`)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(" ".join(header) + "\n")
        for row in rows:
            fh.write(" ".join(f"{v:.10g}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + "\n")


# ---------------------------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------------------------


def _self_test():
    ok = True

    def check(name, got, want, tol):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print(f"{'ok ' if good else 'BAD'} {name}: {got:.10g} (want {want:.10g})")

    check("E Z^4", gauss(lambda z: z ** 4), 3.0, 1e-10)
    check("E relu(Z)^2 (split)", gauss_split(lambda z: np.maximum(z, 0) ** 2), 0.5, 1e-10)
    check("H(0)", float(H(0.0)), 0.5, 1e-15)
    check("Gardner capacity at kappa=0", gardner_capacity(0.0), 2.0, 1e-12)
    check("Gardner capacity by quadrature, kappa=1",
          1.0 / gauss_split(lambda z: np.maximum(z + 1.0, 0) ** 2, a=-1.0), gardner_capacity(1.0), 1e-9)
    check("Cover probability at n = 2d", cover_probability(200, 100), 0.5, 1e-12)
    g = 2.0
    check("mmse Gaussian vs quadrature at gamma=2",
          gauss2(lambda x, z: (x - math.sqrt(g) * (math.sqrt(g) * x + z) / (1 + g)) ** 2), mmse_gaussian(g), 1e-12)
    check("mmse Rademacher at gamma -> 0", mmse_rademacher(1e-6), 1 - 1e-6, 1e-9)
    check("sparse Rademacher with rho = 1 equals Rademacher", mmse_sparse_rademacher(1.3, 1.0),
          mmse_rademacher(1.3), 1e-9)
    q, _ = rs_overlap(0.9)
    check("RS overlap below BBP (lam = 0.9)", q, 0.0, 1e-8)
    q, _ = rs_overlap(1.5)
    m_se = se_z2(1.5, 0.5, 200)[-1]
    check("RS overlap = SE limit at lam = 1.5", m_se, q, 1e-8)
    check("SK RS at beta=0.5, h=0: q", sk_rs_q(0.5, 0.0, q0=0.3), 0.0, 1e-8)
    check("SK RS free entropy at beta=0.5, h=0", sk_rs_free_entropy(0.5, 0.0, q=0.0), math.log(2) + 0.0625, 1e-12)
    check("SK AT at beta=1, h=0 (q = 0)", sk_at(1.0, 0.0, q=0.0), 1.0, 1e-12)
    check("REM at beta_c", rem_free_entropy(math.sqrt(2 * math.log(2))), 2 * math.log(2), 1e-12)
    # Marchenko--Pastur: m(z) against the empirical spectrum
    gen = rng(1)
    X = gen.standard_normal((2000, 1000))
    ev = np.linalg.eigvalsh(X.T @ X / 2000)
    check("MP Stieltjes at z=-1 (gamma=1/2) vs empirical", float(np.mean(1 / (ev + 1))), mp_stieltjes(-1.0, 0.5), 2e-3)
    check("ridge risk -> ridgeless (alpha=2, sigma2=1)", ridge_risk(2.0, 1e-7, 1.0, 1.0), ridgeless_risk(2.0, 1.0, 1.0), 1e-5)
    check("ridge risk -> ridgeless (alpha=0.5, sigma2=0.25)", ridge_risk(0.5, 1e-7, 1.0, 0.25), ridgeless_risk(0.5, 1.0, 0.25), 1e-4)
    sims = [ridge_simulate(800, 400, 0.3, 1.0, 0.25, rng(10 + s)) for s in range(10)]
    m, half = mean_ci(sims)
    check("ridge risk vs simulation (n=800, d=400, lam=0.3)", m, ridge_risk(2.0, 0.3, 1.0, 0.25), 3 * half + 0.01)
    # AMP against state evolution, Z2 synchronization
    gen = rng(7)
    N, lam = 3000, 1.5
    x = gen.choice([-1.0, 1.0], size=N)
    A = spiked_wigner(x, lam, gen)
    xh0 = 0.3 * x + 0.0 * gen.standard_normal(N)   # informative start with overlap 0.3, norm^2 0.09
    est = amp_z2(A, lam, xh0, 12)
    m_amp = float(np.mean(est[-1] * x))
    q, _ = rs_overlap(lam)
    check("AMP overlap after 12 steps vs RS fixed point (N=3000)", m_amp, q, 0.03)
    print("rosettakit self-test", "passed" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    _self_test()
