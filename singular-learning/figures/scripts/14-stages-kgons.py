"""Chapter 14 (Stages): the k-gon critical points of the toy model of superposition, checked.

The toy model of superposition (TMS) of Elhage et al. (2022) in the setting of Chen et al.
(2023): f(x, w) = relu(W^T W x + b) with W in R^{2x6}, b in R^6 (d = 18 parameters); the input
x = mu e_i has a uniform coordinate i in {1..6} and a uniform length mu in [0, 1] (the high-
sparsity limit).  We write ell(w) = E||x - f(x, w)||^2 (Chen et al.'s "L") and H = 3c ell = 18 ell
(their "TMS potential").  The book's population loss is the negative log-likelihood of
p(x | w) ~ exp(-||x - f||^2 / 2), so L = ell / 2 + const and K = L - L(w*) = (ell - ell(w*)) / 2.

This script checks every number of the chapter's "By hand" section:
  1. the closed form of ell against Monte Carlo;
  2. the regular 5-gon and 6-gon: the two equations of the symmetric reduction, their solution
     (l*, b*), the loss, and that the full 18-dimensional gradient vanishes there;
  3. the Hessian of H at the 6-gon, the 5-gon (sixth column vestigial, bias -1/2) and the
     5+-gon (sixth bias 1/12): eigenvalues, number of zero eigenvalues, rank, and
     lambda = rank / 2 by the minimally singular rule;
  4. the Fourier blocks of the 6-gon's Hessian (the symmetry makes it block circulant);
  5. a search for the lowest critical value by gradient descent from 400 random starts;
  6. the crossing of the local free energies of the 5-gon and the 6-gon (Chen et al., App. C)
     under two conventions, and the size of the omitted Laplace constants.
Writes figures/data/14-stages-kgons-spectrum.dat (the three spectra).
Run from the volume root: python figures/scripts/14-stages-kgons.py   (about 20 s)
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import time

import numpy as np
from scipy import optimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sltkit as sk  # noqa: E402

C6 = 6
DATA = os.path.join("figures", "data", "14-stages-kgons-spectrum.dat")


# ---------------------------------------------------------------------------------------------
# The closed-form loss.  For output o and input j put a = G[o, j] = W_o . W_j and beta = b_o.
# Interference (o != j): 3 int_0^1 relu(a mu + beta)^2 dmu; diagonal (o == j):
# 3 int_0^1 (mu - relu(a mu + beta))^2 dmu.  H is the sum over (o, j), ell = H / (3c).
# ---------------------------------------------------------------------------------------------
def terms(W, b):
    """H, dH/dG, dH/db for parameter batches W (S,2,c), b (S,c)."""
    c = W.shape[2]
    G = W.transpose(0, 2, 1) @ W
    beta = np.broadcast_to(b[:, :, None], G.shape)
    a = G
    eye = np.eye(c, dtype=bool)[None]
    with np.errstate(divide="ignore", invalid="ignore"):
        neg = beta <= 0
        actA = neg & (a > 0) & (beta >= -a)                       # upper part active, beta <= 0
        Ta = np.where(actA, (a + beta) ** 3 / a, 0.0)
        Ta_a = np.where(actA, 3 * (a + beta) ** 2 / a - (a + beta) ** 3 / a ** 2, 0.0)
        Ta_b = np.where(actA, 3 * (a + beta) ** 2 / a, 0.0)
        pos = ~neg
        part = pos & (a < -beta)                                  # lower part active, beta > 0
        full = pos & ~part                                        # always active
        Tp = np.where(part, -beta ** 3 / a, 0.0) + np.where(full, a * a + 3 * a * beta + 3 * beta ** 2, 0.0)
        Tp_a = np.where(part, beta ** 3 / a ** 2, 0.0) + np.where(full, 2 * a + 3 * beta, 0.0)
        Tp_b = np.where(part, -3 * beta ** 2 / a, 0.0) + np.where(full, 3 * a + 6 * beta, 0.0)
        actD = neg & (a > 0) & (beta >= -a)
        q = (1 - a) ** 2 - 3 * (1 - a) * beta + 3 * beta ** 2
        q_a = -2 * (1 - a) + 3 * beta
        q_b = -3 * (1 - a) + 6 * beta
        D = np.where(actD, q + beta ** 3 * (1 / a ** 2 + 1 / a), np.where(neg, 1.0, q))
        D_a = np.where(actD, q_a - beta ** 3 * (2 / a ** 3 + 1 / a ** 2), np.where(neg, 0.0, q_a))
        D_b = np.where(actD, q_b + 3 * beta ** 2 * (1 / a ** 2 + 1 / a), np.where(neg, 0.0, q_b))
    H = np.where(eye, D, Ta + Tp).sum((1, 2))
    dHa = np.where(eye, D_a, Ta_a + Tp_a)
    dHb = np.where(eye, D_b, Ta_b + Tp_b).sum(2)
    return H, dHa, dHb


def ell(W, b):
    return terms(W, b)[0] / (3 * W.shape[2])


def grad_H(W, b):
    """Gradient of H with respect to (W, b) as a flat vector per batch element."""
    _, dG, db = terms(W, b)
    dW = W @ (dG + dG.transpose(0, 2, 1))
    return np.concatenate([dW.transpose(0, 2, 1).reshape(len(W), -1), db], 1)   # (x0,y0,x1,y1,..,b)


def unflat(v, c):
    W = v[: 2 * c].reshape(c, 2).T
    return W, v[2 * c:]


def flat(W, b):
    return np.concatenate([W.T.reshape(-1), b])


def kgon(k, c, l, beta, vest=-0.5, pos=()):
    W = np.zeros((2, c))
    b = np.full(c, float(vest))
    for i in range(k):
        th = 2 * np.pi * i / k
        W[:, i] = [l * np.cos(th), l * np.sin(th)]
        b[i] = beta
    for i in pos:
        b[i] = 1.0 / (2 * c)
    return W, b


def mc_ell(W, b, m, seed):
    g = sk.rng(seed)
    c = W.shape[1]
    i = g.integers(0, c, m)
    mu = g.random(m)
    G = W.T @ W
    f = np.maximum(mu[:, None] * G[i, :] + b[None, :], 0.0)
    x = np.zeros((m, c))
    x[np.arange(m), i] = mu
    r = ((x - f) ** 2).sum(1)
    return r.mean(), 2 * r.std() / np.sqrt(m)


# ---------------------------------------------------------------------------------------------
# The symmetric reduction: for the regular k-gon (c = k) with length l (a = l^2) and bias beta,
# only neighbours interfere and H / k = phi(a, beta) = 2 T(a cos(2pi/k), beta) + D(a, beta).
# ---------------------------------------------------------------------------------------------
def phi_and_grad(a, beta, k):
    Ck = np.cos(2 * np.pi / k)
    A = a * Ck
    T = (A + beta) ** 3 / A
    T_A = 3 * (A + beta) ** 2 / A - (A + beta) ** 3 / A ** 2
    T_b = 3 * (A + beta) ** 2 / A
    D = (1 - a) ** 2 - 3 * (1 - a) * beta + 3 * beta ** 2 + beta ** 3 * (1 / a ** 2 + 1 / a)
    D_a = -2 * (1 - a) + 3 * beta - beta ** 3 * (2 / a ** 3 + 1 / a ** 2)
    D_b = -3 * (1 - a) + 6 * beta + 3 * beta ** 2 * (1 / a ** 2 + 1 / a)
    return 2 * T + D, np.array([2 * Ck * T_A + D_a, 2 * T_b + D_b])


def solve_kgon(k, guess):
    sol = optimize.fsolve(lambda z: phi_and_grad(z[0], z[1], k)[1], guess, xtol=1e-12)
    return sol[0], sol[1]


def hessian_H(W, b, h=1e-6):
    v0 = flat(W, b)
    c = W.shape[1]
    d = len(v0)
    Hm = np.zeros((d, d))
    for p in range(d):
        vp = v0.copy(); vp[p] += h
        vm = v0.copy(); vm[p] -= h
        Wp, bp = unflat(vp, c)
        Wm, bm = unflat(vm, c)
        Hm[p] = (grad_H(Wp[None], bp[None])[0] - grad_H(Wm[None], bm[None])[0]) / (2 * h)
    return 0.5 * (Hm + Hm.T)


def fourier_blocks(Hm, k, c):
    """Local frame (radial, tangential, bias) per vertex; returns C0, C1, the deviation from
    block-circulant form, and the eigenvalues of C0 + w^m C1 + w^-m C1^T for m = 0..k-1."""
    d = 3 * c
    Q = np.zeros((d, d))                       # columns: local coordinates -> Euclidean
    for i in range(c):
        th = 2 * np.pi * i / k if i < k else 0.0
        er = np.array([np.cos(th), np.sin(th)])
        es = np.array([-np.sin(th), np.cos(th)])
        Q[2 * i:2 * i + 2, 3 * i] = er
        Q[2 * i:2 * i + 2, 3 * i + 1] = es
        Q[2 * c + i, 3 * i + 2] = 1.0
    Hl = Q.T @ Hm @ Q
    blk = lambda i, j: Hl[3 * i:3 * i + 3, 3 * j:3 * j + 3]
    C0, C1 = blk(0, 0), blk(0, 1)
    dev = max(np.abs(blk(i, j) - blk(0, (j - i) % k)).max() for i in range(k) for j in range(k))
    eig = []
    for m in range(k):
        w = np.exp(2j * np.pi * m / k)
        Ch = C0 + w * C1 + np.conj(w) * C1.T
        eig.append(np.sort(np.linalg.eigvalsh(Ch)))
    return C0, C1, dev, eig


def main():
    t_start = time.perf_counter()
    np.set_printoptions(precision=5, suppress=True, linewidth=120)

    # 2. the symmetric reduction --------------------------------------------------------------
    a6, b6 = solve_kgon(6, (1.74, -0.62))
    a5, b5 = solve_kgon(5, (1.37, -0.28))
    l6, l5 = np.sqrt(a6), np.sqrt(a5)
    phi6 = phi_and_grad(a6, b6, 6)[0]
    phi5 = phi_and_grad(a5, b5, 5)[0]
    print("== the regular k-gons (c = k): solutions of d phi/da = d phi/db = 0")
    print(f"  6-gon: a* = {a6:.6f}  l* = {l6:.6f}  b* = {b6:.6f}  phi = {phi6:.6f}  ell(c=6) = phi/3 = {phi6 / 3:.6f}")
    print(f"  5-gon: a* = {a5:.6f}  l* = {l5:.6f}  b* = {b5:.6f}  phi = {phi5:.6f}  ell(c=5) = phi/3 = {phi5 / 3:.6f}")
    for k, (aa, bb) in ((6, (a6, b6)), (5, (a5, b5))):
        res = np.abs(phi_and_grad(aa, bb, k)[1]).max()
        hh = 1e-6
        J = np.array([(phi_and_grad(aa + hh, bb, k)[1] - phi_and_grad(aa - hh, bb, k)[1]) / (2 * hh),
                      (phi_and_grad(aa, bb + hh, k)[1] - phi_and_grad(aa, bb - hh, k)[1]) / (2 * hh)])
        print(f"  {k}-gon: residual {res:.1e}; Hessian of phi in (a, b) has eigenvalues {np.round(np.linalg.eigvalsh(0.5 * (J + J.T)), 4)}")
    print(f"  chamber checks (6-gon): a cos60 + b = {a6 / 2 + b6:.5f} > 0,  a + b = {a6 + b6:.5f} > 0,  b < 0")
    print(f"  chamber checks (5-gon): a cos72 + b = {a5 * np.cos(2 * np.pi / 5) + b5:.5f} > 0,  a + b = {a5 + b5:.5f} > 0,  b < 0")
    pts = {
        "6-gon": kgon(6, C6, l6, b6),
        "5-gon": kgon(5, C6, l5, b5, vest=-0.5),
        "5+-gon": kgon(5, C6, l5, b5, pos=(5,)),
    }
    lam_true = {"6-gon": 8.5, "5-gon": 7.0, "5+-gon": 8.5}
    print("== the three critical points in c = 6 (d = 18)")
    for name, (W, b) in pts.items():
        e = ell(W[None], b[None])[0]
        g = grad_H(W[None], b[None])[0]
        print(f"  {name:7s} ell = {e:.6f}  (H = {18 * e:.6f})  max |grad H| = {np.abs(g).max():.1e}")
    print(f"  5+-gon: the sixth output contributes 18 b^2 - 3 b + 1 = {18 / 144 - 3 / 12 + 1:.4f} to H at b = 1/12,"
          f" against 1 for the 5-gon: ell drops by {0.125 / 18:.6f}")

    # 1. closed form against Monte Carlo ---------------------------------------------------------
    print("== closed form against Monte Carlo (10^6 draws, +- 2 standard errors)")
    for j, (name, (W, b)) in enumerate(pts.items()):
        m, se = mc_ell(W, b, 1_000_000, 1401 + j)
        print(f"  {name:7s} closed {ell(W[None], b[None])[0]:.5f}   Monte Carlo {m:.5f} +- {se:.5f}")
    g = sk.rng(1404)
    for j in range(3):
        W = g.normal(size=(2, C6)); b = 0.5 * g.normal(size=C6)
        m, se = mc_ell(W, b, 1_000_000, 1410 + j)
        print(f"  random  closed {ell(W[None], b[None])[0]:.5f}   Monte Carlo {m:.5f} +- {se:.5f}")

    # 3. Hessians ------------------------------------------------------------------------------
    print("== Hessian of H = 18 ell (the Hessian of K = (ell - ell*)/2 is this divided by 36)")
    spectra = {}
    for name, (W, b) in pts.items():
        Hm = hessian_H(W, b)
        ev = np.sort(np.linalg.eigvalsh(Hm))
        tol = 1e-6 * ev.max()
        nz = int((np.abs(ev) < tol).sum())
        spectra[name] = ev
        print(f"  {name:7s} eigenvalues: {np.round(ev, 4)}")
        print(f"          zero eigenvalues: {nz} (largest in absolute value {np.abs(ev[:nz]).max():.1e}),"
              f" smallest nonzero {ev[nz]:.4f}, rank {18 - nz}, (d - s)/2 = {(18 - nz) / 2}  (claimed lambda {lam_true[name]})")
        if name == "6-gon":
            Hm6 = Hm
    # 4. Fourier blocks at the 6-gon -------------------------------------------------------------
    C0, C1, dev, eig = fourier_blocks(Hm6, 6, 6)
    print("== Fourier blocks of the 6-gon (local coordinates: radial, tangential, bias)")
    print("  C0 =\n", C0)
    print("  C1 =\n", C1)
    print(f"  deviation from block-circulant form: {dev:.1e}")
    M0 = C0 + C1 + C1.T
    rb = M0[np.ix_([0, 2], [0, 2])]
    print(f"  mode 0: tangential entry C0_tt + 2 C1_tt = {M0[1, 1]:.2e}; radial-bias block {np.round(rb, 4).tolist()},"
          f" trace {np.trace(rb):.4f}, determinant {np.linalg.det(rb):.4f}")
    for m in range(4):
        print(f"  mode {m}: eigenvalues {np.round(eig[m], 4)}")
    allf = np.sort(np.concatenate(eig))
    print(f"  max difference to the full spectrum: {np.abs(allf - spectra['6-gon']).max():.1e}")

    # 5. global search ---------------------------------------------------------------------------
    t0 = time.perf_counter()
    g = sk.rng(1420)
    S = 300

    def fun(v):
        W, b = unflat(v, C6)
        return terms(W[None], b[None])[0][0], grad_H(W[None], b[None])[0]

    fin = []
    for s in range(S):
        v0 = np.concatenate([g.uniform(0.2, 1.2) * g.normal(size=2 * C6), 0.3 * g.normal(size=C6)])
        r = optimize.minimize(fun, v0, jac=True, method="L-BFGS-B", options={"maxiter": 3000, "gtol": 1e-10})
        fin.append(r.fun / 18)
    fin = np.array(fin)
    vals, counts = np.unique(np.round(fin, 4), return_counts=True)
    print(f"== L-BFGS on H from {S} random starts ({time.perf_counter() - t0:.1f}s)")
    print("  final ell (rounded) and counts:", ", ".join(f"{v:.4f} x{c}" for v, c in zip(vals, counts)))
    print(f"  lowest value found: {fin.min():.6f}")

    # 6. free-energy crossings ------------------------------------------------------------------
    print("== the 5 -> 6 crossing of n dl + dlam log n + dc = 0 (Chen et al., App. C)")
    dl = 0.04819 - 0.06874
    dlam = 1.5
    w6 = flat(*pts["6-gon"]); w5 = flat(*kgon(5, C6, l5, b5, vest=0.0))
    dc = 0.5 * (w6 @ w6) - 0.5 * (w5 @ w5)
    print(f"  0.5|w6|^2 = {0.5 * w6 @ w6:.5f}, 0.5|w5|^2 (vestigial column and bias 0) = {0.5 * w5 @ w5:.5f}, dc = {dc:.4f}")
    root = lambda f: optimize.brentq(f, 10, 1e6)
    print(f"  energy n ell:     n_cr = {root(lambda n: n * dl + dlam * np.log(n) + 2.7535):.1f}")
    print(f"  energy n ell / 2: n_cr = {root(lambda n: n * dl / 2 + dlam * np.log(n) + 2.7535):.1f}")
    print(f"  energy n ell, dc = 0: n_cr = {root(lambda n: n * dl + dlam * np.log(n)):.1f}")
    print(f"  energy n ell / 2, dc = 0: n_cr = {root(lambda n: n * dl / 2 + dlam * np.log(n)):.1f}")
    # Laplace constants of a minimally singular phase: -log int_U e^{-n K} phi
    #   = (d - s)/2 log(n / 2 pi) - log A + o(1),  A = int_S phi det(H_perp)^(-1/2) dsigma.
    # Prior N(0, I_18).  6-gon: 120 circles of length 2 pi sqrt(6) l*.  5-gon: 144 copies of a
    # circle of length 2 pi sqrt(5) l5* times the region F of the vestigial (W_v, b_v).
    g = sk.rng(1430)
    m = 2_000_000
    Wv = g.normal(size=(m, 2)); bv = g.normal(size=m)
    W5 = kgon(5, 5, l5, b5)[0]
    inner = Wv @ W5                                              # (m,5)
    inF = (bv < 0) & ((Wv ** 2).sum(1) <= -bv) & (inner.max(1) <= np.minimum(-bv, -b5))
    PhiF = inF.mean()
    print(f"  prior mass of the vestigial region F: {PhiF:.5f} (+- {2 * np.sqrt(PhiF * (1 - PhiF) / m):.5f})")
    for conv, scale in (("energy n ell / 2 (K = (ell - ell*)/2)", 36.0), ("energy n ell", 18.0)):
        ev6 = spectra["6-gon"][1:] / scale
        ev5 = spectra["5-gon"][4:] / scale
        logA6 = (np.log(120) + np.log(2 * np.pi * np.sqrt(6) * l6) - 9 * np.log(2 * np.pi)
                 - 0.5 * (w6 @ w6) - 0.5 * np.log(ev6).sum())
        logA5 = (np.log(144) + np.log(2 * np.pi * np.sqrt(5) * l5) - 7.5 * np.log(2 * np.pi)
                 - 0.5 * (w5 @ w5) + np.log(PhiF) - 0.5 * np.log(ev5).sum())
        c6 = -8.5 * np.log(2 * np.pi) - logA6
        c5 = -7.0 * np.log(2 * np.pi) - logA5
        dfull = c6 - c5
        k = 0.5 if scale == 36.0 else 1.0
        ngrid = np.logspace(1, 5, 4001)
        dF = ngrid * dl * k + dlam * np.log(ngrid) + dfull
        print(f"  {conv}: half log det' (6) = {0.5 * np.log(ev6).sum():.3f}, (5) = {0.5 * np.log(ev5).sum():.3f};"
              f" full dc = {dfull:.3f}; max over n in [10, 1e5] of F6 - F5 = {dF.max():.3f} at n = {ngrid[dF.argmax()]:.0f};"
              f" n with F6 > F5: {ngrid[dF > 0].min() if (dF > 0).any() else 'none'}"
              f" to {ngrid[dF > 0].max() if (dF > 0).any() else 'none'}")

    # the Gaussian-localized estimator of a quadratic K at t = n / log n, n = 1000:
    # each normal eigenvalue h of the Hessian of K contributes (1/2) t h / (t h + gamma)
    t = 1000 / np.log(1000)
    print(f"== quadratic approximation of the localized estimator at t = n/log n = {t:.1f}")
    for name in spectra:
        h = spectra[name] / 36.0
        h = h[h > 1e-6 * h.max()]
        for gam in (1.0, 10.0):
            print(f"  {name:7s} gamma = {gam:4.1f}: sum (1/2) t h / (t h + gamma) = {0.5 * np.sum(t * h / (t * h + gam)):.2f}"
                  f"  (smallest t h = {t * h.min():.2f})")
    # the regular 7-gon (an exercise)
    a7, b7 = solve_kgon(7, (2.1, -0.97))
    print(f"== 7-gon (c = 7): l* = {np.sqrt(a7):.5f}, b* = {b7:.5f}, ell = phi/3 = {phi_and_grad(a7, b7, 7)[0] / 3:.5f},"
          f" a cos(2pi/7) + b = {a7 * np.cos(2 * np.pi / 7) + b7:.4f}, a cos(4pi/7) = {a7 * np.cos(4 * np.pi / 7):.4f}")

    # data file: the three spectra (of the Hessian of K = (ell - ell*)/2), sorted
    rows = [[i + 1, spectra["6-gon"][i] / 36, spectra["5-gon"][i] / 36, spectra["5+-gon"][i] / 36]
            for i in range(18)]
    sk.write_dat(DATA, ["index", "hex", "pent", "pentplus"], rows)
    print(f"wrote {DATA}; total time {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
