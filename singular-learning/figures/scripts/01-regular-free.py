"""Chapter 1 (regular models): the free energy of three regular models, measured by quadrature.

Models, truths and priors (x ~ N(0, I), eps ~ N(0, 1)):
  regular line   y = a x + eps                     truth a0 = 0        prior uniform on [-1, 1]           d = 1
  regular plane  y = a x1 + b x2 + eps             truth (0, 0)        prior uniform on [-1, 1]^2         d = 2
  mixture        (1 - a) N(0, 1) + a N(b, 1)       truth (1/2, 2)      prior uniform on [0, 1] x [-4, 4]  d = 2
The mixture is the running model of Chapter 10, here at a truth where it is regular; the range of b is
wider than the running [-1, 1] so that the posterior stays away from the edge of the box.

For each sample size n and each of R data sets the script computes
  * F_n by quadrature. Linear models: sltkit.free_energy_grid on a box of +-12 posterior standard deviations
    around the least-squares estimate w_hat, clipped to [-1, 1]^d, with the prior density 2^-d of the whole
    box. Mixture: the midpoint rule on a 20 x 20 grid over +-9 standard deviations in coordinates whitened by
    the observed information (checked below against a 48 x 48 grid and a 400 x 1600 grid on the whole box);
  * nS_n = n L_n(w0), the energy of the truth;
  * the Laplace approximation of the chapter's theorem with the observed information J_n = Hessian of L_n
    at w_hat,   nL_n(w_hat) + (d/2) log(n / 2 pi) + (1/2) log det J_n - log phi(w_hat),
    and the Laplace error F_n minus that approximation. For the linear models the error is exactly
    -log Pi_n, Pi_n the Gaussian N(w_hat, S_xx^-1) mass of the box (Computation "the uniform prior"); the
    script prints the largest value of the bound sum_j 2 Phibar((1 - |w_hat_j|)/sd_j) for 1 - Pi_n.
Predictions for the mean of F_n - nS_n:
  * the BIC corollary (realizable, I = J):  (d/2) log(n / (2 pi e)) + (1/2) log det J - log phi(w0);
  * linear models, exactly (the Wishart expectation of log det S_xx):
        -d/2 - (d/2) log(2 pi) + d log 2 + (1/2) [sum_{i<=d} psi((n - i + 1)/2) + d log 2]   + E[-log Pi_n].
Prediction for the mean Laplace error of the mixture: -c/n, with c the second-order coefficient of
Laplace's method (exercise "Laplace's error term") at the population derivatives of L at w0.

Writes figures/data/01-regular-free.dat (columns: n; for line, plane and mixture the mean of F_n - nS_n
with the half-widths of its 95% interval; the exact means; the BIC-corollary predictions) and
figures/data/01-regular-laplace.dat (n; mean Laplace error of the mixture with its interval; n times it;
the prediction -c/n and -c).
Run from the volume root:  python figures/scripts/01-regular-free.py      (one to two minutes)
"""
import itertools
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402
from scipy import integrate, special, stats  # noqa: E402

import sltkit as sk  # noqa: E402

R_LIN = 200
R_MIX = 100
EXP_LIN = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
EXP_MIX = [2.0, 2.5, 3.0, 3.5, 4.0]
A0, B0 = 0.5, 2.0
MIX_BOX = np.array([[0.0, 1.0], [-4.0, 4.0]])
MIX_LOGPRIOR = -math.log(8.0)
OUT = os.path.join("figures", "data")


# ---------------------------------------------------------------------------------------------
# Linear models
# ---------------------------------------------------------------------------------------------
def linear_free_energy(model, d):
    """F_n - nS_n by quadrature; F_n minus its Laplace approximation; a bound for 1 - Pi_n."""
    what = model.least_squares()
    sd = np.sqrt(np.diag(np.linalg.inv(model.Sxx)))
    lo, hi = np.maximum(-1.0, what - 12 * sd), np.minimum(1.0, what + 12 * sd)
    box = list(zip(lo, hi))
    sizes = 4000 if d == 1 else 200
    F_sub = sk.free_energy_grid(model.energy, box, sizes)          # prior uniform on the sub-box
    F = F_sub + d * math.log(2.0) - float(np.sum(np.log(hi - lo)))   # prior density 2^-d instead
    nS = float(model.energy_of_coefficients(np.zeros(d)))
    laplace = model.min_energy() - d / 2 * math.log(2 * math.pi) \
        + 0.5 * np.linalg.slogdet(model.Sxx)[1] + d * math.log(2.0)
    outside = float(np.sum(2 * stats.norm.sf((1 - np.abs(what)) / sd)))
    return F - nS, F - laplace, outside


def linear_exact_mean(n, d):
    """E[F_n - nS_n] for the Gaussian linear model, x ~ N(0, I_d), uniform prior on [-1,1]^d, without the
    term E[-log Pi_n] (at most 2e-8 here)."""
    elogdet = sum(special.digamma((n - i + 1) / 2) for i in range(1, d + 1)) + d * math.log(2)
    return -d / 2 - d / 2 * math.log(2 * math.pi) + d * math.log(2) + 0.5 * elogdet


def linear_corollary(n, d):
    return d / 2 * math.log(n / (2 * math.pi * math.e)) + d * math.log(2)


# ---------------------------------------------------------------------------------------------
# The mixture at the regular truth (1/2, 2)
# ---------------------------------------------------------------------------------------------
_x, _a, _b = sp.symbols("x a b", real=True)
_g = sp.log(1 - _a + _a * sp.exp(_b * _x - _b ** 2 / 2))     # log p(x | a, b) - log phi(x)
DER = {}
for order in range(1, 5):
    for idx in itertools.combinations_with_replacement(range(2), order):
        e = _g
        for i in idx:
            e = sp.diff(e, (_a, _b)[i])
        DER[idx] = sp.lambdify((_x, _a, _b), e, "numpy")


def mix_sample(n, seed):
    gen = sk.rng(seed)
    comp = gen.random(n) < A0
    return gen.standard_normal(n) + B0 * comp


def mix_energy(W, xs):
    return sk.mixture_energy(np.atleast_2d(W), xs)


def mix_grad_hess(w, xs):
    a, b = w
    gr = -np.array([np.sum(DER[(0,)](xs, a, b)), np.sum(DER[(1,)](xs, a, b))])
    hab = np.sum(DER[(0, 1)](xs, a, b))
    H = -np.array([[np.sum(DER[(0, 0)](xs, a, b)), hab], [hab, np.sum(DER[(1, 1)](xs, a, b))]])
    return gr, H


def inside(W):
    return (W[:, 0] > MIX_BOX[0, 0]) & (W[:, 0] < MIX_BOX[0, 1]) & (W[:, 1] > MIX_BOX[1, 0]) & (W[:, 1] < MIX_BOX[1, 1])


def mix_mle(xs):
    """Damped Newton iterations from the truth; returns w_hat, the Hessian of n L_n at w_hat, and the largest
    gradient component there."""
    w = np.array([A0, B0])
    E0 = mix_energy(w, xs)[0]
    for _ in range(100):
        gr, H = mix_grad_hess(w, xs)
        step = np.linalg.solve(H, gr)
        lam = 1.0
        while lam > 1e-12:
            wn = w - lam * step
            if inside(wn[None, :])[0]:
                En = mix_energy(wn, xs)[0]
                if En <= E0 + 1e-10:
                    break
            lam /= 2
        w, E0 = wn, En
        if np.max(np.abs(lam * step)) < 1e-11:
            break
    gr, H = mix_grad_hess(w, xs)
    return w, H, float(np.max(np.abs(gr)))


def mix_posterior_nodes(xs, w_hat, H, N=20, Rad=9.0):
    """Midpoint grid in whitened coordinates u (w = w_hat + L^{-T} u, L L^T = H): nodes w_k and weights
    exp(-(E(w_k) - E(w_hat))) h^2 / (2 pi), whose sum is the ratio of the integral to its Gaussian
    approximation (nodes outside the box get weight 0)."""
    h = 2 * Rad / N
    t = -Rad + h * (np.arange(N) + 0.5)
    U = np.stack(np.meshgrid(t, t, indexing="ij"), -1).reshape(-1, 2)
    L = np.linalg.cholesky(H)
    W = w_hat + np.linalg.solve(L.T, U.T).T
    ok = inside(W)
    E = np.full(len(W), np.inf)
    E[ok] = mix_energy(W[ok], xs)
    E0 = mix_energy(w_hat, xs)[0]
    return W, np.exp(-(E - E0)) * h * h / (2 * math.pi), E0


def mix_free_energy(xs, N=20):
    """F_n - nS_n, F_n minus its Laplace approximation, and the largest gradient component at w_hat."""
    w_hat, H, grmax = mix_mle(xs)
    W, wts, E0 = mix_posterior_nodes(xs, w_hat, H, N=N)
    ratio = float(np.sum(wts))
    laplace = E0 + 0.5 * np.linalg.slogdet(H)[1] - math.log(2 * math.pi) - MIX_LOGPRIOR   # d = 2
    F = laplace - math.log(ratio)
    nS = float(mix_energy([A0, B0], xs)[0])
    return F - nS, F - laplace, grmax


def q_density(t):
    return (1 - A0) * np.exp(-0.5 * t * t) / math.sqrt(2 * math.pi) + A0 * np.exp(-0.5 * (t - B0) ** 2) / math.sqrt(2 * math.pi)


def mix_population_tensors():
    """J = Hessian of L at w0, and the second-order Laplace coefficient
    c = -(1/8) L_ijkl H^ij H^kl + (1/8) L_ijk L_lmn H^ij H^kl H^mn + (1/12) L_ijk L_lmn H^il H^jm H^kn."""
    def pop(idx):
        f = DER[tuple(sorted(idx))]
        return -integrate.quad(lambda t: float(f(t, A0, B0)) * float(q_density(t)), -14, 16, limit=400, epsabs=1e-13)[0]
    J = np.array([[pop((0, 0)), pop((0, 1))], [pop((0, 1)), pop((1, 1))]])
    T3 = np.zeros((2, 2, 2))
    T4 = np.zeros((2, 2, 2, 2))
    for idx in itertools.product(range(2), repeat=3):
        T3[idx] = pop(idx)
    for idx in itertools.product(range(2), repeat=4):
        T4[idx] = pop(idx)
    Hi = np.linalg.inv(J)
    c = (-np.einsum("ijkl,ij,kl->", T4, Hi, Hi) / 8
         + np.einsum("ijk,lmn,ij,kl,mn->", T3, T3, Hi, Hi, Hi) / 8
         + np.einsum("ijk,lmn,il,jm,kn->", T3, T3, Hi, Hi, Hi) / 12)
    return J, c


def main():
    t_start = time.time()
    os.makedirs(OUT, exist_ok=True)

    # ---- the regular line and the regular plane
    rows_lin = {}
    max_err, max_out = 0.0, 0.0
    for d, model in ((1, sk.REGULAR1), (2, sk.REGULAR2)):
        for ex in EXP_LIN:
            n = int(round(10 ** ex))
            vals = []
            for r in range(R_LIN):
                x, y = model.sample([0.0] * d, n, seed=1_000_000 * d + 1000 * int(ex * 2) + r)
                model.fit(x, y)
                v, err, out = linear_free_energy(model, d)
                vals.append(v)
                max_err, max_out = max(max_err, abs(err)), max(max_out, out)
            rows_lin[(d, n)] = sk.mean_ci(vals)
    print(f"linear models: largest |F_n - Laplace| over all data sets = {max_err:.2e}; largest bound for the "
          f"Gaussian mass outside the box (so for -log Pi_n, up to a factor near 1) = {max_out:.1e}")

    # ---- the mixture: first check the quadrature (a finer whitened grid; a grid over the whole box)
    xs = mix_sample(100, seed=17)
    v20, _, _ = mix_free_energy(xs, N=20)
    v48, _, _ = mix_free_energy(xs, N=48)
    na, nb = 400, 1600
    ta = (np.arange(na) + 0.5) / na
    tb = -4.0 + 8.0 * (np.arange(nb) + 0.5) / nb
    Wg = np.stack(np.meshgrid(ta, tb, indexing="ij"), -1).reshape(-1, 2)
    F_box = -special.logsumexp(-mix_energy(Wg, xs)) - math.log((1.0 / na) * (8.0 / nb)) - MIX_LOGPRIOR
    nS = float(mix_energy([A0, B0], xs)[0])
    print(f"quadrature check, mixture, n = 100: F_n - nS_n = {v20:.9f} (20 x 20 whitened grid), {v48:.9f} (48 x 48), "
          f"{F_box - nS:.9f} (400 x 1600 grid on the whole box)")
    xs = mix_sample(10000, seed=18)
    _, e20, _ = mix_free_energy(xs, N=20)
    _, e48, _ = mix_free_energy(xs, N=48)
    print(f"quadrature check, mixture, n = 10^4: F_n - Laplace = {e20:.6e} (20 x 20), {e48:.6e} (48 x 48)")

    J, c_inf = mix_population_tensors()
    print(f"mixture at (1/2, 2): J = {np.round(J, 4).tolist()}, det J = {np.linalg.det(J):.4f}, "
          f"eigenvalues {np.round(np.linalg.eigvalsh(J), 4).tolist()}; second-order Laplace coefficient: "
          f"n E[F_n - Laplace] -> -c = {-c_inf:.4f}")
    rows_mix = {}
    worst_grad = 0.0
    for ex in EXP_MIX:
        n = int(round(10 ** ex))
        vals, errs = [], []
        for r in range(R_MIX):
            xs = mix_sample(n, seed=3_000_000 + 1000 * int(ex * 2) + r)
            v, e, grmax = mix_free_energy(xs)
            worst_grad = max(worst_grad, grmax)
            vals.append(v)
            errs.append(e)
        rows_mix[n] = (sk.mean_ci(vals), sk.mean_ci(errs))
    print(f"mixture: largest |grad n L_n| at the computed MLE = {worst_grad:.1e}")

    # ---- write and print
    mix_const = 0.5 * math.log(np.linalg.det(J)) - MIX_LOGPRIOR
    rows = []
    for ex in EXP_LIN:
        n = int(round(10 ** ex))
        m1, l1, h1 = rows_lin[(1, n)]
        m2, l2, h2 = rows_lin[(2, n)]
        mix = rows_mix.get(n)
        if mix is not None:
            mm, lm, hm = mix[0]
            mix_cols = (mm, mm - lm, hm - mm, math.log(n / (2 * math.pi * math.e)) + mix_const)
        else:
            mix_cols = ("nan", "nan", "nan", "nan")
        rows.append((n, m1, m1 - l1, h1 - m1, linear_exact_mean(n, 1), linear_corollary(n, 1),
                     m2, m2 - l2, h2 - m2, linear_exact_mean(n, 2), linear_corollary(n, 2), *mix_cols))
        print(f"n=1e{ex:.1f}: line {m1:7.3f} [{l1:7.3f},{h1:7.3f}] exact {linear_exact_mean(n, 1):7.3f} "
              f"pred {linear_corollary(n, 1):7.3f} | plane {m2:7.3f} [{l2:7.3f},{h2:7.3f}] exact "
              f"{linear_exact_mean(n, 2):7.3f} pred {linear_corollary(n, 2):7.3f}"
              + (f" | mixture {mix_cols[0]:7.3f} [{mix[0][1]:7.3f},{mix[0][2]:7.3f}] pred {mix_cols[3]:7.3f}"
                 if mix else ""))
    sk.write_dat(os.path.join(OUT, "01-regular-free.dat"),
                 ["n", "line", "linelo", "linehi", "lineexact", "linepred",
                  "plane", "planelo", "planehi", "planeexact", "planepred",
                  "mix", "mixlo", "mixhi", "mixpred"], rows)
    rows = []
    for ex in EXP_MIX:
        n = int(round(10 ** ex))
        me, le, he = rows_mix[n][1]
        rows.append((n, me, me - le, he - me, n * me, -c_inf / n, -c_inf))
        print(f"mixture n=1e{ex:.1f}: E[F_n - Laplace] = {me:.3e} [{le:.3e}, {he:.3e}];  n times it = {n * me:.4f} "
              f"[{n * le:.4f}, {n * he:.4f}]  (prediction {-c_inf:.4f})")
    sk.write_dat(os.path.join(OUT, "01-regular-laplace.dat"), ["n", "err", "errlo", "errhi", "nerr", "pred", "npred"], rows)
    ns = [10 ** e for e in EXP_LIN]
    sl1 = np.polyfit(np.log(ns), [rows_lin[(1, int(round(n)))][0] for n in ns], 1)[0]
    sl2 = np.polyfit(np.log(ns), [rows_lin[(2, int(round(n)))][0] for n in ns], 1)[0]
    nm = [10 ** e for e in EXP_MIX]
    slm = np.polyfit(np.log(nm), [rows_mix[int(round(n))][0][0] for n in nm], 1)[0]
    print(f"least-squares slopes of the means against log n: line {sl1:.3f}, plane {sl2:.3f}, mixture {slm:.3f}")
    print(f"total time {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
