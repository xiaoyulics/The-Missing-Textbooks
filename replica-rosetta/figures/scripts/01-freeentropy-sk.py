"""Chapter 1 (freeentropy): self-averaging, quenched versus annealed, and replica moments in the SK model
by exact enumeration.  Run from the volume root:  python figures/scripts/01-freeentropy-sk.py   (~1 minute)

Model (book normalization, h = 0):  H_N(s) = (beta/sqrt N) sum_{i<j} g_ij s_i s_j,  Z_N = sum_s exp(H_N(s)).
For each N in {4,...,16} and beta in {0.5, 0.8, 1.5} we draw M disorder samples (seed 100 + N) and compute
log Z_N exactly by enumerating the 2^N configurations, and the squared gradient
|grad_g log Z_N|^2 = (beta^2/N) sum_{i<j} <s_i s_j>^2, whose mean bounds Var log Z_N (Gaussian Poincare).

Printed and written:
  * the standard deviation of (1/N) log Z_N with 95% bootstrap intervals, against the Gaussian
    concentration bound beta sqrt((N-1)/2)/N (Theorem thm:freeentropy:concentration) and, for beta < 1,
    against sigma(beta)/N with sigma^2 = -log(1-beta^2)/2 - beta^2/2 (Aizenman-Lebowitz-Ruelle);
  * the quenched free entropy phi_N = E log Z_N / N against the annealed (1/N) log E Z_N
    = log 2 + beta^2 (N-1)/(4N) (exact);
  * at N = 12: the replica free entropy (1/(nN)) log E Z_N^n for real n in (0, 3] by Monte Carlo over
    the disorder (95% bootstrap intervals), against the exact values at n = 1, 2, 3 and the band
    [phi_N, phi_N + n beta^2 (N-1)/(4N)] of Proposition prop:freeentropy:smalln.
Data: figures/data/01-freeentropy-sk-sd.dat, -sk-moments.dat, -sk-exact.dat.
"""
import math
import os
import sys
import time

import numpy as np
from scipy.special import gammaln, logsumexp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DATA = os.path.join(ROOT, "figures", "data")
BETAS = [0.5, 0.8, 1.5]
SIZES = {4: 4000, 6: 4000, 8: 4000, 10: 3000, 12: 3000, 14: 800, 16: 300}
B = 1000  # bootstrap resamples for standard deviations
BM = 400  # bootstrap resamples for the replica moments


def enumerate_sk(N, M, seed):
    """log Z_N and |grad log Z_N|^2 for M disorder samples, for each beta in BETAS."""
    gen = rk.rng(seed)
    S = rk.all_spins(N).astype(np.float64)
    iu = np.triu_indices(N, 1)
    L = np.zeros((M, len(BETAS)))
    G2 = np.zeros((M, len(BETAS)))
    for t in range(M):
        J = np.zeros((N, N))
        J[iu] = gen.standard_normal(len(iu[0]))
        E0 = np.einsum("ki,ij,kj->k", S, J, S, optimize=True) / math.sqrt(N)
        for j, b in enumerate(BETAS):
            lw = b * E0
            lz = logsumexp(lw)
            L[t, j] = lz
            w = np.exp(lw - lz)
            C = (S * w[:, None]).T @ S          # Gibbs correlations <s_i s_j>
            G2[t, j] = b * b / N * float(np.sum(np.triu(C, 1) ** 2))
    return L, G2


def boot_sd(x, gen):
    idx = gen.integers(0, x.size, size=(B, x.size))
    sds = x[idx].std(axis=1, ddof=1)
    return float(x.std(ddof=1)), float(np.quantile(sds, 0.025)), float(np.quantile(sds, 0.975))


def exact_moment(N, beta, n):
    """log E Z_N^n exactly for n = 1, 2, 3 (h = 0), from
    E Z^n = exp(n N b^2/4 - n^2 b^2/4) sum_{s^1..s^n} exp((N b^2/2) sum_{a<b} R_ab^2)."""
    pre = n * N * beta ** 2 / 4 - n * n * beta ** 2 / 4
    if n == 1:
        return pre + N * math.log(2)
    if n == 2:   # fix s^1 = 1 (gauge), R_12 = (2k - N)/N
        k = np.arange(N + 1)
        R = (2 * k - N) / N
        terms = gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1) + N * beta ** 2 / 2 * R ** 2
        return pre + N * math.log(2) + float(logsumexp(terms))
    if n == 3:   # fix s^1 = 1; types of (s^2_i, s^3_i): ++, +-, -+, --
        terms = []
        for n1 in range(N + 1):
            for n2 in range(N + 1 - n1):
                for n3 in range(N + 1 - n1 - n2):
                    n4 = N - n1 - n2 - n3
                    r12, r13, r23 = (n1 + n2 - n3 - n4) / N, (n1 - n2 + n3 - n4) / N, (n1 - n2 - n3 + n4) / N
                    lm = gammaln(N + 1) - gammaln(n1 + 1) - gammaln(n2 + 1) - gammaln(n3 + 1) - gammaln(n4 + 1)
                    terms.append(lm + N * beta ** 2 / 2 * (r12 ** 2 + r13 ** 2 + r23 ** 2))
        return pre + N * math.log(2) + float(logsumexp(terms))
    raise ValueError(n)


def main():
    t0 = time.time()
    gboot = rk.rng(7)
    sd_rows, results = [], {}
    print("N   beta   phi_N (95% CI)        annealed   sd(logZ/N) [95% boot]      bound    Var logZ  E|grad|^2  L^2")
    for N, M in SIZES.items():
        L, G2 = enumerate_sk(N, M, 100 + N)
        results[N] = (L, G2)
        print(f"-- N = {N}: M = {M} disorder samples (seed {100 + N}), {time.time() - t0:.0f} s elapsed")
        row = [N]
        for j, b in enumerate(BETAS):
            x = L[:, j] / N
            m, half = rk.mean_ci(x)
            sd, lo, hi = boot_sd(x, gboot)
            ann = math.log(2) + b * b * (N - 1) / (4 * N)
            bound = b * math.sqrt((N - 1) / 2) / N
            s2 = -0.5 * math.log(1 - b * b) - b * b / 2 if b < 1 else float("nan")
            alr = math.sqrt(s2) / N if b < 1 else float("nan")
            print(f"{N:2d}  {b:3.1f}  {m:.5f} +- {half:.5f}   {ann:.5f}   {sd:.5f} [{lo:.5f}, {hi:.5f}]   {bound:.4f}"
                  f"   {L[:, j].var(ddof=1):.4f}   {G2[:, j].mean():.4f}   {b * b * (N - 1) / 2:.3f}")
            row += [sd, sd - lo, hi - sd, bound, alr, m, half, ann, math.sqrt(G2[:, j].mean()) / N]
        sd_rows.append(row)
    head = ["N"]
    for b in BETAS:
        t = f"{int(round(10 * b)):02d}"
        head += [f"sd{t}", f"lo{t}", f"hi{t}", f"bd{t}", f"alr{t}", f"phi{t}", f"ci{t}", f"ann{t}", f"gr{t}"]
    rk.write_dat(os.path.join(DATA, "01-freeentropy-sk-sd.dat"), head, sd_rows)

    # replica free entropy at N = 12
    N = 12
    L, _ = results[N]
    ngrid = np.array([0.05, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0])
    idx = gboot.integers(0, L.shape[0], size=(BM, L.shape[0]))
    mom_rows = [[n] for n in ngrid]
    print(f"\nreplica free entropy (1/(nN)) log E Z^n at N = {N} (M = {L.shape[0]}):")
    for j, b in enumerate(BETAS):
        lz = L[:, j]
        phiN = lz.mean() / N
        for r, n in enumerate(ngrid):
            est = (logsumexp(n * lz) - math.log(lz.size)) / (n * N)
            bs = (logsumexp(n * lz[idx], axis=1) - math.log(lz.size)) / (n * N)
            lo, hi = np.quantile(bs, [0.025, 0.975])
            up = phiN + n * b * b * (N - 1) / (4 * N)
            mom_rows[r] += [est, est - lo, hi - est, phiN, up]
        for n in [0.05, 0.5, 1.0, 2.0, 3.0]:
            r = int(np.argmin(np.abs(ngrid - n)))
            print(f"  beta={b}: n={n:4.2f}  MC {mom_rows[r][1 + 5 * j]:.5f}  [-{mom_rows[r][2 + 5 * j]:.5f}, "
                  f"+{mom_rows[r][3 + 5 * j]:.5f}]   band [{phiN:.5f}, {mom_rows[r][5 + 5 * j]:.5f}]")
    head = ["n"]
    for b in BETAS:
        t = f"{int(round(10 * b)):02d}"
        head += [f"mc{t}", f"lo{t}", f"hi{t}", f"q{t}", f"up{t}"]
    rk.write_dat(os.path.join(DATA, "01-freeentropy-sk-moments.dat"), head, mom_rows)
    ex_rows = []
    for n in [1, 2, 3]:
        vals = [exact_moment(N, b, n) / (n * N) for b in BETAS]
        ex_rows.append([n] + vals)
        print(f"  exact n={n}: " + "  ".join(f"beta={b}: {v:.5f}" for b, v in zip(BETAS, vals)))
    rk.write_dat(os.path.join(DATA, "01-freeentropy-sk-exact.dat"), ["n"] + [f"ex{int(round(10 * b)):02d}" for b in BETAS], ex_rows)

    # checks quoted in the text
    print("\nchecks:")
    print("  annealed limits log 2 + beta^2/4:", ", ".join(f"{math.log(2) + b * b / 4:.5f}" for b in BETAS))
    for b in BETAS[:2]:
        print(f"  beta={b}: sigma^2(beta) = {-0.5 * math.log(1 - b * b) - b * b / 2:.5f};"
              f"  log second-moment ratio limit = {-b * b / 2 - 0.5 * math.log(1 - b * b):.5f}")
    L16, G16 = results[16]
    for j, b in enumerate(BETAS):
        dev = np.abs(L16[:, j] - L16[:, j].mean()) / 16
        u = 0.1
        print(f"  N=16 beta={b}: largest |logZ/N - mean| = {dev.max():.4f};  fraction > {u}: {np.mean(dev > u):.4f};"
              f"  tail bound 2exp(-N^2u^2/(beta^2(N-1))) = {2 * math.exp(-256 * u * u / (b * b * 15)):.3g}")
    # Carleman: E Z_N^n >= 2^N exp(n^2 beta^2 (N-1)/4) at N = 12, beta = 1.5
    for n in [1, 2, 3]:
        lb = N * math.log(2) + n * n * 1.5 ** 2 * (N - 1) / 4
        print(f"  N=12 beta=1.5 n={n}: log E Z^n exact {exact_moment(12, 1.5, n):.4f} >= diagonal lower bound {lb:.4f};"
              f"  per (nN): {lb / (n * N):.4f}")
    # the limit at beta = 1.5: Guerra's replica-symmetric bound and the one-step (1RSB) Parisi functional
    # P(q, m) = log 2 + (1/m) log E cosh^m(beta sqrt(q) Z) + (beta^2/4)((1-q)^2 - m q^2), an upper bound for all (q, m)
    b = 1.5
    qrs = rk.sk_rs_q(b, 0.0, q0=0.5)
    frs = rk.sk_rs_free_entropy(b, 0.0, q=qrs)

    def p1rsb(par):
        q, m = par
        if not (0.0 < q < 1.0 and 0.0 < m < 1.0):
            return 10.0
        s = b * math.sqrt(q)
        v = rk.gauss(lambda z: np.exp(m * (np.abs(s * z) + np.log1p(np.exp(-2 * np.abs(s * z))) - math.log(2))), 161)
        return math.log(2) + math.log(v) / m + b * b / 4 * ((1 - q) ** 2 - m * q * q)

    from scipy import optimize
    res = optimize.minimize(p1rsb, [0.4, 0.2], method="Nelder-Mead",
                            options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 4000})
    print(f"  beta=1.5: annealed {math.log(2) + b * b / 4:.6f};  RS bound {frs:.6f} (q = {qrs:.5f});"
          f"  1RSB {res.fun:.6f} (q = {res.x[0]:.5f}, m = {res.x[1]:.5f});  RS - 1RSB = {frs - res.fun:.2e}")
    print(f"  N=16 beta=1.5: gap of phi_N to the 1RSB value = {res.fun - results[16][0][:, 2].mean() / 16:.4f}")
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
