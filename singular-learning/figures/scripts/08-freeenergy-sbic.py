"""Chapter 8 (The Free Energy Formula): choosing the rank of a reduced-rank regression by BIC and by sBIC.

The setting of Drton and Plummer (2017, Section 5.1), rerun with our own seeds: M = 10 outputs, N = 15 inputs
(x ~ N(0, I_15)), y = C0 x + N(0, I_10) noise with a coefficient matrix C0 of rank 5 whose nonzero singular values
are 1.2, 1.0, 0.8, 0.6, 0.4 and whose singular vectors are drawn from the Haar measure afresh for every data set.
Model M_i (i = 0, ..., 10) postulates rank(C) <= i, parametrized as C = BA with B of size 10 x i and A of size
i x 15; the noise covariance I_10 is known.

  * maximized log-likelihood of M_i: (1/2) sum_{j <= i} e_j + const, where e_1 >= e_2 >= ... are the eigenvalues
    of Y'X (X'X)^{-1} X'Y (reduced-rank least squares, Eckart-Young);
  * BIC(M_i) = loglik_i - (dim_i / 2) log n, dim_i = i (M + N - i), the dimension of the matrices of rank <= i;
  * sBIC(M_i) = log L'(M_i), the unique positive solution of sum_{j <= i} [L'(M_i) - L'_ij] L'(M_j) = 0, with
    L'_ij = exp(loglik_i) (log n)^(m_ij - 1) / n^(lambda_ij), lambda_ij and m_ij from the Aoyagi-Watanabe formula
    for H = i hidden units and true rank j (Drton and Plummer 2017, Definition 3.1 and Proposition 3.1; uniform
    prior on models). Solved in log space by the recursion L'(M_i) = (-b_i + sqrt(b_i^2 + 4 c_i)) / 2,
    b_i = sum_{j < i} L'(M_j) - L'_ii, c_i = sum_{j < i} L'_ij L'(M_j); checked against mpmath at 60 digits.
200 data sets for each n in {50, 100, 200, 300, 500, 1000, 2000, 5000}; reported: the frequency with which each
criterion selects the true rank 5, with Clopper-Pearson 95% intervals.

Output: figures/data/08-freeenergy-sbic.dat.
Run from the volume root:  python figures/scripts/08-freeenergy-sbic.py   (about 15 s)
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import sltkit as sk  # noqa: E402

M_OUT, N_IN, RANK = 10, 15, 5
SING = np.array([1.2, 1.0, 0.8, 0.6, 0.4])
NS = [50, 100, 200, 300, 500, 1000, 2000, 5000]
REPS = 200
IMAX = min(M_OUT, N_IN)


def aoyagi_watanabe(m_out: int, n_in: int, H: int, r: int) -> tuple[float, int]:
    """Learning coefficient and multiplicity of y = BAx (x in R^n_in, y in R^m_out, H hidden units) at a truth
    of rank r <= H (Aoyagi and Watanabe 2005, in the restatement of Chapter 7). Watanabe (2013) writes M for the
    input and N for the output dimension, as below; the formula is symmetric in M and N, so the order is immaterial."""
    M, N = n_in, m_out
    if N + r <= M + H and M + r <= N + H and H + r <= M + N:
        base = 2 * (H + r) * (M + N) - (M - N) ** 2 - (H + r) ** 2
        if (M + H + N + r) % 2 == 0:
            return base / 8.0, 1
        return (base + 1) / 8.0, 2
    if M + H < N + r:
        return (H * M - H * r + N * r) / 2.0, 1
    if N + H < M + r:
        return (H * N - H * r + M * r) / 2.0, 1
    return M * N / 2.0, 1


def check_table() -> None:
    """Drton and Plummer (2017), Table 1: 5 responses, 3 covariates; m = 2 only for (i, j) = (3, 0)."""
    table = {(0, 0): 0, (1, 0): 1.5, (1, 1): 3.5, (2, 0): 3, (2, 1): 4.5, (2, 2): 6,
             (3, 0): 4.5, (3, 1): 5.5, (3, 2): 6.5, (3, 3): 7.5}
    for (i, j), lam in table.items():
        l, m = aoyagi_watanabe(5, 3, i, j)
        assert abs(l - lam) < 1e-12 and m == (2 if (i, j) == (3, 0) else 1), (i, j, l, m)
    # the regular value on the diagonal: lambda_ii = dim(M_i)/2
    for i in range(IMAX + 1):
        assert abs(aoyagi_watanabe(M_OUT, N_IN, i, i)[0] - i * (M_OUT + N_IN - i) / 2) < 1e-12
    print("Aoyagi-Watanabe formula reproduces Drton-Plummer Table 1 and the diagonal dim/2")


def logaddexp(a: float, b: float) -> float:
    return float(np.logaddexp(a, b))


def log_sum(vals) -> float:
    vals = [v for v in vals if v > -math.inf]
    if not vals:
        return -math.inf
    mx = max(vals)
    return mx + math.log(sum(math.exp(v - mx) for v in vals))


def log_sub(a: float, b: float) -> float:
    """log(e^a - e^b) for a > b."""
    return a + math.log1p(-math.exp(b - a))


def sbic(loglik: np.ndarray, n: int, lam: np.ndarray, mult: np.ndarray) -> np.ndarray:
    """log L'(M_i) for a total order of models 0 < 1 < ... (Drton-Plummer eq. 3.15), in log space."""
    I = len(loglik)
    lp = np.empty((I, I))
    for i in range(I):
        for j in range(i + 1):
            lp[i, j] = loglik[i] - lam[i, j] * math.log(n) + (mult[i, j] - 1) * math.log(math.log(n))
    out = np.empty(I)
    out[0] = lp[0, 0]
    for i in range(1, I):
        lP = log_sum(out[:i])                       # sum_{j<i} L'(M_j)
        lQ = lp[i, i]                               # L'_ii
        lc = log_sum([lp[i, j] + out[j] for j in range(i)])
        if lP >= lQ:                                # b >= 0: L = 2c / (b + sqrt(b^2 + 4c))
            lb = log_sub(lP, lQ) if lP > lQ else -math.inf
            lsq = 0.5 * logaddexp(2 * lb, math.log(4) + lc) if lb > -math.inf else 0.5 * (math.log(4) + lc)
            out[i] = math.log(2) + lc - (logaddexp(lb, lsq) if lb > -math.inf else lsq)
        else:                                       # b < 0: L = (-b + sqrt(b^2 + 4c)) / 2
            lnb = log_sub(lQ, lP)
            lsq = 0.5 * logaddexp(2 * lnb, math.log(4) + lc)
            out[i] = logaddexp(lnb, lsq) - math.log(2)
    return out


def sbic_mpmath(loglik, n, lam, mult) -> list:
    import mpmath as mp
    mp.mp.dps = 60
    I = len(loglik)
    Lp = [[mp.e ** (mp.mpf(loglik[i]) - lam[i, j] * mp.log(n) + (mult[i, j] - 1) * mp.log(mp.log(n)))
           for j in range(i + 1)] for i in range(I)]
    L = [Lp[0][0]]
    for i in range(1, I):
        b = -Lp[i][i] + sum(L[:i])
        c = sum(Lp[i][j] * L[j] for j in range(i))
        L.append((-b + mp.sqrt(b * b + 4 * c)) / 2)
    return [float(mp.log(v)) for v in L]


def haar_columns(rows: int, cols: int, gen: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(gen.standard_normal((rows, cols)))
    return q * np.sign(np.diag(r))


def main() -> None:
    t0 = time.time()
    check_table()
    lam = np.zeros((IMAX + 1, IMAX + 1)); mult = np.ones((IMAX + 1, IMAX + 1), dtype=int)
    for i in range(IMAX + 1):
        for j in range(i + 1):
            lam[i, j], mult[i, j] = aoyagi_watanabe(M_OUT, N_IN, i, j)
    dims = np.array([i * (M_OUT + N_IN - i) for i in range(IMAX + 1)])
    print("lambda_i5 (truth of rank 5) for i = 5..10:", [float(lam[i, 5]) for i in range(5, IMAX + 1)],
          "m:", [int(mult[i, 5]) for i in range(5, IMAX + 1)], " dim_i/2:", [float(dims[i] / 2) for i in range(5, IMAX + 1)])
    print(f"at a truth of rank 4: lambda_44 = {lam[4, 4]:.1f}, lambda_54 = {lam[5, 4]:.1f}; "
          f"BIC charges (dim_5 - dim_4)/2 = {(dims[5] - dims[4]) / 2:.0f} log n = {(dims[5] - dims[4]) / 2 * math.log(300):.1f} at n = 300, "
          f"the marginal likelihood {lam[5, 4] - lam[4, 4]:.0f} log n = {(lam[5, 4] - lam[4, 4]) * math.log(300):.1f}")
    rows, checked = [], False
    for n in NS:
        gen = sk.rng(8_100_000 + n)
        pick_b, pick_s = [], []
        for r in range(REPS):
            C0 = haar_columns(M_OUT, RANK, gen) @ np.diag(SING) @ haar_columns(N_IN, RANK, gen).T
            X = gen.standard_normal((n, N_IN))
            Y = X @ C0.T + gen.standard_normal((n, M_OUT))
            XtY = X.T @ Y
            E = XtY.T @ np.linalg.solve(X.T @ X, XtY)
            e = np.sort(np.linalg.eigvalsh(0.5 * (E + E.T)))[::-1]
            loglik = 0.5 * np.concatenate([[0.0], np.cumsum(e[:IMAX])])
            bic = loglik - dims / 2 * math.log(n)
            sb = sbic(loglik, n, lam, mult)
            if not checked:
                ref = sbic_mpmath(loglik, n, lam, mult)
                print("max |sBIC(log space) - sBIC(mpmath)| =", max(abs(a - b) for a, b in zip(sb, ref)))
                checked = True
            pick_b.append(int(np.argmax(bic))); pick_s.append(int(np.argmax(sb)))
        pick_b, pick_s = np.array(pick_b), np.array(pick_s)
        kb, ks = int(np.sum(pick_b == RANK)), int(np.sum(pick_s == RANK))
        lb, hb = sk.clopper_pearson(kb, REPS); ls, hs = sk.clopper_pearson(ks, REPS)
        rows.append((n, kb / REPS, kb / REPS - lb, hb - kb / REPS, ks / REPS, ks / REPS - ls, hs - ks / REPS))
        hist_b = np.bincount(pick_b, minlength=IMAX + 1); hist_s = np.bincount(pick_s, minlength=IMAX + 1)
        print(f"n={n:5d}: true rank chosen by BIC {kb:3d}/{REPS} [{lb:.2f},{hb:.2f}], by sBIC {ks:3d}/{REPS} [{ls:.2f},{hs:.2f}]"
              f" | BIC ranks {hist_b[:9].tolist()} | sBIC ranks {hist_s[:9].tolist()}")
    sk.write_dat(os.path.join("figures", "data", "08-freeenergy-sbic.dat"),
                 ["n", "bic", "biclo", "bichi", "sbic", "sbiclo", "sbichi"], rows)
    print(f"runtime {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
