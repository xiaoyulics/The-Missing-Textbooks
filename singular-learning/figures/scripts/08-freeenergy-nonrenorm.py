"""Chapter 8 (The Free Energy Formula): where the formula breaks -- Watanabe's non-renormalizable example.

Watanabe (J. Phys. Conf. Ser. 233, 012014, 2010, Section 5): the truth is q = N(0, I_2) and the model is
p(x, y | a) = N((a, sqrt(a^4 - a^2 + 1)), I_2), here with a uniform on [-1, 1]. The model cannot realize q; the
optimal parameter is a = 0, and the energy relative to it is exactly
    n K_n(a) = n a^4 / 2 - sqrt(n) a xi1 - sqrt(n) h(a) xi2,    h(a) = sqrt(a^4 - a^2 + 1) - 1,
with xi1 = sum x_i / sqrt(n) and xi2 = sum y_i / sqrt(n) independent N(0, 1) for every n. The random part
sqrt(n) a xi1 is not bounded by the square root sqrt(n/2) a^2 of the deterministic part, so the sandwich of the
chapter fails. Minimizing n a^4/2 - sqrt(n) a xi1 over the real line gives m2 = -(3/2^(7/3)) |xi1|^(4/3) n^(1/3),
whose mean is -3c n^(1/3) with c = Gamma(7/6) / (2^(5/3) sqrt(pi)), so -3c = -0.4946. Watanabe studies the
learning curves of this example, not its free energy; the scaling of the free energy is the chapter's own
argument (F_n - n L_n(w0) >= min nK_n because the prior is a probability density, and a window of width n^(-1/3)
around the minimizer bounds it by the minimum plus O_p(log n)). A Laplace approximation at the minimizer gives the
heuristic mean -3c n^(1/3) + (1/3) log n + kappa', kappa' = log(3/pi)/2 + (E log|Z| - log 2)/3 + log 2 - 1/12,
printed for comparison only.

For 400 draws of (xi1, xi2) at each n = 10^2, ..., 10^8 the script computes F_n - n L_n(w0) =
-log int exp(-n K_n(a)) da/2 by the trapezoid rule on 100001 points (for |xi1| <= 5 the posterior standard
deviation is at least 5e-4 at n = 1e8, 25 grid steps) and reports its mean divided by n^(1/3) in two ways, with
Student-t 95% intervals: the plain mean, and the control-variate estimate mean(F - m2)/n^(1/3) - 3c, which has
the same expectation and a far smaller variance because m2 is known in closed form for each draw.
Output: figures/data/08-freeenergy-nonrenorm.dat.  Run from the volume root (about a minute).
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy import special  # noqa: E402

import sltkit as sk  # noqa: E402

REPS = 400
EXPONENTS = [2, 3, 4, 5, 6, 7, 8]
GRID = np.linspace(-1.0, 1.0, 100001)
H = np.sqrt(GRID ** 4 - GRID ** 2 + 1.0) - 1.0
LOGW = np.full(GRID.shape, math.log(0.5 * (GRID[1] - GRID[0])))
LOGW[0] += math.log(0.5); LOGW[-1] += math.log(0.5)


def main() -> None:
    t0 = time.time()
    c = math.gamma(7 / 6) / (2 ** (5 / 3) * math.sqrt(math.pi))
    c_min = -3 * c
    elog = -(np.euler_gamma + math.log(2)) / 2                      # E log|Z|, Z ~ N(0, 1)
    kappa = 0.5 * math.log(3 / math.pi) + (elog - math.log(2)) / 3 + math.log(2) - 1 / 12
    print(f"c = Gamma(7/6)/(2^(5/3) sqrt(pi)) = {c:.4f}; limit of E[F]/n^(1/3): -3c = {c_min:.4f}; "
          f"Laplace constant kappa' = {kappa:.3f}")
    rows = []
    for e in EXPONENTS:
        n = 10.0 ** e
        gen = sk.rng(8200 + e)
        scale = n ** (1 / 3)
        F, Emin, M2 = [], [], []
        for _ in range(REPS):
            x1, x2 = gen.standard_normal(2)
            E = 0.5 * n * GRID ** 4 - math.sqrt(n) * GRID * x1 - math.sqrt(n) * H * x2
            F.append(-special.logsumexp(-E + LOGW))
            Emin.append(E.min())
            M2.append(-3 * 2 ** (-7 / 3) * abs(x1) ** (4 / 3) * scale)
        F, Emin, M2 = np.array(F), np.array(Emin), np.array(M2)
        m, lo, hi = sk.mean_ci(F / scale)
        mc, loc, hic = sk.mean_ci((F - M2) / scale)
        mc, loc, hic = mc + c_min, loc + c_min, hic + c_min
        pred = c_min + (math.log(n) / 3 + kappa) / scale
        rows.append((int(n), m, m - lo, hi - m, mc, mc - loc, hic - mc, float(np.mean(Emin)) / scale, c_min, pred,
                     float(np.min(F - Emin))))
        print(f"n=1e{e}: E[F_n - nL_n(w0)] = {np.mean(F):9.3f};  / n^(1/3): plain {m:.3f} [{lo:.3f},{hi:.3f}], "
              f"control variate {mc:.4f} [{loc:.4f},{hic:.4f}]; heuristic {pred:.4f};  "
              f"E[min energy] / n^(1/3) = {np.mean(Emin) / scale:.3f};  min over data sets of F - min energy = "
              f"{np.min(F - Emin):.2f} (must be >= 0)")
    sk.write_dat(os.path.join("figures", "data", "08-freeenergy-nonrenorm.dat"),
                 ["n", "F", "Flo", "Fhi", "Fcv", "Fcvlo", "Fcvhi", "Emin", "Eminlimit", "pred", "gapmin"], rows)
    print(f"runtime {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
