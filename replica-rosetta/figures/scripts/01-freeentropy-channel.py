"""Chapter 1 (freeentropy): the scalar Gaussian channel Y = sqrt(gamma) X + Z.  Free entropy, mutual
information, MMSE, the Nishimori identity and the I-MMSE relation, exactly (Gauss-Hermite quadrature) and by
Monte Carlo.  Run from the volume root:  python figures/scripts/01-freeentropy-channel.py   (~10 s)

Priors: Rademacher (X = +-1) and standard Gaussian.  Posterior mean <x>: tanh(sqrt(gamma) Y) and
sqrt(gamma) Y/(1 + gamma).  Log-evidence (free entropy of one sample): log Z(Y) = log cosh(sqrt(gamma) Y) - gamma/2
and -(1/2) log(1 + gamma) + gamma Y^2/(2(1 + gamma)).
For each gamma in {0.25, 0.5, 1, 2, 4} and each prior, M = 100000 independent draws (seed 1000 g + 10 p) give:
  * the MMSE as the mean squared error of the posterior mean (95% Student-t interval);
  * the MMSE through I-MMSE: 2 (E X^2/2 - psi'(gamma)), with psi' from a centered finite difference of the
    log-evidence with common random numbers (95% interval);
  * the Nishimori identity E[X <x>] = E[<x>^2], for the Bayes posterior and for a mismatched posterior that
    uses the SNR 2 gamma.
Data: figures/data/01-freeentropy-channel.dat (exact curves), -channel-mc.dat (Monte Carlo).
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DATA = os.path.join(ROOT, "figures", "data")
M = 100000


def rademacher(g, X, Z):
    Y = math.sqrt(g) * X + Z
    return np.logaddexp(math.sqrt(g) * Y, -math.sqrt(g) * Y) - math.log(2) - g / 2, np.tanh(math.sqrt(g) * Y)


def gaussian(g, X, Z):
    Y = math.sqrt(g) * X + Z
    return -0.5 * math.log(1 + g) + g * Y ** 2 / (2 * (1 + g)), math.sqrt(g) * Y / (1 + g)


def exact(prior, g):
    """(free entropy psi, mutual information I, mmse) by quadrature (Rademacher) or in closed form."""
    if prior == "rad":
        lc = rk.gauss(lambda z: np.logaddexp(g + math.sqrt(g) * z, -g - math.sqrt(g) * z) - math.log(2), 161)
        return lc - g / 2, g - lc, rk.mmse_rademacher(g)
    return g / 2 - 0.5 * math.log(1 + g), 0.5 * math.log(1 + g), 1 / (1 + g)


def main():
    # exact curves
    rows = []
    for g in np.round(np.linspace(0.0, 5.0, 101), 3):
        if g == 0:
            rows.append([0.0, 0.0, 0.0, 1.0, 1.0])
            continue
        _, Ir, mr = exact("rad", g)
        _, Ig, mg = exact("gau", g)
        rows.append([g, Ir, Ig, mr, mg])
    rk.write_dat(os.path.join(DATA, "01-freeentropy-channel.dat"), ["gamma", "Irad", "Igau", "mmserad", "mmsegau"], rows)
    # quadrature checks of the identities
    for g in (0.5, 2.0):
        I = lambda gg: exact("rad", gg)[1]
        h = 1e-5
        a = rk.gauss(lambda z: np.tanh(g + math.sqrt(g) * z), 161)
        b = rk.gauss(lambda z: np.tanh(g + math.sqrt(g) * z) ** 2, 161)
        print(f"quadrature, Rademacher, gamma={g}: dI/dgamma = {(I(g + h) - I(g - h)) / (2 * h):.10f},"
              f" mmse/2 = {exact('rad', g)[2] / 2:.10f};  E tanh = {a:.10f}, E tanh^2 = {b:.10f}")
    # Monte Carlo
    mc_rows, tex_rows, inside, maxhalf = [], [], [], 0.0
    print("\n gamma prior  mmse exact   MC mmse (95%)        via I-MMSE (95%)      E X<x>     E <x>^2    "
          "mismatched E X<x>', E<x>'^2, mse'")
    for g in (0.25, 0.5, 1.0, 2.0, 4.0):
        row = [g]
        for p, (name, f) in enumerate((("rad", rademacher), ("gau", gaussian))):
            gen = rk.rng(int(1000 * g) + 10 * p)
            X = gen.choice([-1.0, 1.0], size=M) if name == "rad" else gen.standard_normal(M)
            Z = gen.standard_normal(M)
            lz, pm = f(g, X, Z)
            d = 1e-3 * g
            dpsi = (f(g + d, X, Z)[0] - f(g - d, X, Z)[0]) / (2 * d)
            m1, h1 = rk.mean_ci((X - pm) ** 2)
            m2, h2 = rk.mean_ci(dpsi)
            immse, himmse = 2 * (0.5 - m2), 2 * h2          # E X^2 = 1 for both priors
            a, ha = rk.mean_ci(X * pm)
            b, hb = rk.mean_ci(pm ** 2)
            # mismatched posterior: the statistician uses SNR 2 gamma
            Y = math.sqrt(g) * X + Z
            pmm = np.tanh(math.sqrt(2 * g) * Y) if name == "rad" else math.sqrt(2 * g) * Y / (1 + 2 * g)
            am, bm = float(np.mean(X * pmm)), float(np.mean(pmm ** 2))
            msem = float(np.mean((X - pmm) ** 2))
            ex = exact(name, g)[2]
            print(f" {g:4.2f}  {name}   {ex:.5f}   {m1:.5f} +- {h1:.5f}   {immse:.5f} +- {himmse:.5f}"
                  f"   {a:.5f}    {b:.5f}     {am:.5f}, {bm:.5f}, {msem:.5f}")
            row += [ex, m1, h1, immse, himmse, a, ha, b, hb, am, bm]
            inside += [abs(m1 - ex) <= h1, abs(immse - ex) <= himmse]
            if name == "rad":
                maxhalf = max(maxhalf, ha, hb)
                cell = lambda v, hw: f"${v:.4f}({int(round(hw * 1e4))})$"
                tex_rows.append(f"${g:g}$ & ${ex:.4f}$ & {cell(m1, h1)} & {cell(immse, himmse)} & ${a:.4f}$ & "
                                f"${b:.4f}$ & ${am:.4f}$ & ${bm:.4f}$\\\\")
        mc_rows.append(row)
    print("\nall 20 Monte Carlo intervals contain the exact mmse:", all(inside),
          f"({sum(inside)} of {len(inside)});  largest Nishimori half-width (Rademacher): {maxhalf:.5f}")
    print("table rows (Rademacher):")
    for r in tex_rows:
        print("  " + r)
    head = ["gamma"]
    for name in ("rad", "gau"):
        head += [f"ex{name}", f"mc{name}", f"ci{name}", f"im{name}", f"imci{name}", f"xx{name}", f"xxci{name}",
                 f"qq{name}", f"qqci{name}", f"xxmis{name}", f"qqmis{name}"]
    rk.write_dat(os.path.join(DATA, "01-freeentropy-channel-mc.dat"), head, mc_rows)
    # closed forms for the Gaussian prior with a mismatched SNR gamma' (exercise ex:freeentropy:mismatch)
    for g, gp in ((2.0, 0.5), (2.0, 4.0), (2.0, 2.0)):
        print(f"Gaussian prior, gamma={g}, gamma'={gp}: E X xhat = {math.sqrt(gp * g) / (1 + gp):.5f},"
              f" E xhat^2 = {gp * (1 + g) / (1 + gp) ** 2:.5f}, E<x^2>' = {1 / (1 + gp) + gp * (1 + g) / (1 + gp) ** 2:.5f}")


if __name__ == "__main__":
    main()
