"""Chapter 8 (The Free Energy Formula): the normalized free energy F_n - nS_n of five models at the truth 0.

Models (inputs x ~ N(0, I_2), unit noise, uniform prior on [-1, 1]^2, truth y = noise):
    regular plane  y = a x1 + b x2      c = (a, b)        lambda = 1,   m = 1
    quartic        y = a x1 + b^2 x2    c = (a, b^2)      lambda = 3/4, m = 1
    sextic         y = a x1 + b^3 x2    c = (a, b^3)      lambda = 2/3, m = 1
    quartic cup    y = a^2 x1 + b^2 x2  c = (a^2, b^2)    lambda = 1/2, m = 1
    product        y = a b x1           c = ab            lambda = 1/2, m = 2
All five are linear in their coefficient vector c = g(w), so with S = sum x x^T and s = sum x y,
    F_n - nS_n = -log int exp(-c'Sc/2 + c's) phi(w) dw            (truth c0 = 0, y = noise).
Given the inputs, s ~ N(0, S) exactly, and S ~ Wishart(n, I_2); the script draws (S, s) directly by the
Bartlett decomposition S = L L', s = L z with z ~ N(0, I_2), so that rho^2 = s'S^{-1}s = |z|^2 ~ chi^2_2
exactly. The five models share each data set; the product model uses the first input only.

Quadrature (no grid in two dimensions is needed at n = 10^6):
  * regular plane, quartic, sextic: the integral over a in [-1, 1] is done exactly (error functions); the
    integral over b is composite Gauss-Legendre on panels with breakpoints at 0, +-sigma 2^j (sigma =
    n^{-1/(2k)} the scale of b for c2 = b^k), 24 nodes per panel (doubling changes nothing at 1e-14);
  * quartic cup: the same composite rule in a and in b (tensor product) on [0, 1]^2 (symmetry);
  * product model: sltkit.coefficient_free_energy with the density of ab (exact one-dimensional quadrature).
  Checks: against sltkit.free_energy_grid (midpoint grid, 1000^2 and 2000^2 points) at n = 100 and 1000,
  and against the sandwich lemma of the chapter, a deterministic inequality that must hold for every data set
  (checked on 20 data sets per n, with the exact noiseless free energies at (1 -+ 1/2) times the design).

Predictions:
  singular (bronze): lambda log n - (m-1) log log n + E[R], with E[R] the mean of the limit law
     (regular -1 - log(pi/2); quartic -1 - log(2 pi)/2 + log 4 - E log h(Z); sextic -1 - log(2 pi)/2 + log 12
      - E log H(Z); cup -1 + 2 log 2 - 2 E log h(Z); product log(8/pi)/2 - 1/2), with
      h(z) = int_0^inf exp(-(t-z)^2/2) t^{-1/2} dt and H(z) = int_R exp(-(t-z)^2/2)|t|^{-2/3} dt;
  regular (slate dashed): log n - 1 - log(pi/2), Laplace's value for the regular plane, which BIC assigns to
     every model with two parameters up to a constant.

Output: figures/data/08-freeenergy-models.dat (means, 95% Student-t intervals, predictions) and
        figures/data/08-freeenergy-slopes.dat (fitted slopes with 95% intervals).
Run from the volume root:  python figures/scripts/08-freeenergy-models.py   (one to two minutes)
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy import integrate, special  # noqa: E402

import sltkit as sk  # noqa: E402

REPS = 200
EXPONENTS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
NODES = 24
MODELS = ["reg", "quart", "sext", "cup", "prod"]
LAM = {"reg": 1.0, "quart": 0.75, "sext": 2 / 3, "cup": 0.5, "prod": 0.5}
MULT = {"reg": 1, "quart": 1, "sext": 1, "cup": 1, "prod": 2}
NAMES = {"reg": "regular plane y=a x1+b x2", "quart": "quartic y=a x1+b^2 x2", "sext": "sextic y=a x1+b^3 x2",
         "cup": "quartic cup y=a^2 x1+b^2 x2", "prod": "product y=abx"}
GLX, GLW = np.polynomial.legendre.leggauss(NODES)


# ------------------------------------------------------------------------------------------------
# composite Gauss-Legendre rule on [0, 1] with panels refined at the scale sigma
# ------------------------------------------------------------------------------------------------
def panels(sigma: float) -> np.ndarray:
    pts = [0.0]
    j = -4
    while sigma * 2.0 ** j < 1.0:
        pts.append(sigma * 2.0 ** j)
        j += 1
    pts.append(1.0)
    return np.array(pts)


def rule01(sigma: float) -> tuple[np.ndarray, np.ndarray]:
    br = panels(sigma)
    nodes, weights = [], []
    for lo, hi in zip(br[:-1], br[1:]):
        nodes.append(0.5 * (hi - lo) * GLX + 0.5 * (hi + lo))
        weights.append(0.5 * (hi - lo) * GLW)
    return np.concatenate(nodes), np.concatenate(weights)


def rule_sym(sigma: float) -> tuple[np.ndarray, np.ndarray]:
    x, w = rule01(sigma)
    return np.concatenate([-x[::-1], x]), np.concatenate([w[::-1], w])


# ------------------------------------------------------------------------------------------------
# log of int_{-1}^{1} exp(-S11 a^2 / 2 + beta a) da, vectorized in beta
# ------------------------------------------------------------------------------------------------
def log_diff_ndtr(hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
    """log(Phi(hi) - Phi(lo)) for hi > lo, stable in both tails."""
    hi, lo = np.broadcast_arrays(hi, lo)
    out = np.empty(hi.shape)
    upper = lo > 0  # both in the upper tail: use survival functions
    a = special.log_ndtr(-lo[upper]); b = special.log_ndtr(-hi[upper])
    out[upper] = a + np.log1p(-np.exp(b - a))
    lw = ~upper
    a = special.log_ndtr(hi[lw]); b = special.log_ndtr(lo[lw])
    out[lw] = a + np.log1p(-np.exp(b - a))
    return out


def log_gauss_box(S11: float, beta: np.ndarray) -> np.ndarray:
    r = math.sqrt(S11)
    m = beta / S11
    return beta * beta / (2 * S11) + 0.5 * math.log(2 * math.pi / S11) + log_diff_ndtr(r * (1 - m), r * (-1 - m))


# ------------------------------------------------------------------------------------------------
# free energies F - nS = -log int exp(-c'Sc/2 + c's) phi
# ------------------------------------------------------------------------------------------------
def free_linear_a(S: np.ndarray, s: np.ndarray, k: int, n: float, scale: float = 1.0) -> float:
    """Models with c = (a, b^k), a and b uniform on [-1, 1]; S and s may be scaled (sandwich checks)."""
    sigma = n ** (-1.0 / (2 * k))
    b, wb = rule_sym(sigma)
    c2 = b ** k
    S11, S12, S22 = scale * S[0, 0], scale * S[0, 1], scale * S[1, 1]
    beta = s[0] - S12 * c2
    lg = log_gauss_box(S11, beta) + math.log(0.5)                # (1/2) int da
    lg += -0.5 * S22 * c2 * c2 + s[1] * c2 + np.log(0.5 * wb)    # (1/2) db
    return float(-special.logsumexp(lg))


def free_cup(S: np.ndarray, s: np.ndarray, n: float, scale: float = 1.0) -> float:
    sigma = n ** (-0.25)
    x, w = rule01(sigma)
    c1 = (x * x)[:, None]
    c2 = (x * x)[None, :]
    E = 0.5 * scale * (S[0, 0] * c1 * c1 + 2 * S[0, 1] * c1 * c2 + S[1, 1] * c2 * c2) - s[0] * c1 - s[1] * c2
    lw = -E + np.log(w)[:, None] + np.log(w)[None, :]
    return float(-special.logsumexp(lw))


def free_product(S11: float, s1: float, scale: float = 1.0) -> float:
    curv = scale * S11
    return sk.coefficient_free_energy(-s1 * s1 / (2 * curv), curv, lambda c: sk.product_density(c, 2),
                                      -1.0, 1.0, s1 / curv)


def free_all(S: np.ndarray, s: np.ndarray, n: float, scale: float = 1.0) -> dict:
    return {"reg": free_linear_a(S, s, 1, n, scale), "quart": free_linear_a(S, s, 2, n, scale),
            "sext": free_linear_a(S, s, 3, n, scale), "cup": free_cup(S, s, n, scale),
            "prod": free_product(S[0, 0], s[0], scale)}


# ------------------------------------------------------------------------------------------------
# data: S ~ Wishart(n, I_2), s | S ~ N(0, S) (Bartlett)
# ------------------------------------------------------------------------------------------------
def draw(n: int, gen: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    L = np.zeros((2, 2))
    L[0, 0] = math.sqrt(gen.chisquare(n))
    L[1, 1] = math.sqrt(gen.chisquare(n - 1))
    L[1, 0] = gen.standard_normal()
    z = gen.standard_normal(2)
    return L @ L.T, L @ z, float(z @ z)


# ------------------------------------------------------------------------------------------------
# limit constants E[R]
# ------------------------------------------------------------------------------------------------
def h_half(z: float) -> float:
    f = lambda t: math.exp(-0.5 * (t - z) ** 2) / math.sqrt(t)
    mid = max(z, 1.0)
    return integrate.quad(f, 0.0, mid, limit=400)[0] + integrate.quad(f, mid, mid + 40.0, limit=400)[0]


def H_two_thirds(z: float) -> float:
    def side(zz):
        g = lambda t: math.exp(-0.5 * (t - zz) ** 2) * t ** (-2.0 / 3.0)
        mid = max(zz, 1.0)
        return integrate.quad(g, 0.0, mid, limit=400)[0] + integrate.quad(g, mid, mid + 40.0, limit=400)[0]
    return side(z) + side(-z)


def limit_constants() -> dict:
    zs, wz = np.polynomial.hermite_e.hermegauss(120)
    wz = wz / wz.sum()
    elh = float(np.dot(wz, [math.log(h_half(z)) for z in zs]))
    elH = float(np.dot(wz, [math.log(H_two_thirds(z)) for z in zs]))
    return {"reg": -1 - math.log(math.pi / 2),
            "quart": -1 - 0.5 * math.log(2 * math.pi) + math.log(4) - elh,
            "sext": -1 - 0.5 * math.log(2 * math.pi) + math.log(12) - elH,
            "cup": -1 + 2 * math.log(2) - 2 * elh,
            "prod": 0.5 * math.log(8 / math.pi) - 0.5}, elh, elH


def population_free(model: str, n: float) -> float:
    """-log int exp(-n K) phi, exactly (one-dimensional integrals)."""
    Za = 0.5 * integrate.quad(lambda a: math.exp(-0.5 * n * a * a), -1, 1, points=[0])[0]
    if model == "reg":
        return -2 * math.log(Za)
    if model in ("quart", "sext"):
        k = 2 if model == "quart" else 3
        Zb = 0.5 * integrate.quad(lambda b: math.exp(-0.5 * n * b ** (2 * k)), -1, 1, points=[0], limit=200)[0]
        return -math.log(Za) - math.log(Zb)
    if model == "cup":
        Zb = 0.5 * integrate.quad(lambda b: math.exp(-0.5 * n * b ** 4), -1, 1, points=[0], limit=200)[0]
        return -2 * math.log(Zb)
    # product: int exp(-n c^2/2) rho(c) dc
    f = lambda c: math.exp(-0.5 * n * c * c) * float(sk.product_density(c, 2))
    return -math.log(2 * integrate.quad(f, 0, 1, points=[1 / math.sqrt(n)], limit=400)[0])


# ------------------------------------------------------------------------------------------------
def hand_numbers() -> None:
    """The numbers of Computations 8.1-8.2 and of the exercises, recomputed."""
    G = special.gamma
    I6 = G(1 / 6) * G(1.5) / (3 * G(5 / 3))
    C = 2 ** (-1 / 3) * I6
    kappa = -math.log(C * G(5 / 3))
    print(f"[8.1] I6 = {I6:.4f} (quad {integrate.quad(lambda v: math.sqrt(1 - v ** 6), -1, 1)[0]:.4f}), "
          f"V(eps) = {C:.4f} eps^(2/3), Gamma(5/3) = {G(5 / 3):.4f}, constant {kappa:.4f} "
          f"(direct {-math.log(math.sqrt(math.pi / 2) * 2 ** (1 / 6) * G(7 / 6)):.4f})")
    n = 1e4
    lo = 2 / 3 * math.log(0.5 * n) + kappa - 2.0
    hi = 2 / 3 * math.log(1.5 * n) + kappa + 2.0
    etas = np.linspace(0.01, 0.99, 9801)
    width = 2 / 3 * np.log((1 + etas) / (1 - etas)) + 2 / etas
    print(f"[8.2] sandwich at n = 1e4, eta = 1/2, rho^2 = 2: [{lo:.2f}, {hi:.2f}], width {hi - lo:.2f}; "
          f"best eta {etas[np.argmin(width)]:.3f}, width {width.min():.2f}")
    I4 = G(0.25) * G(1.5) / (2 * G(1.75))
    Cq = 0.5 * I4 * 2 ** 0.75
    Cc = 0.25 * math.sqrt(2) * G(0.25) * G(1.25) / G(1.5)
    print(f"[ex] quartic: I4 = {I4:.4f}, V = {Cq:.4f} eps^(3/4), constant {-math.log(Cq * G(1.75)):.3f}; "
          f"cup: V = {Cc:.4f} eps^(1/2), constant {-math.log(Cc * G(1.5)):.3f}")
    print(f"[ex] Gaussian prior on the sextic model: V ~ {2 ** (2 / 3) * I6 / math.pi:.3f} eps^(2/3); "
          f"constant shift log(pi/2) = {math.log(math.pi / 2):.4f}; generic-truth threshold 3^(-3/2) = {3 ** -1.5:.3f}")
    print(f"[ex] H(0) = 2^(1/6) Gamma(1/6) = {2 ** (1 / 6) * G(1 / 6):.3f}; product model: E R_n ~ "
          f"{0.5 * math.log(8 / math.pi) - 0.5:.4f} - gamma/log n = {0.5 * math.log(8 / math.pi) - 0.5 - np.euler_gamma / math.log(1e4):.3f} "
          f"(n = 1e4), {0.5 * math.log(8 / math.pi) - 0.5 - np.euler_gamma / (100 * math.log(10)):.4f} (n = 1e100)")
    for T in (0.0, 2.0, 10.0, 20.0):
        A = math.exp(T / 2) / 10.0
        L = 0.5 * (A - 1 + math.sqrt((1 - A) ** 2 + 4 * A * math.log(100)))
        print(f"[ex] sBIC of two models, n = 100, T = {T:4.1f}: sBIC difference {math.log(L):.2f} "
              f"(log A = {math.log(A):.2f}, log A + log log n = {math.log(A) + math.log(math.log(100)):.2f})")


def check_against_grid() -> None:
    gen = sk.rng(8001)
    for n in (100, 1000):
        S, s, _ = draw(n, gen)
        ours = free_all(S, s, n)
        fns = {"reg": lambda w: np.stack([w[:, 0], w[:, 1]], 1), "quart": lambda w: np.stack([w[:, 0], w[:, 1] ** 2], 1),
               "sext": lambda w: np.stack([w[:, 0], w[:, 1] ** 3], 1), "cup": lambda w: np.stack([w[:, 0] ** 2, w[:, 1] ** 2], 1)}
        line = [f"n={n}:"]
        for key, g in fns.items():
            def energy(w, g=g):
                c = g(w)
                return 0.5 * np.einsum("ki,ij,kj->k", c, S, c) - c @ s
            g1 = sk.free_energy_grid(energy, [(-1, 1), (-1, 1)], 1000)
            g2 = sk.free_energy_grid(energy, [(-1, 1), (-1, 1)], 2000)
            line.append(f"{key} {ours[key]:.5f} grid {g1:.5f}/{g2:.5f}")
        def energy_p(w):
            c = w[:, 0] * w[:, 1]
            return 0.5 * S[0, 0] * c * c - s[0] * c
        g2 = sk.free_energy_grid(energy_p, [(-1, 1), (-1, 1)], 2000)
        line.append(f"prod {ours['prod']:.5f} grid {g2:.5f}")
        print("  ".join(line))


def main() -> None:
    t0 = time.time()
    os.makedirs(os.path.join("figures", "data"), exist_ok=True)
    hand_numbers()
    check_against_grid()
    ER, elh, elH = limit_constants()
    print(f"E log h(Z) = {elh:.4f}, E log H(Z) = {elH:.4f}")
    print("limit constants E[R]: " + ", ".join(f"{k} {v:.4f}" for k, v in ER.items()))
    print(f"population free energies at n = 1e4: sextic {population_free('sext', 1e4):.4f} "
          f"(minus (2/3) log n: {population_free('sext', 1e4) - 2 / 3 * math.log(1e4):.5f}), regular plane "
          f"{population_free('reg', 1e4):.4f}")

    rows, store = [], {}
    eta = 0.5
    sandwich_ok, sandwich_total, onA = 0, 0, 0
    for ex in EXPONENTS:
        n = int(round(10 ** ex))
        gen = sk.rng(80_000 + int(round(ex * 10)))
        vals = {k: [] for k in MODELS}
        rho2 = []
        for r in range(REPS):
            S, s, z2 = draw(n, gen)
            F = free_all(S, s, n)
            for k in MODELS:
                vals[k].append(F[k])
            rho2.append(z2)
            # sandwich: F between the noiseless free energies at (1 -+ eta) and -+ rho^2/(2 eta)
            if r < 20:
                lo = free_all(S, 0 * s, n, scale=1 - eta)
                hi = free_all(S, 0 * s, n, scale=1 + eta)
                for k in MODELS:
                    rr = s[0] ** 2 / S[0, 0] if k == "prod" else z2
                    sandwich_total += 1
                    if lo[k] - rr / (2 * eta) - 1e-9 <= F[k] <= hi[k] + rr / (2 * eta) + 1e-9:
                        sandwich_ok += 1
            ev = np.linalg.eigvalsh(S / n)
            onA += int(ev.min() >= 1 - eta and ev.max() <= 1 + eta)
        store[n] = {k: np.array(v) for k, v in vals.items()}
        store[n]["rho2"] = np.array(rho2)
        row = [n]
        msg = [f"n=1e{ex:.1f}:"]
        for k in MODELS:
            m, lo, hi = sk.mean_ci(vals[k])
            pred = LAM[k] * math.log(n) - (MULT[k] - 1) * math.log(math.log(n)) + ER[k]
            row += [m, m - lo, hi - m, pred]
            msg.append(f"{k} {m:7.3f}+-{hi - m:.3f} (pred {pred:7.3f})")
        row.append(math.log(n) - 1 - math.log(math.pi / 2))
        rows.append(row)
        msg.append(f"mean rho^2 {np.mean(rho2):.3f}")
        print("  ".join(msg))
    print(f"sandwich held in {sandwich_ok}/{sandwich_total} checks; event A_n(1/2) held in {onA}/{REPS * len(EXPONENTS)} data sets")

    header = ["n"]
    for k in MODELS:
        header += [k, k + "lo", k + "hi", k + "pred"]
    header.append("regpredbic")
    sk.write_dat(os.path.join("figures", "data", "08-freeenergy-models.dat"), header, rows)

    # slopes against log n, with 95% intervals (independent data sets at each n)
    logn = np.log([r[0] for r in rows])
    xc = logn - logn.mean()
    wts = xc / np.sum(xc * xc)
    slope_rows = []
    for idx, k in enumerate(MODELS):
        means = np.array([store[n][k].mean() for n in store])
        var_means = np.array([store[n][k].var(ddof=1) / REPS for n in store])
        sl = float(wts @ means)
        se = float(math.sqrt(wts * wts @ var_means))
        loglog = np.log(logn)
        sl_ll = float(wts @ (means + (MULT[k] - 1) * loglog))
        pop = np.array([population_free(k, n) for n in store])
        sl_pop = float(wts @ pop)
        slope_rows.append((idx + 1, LAM[k], MULT[k], sl, 1.96 * se, sl_ll, sl_pop))
        sd_R = float(np.mean([np.std(store[n][k] - LAM[k] * math.log(n) + (MULT[k] - 1) * math.log(math.log(n)), ddof=1)
                              for n in store]))
        print(f"{NAMES[k]:30s} lambda {LAM[k]:.3f} m {MULT[k]}: slope {sl:.3f} +- {1.96 * se:.3f}; "
              f"slope of F + (m-1) loglog n: {sl_ll:.3f}; slope of population free energy {sl_pop:.3f}; mean sd of R_n {sd_R:.3f}")
    sk.write_dat(os.path.join("figures", "data", "08-freeenergy-slopes.dat"),
                 ["model", "lambda", "m", "slope", "ci", "slopell", "slopepop"], slope_rows)

    # Bayes factors on the same data: regular plane against the sextic model (same set of distributions),
    # and quartic against sextic (learning coefficients 3/4 and 2/3)
    outside = 0
    for n in store:
        for k in MODELS:
            m, lo, hi = sk.mean_ci(store[n][k])
            pred = LAM[k] * math.log(n) - (MULT[k] - 1) * math.log(math.log(n)) + ER[k]
            outside += int(not (lo <= pred <= hi)) if MULT[k] == 1 else 0
    print(f"m = 1 models: prediction outside the 95% interval in {outside} of {4 * len(store)} cases")
    for n in store:
        lbf = store[n]["reg"] - store[n]["sext"]
        m, lo, hi = sk.mean_ci(lbf)
        q = store[n]["quart"] - store[n]["sext"]
        print(f"n={n:>7d}: log Bayes factor sextic : regular = {m:.3f} [{lo:.3f},{hi:.3f}] (sd {np.std(lbf, ddof=1):.3f}); "
              f"pred {math.log(n) / 3 + ER['reg'] - ER['sext']:.3f}; P(sextic preferred) = {np.mean(lbf > 0):.3f} | "
              f"sextic : quartic = {q.mean():.3f} (sd {np.std(q, ddof=1):.3f}), P(sextic preferred) = {np.mean(q > 0):.3f}")
    print(f"mean rho^2 over all data sets: {np.mean([store[n]['rho2'].mean() for n in store]):.3f} (chi^2_2 mean 2)")
    print(f"runtime {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
