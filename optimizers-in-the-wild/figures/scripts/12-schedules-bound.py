#!/usr/bin/env python3
"""Figure 12.2 (Optimizers in the Wild, Chapter 12): the last-iterate bound of convex SGD under four
schedules, against SGD on a convex Lipschitz problem.

Problem. The cone f(theta) = E_a |<a, theta - theta*>| = sqrt(2/pi) ||theta - theta*||_2 with
a ~ N(0, I_d), d = 100: convex, nonsmooth at theta*, and the stochastic subgradient
g = sign(<a, theta - theta*>) a has E||g||^2 = d exactly, so G^2 = d. Start at distance D = 1.

Left panel: for every t, the bound of the last-iterate theorem for theta_t (computed from
eta_1..eta_t) and the mean of f(theta_t) - f* over 200 SGD runs; T = 1000, common peak
eta_0 = D/(G sqrt(T)); schedules constant, cosine to zero, linear decay to zero, WSD with a
linear cooldown over the last 20% of the steps.
Right panel: WSD with cooldown fraction c, peak tuned separately for each c (bound: exact minimizer
of the discrete bound; SGD: best of 17 log-spaced peaks between eta_0/4 and 4 eta_0, 40 runs each),
final value relative to linear decay (c = 1).

Output: figures/data/12-schedules-bound.dat, -bound-sgd.dat, -bound-tuned.dat, -bound-tunedcurve.dat.
Printed: the ratios quoted in the text and the entries of the table of the text (J[phi] by quadrature
and the exact discrete bound; no randomness, computed after the data files are written).
Runtime: about 15 seconds (numpy and scipy.integrate, seeds 21 and 22).
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
C0 = np.sqrt(2 / np.pi)


def schedule(name, T, c=0.2):
    """eta_1..eta_T (relative to the peak); eta_T is the free weight of the bound."""
    t = np.arange(1, T + 1)
    if name == "const":
        return np.ones(T)
    if name == "cosine":
        return 0.5 * (1 + np.cos(np.pi * t / (T + 1)))
    if name == "linear":
        return 1 - t / (T + 1)
    if name == "wsd":
        K = int(round(c * T))
        e = np.ones(T)
        e[T - K:] = 1 - np.arange(1, K + 1) / (K + 1)
        return e
    raise ValueError(name)


def bound_terms(eta):
    """bound = D^2 * a + G^2 * b (last-iterate theorem)."""
    S = np.cumsum(eta[::-1])[::-1]
    Q = np.cumsum((eta ** 2)[::-1])[::-1]
    a = 1 / (2 * S[0])
    b = Q[0] / (2 * S[0]) + 0.5 * np.sum(eta[:-1] * Q[:-1] / (S[:-1] * S[1:]))
    return a, b


def bound_along(eta, D, G):
    return np.array([D ** 2 * bound_terms(eta[:t])[0] + G ** 2 * bound_terms(eta[:t])[1]
                     for t in range(1, len(eta) + 1)])


def sgd(etas, d, runs, rng, D=1.0, record=True):
    """SGD on the cone for M schedules x runs; returns mean f(theta_t)-f* for t = 1..T (or final)."""
    M, T = etas.shape
    th = np.zeros((M, runs, d))
    th[:, :, 0] = D
    out = np.empty((M, T)) if record else None
    for t in range(T):
        if record:
            out[:, t] = C0 * np.linalg.norm(th, axis=2).mean(axis=1)
        if t == T - 1:
            break
        a = rng.standard_normal((M, runs, d))
        s = np.sign(np.einsum("mrd,mrd->mr", a, th))
        th -= etas[:, t][:, None, None] * s[:, :, None] * a
    return out if record else C0 * np.linalg.norm(th, axis=2).mean(axis=1)


def main():
    d, T, D = 100, 1000, 1.0
    G = np.sqrt(d)
    eta0 = D / (G * np.sqrt(T))
    names = ["const", "cosine", "linear", "wsd"]
    etas = np.array([eta0 * schedule(nm, T) for nm in names])
    bounds = [bound_along(e, D, G) for e in etas]
    with open(os.path.join(DATA, "12-schedules-bound.dat"), "w") as f:
        f.write("t const cosine linear wsd\n")
        for t in list(range(1, 10)) + list(range(10, T + 1, 5)):
            f.write("%d %.6g %.6g %.6g %.6g\n" % (t, *[b[t - 1] for b in bounds]))
    sim = sgd(etas, d, runs=200, rng=np.random.default_rng(21))
    with open(os.path.join(DATA, "12-schedules-bound-sgd.dat"), "w") as f:
        f.write("t const cosine linear wsd\n")
        for t in [1, 5, 10, 15] + list(range(25, T + 1, 25)):
            f.write("%d %.6g %.6g %.6g %.6g\n" % (t, *sim[:, t - 1]))
    print("final bound:", [round(b[-1], 4) for b in bounds], " final SGD:", np.round(sim[:, -1], 4))
    print("ratio bound/SGD at T:", np.round([b[-1] for b in bounds] / sim[:, -1], 2))
    print("constant plateau eta0*G^2/2 = %.4f, SGD at t=1000: %.4f" % (eta0 * G ** 2 / 2, sim[0, -1]))

    # right panel: tuned peaks, WSD with cooldown fraction c
    cs = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    facs = np.geomspace(0.25, 4, 17)
    rng = np.random.default_rng(22)
    rows = []
    for c in cs:
        shape = schedule("wsd", T, c=c) if c < 1 else schedule("linear", T)
        a, b = bound_terms(shape)
        bound_best = 2 * D * G * np.sqrt(a * b)
        fin = sgd(np.array([f * eta0 * shape for f in facs]), d, runs=40, rng=rng, record=False)
        rows.append((c, fin.min(), facs[np.argmin(fin)], bound_best, np.sqrt(D ** 2 * a / (G ** 2 * b)) / eta0))
    ref_s, ref_b = rows[-1][1], rows[-1][3]
    with open(os.path.join(DATA, "12-schedules-bound-tuned.dat"), "w") as f:
        f.write("c sgdrel boundrel sgdpeak boundpeak\n")
        for (c, s, fs, b, fb) in rows:
            f.write("%.3f %.5g %.5g %.4g %.4g\n" % (c, s / ref_s, b / ref_b, fs, fb))
            print("c=%.2f SGD best %.4f rel %.3f (peak x%.2f) | bound best %.4f rel %.3f (peak x%.2f)" % (
                c, s, s / ref_s, fs, b, b / ref_b, fb))
    # the tuned discrete bound on a fine grid of c (theory curve of the right panel)
    with open(os.path.join(DATA, "12-schedules-bound-tunedcurve.dat"), "w") as f:
        f.write("c boundrel\n")
        for c in np.geomspace(0.02, 1.0, 60):
            shape = schedule("wsd", T, c=c) if c < 1 else schedule("linear", T)
            a, b = bound_terms(shape)
            f.write("%.5g %.6g\n" % (c, 2 * D * G * np.sqrt(a * b) / ref_b))
    for nm in ["const", "cosine"]:
        shape = schedule(nm, T)
        a, b = bound_terms(shape)
        fin = sgd(np.array([f * eta0 * shape for f in facs]), d, runs=40, rng=rng, record=False)
        print("%s: SGD tuned rel %.3f, bound tuned rel %.3f" % (nm, fin.min() / ref_s, 2 * D * G * np.sqrt(a * b) / ref_b))
    table_checks()


def table_checks():
    """Entries of the table of the text: Phi(0), J by quadrature, the limit sqrt(J/Phi(0)) of the best
    bound times sqrt(T)/(DG), and the exact discrete value 2 sqrt(a b T) at T = 10^4 and 10^5 for the
    shape eta_t = phi(t/(T+1)) (no randomness)."""
    from scipy.integrate import quad

    def J_of(phi, brk):
        pts = lambda s: [x for x in brk if x > s] or None
        Phi = lambda s: quad(phi, s, 1, points=pts(s), limit=400)[0]
        Psi = lambda s: quad(lambda u: phi(u) ** 2, s, 1, points=pts(s), limit=400)[0]
        integral = quad(lambda s: phi(s) * Psi(s) / Phi(s) ** 2, 0, 1, points=list(brk) or None, limit=400)[0]
        return Psi(0) / Phi(0) + integral, Phi(0)

    def discrete(phi, T):
        a, b = bound_terms(np.vectorize(phi)(np.arange(1, T + 1) / (T + 1)))
        return 2 * np.sqrt(a * b * T)

    shapes = [("linear decay", lambda s: 1 - s, ()),
              ("cosine to zero", lambda s: 0.5 * (1 + np.cos(np.pi * s)), ())]
    for c in [0.1, 0.2, 0.5]:
        shapes.append(("WSD linear c=%.1f" % c, lambda s, c=c: 1.0 if s <= 1 - c else (1 - s) / c, (1 - c,)))
    shapes.append(("WSD 1-sqrt c=0.2", lambda s: 1.0 if s <= 0.8 else 1 - np.sqrt((s - 0.8) / 0.2), (0.8,)))
    for name, phi, brk in shapes:
        J, P0 = J_of(phi, brk)
        print("table: %-17s Phi(0) %.4f  J %.4f  limit %.4f  discrete %.4f (T=1e4) %.4f (T=1e5)" % (
            name, P0, J, np.sqrt(J / P0), discrete(phi, 10 ** 4), discrete(phi, 10 ** 5)))
    print("table: constant, sqrt(1+H_{T-1}) at T=1e4: %.4f" % np.sqrt(1 + np.sum(1 / np.arange(1, 10 ** 4))))
    rho = 0.1
    for T in [10 ** 4, 10 ** 5]:
        best = discrete(lambda s: rho + (1 - rho) * 0.5 * (1 + np.cos(np.pi * s)), T)
        J = best ** 2 * (rho + (1 - rho) / 2)
        print("table: cosine to 0.1, T=%d: best %.4f, J - 0.1 ln T = %.4f" % (T, best, J - rho * np.log(T)))


if __name__ == "__main__":
    main()
