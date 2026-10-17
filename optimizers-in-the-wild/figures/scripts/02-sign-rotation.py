"""Chapter 2 (Sign Descent), figure fig:sign:rotation.

Gradient descent and sign descent on a diagonal quadratic with heavy-tailed curvatures
h_i = i^(-2), i = 1..d, and on the same quadratic in randomly rotated coordinates.

Panel (a): loss against step.  Curves are exact: gradient descent has the closed form
    L_t = 1/2 sum_i h_i (1 - eta h_i)^(2t) theta_{0,i}^2,
the same in every coordinate system, and sign descent on the diagonal problem has the closed
form of Proposition prop:sign:floor (each coordinate moves by eta towards 0 and then alternates
between the distances r_i and eta - r_i).  Points are runs of the algorithms.

Panel (b): the long-run average of ||grad L||_1 against the step size eta.  Theory lines:
eta/2 * tr H (exact on the diagonal problem) and eta/2 * L_inf of the rotated Hessian (the floor of
Theorem thm:sign:descent), where L_inf = max_s s^T H s over sign vectors s is estimated by
sign power iteration plus single-flip local search (a lower bound on L_inf; the certified lower
bound lmax * ||Q^T e_1||_1^2 of Proposition prop:sign:rotation is printed as well, and so is a
certified upper bound from the semidefinite program of Exercise ex:sign:sdp, which bounds both
L_inf and the best coordinatewise ||L||_1).

Writes figures/data/02-sign-rotation-loss.dat and figures/data/02-sign-rotation-floor.dat.
Runs in a few seconds on one core.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

D = 256            # dimension
A = 2.0            # curvature exponent: h_i = i^(-A)
ETA_GD = 1.0       # 1 / lambda_max
ETA_SIGN = 0.01    # sign-descent step size for panel (a)
T = 10000          # steps for panel (a)
SEED = 2


def setup():
    rng = np.random.default_rng(SEED)
    h = np.arange(1, D + 1, dtype=float) ** (-A)
    z = rng.standard_normal((D, D))
    q, r = np.linalg.qr(z)
    q = q * np.sign(np.diag(r))          # Haar-distributed orthogonal matrix
    hr = (q.T * h) @ q                   # Hessian Q^T H Q of phi -> L(Q phi)
    hr = 0.5 * (hr + hr.T)
    theta0 = rng.standard_normal(D)      # theta^star = 0
    return rng, h, q, hr, theta0


def gd_exact(h, theta0, t):
    return 0.5 * np.sum(h * (1.0 - ETA_GD * h) ** (2 * t) * theta0 ** 2)


def sign_diag_exact(h, theta0, eta, t):
    """Exact distance |x_{t,i}| of sign descent on the separable quadratic (Prop. prop:sign:floor)."""
    a = np.abs(theta0)
    n = np.floor(a / eta)
    r = a - n * eta
    before = a - t * eta
    k = t - n                                        # steps since the cycle was reached
    after = np.where(r == 0.0, 0.0, np.where(k % 2 == 0, r, eta - r))
    return np.where(t <= n, before, after)


def run(hm, x0, eta, steps, kind):
    x = x0.copy()
    loss = np.empty(steps + 1)
    g1 = np.empty(steps + 1)
    for t in range(steps + 1):
        g = hm @ x
        loss[t] = 0.5 * x @ g
        g1[t] = np.abs(g).sum()
        if t == steps:
            break
        if kind == "gd":
            x = x - eta * g
        else:
            x = x - eta * np.sign(g)
    return loss, g1


def linf_estimate(hm, rng, starts=40):
    """Lower bound on max_s s^T H s: sign power iteration, then greedy single flips."""
    best = 0.0
    diag = np.diag(hm)
    for trial in range(starts):
        s = np.sign(rng.standard_normal(D))
        for _ in range(200):
            s2 = np.sign(hm @ s)
            s2[s2 == 0] = 1.0
            if np.array_equal(s2, s):
                break
            s = s2
        hs = hm @ s
        while True:
            gain = -4.0 * s * hs + 4.0 * diag     # change of s^T H s when s_i flips
            i = int(np.argmax(gain))
            if gain[i] <= 1e-12:
                break
            hs -= 2.0 * s[i] * hm[:, i]
            s[i] = -s[i]
        best = max(best, float(s @ hm @ s))
    return best


def sdp_upper_bound(hm, rank=8, sweeps=200, seed=0):
    """Certified upper bound on min{||L||_1 : diag(L) >= H}, hence on max_s s^T H s.

    Burer-Monteiro coordinate ascent ("mixing method") for max tr(HX), X = V V^T with unit rows,
    then the dual candidate L_i = <(HV)_i, v_i>, shifted by -lambda_min(diag(L) - H) if needed so
    that diag(L) - H is positive semidefinite (Exercise ex:sign:sdp).  Uses its own seed.
    """
    v = np.random.default_rng(seed).standard_normal((D, rank))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    off = hm - np.diag(np.diag(hm))
    for _ in range(sweeps):
        for i in range(D):
            w = off[i] @ v
            v[i] = w / np.linalg.norm(w)
    lvec = np.einsum("ij,ij->i", hm @ v, v)
    shift = max(0.0, -float(np.linalg.eigvalsh(np.diag(lvec) - hm).min()))
    return float(lvec.sum() + D * shift)


def floor_average(hm, x0, eta):
    """Average of ||grad||_1 over the final cycle (runs until the orbit has period 2)."""
    x = x0.copy()
    steps = int(40.0 / eta) + 2000
    hist = []
    for t in range(steps):
        x = x - eta * np.sign(hm @ x)
        if t >= steps - 8:
            hist.append(x.copy())
    assert np.allclose(hist[-1], hist[-3], atol=1e-10), "orbit not yet periodic"
    vals = [np.abs(hm @ y).sum() for y in hist[-2:]]
    return 0.5 * (vals[0] + vals[1])


def main():
    rng, h, q, hr, theta0 = setup()
    phi0 = q.T @ theta0
    hd = np.diag(h)
    tr = h.sum()

    # panel (a)
    lgd_d, _ = run(hd, theta0, ETA_GD, T, "gd")
    lgd_r, _ = run(hr, phi0, ETA_GD, T, "gd")
    ls_d, _ = run(hd, theta0, ETA_SIGN, T, "sign")
    ls_r, _ = run(hr, phi0, ETA_SIGN, T, "sign")
    ts = np.unique(np.round(np.logspace(0, np.log10(T), 41)).astype(int))
    with open(os.path.join(DATA, "02-sign-rotation-loss.dat"), "w") as f:
        f.write("t gdexact signexact gddiag gdrot signdiag signrot\n")
        for t in ts:
            xs = sign_diag_exact(h, theta0, ETA_SIGN, t)
            se = 0.5 * np.sum(h * xs ** 2)
            f.write(f"{t:d} {gd_exact(h, theta0, t):.6e} {se:.6e} {lgd_d[t]:.6e} "
                    f"{lgd_r[t]:.6e} {ls_d[t]:.6e} {ls_r[t]:.6e}\n")

    # panel (b)
    linf = linf_estimate(hr, rng)
    cert = h[0] * np.abs(q[0]).sum() ** 2       # lambda_max * ||Q^T e_1||_1^2
    etas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    with open(os.path.join(DATA, "02-sign-rotation-floor.dat"), "w") as f:
        f.write("eta diagtheory rottheory diagsim rotsim\n")
        for eta in etas:
            fd = floor_average(hd, theta0, eta)
            fr = floor_average(hr, phi0, eta)
            f.write(f"{eta:.4g} {eta * tr / 2:.6e} {eta * linf / 2:.6e} {fd:.6e} {fr:.6e}\n")
            print(f"eta={eta}: diag {fd:.6g} (theory {eta * tr / 2:.6g}), "
                  f"rot {fr:.6g} (eta*Linf/2 {eta * linf / 2:.6g})")
    upper = sdp_upper_bound(hr)
    print(f"d={D}, tr H={tr:.6f}, Linf(local search)={linf:.4f}, certified lower bound={cert:.4f}, "
          f"certified upper bound on Linf and on the best ||L||_1 (SDP)={upper:.4f}, "
          f"E-bound 1+2(d-1)/pi={1 + 2 * (D - 1) / np.pi:.4f}, ratio Linf/trH={linf / tr:.2f}, "
          f"(1+2(d-1)/pi) lmax/trH={(1 + 2 * (D - 1) / np.pi) / tr:.2f}")
    print("panel (a): sign descent on the rotated problem from t=1000 on:", ls_r[1000], ls_r[T])
    print("max |exact - simulated| sign diag:",
          max(abs(0.5 * np.sum(h * sign_diag_exact(h, theta0, ETA_SIGN, t) ** 2) - ls_d[t]) for t in ts))
    print("max |GD diag - GD rot| relative:", np.max(np.abs(lgd_d - lgd_r) / lgd_d))


if __name__ == "__main__":
    main()
