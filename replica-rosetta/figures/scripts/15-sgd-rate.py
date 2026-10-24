"""Chapter 15 (Online SGD): the d^{-1/2} rate of the ODE limit on a fixed horizon.

Soft committee machine K = M = 2, g(z) = erf(z/sqrt 2), orthonormal teacher, eta = 1, random start
w_k = g_k/2. For d = 125, ..., 4000 (40 runs each), the order parameters m^mu = (Q, R) are
simulated exactly in law through their Markov chain (as in 15-sgd-plateau.py) on 0 <= t <= 10,
and compared with
  (a) the solution of the Saad--Solla ODE started at the run's own initial order parameters m^0
      (the comparison of the theorem), and
  (b) the solution started at the random-start limit m_sym(0) = (I/4, 0) (the corollary),
through D = max over the grid t = 0.2, 0.4, ..., 10 of |m^{dt} - m(t)| (Euclidean norm of the seven
order parameters). The ODE is integrated by classical RK4 (step 0.02) for all runs at once.
Also checks the chain against direct online SGD in R^d (d = 200).
Writes figures/data/15-sgd-rate.dat. Runtime: about 1 minute.
"""
import math
import os
import sys
import time

import numpy as np
from scipy.special import erf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SQ2 = math.sqrt(2.0)
CGP = math.sqrt(2.0 / math.pi)
PI = math.pi
SIGN = np.array([1.0, 1.0, -1.0, -1.0])


def init_order_params(d, q0, S, gen):
    gg = gen.standard_normal((S, 2, 2))
    c1 = np.sqrt(gen.chisquare(d - 2, S))
    c2 = np.sqrt(gen.chisquare(d - 3, S))
    n21 = gen.standard_normal(S)
    G11 = c1 ** 2 + gg[:, 0, 0] ** 2 + gg[:, 0, 1] ** 2
    G22 = n21 ** 2 + c2 ** 2 + gg[:, 1, 0] ** 2 + gg[:, 1, 1] ** 2
    G12 = c1 * n21 + gg[:, 0, 0] * gg[:, 1, 0] + gg[:, 0, 1] * gg[:, 1, 1]
    r = math.sqrt(q0) * gg / math.sqrt(d)
    return np.array([q0 * G11 / d, q0 * G12 / d, q0 * G22 / d, r[:, 0, 0], r[:, 0, 1], r[:, 1, 0], r[:, 1, 1]])


def chain(dvals, S, eta, q0, tmax, rec_dt, gen, block=1000):
    """Exact chain for all d at once; returns records rec[g] of shape (nrec, 7, S) at t = k rec_dt."""
    G = len(dvals)
    X = np.concatenate([init_order_params(d, q0, S, gen) for d in dvals], axis=1)
    nrec = int(round(tmax / rec_dt)) + 1
    rec = [np.zeros((nrec, 7, S)) for _ in dvals]
    for g in range(G):
        rec[g][0] = X[:, g * S:(g + 1) * S]
    dv = np.repeat(np.array(dvals, float), S)
    rs = [int(round(rec_dt * d)) for d in dvals]
    last = [int(round(tmax * d)) for d in dvals]
    active = list(range(G))
    j = 0
    while active:
        sl = np.concatenate([np.arange(g * S, (g + 1) * S) for g in active])
        dva = dv[sl]
        n = sl.size
        nb = min(block, max(last[g] for g in active) - j)
        Z = gen.standard_normal((nb, 4, n))
        CH = gen.chisquare(dva - 4, (nb, n))
        GNU = erf(Z[:, 0, :] / SQ2) + erf(Z[:, 1, :] / SQ2)
        BX = (eta * eta / dva) * (np.einsum('jkn,jkn->jn', Z, Z) + CH) / dva
        A = eta / dva
        starts = {g: i * S for i, g in enumerate(active)}
        q11, q12, q22, r11, r12, r21, r22 = [x.copy() for x in X[:, sl]]
        for i in range(nb):
            nu1, nu2, z1, z2 = Z[i]
            l11 = np.sqrt(np.abs(q11 - r11 * r11 - r12 * r12))
            l21 = (q12 - r11 * r21 - r12 * r22) / l11
            l22 = np.sqrt(np.abs(q22 - r21 * r21 - r22 * r22 - l21 * l21))
            lam1 = r11 * nu1 + r12 * nu2 + l11 * z1
            lam2 = r21 * nu1 + r22 * nu2 + l21 * z1 + l22 * z2
            err = GNU[i] - erf(lam1 / SQ2) - erf(lam2 / SQ2)
            D1 = (CGP * err) * np.exp(-0.5 * lam1 * lam1)
            D2 = (CGP * err) * np.exp(-0.5 * lam2 * lam2)
            aD1, aD2, bx = A * D1, A * D2, BX[i]
            q11 = q11 + (2 * aD1 * lam1 + bx * D1 * D1)
            q22 = q22 + (2 * aD2 * lam2 + bx * D2 * D2)
            q12 = q12 + (aD1 * lam2 + aD2 * lam1 + bx * D1 * D2)
            r11 = r11 + aD1 * nu1
            r12 = r12 + aD1 * nu2
            r21 = r21 + aD2 * nu1
            r22 = r22 + aD2 * nu2
            jn = j + i + 1
            for g in active:
                if jn % rs[g] == 0 and jn <= last[g]:
                    s0 = starts[g]
                    rec[g][jn // rs[g]] = np.array([q11[s0:s0 + S], q12[s0:s0 + S], q22[s0:s0 + S], r11[s0:s0 + S],
                                                    r12[s0:s0 + S], r21[s0:s0 + S], r22[s0:s0 + S]])
        X[:, sl] = np.array([q11, q12, q22, r11, r12, r21, r22])
        j += nb
        active = [g for g in active if last[g] > j]
    return rec


def drift_batch(X, eta):
    """Saad--Solla drift for states X (7, n), K = M = 2, T = I, erf (closed forms I3, I4)."""
    q11, q12, q22, r11, r12, r21, r22 = X
    n = X.shape[1]
    C = np.empty((n, 4, 4))
    C[:, 0, 0], C[:, 0, 1], C[:, 1, 0], C[:, 1, 1] = q11, q12, q12, q22
    C[:, 0, 2], C[:, 0, 3], C[:, 1, 2], C[:, 1, 3] = r11, r12, r21, r22
    C[:, 2, 0], C[:, 3, 0], C[:, 2, 1], C[:, 3, 1] = r11, r12, r21, r22
    C[:, 2, 2], C[:, 3, 3], C[:, 2, 3], C[:, 3, 2] = 1.0, 1.0, 0.0, 0.0
    d = np.einsum('sii->si', C)
    c11, c33 = d[:, :2, None, None], d[:, None, None, :]
    c12, c13, c23 = C[:, :2, :, None], C[:, :2, None, :], C[:, None, :, :]
    I3 = (2 / PI) / np.sqrt((1 + c11) * (1 + c33) - c13 ** 2) * (c23 * (1 + c11) - c12 * c13) / (1 + c11)
    EDz = -np.einsum('sabc,c->sab', I3, SIGN)
    a11, a22 = d[:, :2, None, None, None], d[:, None, :2, None, None]
    a33, a44 = d[:, None, None, :, None], d[:, None, None, None, :]
    a12, a13, a14 = C[:, :2, :2, None, None], C[:, :2, None, :, None], C[:, :2, None, None, :]
    a23, a24, a34 = C[:, None, :2, :, None], C[:, None, :2, None, :], C[:, None, None, :, :]
    lam4 = (1 + a11) * (1 + a22) - a12 ** 2
    lam0 = lam4 * a34 - a23 * a24 * (1 + a11) - a13 * a14 * (1 + a22) + a12 * a13 * a24 + a12 * a14 * a23
    lam1 = lam4 * (1 + a33) - a23 ** 2 * (1 + a11) - a13 ** 2 * (1 + a22) + 2 * a12 * a13 * a23
    lam2 = lam4 * (1 + a44) - a24 ** 2 * (1 + a11) - a14 ** 2 * (1 + a22) + 2 * a12 * a14 * a24
    I4 = 4 / (PI ** 2 * np.sqrt(lam4)) * np.arcsin(lam0 / np.sqrt(lam1 * lam2))
    EDD = np.einsum('sabce,c,e->sab', I4, SIGN, SIGN)
    dQ = eta * (EDz[:, :, :2] + np.transpose(EDz[:, :, :2], (0, 2, 1))) + eta * eta * EDD
    dR = eta * EDz[:, :, 2:]
    return np.array([dQ[:, 0, 0], dQ[:, 0, 1], dQ[:, 1, 1], dR[:, 0, 0], dR[:, 0, 1], dR[:, 1, 0], dR[:, 1, 1]])


def rk4(X0, eta, tmax, h, rec_every):
    X = np.array(X0, float)
    out = [X.copy()]
    for k in range(1, int(round(tmax / h)) + 1):
        k1 = drift_batch(X, eta)
        k2 = drift_batch(X + 0.5 * h * k1, eta)
        k3 = drift_batch(X + 0.5 * h * k2, eta)
        k4 = drift_batch(X + h * k3, eta)
        X = X + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        if k % rec_every == 0:
            out.append(X.copy())
    return out


def direct_sgd(d, eta, q0, S, nsteps, rec_every, gen):
    """Online SGD in R^d, teacher w*_n = sqrt(d) e_n; returns the order parameters every rec_every steps."""
    W = math.sqrt(q0) * gen.standard_normal((S, 2, d))
    out = []
    for mu in range(1, nsteps + 1):
        x = gen.standard_normal((S, d))
        lam = np.einsum('skd,sd->sk', W, x) / math.sqrt(d)
        err = erf(x[:, :2] / SQ2).sum(1) - erf(lam / SQ2).sum(1)
        D = CGP * np.exp(-0.5 * lam * lam) * err[:, None]
        W += (eta / math.sqrt(d)) * D[:, :, None] * x[:, None, :]
        if mu % rec_every == 0:
            Q = np.einsum('skd,sld->skl', W, W) / d
            R = W[:, :, :2] / math.sqrt(d)
            out.append(np.array([Q[:, 0, 0], Q[:, 0, 1], Q[:, 1, 1], R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1]]))
    return out


def main():
    t0 = time.time()
    eta, q0, S, tmax, rec_dt, h = 1.0, 0.25, 40, 10.0, 0.2, 0.02
    dvals = [125, 250, 500, 1000, 2000, 4000]
    gen = rk.rng(2026)
    rec = chain(dvals, S, eta, q0, tmax, rec_dt, gen)
    t1 = time.time()
    # ODE from each run's own initial state, and from m_sym(0), all at once
    X0 = np.concatenate([r[0] for r in rec] + [np.array([[q0], [0.0], [q0], [0.0], [0.0], [0.0], [0.0]])], axis=1)
    ode = np.array(rk4(X0, eta, tmax, h, int(round(rec_dt / h))))       # (nrec, 7, G*S+1)
    t2 = time.time()
    rows = []
    for g, d in enumerate(dvals):
        mo = ode[:, :, g * S:(g + 1) * S]
        msym = ode[:, :, -1:]
        Down = np.max(np.linalg.norm(rec[g] - mo, axis=1), axis=0)
        Dsym = np.max(np.linalg.norm(rec[g] - msym, axis=1), axis=0)
        m1, h1 = rk.mean_ci(math.sqrt(d) * Down)
        m2, h2 = rk.mean_ci(math.sqrt(d) * Dsym)
        rows.append((d, m1, h1, m2, h2))
        print("d=%5d: sqrt(d) E max|m - m_ode(m^0)| = %.3f +- %.3f;  sqrt(d) E max|m - m_sym| = %.3f +- %.3f;"
              "  E max|m - m_ode| = %.4f" % (d, m1, h1, m2, h2, np.mean(Down)))
    rk.write_dat(os.path.join(OUT, "15-sgd-rate.dat"), ["d", "own", "ownci", "sym", "symci"], rows)
    cown, csym = float(np.mean([r[1] for r in rows])), float(np.mean([r[3] for r in rows]))
    rk.write_dat(os.path.join(OUT, "15-sgd-rate-fit.dat"), ["d", "own", "sym"], [(90.0, cown, csym), (5500.0, cown, csym)])
    print("constants C of C/sqrt(d), averaged over d: %.3f (own start), %.3f (symmetric start)" % (cown, csym))
    print("chain %.0fs, ODE %.0fs" % (t1 - t0, t2 - t1))
    # validation of the chain against direct online SGD in R^d
    dd, Sd, Sc = 200, 200, 400
    direct = direct_sgd(dd, eta, q0, Sd, 8 * dd, 2 * dd, rk.rng(7))
    rc = chain([dd], Sc, eta, q0, 8.0, 2.0, rk.rng(8))[0]
    zmax = 0.0
    for i, t in enumerate((2, 4, 6, 8)):
        a, b = direct[i], rc[i + 1]
        se = np.sqrt(a.var(axis=1, ddof=1) / Sd + b.var(axis=1, ddof=1) / Sc)
        zmax = max(zmax, float(np.max(np.abs(a.mean(axis=1) - b.mean(axis=1)) / se)))
    print("chain vs direct SGD (d=200; %d and %d runs; 7 order parameters at t=2,4,6,8): max |difference|/SE = %.2f"
          % (Sd, Sc, zmax))
    print("runtime %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
