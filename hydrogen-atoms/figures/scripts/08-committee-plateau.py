"""Plateau and specialization in the soft committee machine (Chapter 8, fig:committee:plateau).

Teacher and student: two erf units each, g(z) = erf(z/sqrt 2), orthonormal teacher (T = I),
online SGD on the square loss with step eta/d, eta = 0.5.
Initial order parameters (identical for the ODE and every SGD run):
    R(0) = 0.003 I (a slight preference of student j for teacher j), residual Gram Q - R R^T = 0.01 I.
Theory: the ODE of thm:committee:ode (with its eta^2 term), integrated with LSODA.
Simulation: the exact Markov chain of the order parameters (rem:committee:chain), which has the law of
d-dimensional SGD, for d = 300 and d = 3000, 16 runs each, seed 8; medians over runs.
Output: figures/data/08-committee-plateau-ode.dat   (t eg R11 R12 R21 R22 Q11 Q12)
        figures/data/08-committee-plateau-sgd.dat   (t eg300 eg3000 R11_3000 R12_3000)
Runtime: about 60 seconds.
"""
import os
import numpy as np
from scipy.special import erf
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
ETA, Q0, R0, TMAX, SEED = 0.5, 0.01, 0.003, 300.0, 8
SQ, R2 = np.sqrt(2 / np.pi), np.sqrt(0.5)


# ---------- closed-form Gaussian integrals (lem:committee:erf) ----------
def I3(C, u, v, w):
    return 2 / np.pi * (C[v, w] * (1 + C[u, u]) - C[u, v] * C[u, w]) / (
        (1 + C[u, u]) * np.sqrt((1 + C[u, u]) * (1 + C[w, w]) - C[u, w] ** 2))


def I4(C, u, v, w, z):
    S, W = [u, v], [w, z]
    A = np.eye(2) + C[np.ix_(S, S)]
    Ct = C[np.ix_(W, W)] - C[np.ix_(W, S)] @ np.linalg.solve(A, C[np.ix_(S, W)])
    return (2 / np.pi) ** 2 / np.sqrt(np.linalg.det(A)) * np.arcsin(
        Ct[0, 1] / np.sqrt((1 + Ct[0, 0]) * (1 + Ct[1, 1])))


def rhs(t, z, eta):
    Q = np.array([[z[0], z[2]], [z[2], z[1]]]); R = np.asarray(z[3:]).reshape(2, 2)
    C = np.block([[Q, R], [R.T, np.eye(2)]]); st, te = [0, 1], [2, 3]
    def Edx(j, v):
        return sum(I3(C, j, v, k) for k in te) - sum(I3(C, j, v, l) for l in st)
    dR = [eta * Edx(j, te[k]) for j in st for k in range(2)]
    dQ = {}
    for j, l in [(0, 0), (1, 1), (0, 1)]:
        s2 = sum(I4(C, j, l, a, b) for a in te for b in te) - 2 * sum(I4(C, j, l, a, b) for a in te for b in st) \
            + sum(I4(C, j, l, a, b) for a in st for b in st)
        dQ[j, l] = eta * (Edx(j, l) + Edx(l, j)) + eta ** 2 * s2
    return [dQ[0, 0], dQ[1, 1], dQ[0, 1]] + dR


def egv(Q, R):  # generalization error for T = I, vectorized over the first axis
    dq = np.diagonal(Q, axis1=-2, axis2=-1)
    s = np.pi / 3 + np.arcsin(dq[..., 0] / (1 + dq[..., 0])) + np.arcsin(dq[..., 1] / (1 + dq[..., 1]))
    s = s + 2 * np.arcsin(Q[..., 0, 1] / np.sqrt((1 + dq[..., 0]) * (1 + dq[..., 1])))
    s = s - 2 * np.arcsin(R / np.sqrt(2 * (1 + dq[..., :, None]))).sum((-2, -1))
    return s / np.pi


# ---------- exact chain of the order parameters (m = mstar = 2, T = I) ----------
def chain(dvec, eta, R, Gr, smax, rng, snap_every, block=2048):
    n = R.shape[0]; R = R.copy(); Gr = Gr.copy(); dvec = np.asarray(dvec, float)
    k = eta / dvec; out = []; s = 0
    while True:
        NU = rng.standard_normal((block, n, 2)); XI = rng.standard_normal((block, n, 2))
        CH = rng.chisquare(dvec - 4, (block, n))
        for b in range(block):
            if s % snap_every == 0:
                out.append((Gr + np.einsum('njk,nlk->njl', R, R), R.copy()))
            if s >= smax:
                return out
            nu, xi = NU[b], XI[b]
            a = np.sqrt(Gr[:, 0, 0]); bb = Gr[:, 0, 1] / a; c = np.sqrt(np.maximum(Gr[:, 1, 1] - bb * bb, 0.0))
            r0 = a * xi[:, 0]; r1 = bb * xi[:, 0] + c * xi[:, 1]
            l0 = R[:, 0, 0] * nu[:, 0] + R[:, 0, 1] * nu[:, 1] + r0
            l1 = R[:, 1, 0] * nu[:, 0] + R[:, 1, 1] * nu[:, 1] + r1
            gg = erf(np.stack([nu[:, 0], nu[:, 1], l0, l1]) * R2)
            err = gg[0] + gg[1] - gg[2] - gg[3]
            d0 = SQ * np.exp(-0.5 * l0 * l0) * err * k; d1 = SQ * np.exp(-0.5 * l1 * l1) * err * k
            x2 = xi[:, 0] ** 2 + xi[:, 1] ** 2 + CH[b]
            R[:, 0, :] += d0[:, None] * nu; R[:, 1, :] += d1[:, None] * nu
            Gr[:, 0, 0] += 2 * d0 * r0 + x2 * d0 * d0
            Gr[:, 1, 1] += 2 * d1 * r1 + x2 * d1 * d1
            g01 = d0 * r1 + d1 * r0 + x2 * d0 * d1
            Gr[:, 0, 1] += g01; Gr[:, 1, 0] += g01
            s += 1


def main():
    ts = np.linspace(0, TMAX, 301)
    z0 = [Q0 + R0 ** 2, Q0 + R0 ** 2, 0.0, R0, 0.0, 0.0, R0]
    sol = solve_ivp(rhs, [0, TMAX], z0, args=(ETA,), t_eval=ts, rtol=1e-9, atol=1e-12, method='LSODA')
    y = sol.y
    Q = np.zeros((len(ts), 2, 2)); Q[:, 0, 0] = y[0]; Q[:, 1, 1] = y[1]; Q[:, 0, 1] = Q[:, 1, 0] = y[2]
    R = y[3:].T.reshape(-1, 2, 2)
    e = egv(Q, R)
    with open(os.path.join(DATA, "08-committee-plateau-ode.dat"), "w") as f:
        f.write("t eg R11 R12 R21 R22 Q11 Q12\n")
        for i, t in enumerate(ts):
            f.write(f"{t:.1f} {e[i]:.6e} {R[i,0,0]:.5f} {R[i,0,1]:.5f} {R[i,1,0]:.5f} {R[i,1,1]:.5f} {Q[i,0,0]:.5f} {Q[i,0,1]:.5f}\n")
    rng = np.random.default_rng(SEED)
    n = 16
    res = {}
    for d in (300, 3000):
        Rin = np.tile(R0 * np.eye(2), (n, 1, 1)); Gin = np.tile(Q0 * np.eye(2), (n, 1, 1))
        every = d * 5
        out = chain(np.full(n, d), ETA, Rin, Gin, int(TMAX * d), rng, every)
        res[d] = out
    tt = np.arange(len(res[3000])) * 5.0
    with open(os.path.join(DATA, "08-committee-plateau-sgd.dat"), "w") as f:
        f.write("t eg300 eg3000 R11_3000 R12_3000\n")
        for i, t in enumerate(tt):
            e3 = np.median(egv(*res[300][i])); e30 = np.median(egv(*res[3000][i]))
            Qa, Ra = res[3000][i]
            f.write(f"{t:.1f} {e3:.6e} {e30:.6e} {np.median(Ra[:,0,0]):.5f} {np.median(Ra[:,0,1]):.5f}\n")


if __name__ == "__main__":
    main()
