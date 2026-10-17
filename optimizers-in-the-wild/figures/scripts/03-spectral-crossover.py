"""Chapter 3 figure: when do spectral steps beat Frobenius steps? Matrix least squares, aligned case.

Loss L(W) = 1/2 tr((W - W*) A (W - W*)^T) on 128 x 128 matrices (the layer loss (1/2n)||WX - Y||_F^2 with
Y = W* X and A = X X^T / n). Inputs: A = Q diag(a) Q^T with a_i = i^(-1/2), i = 1..128, so the stable rank of
the inputs is tr A / a_1 = 21.2. Start: W_0 = 0, with W* = -U diag(e) Q^T, e_i = i^(-alpha), so the error is
aligned with the eigenbasis of A ("aligned case"); alpha controls how concentrated the error is on the stiff
directions. Q and U are Haar-random orthogonal matrices (seed 11).

Methods, both with the step sizes of the chapter's theorem (no tuning):
  F  gradient descent W <- W - G / L_F, with L_F = lambda_max(A) = a_1;
  S  spectral descent W <- W - (||G||_nuc / L_op) polar(G), with L_op = tr A (since d_out >= d_in).

Theory (curves): in the aligned case the iterates stay aligned (Proposition prop:spectral:aligned):
GD multiplies e_i by (1 - a_i / a_1), spectral descent is sign descent e <- e - c sign(e) with
c = sum_i a_i |e_i| / sum_i a_i. Closed forms for one step; scalar recursions for many steps.
  onestep_S   exact relative decrease of one spectral step (equal to its guarantee ||G||_nuc^2/(2 L_op L));
  onestep_F   exact relative decrease of one GD step;  guar_F  its guarantee ||G||_F^2/(2 L_F L);
  iters_S, iters_F   steps to reach 1e-2 of the initial loss;
  R           the theorem's ratio of guarantees, nuclear rank of G_0 over stable rank of the inputs.
Simulation (points): the same methods iterated on the dense 128 x 128 matrices, with the polar factor and the
nuclear norm from numpy's SVD; columns sim_onestep_S, sim_onestep_F, sim_iters_S, sim_iters_F.

Output: figures/data/03-spectral-crossover-theory.dat (alpha, R, onestep_S, onestep_F, guar_F, iters_S, iters_F)
        figures/data/03-spectral-crossover-sim.dat    (alpha, sim_onestep_S, sim_onestep_F, sim_iters_S, sim_iters_F)
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")  # the machine is shared: at most two BLAS threads

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
N = 128
INPUT_EXP = 0.5   # input spectrum a_i = i^(-1/2)
TARGET = 1e-2
a = np.arange(1, N + 1, dtype=float) ** (-INPUT_EXP)
LF, LOP = a[0], a.sum()


def scalar_theory(alpha, Tmax=100000):
    e0 = np.arange(1, N + 1, dtype=float) ** (-alpha)
    loss = lambda e: 0.5 * np.sum(a * e * e)
    L0 = loss(e0)
    g = a * e0
    R = (np.abs(g).sum() ** 2 / np.sum(g * g)) / (LOP / LF)
    c0 = np.sum(a * np.abs(e0)) / LOP
    one_S = 1 - loss(e0 - c0 * np.sign(e0)) / L0
    one_F = 1 - loss(e0 * (1 - a / LF)) / L0
    guar_F = np.sum(g * g) / (2 * LF) / L0
    guar_S = np.abs(g).sum() ** 2 / (2 * LOP) / L0
    assert abs(guar_S - one_S) < 1e-12
    e, tF = e0.copy(), None
    for t in range(Tmax):
        if loss(e) <= TARGET * L0:
            tF = t
            break
        e = e * (1 - a / LF)
    e, tS = e0.copy(), None
    for t in range(Tmax):
        if loss(e) <= TARGET * L0:
            tS = t
            break
        e = e - (np.sum(a * np.abs(e)) / LOP) * np.sign(e)
    return R, one_S, one_F, guar_F, tS, tF


def haar(k, rng):
    Q, R = np.linalg.qr(rng.standard_normal((k, k)))
    return Q * np.sign(np.diag(R))


def matrix_sim(alpha, Q, U, Tmax=100000):
    A = (Q * a) @ Q.T
    E0 = (U * np.arange(1, N + 1, dtype=float) ** (-alpha)) @ Q.T   # E = W - W*
    loss = lambda E: 0.5 * np.sum((E @ A) * E)
    L0 = loss(E0)

    def spectral_step(E):
        Us, s, Vt = np.linalg.svd(E @ A)
        return E - (s.sum() / LOP) * (Us @ Vt)

    one_S = 1 - loss(spectral_step(E0)) / L0
    one_F = 1 - loss(E0 - E0 @ A / LF) / L0
    E, tF = E0.copy(), None
    for t in range(Tmax):
        if loss(E) <= TARGET * L0:
            tF = t
            break
        E = E - E @ A / LF
    E, tS = E0.copy(), None
    for t in range(Tmax):
        if loss(E) <= TARGET * L0:
            tS = t
            break
        E = spectral_step(E)
    return one_S, one_F, tS, tF


def crossing(x, y):
    """first x where y changes sign (linear interpolation)."""
    for i in range(len(x) - 1):
        if y[i] == 0:
            return x[i]
        if y[i] * y[i + 1] < 0:
            return x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
    return float("nan")


def main():
    alphas = np.round(np.arange(0, 1.5001, 0.0125), 4)
    rows = [(b,) + scalar_theory(b) for b in alphas]
    with open(os.path.join(DATA, "03-spectral-crossover-theory.dat"), "w") as f:
        f.write("alpha R onestep_S onestep_F guar_F iters_S iters_F\n")
        for r in rows:
            f.write("%.4f %.6e %.6e %.6e %.6e %d %d\n" % r)
    rng = np.random.default_rng(11)
    Q, U = haar(N, rng), haar(N, rng)
    sims = []
    for b in np.round(np.arange(0, 1.5001, 0.1), 4):
        sims.append((b,) + matrix_sim(b, Q, U))
    with open(os.path.join(DATA, "03-spectral-crossover-sim.dat"), "w") as f:
        f.write("alpha sim_onestep_S sim_onestep_F sim_iters_S sim_iters_F\n")
        for r in sims:
            f.write("%.4f %.6e %.6e %d %d\n" % r)
    # agreement and crossovers
    th = {r[0]: r for r in rows}
    dev = max(max(abs(s[1] - th[s[0]][2]), abs(s[2] - th[s[0]][3])) for s in sims)
    mism = [(s[0], s[3], th[s[0]][5], s[4], th[s[0]][6]) for s in sims
            if s[3] != th[s[0]][5] or s[4] != th[s[0]][6]]
    print("stable rank of inputs tr A / a_1 = %.3f" % (LOP / LF))
    print("max |one-step decrease: matrix - theory| = %.1e;  iteration-count mismatches: %s" % (dev, mism))
    B = np.array([r[0] for r in rows])
    print("predicted crossover (R = 1):            alpha = %.3f" % crossing(B, np.array([r[1] - 1 for r in rows])))
    print("one-step crossover (actual decreases):  alpha = %.3f" % crossing(B, np.array([r[2] - r[3] for r in rows])))
    print("multi-step crossover (iterations):      alpha = %.3f" % crossing(B, np.array([r[6] - r[5] for r in rows], float)))
    for r in rows[::8]:
        print("  alpha %.3f  R %.3f  one-step S %.3f F %.3f (guar F %.3f)  iters S %d F %d" % r)


if __name__ == "__main__":
    main()
