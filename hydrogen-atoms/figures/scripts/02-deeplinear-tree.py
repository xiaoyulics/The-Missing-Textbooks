"""Chapter 2 (deeplinear): stage-like learning of hierarchical data by a two-layer linear network.

Data: P = 8 items at the leaves of a binary tree of depth 3; input x = sqrt(P) e_p (so that
E[x x^T] = I), output y in {0,1}^15 with one coordinate per tree node (1 if the item lies below
that node). Then Sigma_yx = Y / sqrt(P), with singular values sqrt(15/8), sqrt(7/8), sqrt(3/8) (x2),
sqrt(1/8) (x4): one mode per level of the tree.
Simulation: full-batch gradient descent (step eta) on (1/2)||Sigma_yx - W2 W1||_F^2 from a small
*random* Gaussian initialization (entries N(0, init^2)), width m = 64, seed 0.
Theory: the logistic solution of the aligned, balanced case, a(t) = s / (1 + (s/a0 - 1) e^{-2 s t}),
with the effective initial strengths a0 read off from the initialization (the eigenvalues of
xi^T xi, xi = (W1(0) V + W2(0)^T U) / 2, within each level).
Writes figures/data/02-deeplinear-tree.dat. Runs in a few seconds.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "02-deeplinear-tree.dat")
rng = np.random.default_rng(0)
P, m, init, eta, T = 8, 64, 0.003, 0.01, 20.0

rows = []
for lev in range(4):
    size = P >> lev
    for j in range(1 << lev):
        rows.append([1.0 if j * size <= p < (j + 1) * size else 0.0 for p in range(P)])
Y = np.array(rows)
Sig = Y / np.sqrt(P)
U, s, Vt = np.linalg.svd(Sig, full_matrices=False)
V = Vt.T
levels = [[0], [1], [2, 3], [4, 5, 6, 7]]
sv = np.array([s[l[0]] for l in levels])
assert np.allclose(sv ** 2 * P, [15, 7, 3, 1])

W1 = rng.standard_normal((m, P)) * init
W2 = rng.standard_normal((Y.shape[0], m)) * init
xi = (W1 @ V + W2.T @ U) / 2
eff = [np.linalg.eigvalsh(xi[:, l].T @ xi[:, l]) for l in levels]
print("effective initial strengths:", [np.round(e, 6) for e in eff])


def theory(t):
    a = [s_ / (1 + (s_ / e - 1) * np.exp(-2 * s_ * t)) for s_, e in zip(sv, eff)]
    frac = [np.mean(ai) / s_ for ai, s_ in zip(a, sv)]
    loss = 0.5 * sum(np.sum((s_ - ai) ** 2) for ai, s_ in zip(a, sv))
    return frac, loss


steps = int(round(T / eta))
out = []
for k in range(steps + 1):
    W = W2 @ W1
    if k % 20 == 0:
        t = k * eta
        M = U.T @ W @ V
        frac = [np.trace(M[np.ix_(l, l)]) / (len(l) * sv[i]) for i, l in enumerate(levels)]
        loss = 0.5 * np.sum((Sig - W) ** 2)
        tf, tl = theory(t)
        out.append([t] + frac + tf + [loss, tl])
    G = W - Sig
    W1, W2 = W1 - eta * W2.T @ G, W2 - eta * G @ W1.T

out = np.array(out)
err = np.abs(out[:, 1:5] - out[:, 5:9]).max()
print("max |simulated - predicted| fraction learned:", round(err, 4))
print("final loss", out[-1, 9])
for i in range(4):
    print("level", i, "t_half predicted", np.log(sv[i] / np.mean(eff[i]) - 1) / (2 * sv[i]))
header = "t sim1 sim2 sim3 sim4 th1 th2 th3 th4 losssim lossth"
np.savetxt(OUT, out, fmt="%.6g", header=header, comments="")
