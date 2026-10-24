"""Chapter 15 (Online SGD): the Saad--Solla ODE for the soft committee machine.

Model (OUTLINE.md, Chapter 15): inputs x ~ N(0, I_d); student sigma = sum_k g(w_k.x/sqrt d), teacher
phi = sum_n g(w*_n.x/sqrt d), g(z) = erf(z/sqrt 2); loss (sigma - phi)^2/2; online SGD
w_k <- w_k + (eta/sqrt d) g'(lambda_k)(phi - sigma) x; order parameters Q = w w^T/d, R = w w*^T/d,
T = w* w*^T/d; time t = (number of samples)/d.

This script
  1. checks the closed forms I2, I3, I4 against tensor Gauss--Hermite quadrature;
  2. computes the symmetric fixed points for K = M = 2 (and K = 3), T = I: the eta -> 0 plateau
     (closed form), and at finite eta the two symmetric fixed points (identical students, A; the
     attracting one, B), with the eigenvalues of the linearization;
  3. checks the closed form of the eta -> 0 escape rate 2K/(pi sqrt(2K-1) (2K+1)^{3/2});
  4. integrates the ODE from the random-start limit m_sym(0) = (q0 I, 0) (eta = 1, q0 = 1/4),
     checks that it stays in the symmetric subspace, and from slightly asymmetric starts
     (escape time against ln(1/delta0));
  5. computes numbers for the exercises (K = M = 1 critical learning rate; linear network).
Writes figures/data/15-sgd-ode-curve.dat and 15-sgd-ode-escape.dat. Runtime: about 30 s.
"""
import math
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

PI = math.pi
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def g(z):
    from scipy.special import erf
    return erf(z / math.sqrt(2))


def gp(z):
    return math.sqrt(2 / PI) * np.exp(-z * z / 2)


# ------------------------------------------------------------------------------------------
# closed forms (all index tuples of a covariance matrix C at once)
# ------------------------------------------------------------------------------------------
def I2mat(C):
    d = np.diag(C)
    return (2 / PI) * np.arcsin(C / np.sqrt(np.outer(1 + d, 1 + d)))


def I3ten(C):
    """I3[a,b,c] = E[g'(z_a) z_b g(z_c)]."""
    d = np.diag(C)
    c11, c33 = d[:, None, None], d[None, None, :]
    c12, c13, c23 = C[:, :, None], C[:, None, :], C[None, :, :]
    lam3 = (1 + c11) * (1 + c33) - c13 ** 2
    return (2 / PI) / np.sqrt(lam3) * (c23 * (1 + c11) - c12 * c13) / (1 + c11)


def I4ten(C):
    """I4[a,b,c,e] = E[g'(z_a) g'(z_b) g(z_c) g(z_e)]."""
    d = np.diag(C)
    c11, c22 = d[:, None, None, None], d[None, :, None, None]
    c33, c44 = d[None, None, :, None], d[None, None, None, :]
    c12, c13, c14 = C[:, :, None, None], C[:, None, :, None], C[:, None, None, :]
    c23, c24, c34 = C[None, :, :, None], C[None, :, None, :], C[None, None, :, :]
    lam4 = (1 + c11) * (1 + c22) - c12 ** 2
    lam0 = lam4 * c34 - c23 * c24 * (1 + c11) - c13 * c14 * (1 + c22) + c12 * c13 * c24 + c12 * c14 * c23
    lam1 = lam4 * (1 + c33) - c23 ** 2 * (1 + c11) - c13 ** 2 * (1 + c22) + 2 * c12 * c13 * c23
    lam2 = lam4 * (1 + c44) - c24 ** 2 * (1 + c11) - c14 ** 2 * (1 + c22) + 2 * c12 * c14 * c24
    return 4 / (PI ** 2 * np.sqrt(lam4)) * np.arcsin(lam0 / np.sqrt(lam1 * lam2))


class SCM:
    """Saad--Solla ODE: dR_in = eta E[D_i nu_n], dQ_ik = eta E[D_i lam_k + D_k lam_i] + eta2 E[D_i D_k]."""

    def __init__(self, K, M, T, eta, eta2=None):
        self.K, self.M, self.T, self.eta = K, M, np.array(T, float), eta
        self.eta2 = eta ** 2 if eta2 is None else eta2
        self.s = np.concatenate([np.ones(K), -np.ones(M)])
        self.iu = np.triu_indices(K)

    def cov(self, Q, R):
        K = self.K
        return np.block([[Q, R], [R.T, self.T]])

    def eg(self, Q, R):
        return 0.5 * self.s @ I2mat(self.cov(Q, R)) @ self.s

    def drift(self, Q, R):
        K = self.K
        C = self.cov(Q, R)
        EDz = -np.einsum('ibc,c->ib', I3ten(C)[:K], self.s)
        EDD = np.einsum('ikce,c,e->ik', I4ten(C)[:K, :K], self.s, self.s)
        return self.eta * (EDz[:, :K] + EDz[:, :K].T) + self.eta2 * EDD, self.eta * EDz[:, K:]

    def pack(self, Q, R):
        return np.concatenate([Q[self.iu], R.ravel()])

    def unpack(self, y):
        nq = len(self.iu[0])
        Q = np.zeros((self.K, self.K))
        Q[self.iu] = y[:nq]
        Q = Q + Q.T - np.diag(np.diag(Q))
        return Q, y[nq:].reshape(self.K, self.M)

    def f(self, t, y):
        return self.pack(*self.drift(*self.unpack(y)))

    def jac(self, y0, h=1e-6):
        n = len(y0)
        J = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            J[:, j] = (self.f(0, y0 + e) - self.f(0, y0 - e)) / (2 * h)
        return J


def check_closed_forms():
    """Closed forms against tensor Gauss--Hermite quadrature (44 nodes per axis)."""
    gen = rk.rng(5)
    A = gen.normal(size=(4, 4)) * 0.45
    C = A @ A.T + 0.1 * np.eye(4)
    x, w = np.polynomial.hermite_e.hermegauss(44)
    w = w / math.sqrt(2 * PI)
    X = np.stack(np.meshgrid(x, x, x, x, indexing='ij'), 0).reshape(4, -1)
    W = np.einsum('i,j,k,l->ijkl', w, w, w, w).ravel()
    u, v, z3, z4 = np.linalg.cholesky(C) @ X
    I2, I3, I4 = I2mat(C), I3ten(C), I4ten(C)
    errs = [abs(np.sum(W * g(u) * g(v)) - I2[0, 1]),
            abs(np.sum(W * gp(u) * v * g(z3)) - I3[0, 1, 2]),
            abs(np.sum(W * gp(u) * v * g(u)) - I3[0, 1, 0]),
            abs(np.sum(W * gp(u) * gp(v) * g(z3) * g(z4)) - I4[0, 1, 2, 3]),
            abs(np.sum(W * gp(u) * gp(u) * g(v) * g(v)) - I4[0, 0, 1, 1]),
            abs(np.sum(W * gp(u) * gp(v) * g(u) * g(v)) - I4[0, 1, 0, 1])]
    print("1. closed forms vs quadrature: max abs error %.1e (I2, I3, I4, including repeated indices)" % max(errs))
    # one-dimensional check of the degenerate I4 with the kit's quadrature
    c = 1.0
    q = rk.gauss(lambda z: gp(math.sqrt(c) * z) ** 2 * g(math.sqrt(c) * z) ** 2, 151)
    print("   E[g'(u)^2 g(u)^2], u ~ N(0,1): quadrature %.12f, closed form %.12f" % (q, I4ten(np.array([[c]]))[0, 0, 0, 0]))


def build_sym(K, xx):
    q, c, r, s = xx
    return np.full((K, K), c) + (q - c) * np.eye(K), np.full((K, K), s) + (r - s) * np.eye(K)


def sym_fixed_point(K, eta, guess):
    model = SCM(K, K, np.eye(K), eta)

    def F(xx):
        dQ, dR = model.drift(*build_sym(K, xx))
        return [dQ[0, 0], dQ[0, 1], dR[0, 0], dR[0, 1]]
    xx = fsolve(F, guess, xtol=1e-13)
    Q, R = build_sym(K, xx)
    y = model.pack(Q, R)
    ev = np.linalg.eigvals(model.jac(y))
    return xx, model.eg(Q, R), np.sort(ev.real)[::-1], float(np.max(np.abs(F(xx))))


def main():
    t_start = time.time()
    check_closed_forms()

    # ---------------- 2. symmetric fixed points ----------------
    print("2. symmetric fixed points, K = M, T = I")
    table = []
    for K in (2, 3):
        q0, s0 = 1 / (2 * K - 1), 1 / math.sqrt(K * (2 * K - 1))
        eps0 = (K / PI) * (PI / 6 - K * math.asin(1 / (2 * K)))
        lam1 = 2 * K / (PI * math.sqrt(2 * K - 1) * (2 * K + 1) ** 1.5)
        print("   K=%d eta->0: Q=C=%.6f R=S=%.6f eps*=%.6f, escape rate/eta = 2K/(pi sqrt(2K-1)(2K+1)^1.5) = %.7f"
              % (K, q0, s0, eps0, lam1))
        # numerical check of the eta -> 0 limits with the first-order ODE (eta2 = 0, eta = 1)
        model0 = SCM(K, K, np.eye(K), 1.0, eta2=0.0)
        Q, R = build_sym(K, [q0, q0, s0, s0])
        dQ, dR = model0.drift(Q, R)
        ev0 = np.sort(np.linalg.eigvals(model0.jac(model0.pack(Q, R))).real)[::-1]
        print("     first-order drift at the plateau: max |.| = %.1e; largest eigenvalue %.7f; eg %.6f"
              % (max(np.max(np.abs(dQ)), np.max(np.abs(dR))), ev0[0], model0.eg(Q, R)))
        if K == 2:
            table.append(("0", q0, q0, s0, eps0, lam1, "A=B"))
            for eta in (0.1, 0.5, 1.0):
                for name, guess in (("A", [0.34, 0.34, 0.41, 0.41]), ("B", [0.40, 0.32, 0.42, 0.42])):
                    xx, eps, ev, res = sym_fixed_point(K, eta, guess)
                    npos = int(np.sum(ev > 1e-9))
                    print("   K=2 eta=%.2f fixed point %s: (Q,C,R,S)=(%.6f, %.6f, %.6f, %.6f) eps=%.7f residual %.0e;"
                          " %d positive eigenvalues, largest %.6f (= %.5f eta); next %.6f"
                          % (eta, name, *xx, eps, res, npos, ev[0], ev[0] / eta, ev[1]))
                    table.append(("%.1f" % eta, *xx[[0, 1, 2]], eps, ev[0], name))
        else:
            for eta in (0.1, 0.5):
                xx, eps, ev, res = sym_fixed_point(K, eta, [0.2, 0.2, 0.26, 0.26])
                print("   K=3 eta=%.2f identical-student fixed point: eps=%.6f, largest eigenvalue %.6f" % (eta, eps, ev[0]))

    # ---------------- 4. the ODE from the random-start limit ----------------
    # (integrated to t = 150 only: along the unstable specialization direction, rounding errors of
    # order 1e-16 grow like exp(0.059 t) and would leave the invariant subspace near t = 600)
    eta, q0 = 1.0, 0.25
    model = SCM(2, 2, np.eye(2), eta)
    y0 = model.pack(q0 * np.eye(2), np.zeros((2, 2)))
    sol = solve_ivp(model.f, (0, 150), y0, method='LSODA', rtol=1e-11, atol=1e-13, dense_output=True)
    ts = np.concatenate([np.linspace(0, 10, 101), np.linspace(10.5, 150, 280)])
    rows = []
    asym = 0.0
    for t in ts:
        Q, R = model.unpack(sol.sol(t))
        asym = max(asym, abs(Q[0, 0] - Q[1, 1]), np.max(np.abs(R - R[0, 0])))
        rows.append((t, model.eg(Q, R)))
    rk.write_dat(os.path.join(OUT, "15-sgd-ode-curve.dat"), ["t", "eg"], rows)
    xxB, epsB = sym_fixed_point(2, eta, [0.40, 0.32, 0.42, 0.42])[:2]
    rk.write_dat(os.path.join(OUT, "15-sgd-ode-plateau.dat"), ["t", "eg"], [(0.0, epsB), (160.0, epsB)])
    egt = {t: model.eg(*model.unpack(sol.sol(t))) for t in (2, 5, 10, 20, 40, 80, 150)}
    print("4. symmetric ODE (eta=1, q0=1/4): eg(0)=%.4f; " % rows[0][1]
          + " ".join("eg(%g)=%.5f" % (t, e) for t, e in egt.items())
          + "; max deviation from the symmetric subspace on [0,150]: %.1e" % asym)
    Q150, R150 = model.unpack(sol.sol(150))
    print("   state at t=150: Q11=%.5f Q12=%.5f R=%.5f (fixed point B: 0.435085 0.310213 0.426138)"
          % (Q150[0, 0], Q150[0, 1], R150[0, 0]))
    # escape from slightly asymmetric starts R(0) = delta0 * I (which has a component along the
    # specialization direction E = I - J/2)
    lamB = sym_fixed_point(2, eta, [0.40, 0.32, 0.42, 0.42])[2][0]
    erows = []
    def below_half(t, y):
        return model.eg(*model.unpack(y)) - 0.020657 / 2
    below_half.terminal = True
    below_half.direction = -1
    for delta0 in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5):
        yy = model.pack(q0 * np.eye(2), delta0 * np.eye(2))
        s2 = solve_ivp(model.f, (0, 500), yy, method='LSODA', rtol=1e-11, atol=1e-14, events=below_half)
        tesc = float(s2.t_events[0][0])
        erows.append((math.log(1 / delta0), tesc, tesc - math.log(1 / delta0) / lamB))
        print("   delta0=%.0e: t_esc=%.2f, t_esc - ln(1/delta0)/lambda = %.2f" % (delta0, tesc, erows[-1][2]))
    rk.write_dat(os.path.join(OUT, "15-sgd-ode-escape.dat"), ["lninv", "tesc", "shift"], erows)
    print("   escape rate at fixed point B (eta=1): %.6f; slope 1/(2 lambda) = %.3f; spread 1.1107/lambda = %.2f"
          % (lamB, 1 / (2 * lamB), 1.1107 / lamB))
    print("   runtime %.1fs" % (time.time() - t_start))


if __name__ == "__main__":
    main()
