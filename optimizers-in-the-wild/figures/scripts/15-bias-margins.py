"""Figure 15.2 and Table 15.1 of Optimizers in the Wild (Chapter 15, key `bias`).

Normalized steepest descent on the exponential loss of separable data, in four norms:
    l2 (normalized gradient descent), l-infinity (sign descent), l1 (greedy coordinate
    descent), and the spectral norm on a multiclass linear classifier (orthogonalized descent,
    Muon without momentum).
For each norm, the exact maximum margin gamma is computed by quadratic programming (l2),
linear programming (l1, l-infinity) or, for the spectral norm, by Kelley's cutting-plane method
on the dual problem min_p ||sum_i p_i Z_i||_nuc, which brackets gamma between a primal-feasible
margin and a dual value (bracket width printed; below 1e-7).
Each optimizer runs T = 10^5 steps from zero with two step-size schedules,
    decaying  eta_t = 1/(R sqrt(t+1))       constant  eta_t = 0.05/R,
where R = max_i ||z_i||_* (dual norm). Recorded: the normalized margin min_i <theta_t, z_i>/||theta_t||
at 41 log-spaced steps, and the lower bound of the margin theorem,
    gamma - (log n + R^2/2 sum_{s<t} eta_s^2) / sum_{s<t} eta_s.
Output: figures/data/15-bias-margins.dat, figures/data/15-bias-margins-gamma.dat; the cross-norm
margins of Table 15.1 are printed. numpy and scipy only; deterministic seeds; about 15 s on a laptop CPU.
"""
import os
import numpy as np
from scipy.optimize import linprog, minimize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
T = 100_000
REC = np.unique(np.logspace(0, 5, 41).astype(int))


# ---------------------------------------------------------------- max-margin solvers
def mm_linf(Z):
    n, d = Z.shape
    r = linprog(np.r_[np.zeros(d), -1.0], A_ub=np.hstack([-Z, np.ones((n, 1))]), b_ub=np.zeros(n),
                bounds=[(-1, 1)] * d + [(None, None)], method="highs")
    return -r.fun, r.x[:d]


def mm_l1(Z):
    n, d = Z.shape
    A = np.vstack([np.hstack([-Z, Z, np.ones((n, 1))]), np.r_[np.ones(2 * d), 0.0][None, :]])
    r = linprog(np.r_[np.zeros(2 * d), -1.0], A_ub=A, b_ub=np.r_[np.zeros(n), 1.0],
                bounds=[(0, None)] * (2 * d) + [(None, None)], method="highs")
    return -r.fun, r.x[:d] - r.x[d:2 * d]


def mm_l2(Z):
    w0 = mm_linf(Z)[1]
    w0 = w0 / np.min(Z @ w0)
    r = minimize(lambda w: 0.5 * w @ w, w0, jac=lambda w: w, method="SLSQP",
                 constraints=[{"type": "ineq", "fun": lambda w: Z @ w - 1, "jac": lambda w: Z}],
                 options={"ftol": 1e-15, "maxiter": 2000})
    S = np.where(Z @ r.x < 1 + 1e-6)[0]
    a = np.linalg.lstsq(Z[S] @ Z[S].T, np.ones(len(S)), rcond=None)[0]
    w = Z[S].T @ a
    assert np.all(a > -1e-12) and np.min(Z @ w) > 1 - 1e-10   # KKT conditions hold exactly
    return 1 / np.linalg.norm(w), w / np.linalg.norm(w)


def polar(G):
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    r = int(np.sum(s > 1e-12 * s[0]))
    return U[:, :r] @ Vt[:r]


def opn(G):
    return np.linalg.svd(G, compute_uv=False)[0]


def nucn(G):
    return np.linalg.svd(G, compute_uv=False).sum()


def mm_op(Zs, iters=2000, tol=1e-7):
    """Kelley's cutting planes for min_p ||sum p_i Z_i||_nuc; returns (lower, upper, W)."""
    N = len(Zs)
    Zflat = Zs.reshape(N, -1)
    p = np.ones(N) / N
    cuts, rows, up, low, Wb = [], [], np.inf, -np.inf, None
    for _ in range(iters):
        M = np.tensordot(p, Zs, axes=1)
        up = min(up, nucn(M))
        cuts.append(polar(M))
        rows.append(Zflat @ cuts[-1].ravel())          # <W_k, Z_i> for the new cut
        C = np.array(rows)
        r = linprog(np.r_[np.zeros(N), 1.0], A_ub=np.hstack([C, -np.ones((len(cuts), 1))]),
                    b_ub=np.zeros(len(cuts)), A_eq=np.r_[np.ones(N), 0.0][None, :], b_eq=[1.0],
                    bounds=[(0, None)] * N + [(None, None)], method="highs")
        p = r.x[:N]
        lam = np.maximum(-r.ineqlin.marginals, 0)
        W = np.tensordot(lam / lam.sum(), np.array(cuts), axes=1)
        W = W / opn(W)
        m = np.min(Zflat @ W.ravel())
        if m > low:
            low, Wb = m, W
        if up - low < tol:
            break
    return low, up, Wb


# ---------------------------------------------------------------- data
rng = np.random.default_rng(11)
d, n = 5, 16
wt = rng.standard_normal(d)
X = rng.standard_normal((400, d))
X /= np.linalg.norm(X, axis=1, keepdims=True)
y = np.sign(X @ wt)
keep = np.abs(X @ wt) / np.linalg.norm(wt) > 0.15
X, y = X[keep][:n], y[keep][:n]
Z = y[:, None] * X                                     # signed examples, ||z_i||_2 = 1

rng = np.random.default_rng(3)
K, dm, nm = 3, 4, 12
Wt = rng.standard_normal((K, dm))
Xm = rng.standard_normal((200, dm))
Xm /= np.linalg.norm(Xm, axis=1, keepdims=True)
sc = Xm @ Wt.T
ym = sc.argmax(1)
srt = np.sort(sc, 1)
keep = srt[:, -1] - srt[:, -2] > 0.25
Xm, ym = Xm[keep][:nm], ym[keep][:nm]
Zs = []                                                 # Z_ic = (e_y - e_c) x^T / sqrt 2, nuclear norm 1
for i in range(nm):
    for c in range(K):
        if c != ym[i]:
            e = np.zeros(K); e[ym[i]] = 1; e[c] = -1
            Zs.append(np.outer(e, Xm[i]) / np.sqrt(2))
Zs = np.array(Zs)
N = len(Zs)

# ---------------------------------------------------------------- geometry
NORM = {"l2": lambda v: np.linalg.norm(v), "linf": lambda v: np.max(np.abs(v)), "l1": lambda v: np.sum(np.abs(v))}
DUAL = {"l2": lambda v: np.linalg.norm(v), "linf": lambda v: np.sum(np.abs(v)), "l1": lambda v: np.max(np.abs(v))}


def lmo(g, norm):
    if norm == "l2":
        return -g / np.linalg.norm(g)
    if norm == "linf":
        return -np.sign(g)
    if norm == "l1":
        j = int(np.argmax(np.abs(g))); u = np.zeros_like(g); u[j] = -np.sign(g[j]); return u
    if norm == "op":
        return -polar(g)


def weights(m):
    w = np.exp(-(m - m.min()))
    return w / w.sum()


RECSET = set(int(k) for k in REC)


def run(norm, etas, A, shape=None):
    """normalized steepest descent on the exponential loss of the signed examples A (rows)."""
    th = np.zeros(A.shape[1])
    AT = np.ascontiguousarray(A.T)
    out = []
    for t in range(T):
        m = A @ th
        w = np.exp(m.min() - m)
        g = -(AT @ w)                                   # gradient up to a positive factor
        if norm == "op":
            th += etas[t] * lmo(g.reshape(shape), "op").ravel()
        else:
            th += etas[t] * lmo(g, norm)
        if t + 1 in RECSET:
            nn = opn(th.reshape(shape)) if norm == "op" else NORM[norm](th)
            out.append(np.min(A @ th) / nn)
    return np.array(out), th


gam = {"l2": mm_l2(Z), "linf": mm_linf(Z), "l1": mm_l1(Z)}
R = {k: max(DUAL[k](z) for z in Z) for k in gam}
lo, up, Wst = mm_op(Zs)
gam["op"] = (lo, Wst.ravel())
R["op"] = max(nucn(Zi) for Zi in Zs)
print("max margins:", {k: round(v[0], 6) for k, v in gam.items()}, " R:", {k: round(v, 4) for k, v in R.items()})
print(f"spectral max margin bracket [{lo:.9f}, {up:.9f}]")

A = Zs.reshape(N, -1)
cols, table = ["t"], [REC.astype(float)]
finals = {}
for norm in ["l2", "linf", "l1", "op"]:
    nexp = N if norm == "op" else n
    for sched in ["dec", "con"]:
        etas = (1.0 / np.sqrt(np.arange(T) + 1) if sched == "dec" else 0.05 * np.ones(T)) / R[norm]
        if norm == "op":
            marg, th = run("op", etas, A, shape=(K, dm))
        else:
            marg, th = run(norm, etas, Z)
        S = np.cumsum(etas)[REC - 1]
        Q = np.cumsum(etas ** 2)[REC - 1]
        bound = gam[norm][0] - (np.log(nexp) + R[norm] ** 2 * Q / 2) / S
        assert np.all(marg >= bound - 1e-12) and np.all(marg <= gam[norm][0] + 1e-9)
        cols += [f"{norm}{sched}", f"{norm}{sched}b"]
        table += [marg, bound]
        if sched == "dec":
            finals[norm] = th
        C = (gam[norm][0] - marg) * S                  # gap times the distance travelled
        print(f"{norm:4s} {sched}: gap*tau at t=1e3,1e4,1e5: " +
              " ".join(f"{C[np.searchsorted(REC, k)]:.3f}" for k in (1000, 10000, 100000)) +
              f"  (log n = {np.log(nexp):.3f})")
        print(f"{norm:4s} {sched}: gap at t=1e3,1e4,1e5: " +
              " ".join(f"{gam[norm][0]-marg[np.searchsorted(REC, k)]:.2e}" for k in (1000, 10000, 100000)) +
              "   bound gap: " + " ".join(f"{gam[norm][0]-bound[np.searchsorted(REC, k)]:.2e}" for k in (1000, 10000, 100000)))

with open(os.path.join(DATA, "15-bias-margins.dat"), "w") as f:
    f.write(" ".join(cols) + "\n")
    for row in np.array(table).T:
        f.write(f"{int(row[0])} " + " ".join(f"{v:.6f}" for v in row[1:]) + "\n")
with open(os.path.join(DATA, "15-bias-margins-gamma.dat"), "w") as f:
    f.write("t l2 linf l1 op\n")
    for t in (1, T):
        f.write(f"{t} " + " ".join(f"{gam[k][0]:.6f}" for k in ("l2", "linf", "l1", "op")) + "\n")

# ---------------------------------------------------------------- Table 15.1: cross-norm margins
print("\nTable: normalized margins of the final iterates (decaying steps, T = 1e5)")
print("vectors      l2-margin  linf-margin  l1-margin")
for norm, name in [("l2", "normalized GD"), ("linf", "sign descent"), ("l1", "coordinate descent")]:
    th = finals[norm]
    print(f"{name:20s} " + "  ".join(f"{np.min(Z@th)/NORM[k](th):.4f}" for k in ("l2", "linf", "l1")))
print(f"{'max margin':20s} " + "  ".join(f"{gam[k][0]:.4f}" for k in ("l2", "linf", "l1")))
# matrices: Frobenius (normalized GD on W), entrywise max norm (sign descent), spectral (orthogonalized)
gF, _ = mm_l2(A)
gM, _ = mm_linf(A)
mats = {"op": finals["op"]}
for norm in ["l2", "linf"]:
    Rm = max(DUAL[norm](a) for a in A)
    _, th = run(norm, (1.0 / np.sqrt(np.arange(T) + 1)) / Rm, A)
    mats[norm] = th
print("matrices     Frobenius  max-entry  spectral")
for norm, name in [("l2", "normalized GD"), ("linf", "sign descent"), ("op", "orthogonalized")]:
    th = mats[norm]
    W = th.reshape(K, dm)
    print(f"{name:20s} {np.min(A@th)/np.linalg.norm(th):.4f}  {np.min(A@th)/np.max(np.abs(th)):.4f}  "
          f"{np.min(A@th)/opn(W):.4f}")
print(f"{'max margin':20s} {gF:.4f}  {gM:.4f}  {lo:.4f}")
