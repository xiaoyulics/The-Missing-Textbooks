"""Chapter 14: the three-parameter caricature (linear attention, square loss).

Features phi_U (in-context unigram frequencies) and phi_B (bigram counts out of the
current token, divided by T) are computed on a pool of sampled Markov sequences
(K=3 states, Dirichlet(1) rows, uniform first token, query position T=24).
The theory is the gradient flow of L(c,ab) with the pool moments (solved by an ODE
solver); the simulation is minibatch SGD on (c,a,b) with batches from the pool.
Writes figures/data/14-induction-caricature-{curve,escape}.dat. Runs in about 20 s.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy.integrate import solve_ivp

out = os.path.join(os.path.dirname(__file__), "..", "data")
rng = np.random.default_rng(0)
K, T, M = 3, 24, 400000
P = rng.dirichlet(np.ones(K), size=(M, K))
x = np.zeros((M, T + 1), int); x[:, 0] = rng.integers(K, size=M)
u = rng.random((M, T + 1)); ar = np.arange(M)
for t in range(1, T + 1):
    x[:, t] = (u[:, t, None] > np.cumsum(P[ar, x[:, t - 1]], 1)).sum(1)
X = x[:, :T]; Y = np.eye(K)[x[:, T]]
phiU = np.eye(K)[X].mean(1)
match = X[:, :-1] == X[:, -1][:, None]
phiB = (np.eye(K)[X[:, 1:]] * match[:, :, None]).sum(1) / T
A = (phiU * phiU).sum(1).mean(); B = (phiU * phiB).sum(1).mean(); C = (phiB * phiB).sum(1).mean()
gU = (phiU * Y).sum(1).mean(); gB = (phiB * Y).sum(1).mean()
cs = gU / A; G = gB - B * cs; D = A * C - B * B; vss = A * G / D
L = lambda c, v: 1 - 2 * c * gU - 2 * v * gB + A * c * c + 2 * B * c * v + C * v * v
print(f"A={A:.4f} B={B:.4f} C={C:.4f} gU={gU:.4f} gB={gB:.4f} c*={cs:.4f} G={G:.4f} v**={vss:.4f}")
Lss = L((gU * C - gB * B) / D, vss)
print(f"L(0,0)=1  L_U={L(cs,0):.4f}  L**={L((gU*C-gB*B)/D, vss):.4f}")

def rhs(t, z):
    c, a, b = z; r = 2 * (gB - B * c - C * a * b)
    return [2 * (gU - A * c - B * a * b), b * r, a * r]

def ode(eps, tmax=400):
    s = solve_ivp(rhs, [0, tmax], [0, eps, eps], rtol=1e-10, atol=1e-14, dense_output=True)
    return s.sol

def sgd(eps, seed, lr=0.05, batch=256, steps=6000):
    g = np.random.default_rng(seed); c, a, b = 0.0, eps, eps; hist = []
    for k in range(steps + 1):
        hist.append((k * lr, c, a * b))
        i = g.integers(M, size=batch)
        r = c * phiU[i] + a * b * phiB[i] - Y[i]
        dc = 2 * (r * phiU[i]).sum(1).mean(); dv = 2 * (r * phiB[i]).sum(1).mean()
        c, a, b = c - lr * dc, a - lr * b * dv, b - lr * a * dv
    return np.array(hist)

def tau(tt, v):
    j = np.argmax(v >= vss / 2)
    return np.interp(vss / 2, [v[j - 1], v[j]], [tt[j - 1], tt[j]])

# left panel: loss curve for eps = 1e-3
sol = ode(1e-3); h = sgd(1e-3, 1)
tt = np.concatenate([np.linspace(0, 10, 41)[:-1], np.linspace(10, 200, 191)]); z = sol(tt)
with open(os.path.join(out, "14-induction-caricature-curve.dat"), "w") as f:
    f.write("t ode c v sgdt sgd\n")   # ode, sgd: excess loss L - L** (log axis)
    for k, t in enumerate(tt):
        j = min(int(round(t / 0.05)), len(h) - 1)
        f.write(f"{t:.2f} {L(z[0,k], z[1,k]*z[2,k])-Lss:.4e} {z[0,k]:.5f} {z[1,k]*z[2,k]:.6f} "
                f"{h[j,0]:.2f} {max(L(h[j,1], h[j,2])-Lss,1e-7):.4e}\n")
# right panel: escape time versus eps
with open(os.path.join(out, "14-induction-caricature-escape.dat"), "w") as f:
    f.write("log10inv ode sgdmean sgdmin sgdmax asym\n")
    for e10 in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]:
        eps = 10.0 ** (-e10)
        tt = np.linspace(0, 400, 400001); z = ode(eps)(tt); to = tau(tt, z[1] * z[2])
        ts = [tau(hh[:, 0], hh[:, 2]) for hh in (sgd(eps, s, steps=int(1.3 * to / 0.05) + 400) for s in range(5))]
        f.write(f"{e10} {to:.3f} {np.mean(ts):.3f} {min(ts):.3f} {max(ts):.3f} {np.log(1/eps**2)/(4*G):.3f}\n")
        print(e10, to, np.mean(ts), min(ts), max(ts), np.log(1 / eps**2) / (4 * G))
