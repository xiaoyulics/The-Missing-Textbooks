"""Figure of Chapter 6 (Lazy Training): learning by modes in a wide two-layer network.

Training set: n = 32 equally spaced points on the unit circle, x_i = (cos t_i, sin t_i),
augmented with a bias coordinate, xt_i = (x_i, 1)/sqrt(2), so that ||xt_i|| = 1.
Labels y_i = cos t_i + cos 2t_i + cos 3t_i.
Network: f(x; W) = m^{-1/2} sum_r a_r tanh(<w_r, xt>), first layer trained, a_r = +-1 fixed.
Antisymmetric initialization: w_{r+m/2} = w_r ~ N(0, I_3), a_r = +1 for the first half and
-1 for the second, so that f(.; W_0) = 0 and the initial residual is -y.
Training: full-batch gradient descent on (1/2)||F(W) - y||^2 with step eta = 0.02.

Because the inputs are equally spaced, the infinite-width tangent kernel K_inf is circulant,
its eigenvectors on the training set are the Fourier modes cos(k t_i), sin(k t_i), and its
eigenvalues lam_k are the discrete Fourier transform of its first row. The linearized
model, run by gradient descent, multiplies the k-th Fourier coefficient of the residual
by (1 - eta lam_k) per step: that is the curve. The points are the ratio
<cos(k t), r_t> / <cos(k t), r_0> for networks of width m = 16384 and m = 64.

K_inf(x, x') = <xt, xt'> E[tanh'(u) tanh'(v)], (u, v) standard Gaussians with correlation
<xt, xt'>, computed by Gauss-Hermite quadrature (80 x 80 nodes).

Writes figures/data/06-lazy-modes-theory.dat and figures/data/06-lazy-modes-sim.dat.
Deterministic (seed 6); runs in about 15 seconds on a laptop CPU.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
N, ETA, T = 32, 0.02, 30.0
FREQS = [1, 2, 3]
WIDTHS = [16384, 64]

z, wq = np.polynomial.hermite_e.hermegauss(80)
wq = wq / wq.sum()

def dtanh(u):
    return 1.0 / np.cosh(u) ** 2

def kappa(rho):
    """E[tanh'(u) tanh'(v)] for standard Gaussians u, v with correlation rho."""
    s = np.sqrt(max(1.0 - rho * rho, 0.0))
    U = z[:, None]
    V = rho * z[:, None] + s * z[None, :]
    return float(np.sum(wq[:, None] * wq[None, :] * dtanh(U) * dtanh(V)))

theta = 2 * np.pi * np.arange(N) / N
Xt = np.stack([np.cos(theta), np.sin(theta), np.ones(N)], 1) / np.sqrt(2)
G = Xt @ Xt.T
Kinf = G * np.vectorize(kappa)(G)
lam = np.real(np.fft.fft(Kinf[0]))           # eigenvalue of frequency k is lam[k]
y = sum(np.cos(k * theta) for k in FREQS)

def run(m, seed):
    rng = np.random.default_rng(seed)
    h = m // 2
    W = rng.standard_normal((h, 3))
    W = np.vstack([W, W])
    a = np.r_[np.ones(h), -np.ones(h)]
    steps = int(round(T / ETA))
    record = sorted(set(np.r_[np.arange(0, 51, 2), np.unique(np.round(np.geomspace(1, steps, 70)).astype(int))]))
    out = []
    for s in range(steps + 1):
        P = Xt @ W.T
        r = (np.tanh(P) @ a) / np.sqrt(m) - y
        if s in record:
            out.append([s * ETA] + [(np.cos(k * theta) @ r) / (np.cos(k * theta) @ (-y)) for k in FREQS])
        W -= ETA * (((r[:, None] * dtanh(P)) * a[None, :]).T @ Xt) / np.sqrt(m)
    return np.array(out)

def main():
    tt = np.linspace(0, T, 301)
    with open(os.path.join(DATA, "06-lazy-modes-theory.dat"), "w") as fh:
        fh.write("t " + " ".join("k%d" % k for k in FREQS) + "\n")
        for t in tt:
            fh.write("%.4f " % t + " ".join("%.6e" % (1 - ETA * lam[k]) ** (t / ETA) for k in FREQS) + "\n")
    sims = {m: run(m, 6) for m in WIDTHS}
    with open(os.path.join(DATA, "06-lazy-modes-sim.dat"), "w") as fh:
        fh.write("t " + " ".join("m%d_k%d" % (m, k) for m in WIDTHS for k in FREQS) + "\n")
        for i in range(len(sims[WIDTHS[0]])):
            vals = [sims[m][i, j + 1] for m in WIDTHS for j in range(len(FREQS))]
            # residual ratios below 1e-3 are dominated by finite-width mode mixing; mark them nan
            fh.write("%.4f " % sims[WIDTHS[0]][i, 0] + " ".join("%.6e" % v if v > 1e-3 else "nan" for v in vals) + "\n")
    print("eigenvalues lam_0..lam_4:", np.round(lam[:5], 5))
    print("smallest eigenvalue of K_inf:", np.linalg.eigvalsh(Kinf).min())
    for m in WIDTHS:
        s = sims[m]
        for j, k in enumerate(FREQS):
            pred = (1 - ETA * lam[k]) ** (s[:, 0] / ETA)
            mask = pred > 1e-2
            print("m=%d k=%d max |log ratio| (pred>1e-2): %.3f" % (m, k, np.max(np.abs(np.log(np.abs(s[mask, j + 1]) / pred[mask])))))

if __name__ == "__main__":
    main()
