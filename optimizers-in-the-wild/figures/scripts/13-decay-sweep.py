"""Chapter 13, figure "equilibrium", panel (b): the equilibrium angle per step against eta*lambda in a
small network with LayerNorm, for SGD, AdamW without momentum and AdamW with beta1 = 0.9.

Network: x in R^16 -> W1 (32x16) -> LayerNorm -> ReLU -> W2 (32x32) -> LayerNorm -> ReLU -> W3 (8x32) ->
softmax cross-entropy, on 4096 Gaussian inputs labelled by a random tanh teacher (labels sampled from
its softmax). The LayerNorms have no gain or bias, so the loss is exactly scale invariant in W1 and in
W2 (up to the LayerNorm epsilon 1e-12). Decoupled weight decay is applied to W1 and W2 only; W3 is
trained by the same optimizer without decay. Batch size 16, 10000 steps; all runs see the same batches.

For each optimizer and each eta*lambda in {5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2} two (eta, lambda) pairs with
the same product are run. Measured: the root mean square of the angle between W_t and W_{t+1} over the
last 4000 steps, for W1 and W2.
Theory: arccos(1 - eta*lambda) (Proposition prop:decay:average, exact for orthogonal updates at a
stationary norm), and for momentum beta = 0.9 the leading-order heuristic
sqrt(2 eta lambda (1-beta)/(1+beta)) derived in the chapter.
Writes figures/data/13-decay-sweep.dat. Runtime about 45 s on two threads.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")
LN_EPS = 1e-12


def ln_forward(z):
    zc = z - z.mean(-1, keepdims=True)
    sd = np.sqrt((zc ** 2).mean(-1, keepdims=True) + LN_EPS)
    return zc / sd, sd


def ln_backward(dn, n, sd):
    return (dn - dn.mean(-1, keepdims=True) - n * (dn * n).mean(-1, keepdims=True)) / sd


def loss_and_grads(W1, W2, W3, X, Y):
    """Batched over runs: W1 (R,H,D), W2 (R,H,H), W3 (R,K,H), X (B,D), Y (B,)."""
    z1 = np.einsum("bd,rhd->rbh", X, W1)
    n1, s1 = ln_forward(z1)
    a1 = np.maximum(n1, 0)
    z2 = np.matmul(a1, np.transpose(W2, (0, 2, 1)))
    n2, s2 = ln_forward(z2)
    a2 = np.maximum(n2, 0)
    lo = np.matmul(a2, np.transpose(W3, (0, 2, 1)))
    lo = lo - lo.max(-1, keepdims=True)
    p = np.exp(lo)
    p /= p.sum(-1, keepdims=True)
    Bsz = X.shape[0]
    dlo = p.copy()
    dlo[:, np.arange(Bsz), Y] -= 1
    dlo /= Bsz
    loss = -np.log(p[:, np.arange(Bsz), Y]).mean(1)
    dW3 = np.matmul(np.transpose(dlo, (0, 2, 1)), a2)
    dz2 = ln_backward(np.matmul(dlo, W3) * (n2 > 0), n2, s2)
    dW2 = np.matmul(np.transpose(dz2, (0, 2, 1)), a1)
    dz1 = ln_backward(np.matmul(dz2, W2) * (n1 > 0), n1, s1)
    dW1 = np.einsum("rbh,bd->rhd", dz1, X)
    return loss, (dW1, dW2, dW3)


def main():
    rng = np.random.default_rng(0)
    D, H, K, Bsz, n = 16, 32, 8, 16, 4096
    X_all = rng.normal(size=(n, D))
    A = rng.normal(size=(D, 32)) / np.sqrt(D)
    Bt = rng.normal(size=(32, K)) / np.sqrt(32)
    logit = 3 * np.tanh(X_all @ A) @ Bt
    P = np.exp(logit - logit.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    Y_all = np.array([rng.choice(K, p=p) for p in P])

    els = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
    opts = [("sgd", (0.2, 0.8), 0.0), ("adamz", (2e-3, 8e-3), 0.0), ("adamm", (2e-3, 8e-3), 0.9)]
    cfg = [(name, eta, el / eta, b1, pair) for name, etas, b1 in opts for el in els
           for pair, eta in zip("ab", etas)]
    R = len(cfg)
    eta = np.array([c[1] for c in cfg])[:, None, None]
    lam = np.array([c[2] for c in cfg])[:, None, None]
    b1 = np.array([c[3] for c in cfg])[:, None, None]
    is_adam = np.array([c[0] != "sgd" for c in cfg])[:, None, None]
    lr_out = np.where(is_adam, 2e-3, 0.2)        # output layer, no decay
    b2 = 0.95
    W = [np.repeat(rng.normal(size=(1, H, D)) / np.sqrt(D), R, 0),
         np.repeat(rng.normal(size=(1, H, H)) / np.sqrt(H), R, 0),
         np.repeat(0.1 * rng.normal(size=(1, K, H)) / np.sqrt(H), R, 0)]
    M = [np.zeros_like(w) for w in W]
    V = [np.zeros_like(w) for w in W]
    T, win = 10000, 4000
    ang2 = np.zeros((2, R))
    tan2u = np.zeros((2, R))
    usum = np.zeros((2, R))
    u_start = np.zeros((2, R))
    for t in range(T):
        idx = rng.integers(0, n, size=Bsz)
        loss, G = loss_and_grads(W[0], W[1], W[2], X_all[idx], Y_all[idx])
        old = [W[0].copy(), W[1].copy()]
        for i in range(3):
            g = G[i]
            M[i] = b1 * M[i] + (1 - b1) * g
            V[i] = b2 * V[i] + (1 - b2) * g * g
            u_adam = (M[i] / (1 - b1 ** (t + 1))) / (np.sqrt(V[i] / (1 - b2 ** (t + 1))) + 1e-16)
            u = np.where(is_adam, u_adam, g)
            if i < 2:
                W[i] = (1 - eta * lam) * W[i] - eta * u
            else:
                W[i] = W[i] - lr_out * u
        if t >= T - win:
            for i in range(2):
                nu = np.sum(old[i] ** 2, axis=(1, 2))
                nn = np.sum(W[i] ** 2, axis=(1, 2))
                c = np.sum(old[i] * W[i], axis=(1, 2)) / np.sqrt(nu * nn)
                w = np.arccos(np.clip(c, -1, 1))
                ang2[i] += w ** 2
                tan2u[i] += nu * np.tan(w) ** 2
                usum[i] += nu
                if t == T - win:
                    u_start[i] = nu
    u_end = [np.sum(W[i] ** 2, axis=(1, 2)) for i in range(2)]
    rms = np.sqrt(ang2 / win)

    names = []
    table = {}
    for r, (name, e, l, bb, pair) in enumerate(cfg):
        for i in range(2):
            key = f"{name}{i + 1}{pair}"
            table.setdefault(key, {})[e * l] = rms[i, r]
    with open(os.path.join(OUT, "13-decay-sweep.dat"), "w") as f:
        keys = sorted(table)
        f.write("el exact mom " + " ".join(keys) + "\n")
        for el in els:
            mom = np.sqrt(2 * el * (1 - 0.9) / (1 + 0.9))
            f.write(f"{el:.6g} {np.arccos(1 - el):.6g} {mom:.6g} "
                    + " ".join(f"{table[k][el]:.6g}" for k in keys) + "\n")
    print("final mean training loss", float(loss.mean()))
    for r, (name, e, l, bb, pair) in enumerate(cfg):
        el = e * l
        a = (1 - el) ** 2
        msg = []
        for i in range(2):
            wmean = tan2u[i, r] / usum[i, r]
            msg.append(f"W{i + 1}: rms/arccos {rms[i, r] / np.arccos(1 - el):.3f}, "
                       f"u-weighted tan^2/((1-a)/a) {wmean / ((1 - a) / a):.3f}, "
                       f"norm drift {u_end[i][r] / u_start[i, r]:.3f}")
        print(f"{name:6s} eta={e:<7g} lam={l:<8.4g} " + " | ".join(msg))


if __name__ == "__main__":
    main()
