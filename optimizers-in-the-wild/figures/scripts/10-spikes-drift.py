"""Drift of the output logits along the flat direction of the cross-entropy (Section 10.5, Exercise).

A bias-only softmax model (a unigram language model) over K = 1000 classes with Zipf label
frequencies q_k proportional to 1/k is trained from b = 0 on minibatches of B labels.
The cross-entropy is invariant under b -> b + c 1 (Proposition prop:spikes:zloss), and every
gradient sums to zero over the classes, so SGD keeps the mean of b fixed exactly. Coordinatewise
normalized methods do not. For each setting the script reports the drift rate of log Z per step,
in units of the step size eta, over the second half of the run (T = 20000 steps, eta = 0.01,
beta1 = 0.9), and the excess cross-entropy at the end (cross-entropy minus the entropy of q).

Output: figures/data/10-spikes-drift.dat. Deterministic seeds; about 20 s on one core.
"""
import os
import numpy as np

K = 1000
T = 20000
ETA = 1e-2
B1 = 0.9
Q = 1.0 / np.arange(1, K + 1)
Q /= Q.sum()
ENTROPY = -(Q * np.log(Q)).sum()


def lse(b):
    m = b.max()
    return m + np.log(np.exp(b - m).sum())


def run(method="adam", batch=64, b2=0.95, alpha=0.0, wd=0.0, seed=0):
    rng = np.random.default_rng(seed)
    b = np.zeros(K)
    m = np.zeros(K)
    v = np.zeros(K)
    eta = 1.0 if method == "sgd" else ETA
    half = None
    for t in range(1, T + 1):
        if batch is None:
            y = Q
        else:
            y = np.bincount(rng.choice(K, size=batch, p=Q), minlength=K) / batch
        lz = lse(b)
        p = np.exp(b - lz)
        g = p - y + 2.0 * alpha * lz * p
        if method == "sgd":
            b -= eta * g
        elif method == "sign":
            b = (1 - eta * wd) * b - eta * np.sign(g)
        else:
            m = B1 * m + (1 - B1) * g
            v = b2 * v + (1 - b2) * g * g
            mhat = m / (1 - B1 ** t)
            vhat = v / (1 - b2 ** t)
            b = (1 - eta * wd) * b - eta * mhat / (np.sqrt(vhat) + 1e-8)
        if t == T // 2:
            half = lse(b)
    rate = (lse(b) - half) / (T - T // 2) / eta
    excess = -(Q * (b - lse(b))).sum() - ENTROPY
    return rate, excess, b.mean()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "data", "10-spikes-drift.dat")
    settings = [
        ("sgd", dict(method="sgd", batch=64)),
        ("adam-full", dict(batch=None)),
        ("adam-B16", dict(batch=16)),
        ("adam-B64", dict(batch=64)),
        ("adam-B256", dict(batch=256)),
        ("adam-B1024", dict(batch=1024)),
        ("adam-B4096", dict(batch=4096)),
        ("adam-b2-0.99", dict(b2=0.99)),
        ("adam-b2-0.999", dict(b2=0.999)),
        ("adam-z1e-4", dict(alpha=1e-4)),
        ("adam-z1e-2", dict(alpha=1e-2)),
        ("adam-z1e-1", dict(alpha=1e-1)),
        ("adamw-0.01", dict(wd=0.01)),
        ("adamw-0.1", dict(wd=0.1)),
        ("sign-B64", dict(method="sign", batch=64)),
    ]
    with open(out, "w") as fh:
        fh.write("setting driftrate excessCE meanbias\n")
        for i, (name, kw) in enumerate(settings):
            rate, excess, mean = run(seed=i, **kw)
            print(f"{name:14s} drift of log Z per step / eta = {rate:+.4f}   excess CE = {excess:.4f}   mean bias = {mean:+.3f}")
            fh.write(f"{name} {rate:.5g} {excess:.5g} {mean:.5g}\n")
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
