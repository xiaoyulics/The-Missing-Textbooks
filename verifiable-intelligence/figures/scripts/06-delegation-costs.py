#!/usr/bin/env python3
"""Chapter 6 (delegation): what the verifier and the honest prover of the GKR protocol spend.

Runs Protocol 6.1 (the implementation of 06-delegation-gkr.py) on the batch-network circuit
y_t = sum_kappa (sum_j W[kappa, j] X[j, t])^2 for N = 2, 4, ..., 64 (hidden width N, input dimension N,
batch of N inputs; depth D = 2 log2 N + 2), over F_p with p = 2^31 - 1, on quantized inputs: every
weight and input entry is an integer in {-7, ..., 7}, so every gate value lies in (-p/2, p/2) and the
field computation equals the integer computation (checked). One honest run per size, accepted.
Records:
  S       gates of the circuit, inputs included; n = 2N^2 inputs
  evalops field operations to evaluate the circuit (one per non-input gate)
  pops    field operations of the honest prover (instrumented; evaluation of the circuit included)
  vops    field operations of the verifier with the closed-form wiring (structured-wiring proposition)
  vgen    field operations of the verifier that evaluates the wiring by summing over the gates
  vnoin   vops minus the two multilinear extensions it evaluates (outputs and input)
  psent   field elements sent by the prover; vsent by the verifier
  ptime, vtime  wall-clock seconds of the prover and of the verifier (closed-form wiring)
Writes figures/data/06-delegation-costs.dat. Run from the volume root (a few seconds on a laptop):
    python3 figures/scripts/06-delegation-costs.py
"""
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import protokit as pk  # noqa: E402

_spec = importlib.util.spec_from_file_location("gkr", os.path.join(HERE, "06-delegation-gkr.py"))
gkr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gkr)

P = pk.PRIME
OUT = "figures/data/06-delegation-costs.dat"


class Timed(gkr.Honest):
    """The honest prover, with its wall-clock time accumulated."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.secs = 0.0
        self.round_ops = 0                                   # operations spent in sumcheck rounds

    def _t(self, f, *a):
        t0 = time.perf_counter()
        out = f(*a)
        self.secs += time.perf_counter() - t0
        return out

    def _r(self, f, *a):
        before = self.ops.n
        out = self._t(f, *a)
        self.round_ops += self.ops.n - before
        return out

    def start_phase(self, *a):
        return self._t(super().start_phase, *a)

    def sum_message(self, *a):
        return self._r(super().sum_message, *a)

    def sum_receive(self, *a):
        return self._r(super().sum_receive, *a)

    def line_message(self, *a):
        return self._t(super().line_message, *a)


def main():
    t0 = time.time()
    rows = []
    for m in range(1, 7):
        N = 1 << m
        L = gkr.network_circuit(m)
        D = len(L) - 1
        S, n = gkr.size(L), L[D]["S"]
        g = pk.rng(60 + m)
        Wm = g.integers(-7, 8, size=(N, N))
        Xm = g.integers(-7, 8, size=(N, N))
        x = np.concatenate([Wm.ravel(), Xm.ravel()]) % P
        Yint = ((Wm @ Xm) ** 2).sum(axis=0)              # the network on the integers
        assert int(np.abs(Yint).max()) < P // 2
        pops = gkr.Ops()
        tp = time.perf_counter()
        Wv = gkr.evaluate(L, x, P, pops)
        teval = time.perf_counter() - tp
        assert np.array_equal(Wv[0], Yint % P)           # the field computation is the integer one
        evalops = pops.n
        prov = Timed(L, Wv, P, pops)
        vops = gkr.Ops()
        tr = time.perf_counter()
        acc, psent, vsent = gkr.run(L, x, prov, P, 1, pk.rng(600 + m), vops=vops)
        ttotal = time.perf_counter() - tr
        assert acc.all()
        vgen = gkr.Ops()
        acc2, _, _ = gkr.run(L, x, gkr.Honest(L, Wv, P), P, 1, pk.rng(600 + m), wiring="generic", vops=vgen)
        assert acc2.all()
        k = [l["k"] for l in L]
        vnoin = vops.n - 3 * ((1 << k[0]) - 1) - 3 * ((1 << k[D]) - 1)
        rows.append([N, D, S, n, evalops, pops.n, vops.n, vgen.n, vnoin, psent, vsent,
                     prov.secs + teval, ttotal - prov.secs])
        gate_rounds = sum(2 * k[i + 1] * L[i]["S"] for i in range(D))
        rounds = 1 + sum(2 * k[i] + 1 for i in range(1, D + 1))
        print(f"N={N:3d} D={D:2d} S={S:7d} n={n:5d} eval={evalops:7d} prover={pops.n:10d} "
              f"({pops.n / S:.1f} per gate, {pops.n / (S * np.log2(S)):.2f} per gate*log2 S; "
              f"{prov.round_ops / gate_rounds:.1f} per gate per sumcheck round) "
              f"verifier={vops.n:6d} (input extension {3 * ((1 << k[D]) - 1)}, without the two extensions {vnoin}) "
              f"generic={vgen.n:8d} ({vgen.n / evalops:.1f} x evaluation) "
              f"sent P->V {psent}, V->P {vsent}, rounds {rounds}; bound {k[0] + 5 * sum(k[1:])}/p = "
              f"{(k[0] + 5 * sum(k[1:])) / P:.2e}; time P {prov.secs + teval:.3f}s V {ttotal - prov.secs:.4f}s")
    pk.write_dat(OUT, ["N", "D", "S", "n", "evalops", "pops", "vops", "vgen", "vnoin", "psent", "vsent",
                       "ptime", "vtime"], rows)
    a, b = rows[0], rows[-1]
    print(f"from N={a[0]} to N={b[0]}: circuit x{b[2] / a[2]:.0f}, prover x{b[5] / a[5]:.0f}, "
          f"verifier x{b[6] / a[6]:.0f}")
    c = rows[-2]
    print(f"verifier slope on log-log axes between the last two sizes: "
          f"{np.log(b[6] / c[6]) / np.log(b[2] / c[2]):.2f}")
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
