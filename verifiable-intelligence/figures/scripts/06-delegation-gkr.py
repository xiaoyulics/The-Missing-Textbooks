#!/usr/bin/env python3
"""Chapter 6 (delegation): the GKR protocol against honest and cheating provers.

The circuit. For N = 2^m, a layered arithmetic circuit over F_p that evaluates a two-layer network
with square activation and unit output weights, y_t = sum_kappa (sum_j W[kappa, j] X[j, t])^2, on a
batch of N inputs (the columns of X, each of dimension N; hidden width N). Layers, from the output:
  layers 0..m-1      additions: partial sums over kappa            (k_i = m + i)
  layer m            multiplications: squares Q = U * U            (k_m = 2m)
  layers m+1..2m     additions: partial sums over j; m+1 holds U   (k_i = m + i - 1)
  layer 2m+1         multiplications: products W[kappa, j] X[j, t] (k = 3m)
  layer 2m+2 = D     the input: W, then X; 2N^2 field elements     (k_D = 2m + 1)
Gate labels are bit strings, most significant bit first: (kappa, j, t) in the products layer,
(s, alpha, beta) in the input layer (s = 0 for W, s = 1 for X).

The protocol is Protocol 6.1 of the chapter: the prover sends the outputs; the verifier evaluates
their multilinear extension at a random point; then, layer by layer, a sumcheck over the wiring
identity (2 k_{i+1} rounds, messages of degree 2 sent as 3 coefficients) and a line through the
two points at which it ends (a message of degree k_{i+1} sent as k_{i+1}+1 coefficients); finally
the verifier evaluates the multilinear extension of the input at one point. The verifier evaluates
the wiring predicates' extensions in closed form (the chapter's proposition on structured wiring;
checked here against the generic sum over gates).

Provers:
  honest        sends the true outputs and the prescribed messages;
  root-planting claims output 0 plus 1; in each sumcheck round it keeps the prescribed values at
                0 and 2 and changes the value at 1 so that the check passes (it escapes, and turns
                honest, if r is 0 or 2); in each line step it changes q at one of the nodes 0, 1
                only (it escapes if t* is one of the other k nodes 0..k); it attains the product
                bound below up to degenerate events of probability O(k/p) per line step (in the
                worked run, a variant that keeps the values at 1 and 2 in the first round of a
                phase avoids them and attains the product bound exactly);
  fault         computes the circuit with square gate 0 (layer m) increased by 1, claims the
                resulting outputs, and runs the honest prover on its own gate values; it passes
                exactly when chi_0(z_m) = 0, with probability 1 - (1 - 1/p)^(2m).
Proved bounds (the chapter's GKR theorem): every prover is accepted with probability at most
  1 - (1-1/p)^{k_0} prod_{i=1}^{D} (1-2/p)^{2k_i} (1-k_i/p)  <=  (k_0 + 5 sum_{i>=1} k_i) / p.

Writes figures/data/06-delegation-gkr.dat (measured rates) and 06-delegation-gkr-theory.dat
(curves), and prints the worked run of the chapter (N = 1 over F_13). Run from the volume root
(about 25 s on a laptop):
    python3 figures/scripts/06-delegation-gkr.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import protokit as pk  # noqa: E402


# --------------------------------------------------------------------------- operation counter
class Ops:
    """Counts field operations (additions, subtractions, multiplications) of one party in one run."""

    def __init__(self):
        self.n = 0

    def __call__(self, k):
        self.n += int(k)


NOCOUNT = Ops()


# --------------------------------------------------------------------------- the circuit
def network_circuit(m):
    """Layers 0..D of the batch-network circuit for N = 2^m. A non-input layer is a dict with k,
    op ('add' or 'mult'), in1, in2 (gate g has label g), S, and kind (for the closed-form wiring)."""
    N = 1 << m
    D = 2 * m + 2
    L = [None] * (D + 1)
    for l in range(m):                                  # kappa-sums, layer l: labels (kappa', t)
        kp, t = np.meshgrid(np.arange(1 << l), np.arange(N), indexing="ij")
        L[l] = dict(k=m + l, op="add", kind=("ksum", l),
                    in1=((2 * kp) * N + t).ravel(), in2=((2 * kp + 1) * N + t).ravel())
    lab = np.arange(N * N)                              # squares, layer m: labels (kappa, t)
    L[m] = dict(k=2 * m, op="mult", kind=("square",), in1=lab.copy(), in2=lab.copy())
    for l in range(m):                                  # j-sums, layer m+1+l: labels (kappa, j', t)
        nj = 1 << l
        kk, jj, tt = np.meshgrid(np.arange(N), np.arange(nj), np.arange(N), indexing="ij")
        L[m + 1 + l] = dict(k=2 * m + l, op="add", kind=("jsum", l),
                            in1=(kk * (2 * nj) * N + (2 * jj) * N + tt).ravel(),
                            in2=(kk * (2 * nj) * N + (2 * jj + 1) * N + tt).ravel())
    kk, jj, tt = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    L[2 * m + 1] = dict(k=3 * m, op="mult", kind=("prod",),       # products W[kappa,j] X[j,t]
                        in1=(kk * N + jj).ravel(), in2=(N * N + jj * N + tt).ravel())
    L[D] = dict(k=2 * m + 1, kind=("input",), S=2 * N * N)
    for i in range(D):
        L[i]["S"] = L[i]["in1"].shape[0]
        assert L[i]["S"] == 1 << L[i]["k"]
    return L


def evaluate(L, x, p, ops=NOCOUNT):
    """Gate values W_0..W_D of the circuit on input x (length 2^{k_D}), reduced mod p."""
    D = len(L) - 1
    W = [None] * (D + 1)
    W[D] = np.asarray(x, dtype=np.int64) % p
    for i in range(D - 1, -1, -1):
        a, b = W[i + 1][L[i]["in1"]], W[i + 1][L[i]["in2"]]
        W[i] = (a + b) % p if L[i]["op"] == "add" else a * b % p
        ops(L[i]["S"])
    return W


def size(L):
    return sum(l["S"] for l in L)


# --------------------------------------------------------------------------- multilinear tools
def chi_table(z, p, ops=NOCOUNT):
    """z: (T, k) -> (T, 2^k), entry a = chi_a(z) = prod_u (z_u a_u + (1-z_u)(1-a_u)), MSB first."""
    T, k = z.shape
    E = np.ones((T, 1), dtype=np.int64)
    for u in range(k):
        hi = E * z[:, u:u + 1] % p
        lo = (E - hi) % p
        E = np.stack([lo, hi], axis=-1).reshape(T, -1)
        ops(E.shape[1])                       # one multiplication and one subtraction per pair
    return E


def mle(table, z, p, ops=NOCOUNT):
    """Multilinear extension of table (length 2^k, or (T, 2^k)) at the points z (T, k), by folding
    the most significant variable first (3 operations per pair)."""
    T, k = z.shape
    v = np.broadcast_to(np.asarray(table, dtype=np.int64) % p, (T, 1 << k)).copy()
    for u in range(k):
        h = v.shape[1] // 2
        v = (v[:, :h] + (v[:, h:] - v[:, :h]) % p * z[:, u:u + 1]) % p
        ops(3 * h)
    return v[:, 0]


def eq2(x, y, p):
    return (x * y + (1 - x) * (1 - y)) % p


def eq3(x, y, w, p):
    return (x * y % p * w + (1 - x) * (1 - y) % p * ((1 - w) % p)) % p


def _prod3(zs, bs, cs, p, ops):
    acc = np.ones(zs.shape[0], dtype=np.int64)
    for u in range(zs.shape[1]):
        acc = acc * eq3(zs[:, u], bs[:, u], cs[:, u], p) % p
    ops(9 * zs.shape[1])          # per factor: 3 subtractions, 4 multiplications, 1 addition; 1 product
    return acc


def wiring_closed(L, i, z, b, c, p, ops=NOCOUNT):
    """(add~_i, mult~_i) at (z, b, c), each of shape (T,), by the closed forms of the chapter's
    proposition on structured wiring (products of eq factors over blocks of bits)."""
    T = z.shape[0]
    kind = L[i]["kind"]
    zero = np.zeros(T, dtype=np.int64)
    if kind[0] == "square":                   # mult(a, b, c) = [b = a][c = a]
        return zero, _prod3(z, b, c, p, ops)
    if kind[0] == "ksum":                     # z = (kappa' [l], t); b, c = (kappa' bit, t)
        l = kind[1]
        val = _prod3(z[:, :l], b[:, :l], c[:, :l], p, ops)
        val = val * ((1 - b[:, l]) % p) % p * c[:, l] % p
        val = val * _prod3(z[:, l:], b[:, l + 1:], c[:, l + 1:], p, ops) % p
        ops(4)
        return val, zero
    if kind[0] == "jsum":                     # z = (kappa [m], j' [l], t [m]); b, c = (kappa, j' bit, t)
        l = kind[1]
        m = (z.shape[1] - l) // 2
        val = _prod3(z[:, :m], b[:, :m], c[:, :m], p, ops)
        val = val * _prod3(z[:, m:m + l], b[:, m:m + l], c[:, m:m + l], p, ops) % p
        val = val * ((1 - b[:, m + l]) % p) % p * c[:, m + l] % p
        val = val * _prod3(z[:, m + l:], b[:, m + l + 1:], c[:, m + l + 1:], p, ops) % p
        ops(6)
        return val, zero
    if kind[0] == "prod":                     # z = (kappa, j, t); b = (0, kappa, j); c = (1, j, t)
        m = z.shape[1] // 3
        val = (1 - b[:, 0]) % p * c[:, 0] % p
        for u in range(m):
            val = val * eq2(z[:, u], b[:, 1 + u], p) % p                  # kappa = alpha(b)
            val = val * eq2(z[:, 2 * m + u], c[:, 1 + m + u], p) % p      # t = beta(c)
        val = val * _prod3(z[:, m:2 * m], b[:, 1 + m:], c[:, 1:1 + m], p, ops) % p   # j = beta(b) = alpha(c)
        ops(3 + 2 * 7 * m)
        return zero, val
    raise ValueError(kind)


def wiring_generic(L, i, z, b, c, p, ops=NOCOUNT):
    """(add~_i, mult~_i) at (z, b, c) as the sum over gates of chi_a(z) chi_in1(b) chi_in2(c):
    O(2^{k_i} + 2^{k_{i+1}} + S_i) operations, what a verifier must spend without structure."""
    Ez, Eb, Ec = chi_table(z, p, ops), chi_table(b, p, ops), chi_table(c, p, ops)
    S = L[i]["S"]
    t = Ez[:, :S] * Eb[:, L[i]["in1"]] % p * Ec[:, L[i]["in2"]] % p
    ops(3 * S)
    val = t.sum(axis=1) % p
    zero = np.zeros(z.shape[0], dtype=np.int64)
    return (val, zero) if L[i]["op"] == "add" else (zero, val)


# --------------------------------------------------------------------------- polynomials
_VINV = {}


def _vandermonde_inverse(n, p):
    """Matrix taking values at 0..n-1 to coefficients (lowest degree first), mod p."""
    key = (n, p)
    if key not in _VINV:
        rows = [[0] * n for _ in range(n)]
        for j in range(n):
            poly, den = [1], 1                      # prod_{k != j} (X - k), lowest degree first
            for k in range(n):
                if k != j:
                    poly = [(-k * poly[0]) % p] + [(poly[a - 1] - k * poly[a]) % p for a in range(1, len(poly))] + [poly[-1]]
                    den = den * (j - k) % p
            inv = pow(den, p - 2, p)
            for a in range(n):
                rows[a][j] = poly[a] * inv % p
        _VINV[key] = np.array(rows, dtype=np.int64)
    return _VINV[key]


def coeffs_from_values(vals, p, ops=NOCOUNT):
    """vals (T, n): values at 0..n-1 -> coefficients (T, n), lowest degree first (prover's work)."""
    T, n = vals.shape
    Vi = _vandermonde_inverse(n, p)
    out = np.zeros_like(vals)
    for a in range(n):
        acc = np.zeros(T, dtype=np.int64)
        for j in range(n):
            acc = (acc + vals[:, j] * Vi[a, j]) % p
        out[:, a] = acc
    ops(2 * n * n)
    return out


def horner(coef, r, p, ops=NOCOUNT):
    """coef (T, d+1), r (T,) -> values (T,)."""
    acc = np.zeros(coef.shape[0], dtype=np.int64)
    for a in range(coef.shape[1] - 1, -1, -1):
        acc = (acc * r + coef[:, a]) % p
    ops(2 * (coef.shape[1] - 1))
    return acc


def _inv(a, p):
    return np.array([pow(int(v), p - 2, p) if v else 0 for v in a], dtype=np.int64)


# --------------------------------------------------------------------------- provers
class Honest:
    """The prescribed prover, run on the gate values W it holds (the true values for the honest
    prover; corrupted ones for the fault-injecting prover). Implements the linear-pass prover of
    the GKR theorem, part (iii): one pass over the gates of layer i and one fold of a table per round."""

    def __init__(self, L, W, p, ops=NOCOUNT):
        self.L, self.W, self.p, self.ops = L, W, p, ops

    def outputs(self):
        return self.W[0]

    def start_phase(self, i, z, claim):
        L, p = self.L, self.p
        T = z.shape[0]
        self.i, self.K, self.z = i, L[i + 1]["k"], z
        self.add = L[i]["op"] == "add"
        self.in1, self.in2 = L[i]["in1"], L[i]["in2"]
        self.w = chi_table(z, p, self.ops)[:, :L[i]["S"]]      # chi_a(z) for each gate a
        self.beta = np.ones_like(self.w)
        self.gamma = np.ones_like(self.w)
        self.tab = np.broadcast_to(self.W[i + 1], (T, 1 << self.K)).copy()
        self.Wc = self.W[i + 1][self.in2]
        self.j = 0
        self.rs = []

    def _values(self):
        """Values at 0, 1, 2 of the prescribed sumcheck message of the current round: (T, 3)."""
        p, K, j = self.p, self.K, self.j
        phase1 = j < K
        jj = j if phase1 else j - K                              # variable bound in this round
        lab = self.in1 if phase1 else self.in2
        half = 1 << (K - jj - 1)
        bit = (lab >> (K - jj - 1)) & 1
        suf = lab & (half - 1)
        B0, B1 = self.tab[:, suf], self.tab[:, half + suf]
        B2 = (2 * B1 - B0) % p                                   # the table's line at X = 2
        pre = self.w * self.beta % p
        if not phase1:
            pre = pre * self.gamma % p
        chis = [(1 - bit), bit, np.where(bit == 1, 2, p - 1)]    # chi_bit(X) at X = 0, 1, 2
        out = np.zeros((self.w.shape[0], 3), dtype=np.int64)
        for X, Bx in enumerate((B0, B1, B2)):
            if phase1:
                inner = (Bx + self.Wc) % p if self.add else Bx * self.Wc % p
            else:
                ob = self.omega_b[:, None]
                inner = (ob + Bx) % p if self.add else ob * Bx % p
            out[:, X] = (pre * chis[X] % p * inner % p).sum(axis=1) % p
        S = self.w.shape[1]
        self.ops(2 * S + (1 if phase1 else 2) * S + 3 * 4 * S)   # B2; pre; per X: chi, inner, product, sum
        return out

    def sum_message(self, claim):
        return coeffs_from_values(self._values(), self.p, self.ops)

    def sum_receive(self, r):
        p, K, j = self.p, self.K, self.j
        phase1 = j < K
        jj = j if phase1 else j - K
        lab = self.in1 if phase1 else self.in2
        bit = (lab >> (K - jj - 1)) & 1
        rr = r[:, None]
        fac = np.where(bit == 1, rr, (1 - rr) % p)
        if phase1:
            self.beta = self.beta * fac % p
        else:
            self.gamma = self.gamma * fac % p
        half = self.tab.shape[1] // 2
        self.tab = (self.tab[:, :half] + (self.tab[:, half:] - self.tab[:, :half]) % p * rr) % p
        self.ops(self.w.shape[1] + 3 * half + 1)
        self.rs.append(r)
        self.j += 1
        if self.j == K:                                          # b bound: keep W~(b*), restart the table
            self.omega_b = self.tab[:, 0].copy()
            self.tab = np.broadcast_to(self.W[self.i + 1], (r.shape[0], 1 << K)).copy()
        if self.j == 2 * K:
            self.omega_c = self.tab[:, 0].copy()

    def _bc(self):
        if self.K == 0:
            e = np.zeros((self.w.shape[0], 0), dtype=np.int64)
            return e, e
        R = np.stack(self.rs, axis=1)
        return R[:, :self.K], R[:, self.K:]

    def line_values(self):
        """Values at t = 0..K of W~_{i+1} on the line through b* (t = 0) and c* (t = 1)."""
        p, K = self.p, self.K
        T = self.w.shape[0]
        if K == 0:
            v = np.full(T, self.W[self.i + 1][0], dtype=np.int64)
            self.omega_b = self.omega_c = v
            return v[:, None]
        b, c = self._bc()
        vals = [self.omega_b, self.omega_c]
        for t in range(2, K + 1):
            pt = (b + t * (c - b)) % p
            self.ops(3 * K)
            vals.append(mle(self.W[self.i + 1], pt, p, self.ops))
        return np.stack(vals, axis=1)

    def line_message(self, claim):
        return coeffs_from_values(self.line_values(), self.p, self.ops)


class RootPlanting(Honest):
    """Claims a false output; plants roots in every message while its current claim is false.
    target: the claimed value of output 0 (default: the true value plus 1)."""

    def __init__(self, L, W, p, ops=NOCOUNT, target=None, first_keep=(0, 2)):
        super().__init__(L, W, p, ops)
        self.target = target
        self.first_keep = first_keep                  # nodes kept in the first round of each phase
        self.stuck = {}                               # phase -> runs in which no line message can pass

    def outputs(self):
        y = self.W[0].copy()
        y[0] = (y[0] + 1) % self.p if self.target is None else self.target % self.p
        return y

    def sum_message(self, claim):
        v = self._values()
        if self.j == 0 and self.first_keep == (1, 2):
            v[:, 0] = (claim - v[:, 1]) % self.p      # keep the values at 1 and 2; fix the value at 0
        else:
            v[:, 1] = (claim - v[:, 0]) % self.p      # keep the values at 0 and 2; fix the value at 1
        return coeffs_from_values(v, self.p, self.ops)

    def line_message(self, claim):
        p, K = self.p, self.K
        vals = self.line_values()
        w0, w1 = vals[:, 0], vals[:, min(1, K)]          # for K = 0 the line is a single point
        b, c = self._bc()
        A, M = wiring_closed(self.L, self.i, self.z, b, c, p)
        eps = (claim - (A * ((w0 + w1) % p) + M * w0 % p * w1)) % p
        c0 = (A + M * w0) % p                         # keeping q(0), q(1) must change by eps / c0
        c1 = (A + M * w1) % p                         # keeping q(1), q(0) must change by eps / c1
        lie = eps != 0
        self.stuck[self.i] = self.stuck.get(self.i, 0) + int((lie & (c0 == 0) & (c1 == 0) & (M == 0)).sum())
        new = vals.copy()
        if K == 0:                                    # q is a constant: solve 2Aq + Mq^2 = claim
            for s in np.nonzero(lie)[0]:
                sols = [q for q in range(p) if (2 * A[s] * q + M[s] * q * q - claim[s]) % p == 0]
                if sols:
                    new[s, 0] = sols[0]
        else:
            u0 = lie & (c0 != 0)
            u1 = lie & (c0 == 0) & (c1 != 0)
            both = lie & (c0 == 0) & (c1 == 0) & (M != 0)
            new[u0, 1] = (w1[u0] + eps[u0] * _inv(c0[u0], p)) % p
            new[u1, 0] = (w0[u1] + eps[u1] * _inv(c1[u1], p)) % p
            new[both, 0] = (w0[both] + 1) % p
            new[both, 1] = (w1[both] + eps[both] * _inv(M[both], p)) % p
        return coeffs_from_values(new, p, self.ops)


def faulty_values(L, x, p, layer, gate=0, delta=1):
    """Gate values of a computation in which one gate of `layer` is off by delta, with the layers
    above it recomputed from the corrupted values."""
    W = evaluate(L, x, p)
    W[layer] = W[layer].copy()
    W[layer][gate] = (W[layer][gate] + delta) % p
    for i in range(layer - 1, -1, -1):
        a, b = W[i + 1][L[i]["in1"]], W[i + 1][L[i]["in2"]]
        W[i] = (a + b) % p if L[i]["op"] == "add" else a * b % p
    return W


# --------------------------------------------------------------------------- the verifier
def run(L, x, prover, p, T, gen, wiring="closed", vops=NOCOUNT, trace=None, coins=None):
    """Run Protocol 6.1 T times in parallel with independent coins. Returns (accept (T,) bool,
    field elements sent by the prover in one run, field elements sent by the verifier)."""
    D = len(L) - 1
    k = [l["k"] for l in L]
    wire = wiring_closed if wiring == "closed" else wiring_generic
    sent = recv = 0

    def coin(shape):
        if coins is not None:
            return np.broadcast_to(np.asarray(coins.pop(0), dtype=np.int64), shape).copy()
        return gen.integers(0, p, size=shape, dtype=np.int64)

    y = np.asarray(prover.outputs(), dtype=np.int64) % p
    sent += y.shape[0]
    z = coin((T, k[0]))
    recv += k[0]
    claim = mle(y, z, p, vops)
    alive = np.ones(T, dtype=bool)
    if trace is not None:
        trace.append(dict(step="output", y=[int(v) for v in y], claim=int(claim[0])))
    for i in range(D):
        K = k[i + 1]
        prover.start_phase(i, z, claim)
        rs = []
        for j in range(2 * K):
            g = prover.sum_message(claim)                    # 3 coefficients
            sent += 3
            ok = (2 * g[:, 0] + g[:, 1] + g[:, 2]) % p == claim
            alive &= ok
            vops(3)
            r = coin((T,))
            recv += 1
            new = horner(g, r, p, vops)
            if trace is not None:
                trace.append(dict(step=f"phase {i} round {j + 1}", g=[int(v) for v in g[0]], ok=bool(ok[0]), r=int(r[0]), claim=int(new[0])))
            claim = new
            prover.sum_receive(r)
            rs.append(r)
        q = prover.line_message(claim)                       # K+1 coefficients
        sent += K + 1
        if K:
            R = np.stack(rs, axis=1)
            b, c = R[:, :K], R[:, K:]
        else:
            b = c = np.zeros((T, 0), dtype=np.int64)
        A, M = wire(L, i, z, b, c, p, vops)
        q0 = q[:, 0]
        q1 = q.sum(axis=1) % p
        ok = claim == (A * ((q0 + q1) % p) + M * q0 % p * q1) % p
        alive &= ok
        vops(K + 5)
        t = coin((T,))
        recv += 1
        z = (b + (c - b) % p * t[:, None]) % p
        vops(3 * K)
        claim = horner(q, t, p, vops)
        if trace is not None:
            trace.append(dict(step=f"phase {i} line", q=[int(v) for v in q[0]], A=int(A[0]), M=int(M[0]), ok=bool(ok[0]), t=int(t[0]), z=[int(v) for v in z[0]], claim=int(claim[0])))
    final = mle(x, z, p, vops) == claim
    if trace is not None:
        trace.append(dict(step="final", xz=int(mle(x, z, p)[0]), claim=int(claim[0]), ok=bool(final[0])))
    return alive & final, sent, recv


# --------------------------------------------------------------------------- self-tests
def self_test():
    p = 1009
    g = pk.rng(1)
    for m in (0, 1, 2, 3):
        L = network_circuit(m)
        D = len(L) - 1
        x = g.integers(0, p, size=L[D]["S"])
        W = evaluate(L, x, p)
        for i in range(D):
            K = L[i + 1]["k"]
            z = g.integers(0, p, size=(3, L[i]["k"]))
            b = g.integers(0, p, size=(3, K))
            c = g.integers(0, p, size=(3, K))
            A1, M1 = wiring_closed(L, i, z, b, c, p)
            A2, M2 = wiring_generic(L, i, z, b, c, p)
            assert np.all(A1 == A2) and np.all(M1 == M2), ("closed form", m, i)
            if m <= 1:                                       # the wiring identity of the chapter
                tot = np.zeros(3, dtype=np.int64)
                for bb in range(1 << K):
                    for cc in range(1 << K):
                        bv = np.array([[(bb >> (K - 1 - u)) & 1 for u in range(K)]] * 3, dtype=np.int64).reshape(3, K)
                        cv = np.array([[(cc >> (K - 1 - u)) & 1 for u in range(K)]] * 3, dtype=np.int64).reshape(3, K)
                        A, M = wiring_closed(L, i, z, bv, cv, p)
                        wb, wc = W[i + 1][bb], W[i + 1][cc]
                        tot = (tot + A * ((wb + wc) % p) + M * (wb * wc % p)) % p
                assert np.all(tot == mle(W[i], z, p)), ("identity", m, i)
        acc, _, _ = run(L, x, Honest(L, W, p), p, 40, g)
        assert acc.all(), ("completeness", m)
    print("self-test passed: closed-form wiring = generic, wiring identity, completeness")


# --------------------------------------------------------------------------- the worked run
def worked_run():
    """N = 1: y = (w x)^2 over F_13 with w = 2, x = 3; coins t = 0 in phase 0; r = (4, 7), t* = 5."""
    p = 13
    L = network_circuit(0)
    x = np.array([2, 3])
    W = evaluate(L, x, p)
    print("worked run over F_13: gate values by layer", [list(map(int, w)) for w in W])
    for name, prov in (("honest", Honest(L, W, p)), ("root-planting, claims 12", RootPlanting(L, W, p, target=12)),
                       ("claims 11", RootPlanting(L, W, p, target=11))):
        tr = []
        coins = [np.zeros((1, 0), dtype=np.int64), 0, 4, 7, 5]
        acc, _, _ = run(L, x, prov, p, 1, None, coins=coins, trace=tr)
        print(f"  {name}: accepted = {bool(acc[0])}")
        for row in tr:
            print("     ", row)
    # exact acceptance of the root-planting prover claiming 12, over all coins (phase 1 only); the
    # variant that keeps the honest values at 1 and 2 in round 1 attains the product bound exactly
    cnts = {}
    for keep in ((0, 2), (1, 2)):
        cnt = 0
        for r1 in range(p):
            for r2 in range(p):
                for t in range(p):
                    acc, _, _ = run(L, x, RootPlanting(L, W, p, target=12, first_keep=keep), p, 1, None,
                                    coins=[np.zeros((1, 0), dtype=np.int64), 0, r1, r2, t])
                    cnt += int(acc[0])
        cnts[keep] = cnt
    cnt = cnts[(0, 2)]
    print(f"  root-planting prover claiming 12: accepted on {cnt} of {p ** 3} coin triples "
          f"({cnt / p ** 3:.4f}); product bound 1-(11/13)^2(12/13) = {1 - (11 / 13) ** 2 * (12 / 13):.4f} "
          f"= {round((1 - (11 / 13) ** 2 * (12 / 13)) * p ** 3)}/{p ** 3}, sum bound 5/13 = {5 / 13:.4f}")
    print(f"  the same liar keeping the honest values at 1 and 2 in round 1: accepted on {cnts[(1, 2)]} "
          f"of {p ** 3} coin triples")


def run_t0(L, x, prover, p, T, gen):
    """Protocol 6.1 with a verifier that always takes t* = 0 (exercise: a verifier that checks one point)."""
    D = len(L) - 1
    k = [l["k"] for l in L]
    coins = [gen.integers(0, p, size=(T, k[0]), dtype=np.int64)]
    for i in range(D):
        coins += [gen.integers(0, p, size=T, dtype=np.int64) for _ in range(2 * k[i + 1])]
        coins.append(np.zeros(T, dtype=np.int64))
    return run(L, x, prover, p, T, None, coins=coins)


def primes_from(lo):
    n = lo
    while any(n % d == 0 for d in range(2, int(n ** 0.5) + 1)):
        n += 1
    return n


def theory(k, m, p):
    prod = 1 - (1 - 1 / p) ** k[0] * np.prod([(1 - 2 / p) ** (2 * ki) * (1 - ki / p) for ki in k[1:]])
    return prod, 1 - (1 - 1 / p) ** k[m], (k[0] + 5 * sum(k[1:])) / p


def main():
    t0 = time.time()
    self_test()
    worked_run()
    m = 2
    L = network_circuit(m)
    D = len(L) - 1
    k = [l["k"] for l in L]
    total = k[0] + 5 * sum(k[1:])
    print(f"circuit N = {1 << m}: depth D = {D}, size S = {size(L)}, n = {L[D]['S']}, k_i = {k}, "
          f"sum bound = {total}/p")
    xint = pk.rng(6).integers(-7, 8, size=L[D]["S"])
    print("  input W then X:", list(map(int, xint)))
    primes = [primes_from(v) for v in (150, 250, 400, 630, 1000, 1600, 2500, 4000, 6300, 10000)]
    print("  primes:", primes)
    rows = []
    TRIALS, HTRIALS = 20000, 2000
    for idx, p in enumerate(primes):
        x = xint % p
        W = evaluate(L, x, p)
        res = {}
        for s, (name, prov, n) in enumerate((("honest", Honest(L, W, p), HTRIALS),
                                             ("plant", RootPlanting(L, W, p), TRIALS),
                                             ("fault", Honest(L, faulty_values(L, x, p, layer=m), p), TRIALS))):
            acc, _, _ = run(L, x, prov, p, n, pk.rng(600 + 10 * idx + s))
            kk = int(acc.sum())
            lo, hi = pk.clopper_pearson(kk, n)
            res[name] = (kk / n, lo, hi, kk, n)
            if name == "plant" and p in (151, 1009):
                print(f"  p={p}: root-planting liar stuck at a line step (wiring extensions vanish), "
                      f"fraction of runs per phase: " + ", ".join(f"{prov.stuck.get(i, 0) / n:.4f}" for i in range(D)))
        prod, fault, bound = theory(k, m, p)
        print(f"  p={p:6d}: honest {res['honest'][0]:.3f} | root-planting {res['plant'][0]:.4f} "
              f"[{res['plant'][1]:.4f}, {res['plant'][2]:.4f}] product bound {prod:.4f} sum bound {bound:.4f} "
              f"| fault {res['fault'][0]:.4f} [{res['fault'][1]:.4f}, {res['fault'][2]:.4f}] exact {fault:.4f}"
              f"  ({time.time() - t0:.0f} s)")
        rows.append([p, res["honest"][0],
                     res["plant"][0], res["plant"][2] - res["plant"][0], res["plant"][0] - res["plant"][1],
                     res["fault"][0], res["fault"][2] - res["fault"][0], res["fault"][0] - res["fault"][1],
                     prod, fault, bound, res["plant"][3], res["fault"][3]])
    pk.write_dat("figures/data/06-delegation-gkr.dat",
                 ["p", "honest", "plant", "plantep", "plantem", "fault", "faultep", "faultem",
                  "exactplant", "exactfault", "bound", "plantk", "faultk"], rows)
    th = []
    for pp in np.geomspace(primes[0] * 0.9, primes[-1] * 1.1, 80):
        prod, fault, bound = theory(k, m, pp)
        th.append([pp, min(bound, 1.0), prod, fault])
    pk.write_dat("figures/data/06-delegation-gkr-theory.dat", ["p", "bound", "product", "fault"], th)
    # exercise: a verifier that always takes t* = 0, against the root-planting prover
    for idx, p in enumerate((151, 1009)):
        x = xint % p
        W = evaluate(L, x, p)
        acc, _, _ = run_t0(L, x, RootPlanting(L, W, p), p, TRIALS, pk.rng(700 + idx))
        print(f"  verifier with t* = 0, p={p}: root-planting prover accepted in {acc.mean():.4f} of {TRIALS} runs")
    # exercise: spot checks of a transcript with one wrong gate
    g4 = size(L) - L[D]["S"]
    L64 = network_circuit(6)
    g64 = size(L64) - L64[-1]["S"]
    print(f"  spot checks, N=4 ({g4} non-input gates): q=10 -> {1 - (1 - 1 / g4) ** 10:.3f}, "
          f"q=100 -> {1 - (1 - 1 / g4) ** 100:.3f}; N=64 ({g64} gates) needs "
          f"q={int(np.ceil(np.log(0.01) / np.log(1 - 1 / g64)))} for 0.99")
    print(f"  N=4: skipping the last line gives bound {k[0] + 5 * sum(k[1:D]) + 4 * k[D]}/p instead of {total}/p")
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
