"""Figure 10.1: clipped versus fixed-step gradient descent on a cliff followed by a plateau.

The loss is f(x, y) = f_ext(x) + f_ph(y) with L0 = L1 = 1, where
  f_ext(x) = (L0/L1^2) (exp(L1 |x|) - 1 - L1 |x|)      (the extremal (L0,L1)-smooth function: a cliff)
  f_ph(y)  = (4 eps^2/L0) (sqrt(1 + (L0 y/(2 eps))^2) - 1)   (a plateau of slope below 2 eps)
The sum is (L0,L1)-smooth in the Euclidean norm. Gradient descent starts at (x0, y0) with
|f_ext'(x0)| = M and f_ph(y0) = Dy, and stops when the gradient norm is at most eps.

For each initial gradient norm M the script records
  * the largest step size for which fixed-step GD does not diverge (bisection), against the exact
    threshold 2 log(1 + L1 M/L0)/(L1 M);
  * the largest step size for which clipped GD (gamma = 1/(3 L1)) converges to the minimizer (scan,
    gradient norm below 1e-9), against the band
    [L1 gamma/(L0 (exp(L1 gamma/2) - 1)), 2/L0] predicted for every start;
  * the number of steps to reach |grad| <= eps for GD at 0.9 times its threshold and for clipped GD
    with eta = 1/(2 L0), gamma = 1/(3 L1), against the predictions T_flow/eta (plus the number of
    clipped steps on the cliff, 3 (log(1 + P) - log(5/3)), for clipped GD).

Output: figures/data/10-spikes-separation.dat. Deterministic (no randomness); about 20 s on one core.
"""
import math
import os

L0 = 1.0
L1 = 1.0
EPS = 0.05
DY = 1.0
GAMMA = 1.0 / (3.0 * L1)
ETA_CLIP = 1.0 / (2.0 * L0)
KAPPA = 0.9  # GD runs at this fraction of its divergence threshold


def fext_p(x):
    return math.copysign((L0 / L1) * math.expm1(L1 * abs(x)), x)


def fph_p(y):
    s = L0 * y / (2.0 * EPS)
    return 2.0 * EPS * s / math.sqrt(1.0 + s * s)


def y_start(D):
    a = L0 * D / (4.0 * EPS ** 2)
    return 2.0 * EPS * math.sqrt((1.0 + a) ** 2 - 1.0) / L0


def flow_time(D):
    """Gradient-flow time on the plateau from f_ph = D down to |f_ph'| = eps (closed form)."""
    a = L0 * D / (4.0 * EPS ** 2)
    s0 = math.sqrt((1.0 + a) ** 2 - 1.0)
    se = 1.0 / math.sqrt(3.0)
    F = lambda s: math.sqrt(1.0 + s * s) + math.log(s / (1.0 + math.sqrt(1.0 + s * s)))
    return (F(s0) - F(se)) / L0


def steps_to_eps(eta, x0, y0, gamma=None, nmax=20_000_000, tol=EPS):
    """Steps of (clipped) GD until the gradient norm is <= tol; -1 if it diverges or does not finish."""
    x, y = x0, y0
    for t in range(nmax):
        if abs(x) > 2.0 * x0 + 10.0 / L1:
            return -1
        gx, gy = fext_p(x), fph_p(y)
        n = math.hypot(gx, gy)
        if n <= tol:
            return t
        et = eta if (gamma is None or n == 0.0) else min(eta, gamma / n)
        x -= et * gx
        y -= et * gy
    return -1


def gd_diverges(eta, x0, y0, steps=400):
    """Exact criterion from the proof: below the threshold |x_t| decreases strictly; at or above it,
    |x_1| >= |x_0|. Checked over a few steps of the actual 2-D iteration."""
    x, y = x0, y0
    for t in range(steps):
        gx, gy = fext_p(x), fph_p(y)
        x -= eta * gx
        y -= eta * gy
        if abs(x) >= x0:
            return True
        if abs(x) > 2.0 * x0 + 10.0 / L1:
            return True
    return False


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "data", "10-spikes-separation.dat")
    y0 = y_start(DY)
    tflow = flow_time(DY)
    clip_lo = L1 * GAMMA / (L0 * math.expm1(L1 * GAMMA / 2.0))
    clip_hi = 2.0 / L0
    Ms = [10 ** (k / 4.0) for k in range(0, 25)]  # 1 ... 1e6
    rows = []
    for M in Ms:
        P = L1 * M / L0
        x0 = math.log1p(P) / L1
        thr = 2.0 * math.log1p(P) / (L1 * M)
        # bisection for GD's divergence threshold
        lo, hi = 0.0, 2.5 / L0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if gd_diverges(mid, x0, y0):
                hi = mid
            else:
                lo = mid
        thr_sim = 0.5 * (lo + hi)
        # scan for clipped GD's largest convergent step (all smaller grid values must converge too)
        clip_sim = None
        eta = 1.50
        while eta <= 2.2:
            if steps_to_eps(eta, x0, y0, gamma=GAMMA, nmax=200_000, tol=1e-9) < 0:
                break
            clip_sim = eta
            eta += 0.0025
        # steps to eps
        eta_gd = KAPPA * thr
        n_gd = steps_to_eps(eta_gd, x0, y0)
        n_clip = steps_to_eps(ETA_CLIP, x0, y0, gamma=GAMMA)
        gd_theory = tflow / eta_gd
        wall = max(0.0, (math.log1p(P) - math.log(1.0 + (GAMMA / ETA_CLIP) * L1 / L0)) / (L1 * GAMMA))
        clip_theory = tflow / ETA_CLIP + wall
        rows.append((M, gd_theory, n_gd, clip_theory, n_clip, thr, thr_sim, clip_sim, clip_lo, clip_hi))
        print(f"M={M:10.3g}  thr={thr:.5g} (bisect {thr_sim:.5g})  clip-thr={clip_sim}  "
              f"GD steps {n_gd} (theory {gd_theory:.0f})  clipped {n_clip} (theory {clip_theory:.1f})")
    with open(out, "w") as fh:
        fh.write("M gdtheory gdsim cliptheory clipsim thrtheory thrsim clipthr cliplo cliphi\n")
        for r in rows:
            fh.write(" ".join(f"{v:.6g}" for v in r) + "\n")
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
