"""Prologue figure: gradient descent and sign descent on one diagonal quadratic.

L(theta) = 1/2 sum_i h_i theta_i^2 with h_i = 1/i^2, i = 1..d, d = 100, so h_max = 1, h_min = 1e-4, tr H = 1.635.
Initial point theta_0,i uniform on [0.5, 1.5] (seed 0).

Runs (deterministic, so "simulation" means iterating the update rule):
  gd      gradient descent, eta = 1.9 (just below the stability threshold 2/h_max = 2);
          theory curve = normal-mode formula 1/2 sum_i h_i (1 - eta h_i)^(2t) theta_0,i^2.
  sign    sign descent theta <- theta - eta sign(theta), eta = 0.02 (constant);
          theory curve = the coordinatewise closed form (|theta_i| falls by eta per step until it is below
          eta, then alternates between a_i and a_i - eta, with a_i = |theta_0,i| mod eta).
  signlin sign descent with the linearly decaying schedule eta_t = eta (1 - t/T), T = 400.
Also printed: the predicted arrival time max_i |theta_0,i| / eta, the expected cycle-averaged floor
tr(H) eta^2 / 6 for remainders |theta_0,i| mod eta uniform on [0, eta), the two losses of the sign
two-cycle, and the first step at which gradient descent goes below their mean (about 6500).
Output: figures/data/00-prologue-quadratic.dat (t, gd_exact, gd_sim, sign_exact, sign_sim, signlin_sim)
"""
import os
import numpy as np

d = 100
h = 1.0 / np.arange(1, d + 1) ** 2
rng = np.random.default_rng(0)
theta0 = rng.uniform(0.5, 1.5, size=d)
T = 400

def loss(theta):
    return 0.5 * np.sum(h * theta ** 2)

# gradient descent
eta_gd = 1.9
gd_exact = np.array([0.5 * np.sum(h * (1 - eta_gd * h) ** (2 * t) * theta0 ** 2) for t in range(T + 1)])
theta = theta0.copy(); gd_sim = [loss(theta)]
for t in range(T):
    theta = theta - eta_gd * h * theta
    gd_sim.append(loss(theta))

# sign descent, constant step
eta = 0.02
def sign_closed_form(t):
    out = np.empty(d)
    for i in range(d):
        x0 = theta0[i]
        k = int(np.floor(x0 / eta))          # steps while |theta| >= eta
        if t <= k:
            out[i] = x0 - t * eta
        else:
            a = x0 - k * eta                  # in [0, eta)
            out[i] = a if (t - k) % 2 == 0 else a - eta
    return out
sign_exact = [loss(sign_closed_form(t)) for t in range(T + 1)]
theta = theta0.copy(); sign_sim = [loss(theta)]
for t in range(T):
    theta = theta - eta * np.sign(theta)
    sign_sim.append(loss(theta))

# sign descent, linear decay to zero
theta = theta0.copy(); signlin_sim = [loss(theta)]
for t in range(T):
    theta = theta - eta * (1 - t / T) * np.sign(theta)
    signlin_sim.append(loss(theta))

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "..", "data", "00-prologue-quadratic.dat")
with open(out, "w") as f:
    f.write("t gd_exact gd_sim sign_exact sign_sim signlin_sim\n")
    for t in range(T + 1):
        f.write(f"{t} {gd_exact[t]:.6e} {gd_sim[t]:.6e} {sign_exact[t]:.6e} {sign_sim[t]:.6e} {signlin_sim[t]:.6e}\n")
print("arrival time max|theta0|/eta =", theta0.max() / eta)
print("floor guide tr(H) eta^2/6 =", h.sum() * eta ** 2 / 6)
print("max |gd_exact - gd_sim| =", np.max(np.abs(gd_exact - np.array(gd_sim))))
print("max |sign_exact - sign_sim| =", np.max(np.abs(np.array(sign_exact) - np.array(sign_sim))))
print("final losses: gd %.3e sign %.3e signlin %.3e" % (gd_sim[-1], sign_sim[-1], signlin_sim[-1]))
# numbers quoted in the caption and in the text after the figure
lo, hi = sorted(sign_sim[-2:])
mean_cycle = 0.5 * (lo + hi)
print("sign two-cycle losses %.3e and %.3e, mean %.3e" % (lo, hi, mean_cycle))
t_gd = next(t for t in range(10 ** 6)
            if 0.5 * np.sum(h * (1 - eta_gd * h) ** (2 * t) * theta0 ** 2) < mean_cycle)
print("first step at which gradient descent goes below that mean:", t_gd)
