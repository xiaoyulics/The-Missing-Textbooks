"""Gradient descent on a two-dimensional quadratic: the prologue's first hydrogen atom.

L(theta) = (1/2)(lam1*x^2 + lam2*y^2) with lam1 = 1, lam2 = 10, so lambda_max = 10 and the
stability threshold is eta = 2/lambda_max = 0.2. Writes trajectories for three step sizes
(eta = 0.05, 0.17, 0.205) and the loss along each, together with the exact prediction
L_t = (1/2) sum_k lam_k (1 - eta lam_k)^(2t) e_k(0)^2.
"""
import numpy as np, os
out = os.path.join(os.path.dirname(__file__), '..', 'data')
lam = np.array([1.0, 10.0])
theta0 = np.array([-2.6, 0.55])
etas = {'a': 0.05, 'b': 0.17, 'c': 0.205}
T = 40
for tag, eta in etas.items():
    th = theta0.copy(); rows = []
    for t in range(T + 1):
        loss = 0.5 * np.sum(lam * th**2)
        exact = 0.5 * np.sum(lam * ((1 - eta * lam)**(2 * t)) * theta0**2)
        rows.append((t, th[0], th[1], loss, exact))
        th = th - eta * lam * th
    with open(os.path.join(out, f'00-prologue-gd-{tag}.dat'), 'w') as f:
        f.write('t x y loss exact\n')
        for r in rows:
            f.write('%d %.6f %.6f %.6e %.6e\n' % r)
print('ok')
