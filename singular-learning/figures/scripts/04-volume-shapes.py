"""04-volume-shapes.py -- Chapter 4 (Volume), "By hand": the sets of nearly true parameters.

Writes the boundary of {K < eps} inside the prior box [-1, 1]^2 for eps = 10^-1, 10^-2, 10^-3 and
four models, as closed polygons of 400 points (columns x1 y1 x2 y2 x3 y3, one pair per eps):

    figures/data/04-volume-shapes-reg.dat    regular plane  K = (a^2 + b^2)/2   (discs)
    figures/data/04-volume-shapes-cup.dat    quartic cup    K = (a^4 + b^4)/2   (squircles)
    figures/data/04-volume-shapes-prod.dat   product model  K = a^2 b^2 / 2     (crosses)
    figures/data/04-volume-shapes-quart.dat  quartic model  K = (a^2 + b^4)/2   (lenses)

and prints the prior mass V(eps) of each set: exactly, and by the shoelace area of the polygon.
Run from the volume root: python figures/scripts/04-volume-shapes.py   (about 1 s)
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sltkit as sk  # noqa: E402

OUT = "figures/data"
NPTS = 400
EPS = (1e-1, 1e-2, 1e-3)
G = math.gamma
C_CUP = math.sqrt(2) * G(1.25) ** 2 / G(1.5)
C_QUA = 2 ** 0.75 * G(1.5) * G(1.25) / G(1.75)


def ssqrt(v):
    """sign(v) |v|^(1/2)"""
    return np.sign(v) * np.sqrt(np.abs(v))


def curve(model, eps):
    t = np.linspace(0.0, 2 * np.pi, NPTS, endpoint=False)
    if model == "reg":
        r = math.sqrt(2 * eps)
        return r * np.cos(t), r * np.sin(t)
    if model == "cup":
        R = (2 * eps) ** 0.25
        return R * ssqrt(np.cos(t)), R * ssqrt(np.sin(t))
    if model == "quart":
        return math.sqrt(2 * eps) * np.cos(t), (2 * eps) ** 0.25 * ssqrt(np.sin(t))
    if model == "prod":
        s = math.sqrt(2 * eps)
        n_arc = NPTS // 4 - 2                      # points per hyperbola arc; 2 corner points per quadrant
        a = np.exp(np.linspace(0.0, math.log(s), n_arc))   # from a = 1 down to a = s
        xs, ys = [], []
        for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            # in quadrant (sx, sy) the boundary runs along |b| = s/|a|; traverse counterclockwise
            arc_a, arc_b = a, s / a
            if sx * sy < 0:
                arc_a, arc_b = arc_a[::-1], arc_b[::-1]
            xs.extend(sx * arc_a); ys.extend(sy * arc_b)
            # the box edge to the next quadrant
            if (sx, sy) == (1, 1):
                xs.extend([s, -s]); ys.extend([1.0, 1.0])
            elif (sx, sy) == (-1, 1):
                xs.extend([-1.0, -1.0]); ys.extend([s, -s])
            elif (sx, sy) == (-1, -1):
                xs.extend([-s, s]); ys.extend([-1.0, -1.0])
            else:
                xs.extend([1.0, 1.0]); ys.extend([-s, s])
        return np.array(xs), np.array(ys)
    raise ValueError(model)


def exact_V(model, eps):
    if model == "reg":
        return math.pi / 2 * eps
    if model == "cup":
        return C_CUP * eps ** 0.5
    if model == "quart":
        return C_QUA * eps ** 0.75
    s = math.sqrt(2 * eps)
    return s * (1 + math.log(1 / s))


def main():
    os.makedirs(OUT, exist_ok=True)
    for model in ("reg", "cup", "prod", "quart"):
        cols = []
        for eps in EPS:
            x, y = curve(model, eps)
            assert len(x) == NPTS, (model, len(x))
            area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            print(f"{model:6s} eps={eps:g}: V exact {exact_V(model, eps):.4f}, polygon {area / 4:.4f}")
            cols += [x, y]
        sk.write_dat(f"{OUT}/04-volume-shapes-{model}.dat", ["x1", "y1", "x2", "y2", "x3", "y3"],
                     np.column_stack(cols))


if __name__ == "__main__":
    main()
