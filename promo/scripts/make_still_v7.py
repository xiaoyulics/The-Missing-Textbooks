#!/usr/bin/env python3
"""
The README still for volume VII (docs/img/stills/v7.jpg, 1280 x 720).

Volume VII has no act in the film, so its still is drawn from the book's own computation: the largest
margin of n = alpha d random labeled Gaussian points in dimension d = 200 (the prologue's measurements by
quadratic programs, replica-rosetta/figures/data/00-prologue-margin.dat), against Gardner's replica formula,
glowing lapis where it is a theorem (alpha < 2, margin >= 0) and dotted carnelian where it is only an
upper bound (alpha > 2). Needs NumPy and Pillow.
Run from the repo root:  python3 promo/scripts/make_still_v7.py
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "replica-rosetta", "figures", "data")
W, H = 1280, 720
X0, X1, Y0, Y1 = 640, 1210, 110, 630          # plot box on the canvas
AMIN, AMAX, KMIN, KMAX = 0.0, 4.0, -1.2, 2.1   # alpha and kappa ranges
LAPIS, CARNELIAN, INK = (96, 150, 222), (222, 112, 82), (236, 240, 246)


def first_font(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[-1]


SERIF_IT = first_font("/System/Library/Fonts/Palatino.ttc", "C:/Windows/Fonts/palai.ttf",
                      "/usr/share/fonts/opentype/tex-gyre/texgyrepagella-italic.otf")
MONO = first_font("/System/Library/Fonts/Menlo.ttc", "C:/Windows/Fonts/consola.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")


def face(path, size, want=None):
    """A font face; in a .ttc collection, the face whose name contains `want`."""
    if path.endswith(".ttc") and want:
        for i in range(8):
            try:
                f = ImageFont.truetype(path, size, index=i)
                if want in " ".join(f.getname()).lower():
                    return f
            except Exception:
                break
    return ImageFont.truetype(path, size)


def table(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        head = fh.readline().split()
        rows = np.array([[float(v) for v in line.split()] for line in fh if line.strip()])
    return {h: rows[:, i] for i, h in enumerate(head)}


def to_px(alpha, kappa):
    x = X0 + (np.asarray(alpha) - AMIN) / (AMAX - AMIN) * (X1 - X0)
    y = Y1 - (np.asarray(kappa) - KMIN) / (KMAX - KMIN) * (Y1 - Y0)
    return x, y


def glow(layer, colour, strengths=(0.9, 2.2, 4.0), radii=(0.0, 1.8, 7.0)):
    """Additive light from a white-on-black line layer, in one colour."""
    a = [np.asarray(layer.filter(ImageFilter.GaussianBlur(r)) if r else layer, float) / 255 for r in radii]
    c = np.array(colour, float)[None, None, :]
    return sum(np.clip(ai * s, 0, 1)[..., None] * c * w for ai, s, w in zip(a, strengths, (1.0, 0.8, 0.45)))


def main():
    # background: a dark granite gradient
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot((xx - 925) / W, (yy - 370) / H)
    base = np.array([10, 9, 14], float)[None, None, :] + np.array([16, 13, 20], float)[None, None, :] * np.clip(1 - 1.5 * r, 0, 1)[..., None]
    img = Image.fromarray(base.astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)

    # axes and the guide lines kappa = 0, alpha = 2
    ax_col, guide = (58, 52, 66), (74, 66, 84)
    x0, y0 = to_px(AMIN, 0.0)
    x1, _ = to_px(AMAX, 0.0)
    d.line([x0, y0, x1, y0], fill=guide, width=1)
    x2, ytop = to_px(2.0, KMAX)
    _, ybot = to_px(2.0, KMIN)
    for yv in np.arange(ytop, ybot, 9):
        d.line([x2, yv, x2, yv + 4], fill=guide, width=1)
    d.line([X0, Y0, X0, Y1], fill=ax_col, width=1)
    d.line([X0, Y1, X1, Y1], fill=ax_col, width=1)

    # Gardner's formula: lapis where proved, dotted carnelian where only an upper bound
    pos, neg = table("00-prologue-margin-curve.dat"), table("00-prologue-margin-curveneg.dat")
    lay_p = Image.new("L", (W, H), 0)
    lay_n = Image.new("L", (W, H), 0)
    keep = (pos["kappa"] <= KMAX) & (pos["alpha"] >= AMIN)
    px, py = to_px(pos["alpha"][keep], pos["kappa"][keep])
    ImageDraw.Draw(lay_p).line(list(zip(px, py)), fill=255, width=3)
    keep = neg["alpha"] <= AMAX
    nx, ny = to_px(neg["alpha"][keep], neg["kappa"][keep])
    dn = ImageDraw.Draw(lay_n)
    pts = list(zip(nx, ny))
    seg = np.cumsum([0] + [np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)])
    for s in np.arange(0, seg[-1], 11.0):            # dots every 11 px along the curve
        i = min(np.searchsorted(seg, s, side="right") - 1, len(pts) - 2)
        t = (s - seg[i]) / max(seg[i + 1] - seg[i], 1e-9)
        x = pts[i][0] + t * (pts[i + 1][0] - pts[i][0])
        y = pts[i][1] + t * (pts[i + 1][1] - pts[i][1])
        dn.ellipse([x - 2.2, y - 2.2, x + 2.2, y + 2.2], fill=255)
    light = glow(lay_p, LAPIS) + glow(lay_n, CARNELIAN, strengths=(0.9, 1.6, 2.6))
    img = Image.fromarray(np.clip(np.asarray(img, float) + light, 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)

    # the measurements: largest margins at d = 200, with 95% intervals
    m = table("00-prologue-margin.dat")
    mx, my = to_px(m["alpha"], m["kappa"])
    _, lo = to_px(m["alpha"], m["kappa"] - m["half"])
    _, hi = to_px(m["alpha"], m["kappa"] + m["half"])
    for x, y, a, b in zip(mx, my, lo, hi):
        d.line([x, a, x, b], fill=(200, 204, 212), width=1)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=INK, outline=(14, 12, 18))

    # captions
    big = face(SERIF_IT, 58, "italic")
    small = face(MONO, 19)
    tiny = face(MONO, 16)
    d.text((80, 232), "One number,", font=big, fill=INK)
    d.text((80, 304), "two languages.", font=big, fill=LAPIS)
    grey = (128, 122, 140)
    d.text((84, 412), "the capacity of the perceptron: \u03b1 = 2", font=small, fill=grey)
    d.text((84, 442), "replicas (Gardner 1988), proved by comparison", font=small, fill=grey)
    d.text((84, 472), "largest margin at d = 200, quadratic programs", font=small, fill=grey)
    d.text((84, 502), "solid: a theorem; dotted: an upper bound only", font=small, fill=grey)
    d.text((x2 + 7, Y1 - 24), "\u03b1 = 2", font=tiny, fill=(110, 102, 122))
    d.text((X1 - 18, y0 + 6), "\u03b1", font=tiny, fill=(110, 102, 122))
    d.text((X0 + 8, Y0 - 4), "\u03ba", font=tiny, fill=(110, 102, 122))

    dest = os.path.join(ROOT, "docs", "img", "stills", "v7.jpg")
    img.save(dest, quality=86, optimize=True, progressive=True)
    print("wrote", dest)


if __name__ == "__main__":
    main()
