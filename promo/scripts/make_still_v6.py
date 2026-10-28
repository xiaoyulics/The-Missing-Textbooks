#!/usr/bin/env python3
"""
The README still for volume VI (docs/img/stills/v6.jpg, 1280 x 720).

Volume VI has no act in the film, so its still is drawn from the book's own computation: exact draws
from the posterior of y = abx fitted to pure noise (the sampler of the book's shared kit,
singular-learning/figures/scripts/sltkit.py), which crowd onto the cross ab = 0 instead of forming the
Gaussian blob that the regular theory expects. Needs NumPy and Pillow.
Run from the repo root:  python3 promo/scripts/make_still_v6.py
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "singular-learning", "figures", "scripts"))
import sltkit as sk  # noqa: E402

W, H = 1280, 720
N, DRAWS, SEED = 10_000, 9000, 6


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


def draws():
    x, y = sk.PRODUCT.sample([0.0, 0.0], N, seed=SEED)
    sk.PRODUCT.fit(x, y)
    return sk.sample_product_posterior(sk.PRODUCT, DRAWS, seed=100 + SEED)


def main():
    # background: a dark radial gradient
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy, side = 915, 360, 560                       # the square [-1, 1]^2 on the canvas
    r = np.hypot((xx - cx) / W, (yy - cy) / H)
    base = np.array([7, 9, 15], float)[None, None, :] + np.array([12, 13, 18], float)[None, None, :] * np.clip(1 - 1.6 * r, 0, 1)[..., None]
    img = Image.fromarray(base.astype(np.uint8), "RGB")

    # faint frame of the prior's square and the cross ab = 0, in slate
    d = ImageDraw.Draw(img)
    half = side // 2
    d.rectangle([cx - half, cy - half, cx + half, cy + half], outline=(38, 50, 64), width=1)
    d.line([cx - half, cy, cx + half, cy], fill=(46, 64, 84), width=1)
    d.line([cx, cy - half, cx, cy + half], fill=(46, 64, 84), width=1)

    # the draws, splatted additively and given a glow
    ab = draws()
    px = np.round(cx + ab[:, 0] * half).astype(int)
    py = np.round(cy - ab[:, 1] * half).astype(int)
    acc = np.zeros((H, W), float)
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    np.add.at(acc, (py[ok], px[ok]), 1.0)
    core = Image.fromarray(np.clip(acc * 255, 0, 255).astype(np.uint8), "L")
    halo1 = core.filter(ImageFilter.GaussianBlur(1.6))
    halo2 = core.filter(ImageFilter.GaussianBlur(7))
    a0 = np.asarray(core, float) / 255
    a1 = np.asarray(halo1, float) / 255
    a2 = np.asarray(halo2, float) / 255
    light = (np.array([255, 226, 170])[None, None, :] * np.clip(a0 * 0.55, 0, 1)[..., None]
             + np.array([232, 160, 64])[None, None, :] * np.clip(a1 * 2.6, 0, 1)[..., None]
             + np.array([150, 96, 18])[None, None, :] * np.clip(a2 * 5.0, 0, 1)[..., None])
    out = np.clip(np.asarray(img, float) + light, 0, 255).astype(np.uint8)
    img = Image.fromarray(out, "RGB")

    # captions
    d = ImageDraw.Draw(img)
    big = face(SERIF_IT, 58, "italic")
    small = face(MONO, 19)
    d.text((80, 252), "Two parameters", font=big, fill=(236, 240, 246))
    d.text((80, 324), "that learn like one.", font=big, fill=(232, 160, 64))
    d.text((84, 432), f"y = abx fitted to pure noise", font=small, fill=(120, 140, 160))
    d.text((84, 462), f"{DRAWS} exact posterior draws, n = 10⁴", font=small, fill=(120, 140, 160))
    d.text((84, 492), "λ = 1/2, m = 2", font=small, fill=(120, 140, 160))
    d.text((cx + half - 30, cy + 8), "a", font=small, fill=(90, 110, 130))
    d.text((cx + 16, cy - half + 2), "b", font=small, fill=(90, 110, 130))

    dest = os.path.join(ROOT, "docs", "img", "stills", "v6.jpg")
    img.save(dest, quality=86, optimize=True, progressive=True)
    print("wrote", dest)


if __name__ == "__main__":
    main()
