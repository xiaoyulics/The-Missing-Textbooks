#!/usr/bin/env python3
"""
Builds the images used by the repository README (docs/img/): banner, social preview, one still per volume,
cover thumbnails with a soft shadow, and the shelf in a light and a dark variant.

Inputs are produced by the film: promo/out/posters*/ (stills) and promo/public/covers/ (scripts/prepare_assets.py).
Run from the repo root:  python3 promo/scripts/make_readme_assets.py   (stills named below must exist)
"""
import os, sys
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
P = os.path.join(ROOT, "promo")
OUT = os.path.join(ROOT, "docs", "img")
os.makedirs(f"{OUT}/covers", exist_ok=True)
os.makedirs(f"{OUT}/stills", exist_ok=True)

SCRATCH = sys.argv[1] if len(sys.argv) > 1 else None   # folder holding the extra stills (frames 680, 1640, 2216 of PromoEN)

# ---------------------------------------------------------------- banner and social preview
zh = Image.open(f"{P}/out/posters/1-title-lockup.png").convert("RGB")
banner = zh.crop((0, 92, 1920, 872)).resize((1800, 731), Image.LANCZOS)
banner.save(f"{OUT}/banner.jpg", quality=88, optimize=True, progressive=True)

en = Image.open(f"{P}/out/posters-en/1-title-lockup.png").convert("RGB")
en.crop((0, 60, 1920, 1020)).resize((1280, 640), Image.LANCZOS).save(f"{OUT}/social-preview.jpg", quality=90, optimize=True)

# ---------------------------------------------------------------- one still per volume (1280 x 720)
stills = {
    1: f"{SCRATCH}/PromoEN-00680.png",
    2: f"{P}/out/posters-en/3-balmer-spectrum.png",
    3: f"{SCRATCH}/PromoEN-01640.png",
    4: f"{SCRATCH}/PromoEN-02216.png",
    5: f"{P}/out/posters-en/6-caught.png",
}
for n, src in stills.items():
    Image.open(src).convert("RGB").resize((1280, 720), Image.LANCZOS).save(f"{OUT}/stills/v{n}.jpg", quality=84, optimize=True, progressive=True)

# ---------------------------------------------------------------- cover thumbnails with a soft shadow
CW, CH = 330, 440          # cover size in the shelf image (the page is 595 x 794 pt, aspect 0.75)
def cover_with_shadow(n, dark=False, size=(CW, CH)):
    im = Image.open(f"{P}/public/covers/v{n}.jpg").convert("RGB").resize(size, Image.LANCZOS)
    pad = 44
    canvas = Image.new("RGBA", (size[0] + 2 * pad, size[1] + 2 * pad), (0, 0, 0, 0))
    sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rectangle([pad, pad + 14, pad + size[0], pad + size[1] + 14], fill=(0, 0, 0, 120 if not dark else 190))
    sh = sh.filter(ImageFilter.GaussianBlur(16))
    canvas = Image.alpha_composite(canvas, sh)
    canvas.paste(im, (pad, pad))
    edge = (48, 54, 61, 255) if dark else (208, 215, 222, 255)
    ImageDraw.Draw(canvas).rectangle([pad, pad, pad + size[0] - 1, pad + size[1] - 1], outline=edge, width=1)
    return canvas

for n in range(1, 6):
    c = cover_with_shadow(n, size=(300, 400))
    c.save(f"{OUT}/covers/v{n}.png", optimize=True)

# ---------------------------------------------------------------- the shelf (light and dark variants)
def font(path, size, index=0):
    try:
        return ImageFont.truetype(path, size, index=index)
    except Exception:
        return ImageFont.load_default()

def find_face(path, want):
    for i in range(8):
        try:
            f = ImageFont.truetype(path, 20, index=i)
            if want in " ".join(f.getname()).lower():
                return i
        except Exception:
            break
    return 0

PAL = "/System/Library/Fonts/Palatino.ttc"
HEL = "/System/Library/Fonts/HelveticaNeue.ttc"
i_reg, i_ital = find_face(PAL, "regular"), find_face(PAL, "italic")
VOLS = [
    ("I", ["Dissolving", "Problems"], 333, "#1b8aa8", "#52c7e8"),
    ("II", ["The Hydrogen Atoms", "of Deep Learning"], 394, "#6a45d6", "#a78bff"),
    ("III", ["Optimizers", "in the Wild"], 369, "#7d8d22", "#c4d453"),
    ("IV", ["Deep Learning as an", "Experimental Science"], 365, "#c0432a", "#ff8a6e"),
    ("V", ["Verifiable", "Intelligence"], 357, "#3a44b5", "#9aa4ff"),
]
def shelf(dark):
    W, H = 1800, 740
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gap = (W - 5 * 308) // 6
    fg = "#e6edf3" if dark else "#1f2328"
    mute = "#8b949e" if dark else "#656d76"
    f_num, f_title, f_pg = font(PAL, 50, i_reg), font(PAL, 29, i_reg), font(HEL, 21)
    d = ImageDraw.Draw(img)
    for k, (roman, lines, pages, acc_l, acc_d) in enumerate(VOLS):
        x = gap + k * (308 + gap)
        c = cover_with_shadow(k + 1, dark=dark, size=(308, 411))
        img.alpha_composite(c, (x - 44, 0))
        cx = x + 154
        acc = acc_d if dark else acc_l
        y = 470
        w = d.textlength(roman, font=f_num)
        d.text((cx - w / 2, y), roman, font=f_num, fill=acc)
        d.rectangle([cx - 22, y + 70, cx + 22, y + 72], fill=acc)
        yy = y + 92
        for ln in lines:
            w = d.textlength(ln, font=f_title)
            d.text((cx - w / 2, yy), ln, font=f_title, fill=fg)
            yy += 37
        label = f"{pages} PAGES"
        label = " ".join(label)       # a little letter-spacing
        w = d.textlength(label, font=f_pg)
        d.text((cx - w / 2, yy + 14), label, font=f_pg, fill=mute)
    return img
shelf(False).save(f"{OUT}/shelf-light.png", optimize=True)
shelf(True).save(f"{OUT}/shelf-dark.png", optimize=True)
print("README images written to", OUT)
for root, _, files in os.walk(OUT):
    for f in sorted(files):
        p = os.path.join(root, f)
        print(f"  {os.path.relpath(p, OUT):28s} {os.path.getsize(p)/1024:7.0f} KB")
