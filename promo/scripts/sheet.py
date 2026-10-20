#!/usr/bin/env python3
"""Contact sheet: sheet.py out.png cols thumbw file1 file2 ..."""
import sys, re
from PIL import Image, ImageDraw, ImageFont
out, cols, tw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
files = sys.argv[4:]
ims = [Image.open(f).convert("RGB") for f in files]
w, h = ims[0].size
th = int(tw * h / w)
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (cols * tw, rows * th), (20, 20, 20))
d = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 18)
except Exception:
    font = ImageFont.load_default()
for i, (f, im) in enumerate(zip(files, ims)):
    im = im.resize((tw, th), Image.LANCZOS)
    x, y = (i % cols) * tw, (i // cols) * th
    sheet.paste(im, (x, y))
    m = re.search(r"-(\d+)\.(png|jpg)$", f)
    lab = m.group(1) if m else f
    d.rectangle([x, y, x + 64, y + 24], fill=(0, 0, 0))
    d.text((x + 4, y + 2), lab, fill=(255, 255, 0), font=font)
sheet.save(out)
print(out, sheet.size)
