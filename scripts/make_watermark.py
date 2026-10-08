"""Buat background putih logo watermark jadi transparan (flood-fill dari tepi)."""
import sys
import fitz
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
pm = fitz.Pixmap(src)
if pm.alpha:
    pm = fitz.Pixmap(pm, 0)  # buang alpha lama
if pm.colorspace.n != 3:
    pm = fitz.Pixmap(fitz.csRGB, pm)
w, h = pm.width, pm.height
rgb = np.frombuffer(pm.samples, dtype=np.uint8).reshape(h, w, 3).copy()

near_white = rgb.min(axis=2) >= 232
reach = np.zeros((h, w), bool)
reach[0, :] = near_white[0, :]; reach[-1, :] = near_white[-1, :]
reach[:, 0] = near_white[:, 0]; reach[:, -1] = near_white[:, -1]

def dilate(m):
    o = m.copy()
    o[1:, :] |= m[:-1, :]; o[:-1, :] |= m[1:, :]
    o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]
    return o

for _ in range(max(h, w) * 2):
    nxt = dilate(reach) & near_white
    if (nxt == reach).all():
        break
    reach = nxt

alpha = np.full((h, w), 255, np.uint8)
alpha[reach] = 0
# haluskan tepi: piksel non-latar yang bersebelahan dengan latar -> setengah transparan
edge = dilate(reach) & ~reach
alpha[edge] = 150

rgba = np.dstack([rgb, alpha])
out = fitz.Pixmap(fitz.csRGB, w, h, rgba.tobytes(), True)
out.save(dst)
print(f"OK {w}x{h} transparan={reach.mean()*100:.1f}% -> {dst}")
