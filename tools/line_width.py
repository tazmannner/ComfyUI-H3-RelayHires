#!/usr/bin/env python3
"""Typical ink-line width (px) of anime frames, robust to grey/anti-aliased lines: along rows and columns, every local
luma minimum that sits >= CONTRAST below the brightest pixel within 8 px on BOTH sides is an ink line; its width is
the full width at half that contrast (sub-pixel, linear interpolation). Reports the median over thin lines (<= 12 px).
  line_width.py file.mp4 [frames=20,60,100]"""
import subprocess, sys
import numpy as np

CONTRAST, REACH, MAXW = 60, 8, 12


def frames(path, idx):
    w, h = map(int, subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True).stdout.split(","))
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"select='{sel}'", "-vsync", "0", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32), w


def widths(a):
    """a: rows x n luma. Returns FWHM widths of ink lines found along each row."""
    out = []
    n = a.shape[1]
    # candidate minima: strictly lower than left neighbour, not higher than right one
    m = np.zeros_like(a, dtype=bool)
    m[:, 1:-1] = (a[:, 1:-1] < a[:, :-2]) & (a[:, 1:-1] <= a[:, 2:])
    rows, cols = np.nonzero(m)
    for r, c in zip(rows, cols):
        lo, hi = max(0, c - REACH), min(n, c + REACH + 1)
        v = a[r, c]
        L, R = a[r, lo:c].max(initial=-1), a[r, c + 1:hi].max(initial=-1)
        bg = min(L, R)
        if bg - v < CONTRAST:
            continue
        half = v + (bg - v) / 2
        # walk left/right to the half level, interpolate
        i = c
        while i > lo and a[r, i - 1] < half:
            i -= 1
        if i == lo:
            continue
        xl = i - (half - a[r, i]) / (a[r, i - 1] - a[r, i] + 1e-6)
        j = c
        while j < hi - 1 and a[r, j + 1] < half:
            j += 1
        if j == hi - 1:
            continue
        xr = j + (half - a[r, j]) / (a[r, j + 1] - a[r, j] + 1e-6)
        wd = xr - xl
        if 0.5 <= wd <= MAXW:
            out.append(wd)
    return out


if __name__ == "__main__":
    path = sys.argv[1]
    idx = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "20,60,100").split(",")]
    fr, w = frames(path, idx)
    r = []
    for f in fr:
        r += widths(f[::3]) + widths(f.T[::3])  # every 3rd row/column is plenty
    r = np.array(r)
    s = 1920 / w
    print(f"{path.split('/')[-1][:58]:58s} {w:4d}w: ink line {np.median(r) * s:.2f} px at 1920 "
          f"(p25 {np.percentile(r, 25) * s:.2f}, p75 {np.percentile(r, 75) * s:.2f}, n={len(r)})")
