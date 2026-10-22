#!/usr/bin/env python3
"""Boil/flicker on still content: per consecutive frame pair, mean |diff| over the 10% of 32x32 blocks that change
least across the whole clip (static background). Prints the mean, and the mean at every k-th frame pair vs the rest
(k = 8: a +-2 latent-frame attention chunk is 8 video frames; a chunk-edge artifact would show as a ratio > 1).
  static_flicker.py file.mp4 [width=960]"""
import subprocess, sys
import numpy as np

path = sys.argv[1]
W = int(sys.argv[2]) if len(sys.argv) > 2 else 960
w0, h0 = map(int, subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                  "-of", "csv=p=0", path], capture_output=True, text=True).stdout.split(","))
H = round(h0 * W / w0 / 2) * 2
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={W}:{H}:flags=area", "-f", "rawvideo",
                      "-pix_fmt", "gray", "-"], capture_output=True).stdout
f = np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)
d = np.abs(np.diff(f, axis=0))                                   # T-1 x H x W
B = 16
hb, wb = H // B, W // B
blk = d[:, :hb * B, :wb * B].reshape(len(d), hb, B, wb, B).mean(axis=(2, 4))  # T-1 x hb x wb
total = blk.mean(axis=0)
static = total <= np.quantile(total, 0.10)
series = blk[:, static].mean(axis=1)
idx = np.arange(len(series))
for k in (4, 8):
    on = series[(idx + 1) % k == 0].mean()
    off = series[(idx + 1) % k != 0].mean()
    print(f"{path.split('/')[-1][:56]:56s} static-block flicker {series.mean():.3f}  every-{k}th/rest {on / off:.2f}")
