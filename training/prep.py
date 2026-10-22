#!/usr/bin/env python3
"""Relay Hires: build (LR, HR) anime training pairs for the AnimeSR upscaler. Runs with ComfyUI's Python (it imports
comfy to load H3's video VAE) and a GPU:  tar -cf - <1080p clips/stills> | python training/prep.py > pairs.tar

stdin : tar stream of source clips (.mp4) and stills (.png), 1920x1080 (other sizes are skipped)
stdout: tar stream of pairs  {train,val}/<id>_hr.png (384x384) + <id>_lr.png (0.75R x 0.75R)
        and full val frames  full/<id>_hr.png (1920x1080) + full/<id>_<w>x<h>.png (LR round trips)
stderr: progress

LR = what H3's pass 1 hands the upscaler: a picture at 0.47-0.72x the target scale, decoded by H3's video VAE.
So: take a 512 px region (384 px crop + 64 px margin), downscale it to R x R (R = 240..368 step 16, s = R/512)
with antialiasing, round-trip it through the full H3 video VAE in video mode (5-frame windows; a still is repeated
5x, because still mode shifts colours) and keep the centre (the margin absorbs the VAE's border effects). Clips give frames 1-4 of a
5-frame window (one 4-frame latent token, like every frame after the first in a render).
"""
import hashlib, io, os, random, sys, tarfile, time

ARGS = sys.argv[1:]
# ComfyUI root: COMFYUI_DIR, else three levels up (custom_nodes/ComfyUI-H3-RelayHires/training/prep.py)
COMFY = os.environ.get("COMFYUI_DIR") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, COMFY); os.chdir(COMFY)
sys.argv = [sys.argv[0]] + (["--cpu"] if os.environ.get("PREP_CPU") else [])
import comfy.options
comfy.options.enable_args_parsing()
import av, cv2, numpy as np, torch
import comfy.sd, comfy.utils, folder_paths
import torch.nn.functional as F

SEED = 1234
WINDOWS_PER_CLIP = int(os.environ.get("WINDOWS", 3))
CROPS_PER_STILL = int(os.environ.get("STILL_CROPS", 2))
VAL_EVERY = 25          # 1 source in 25 (by name hash) is held out
REG, CROP = 512, 384
RS = list(range(240, 369, 16))
FULL_SIZES = [(960, 544), (1120, 640), (1344, 768)]

rng = random.Random(SEED)
log = lambda *a: print(*a, file=sys.stderr, flush=True)
VAE_NAME = os.environ.get("VAE_NAME", "minimax_h3_video_vae_fp16.safetensors")  # the H3 video VAE in models/vae
vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path("vae", VAE_NAME)))
out = tarfile.open(fileobj=sys.stdout.buffer, mode="w|")
n_pairs = {"train": 0, "val": 0}


def put(name, img):  # img: HxWx3 uint8 RGB
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    ti = tarfile.TarInfo(name)
    ti.size = len(buf)
    ti.mtime = int(time.time())
    out.addfile(ti, io.BytesIO(buf.tobytes()))


def roundtrip(frames):
    """frames: F x H x W x 3 float [0,1] (F = 1 or 5) -> same through the H3 video VAE"""
    with torch.inference_mode():
        z = vae.encode(frames)
        y = vae.decode(z)
    y = y.reshape(-1, *y.shape[-3:])[: frames.shape[0]]
    return y.float().clamp(0, 1).cpu()


def down(x, r):
    """N x H x W x 3 float -> N x r x r x 3, antialiased bicubic"""
    t = x.permute(0, 3, 1, 2)
    t = F.interpolate(t, size=(r, r), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
    return t.permute(0, 2, 3, 1)


def detail(img):  # mean gradient magnitude of a grey 4x-downscaled image
    g = cv2.cvtColor(cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())


def pick_region(frame):
    """512 px region; 3 in 4 times the most detailed of 4 random candidates (lines), else random (flat fills)"""
    H, W = frame.shape[:2]
    cands = [(rng.randrange(0, H - REG + 1), rng.randrange(0, W - REG + 1)) for _ in range(4)]
    if rng.random() < 0.75:
        return max(cands, key=lambda c: detail(frame[c[0]:c[0] + REG, c[1]:c[1] + REG]))
    return cands[0]


def emit_pairs(split, sid, hr_frames, lr_frames, r):
    m = (REG - CROP) // 2
    lc, lm = CROP * r // REG, (r - CROP * r // REG) // 2
    for k in range(hr_frames.shape[0]):
        n = n_pairs[split]
        n_pairs[split] += 1
        put(f"{split}/{sid}_{n:06d}_hr.png", hr_frames[k, m:m + CROP, m:m + CROP])
        put(f"{split}/{sid}_{n:06d}_lr.png", (lr_frames[k, lm:lm + lc, lm:lm + lc].numpy() * 255 + 0.5).astype(np.uint8))


def do_frames(split, sid, frames, keep):
    """frames: 5 x (HxWx3 uint8, 1080p) as a video window (a still is repeated 5x: H3 never decodes in still
    mode, which shifts colours). Emits the pairs of the window frames listed in `keep`."""
    f = np.stack(frames)
    y, x = pick_region(f[len(f) // 2])
    reg = f[:, y:y + REG, x:x + REG]
    r = rng.choice(RS)
    lr = roundtrip(down(torch.from_numpy(reg).float() / 255, r))
    emit_pairs(split, sid, reg[keep], lr[keep], r)


def do_full(sid, frame):
    put(f"full/{sid}_hr.png", frame)
    x = torch.from_numpy(frame).float()[None] / 255
    for w, h in FULL_SIZES:
        t = F.interpolate(x.permute(0, 3, 1, 2), size=(h, w), mode="bicubic", antialias=True, align_corners=False)
        lr = roundtrip(t.clamp(0, 1).permute(0, 2, 3, 1).repeat(5, 1, 1, 1))  # video mode, see do_frames
        put(f"full/{sid}_{w}x{h}.png", (lr[2].numpy() * 255 + 0.5).astype(np.uint8))


def decode_clip(data):
    c = av.open(io.BytesIO(data))
    fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    c.close()
    return fr


t0, n_src, n_full = time.time(), 0, 0
src = tarfile.open(fileobj=sys.stdin.buffer, mode="r|")
for ti in src:
    if not ti.isfile() or not ti.name.lower().endswith((".mp4", ".png")):
        continue
    data = src.extractfile(ti).read()
    name = ti.name
    sid = hashlib.md5(name.encode()).hexdigest()[:10]
    split = "val" if int(sid, 16) % VAL_EVERY == 0 else "train"
    try:
        if name.lower().endswith(".png"):
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.shape[:2] != (1080, 1920):
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for _ in range(CROPS_PER_STILL):
                do_frames(split, sid, [img] * 5, keep=[2])
            if split == "val" and n_full < 24:
                do_full(sid, img); n_full += 1
        else:
            fr = decode_clip(data)
            if len(fr) < 5 or fr[0].shape[:2] != (1080, 1920):
                continue
            for _ in range(WINDOWS_PER_CLIP):
                s = rng.randrange(0, len(fr) - 4)
                do_frames(split, sid, fr[s:s + 5], keep=[1, 2, 3, 4])
            if split == "val" and n_full < 24:
                do_full(sid, fr[len(fr) // 2]); n_full += 1
    except Exception as e:  # a broken source must not kill the run
        log(f"skip {name}: {e}")
        continue
    n_src += 1
    if n_src % 25 == 0:
        log(f"{n_src} sources, pairs {n_pairs}, full {n_full}, {time.time() - t0:.0f}s")
out.close()
log(f"done: {n_src} sources, pairs {n_pairs}, full {n_full}, {time.time() - t0:.0f}s")
