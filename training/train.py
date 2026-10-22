#!/usr/bin/env python3
"""Train AnimeSR (Relay Hires):  cat pairs.tar | ITERS=12000 OUT=run python training/train.py   (needs torch, cv2,
safetensors and a GPU; animesr_v1 = 12k iterations, ~22 min on an RTX 5090)
stdin : the pairs tar written by prep.py
env   : ITERS (40000) BATCH (16) LR (3e-4) OUT (container dir for checkpoints/eval) GRAD_W (0.5) INIT (ckpt to resume)
Saves OUT/animesr_<step>.safetensors (+ animesr_best.safetensors by val PSNR) and OUT/train.log."""
import io, math, os, random, sys, tarfile, time
from collections import defaultdict

import cv2, numpy as np, torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the pack root (animesr.py)
from animesr import AnimeSR, load

ITERS, BATCH, LR = int(os.environ.get("ITERS", 40000)), int(os.environ.get("BATCH", 16)), float(os.environ.get("LR", 3e-4))
GRAD_W = float(os.environ.get("GRAD_W", 0.5))
THICK_P = float(os.environ.get("THICK_P", 0.8))  # share of batches with H3-thick lines
OUT = os.environ.get("OUT", "animesr_run")
os.makedirs(OUT, exist_ok=True)
logf = open(os.path.join(OUT, "train.log"), "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, file=sys.stderr, flush=True)
    logf.write(s + "\n"); logf.flush()


# ---- data: everything in RAM as uint8, grouped by LR size so a batch stacks ----
t0 = time.time()
pairs = {"train": defaultdict(lambda: ([], [])), "val": defaultdict(lambda: ([], []))}
pending, full = {}, {}
src = tarfile.open(fileobj=sys.stdin.buffer, mode="r|")
for ti in src:
    if not ti.isfile():
        continue
    img = cv2.imdecode(np.frombuffer(src.extractfile(ti).read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    split, base = ti.name.split("/", 1)
    if split == "full":
        full[base[:-4]] = img
        continue
    key, kind = base.rsplit("_", 1)
    pending.setdefault(key, {})[kind[:2]] = img
    if len(pending[key]) == 2:
        p = pending.pop(key)
        g = pairs[split][p["lr"].shape[0]]
        g[0].append(torch.from_numpy(p["lr"]))
        g[1].append(torch.from_numpy(p["hr"]))
data = {s: {r: (torch.stack(a), torch.stack(b)) for r, (a, b) in d.items()} for s, d in pairs.items()}
n_tr = sum(v[0].shape[0] for v in data["train"].values())
n_va = sum(v[0].shape[0] for v in data["val"].values())
log(f"loaded {n_tr} train / {n_va} val pairs, {len(full)} full frames, sizes {sorted(data['train'])}, {time.time() - t0:.0f}s")
sizes = sorted(data["train"])
weights = [data["train"][r][0].shape[0] for r in sizes]

dev = "cuda"
torch.backends.cudnn.benchmark = True
model = AnimeSR().to(dev).to(memory_format=torch.channels_last)
if os.environ.get("INIT"):
    model.load_state_dict(load(os.environ["INIT"]).state_dict())
log(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M, cfg {model.cfg}")
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(i, ITERS) / ITERS)))
SOB = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=dev) / 4


def grads(x):
    k = torch.stack([SOB, SOB.t()])[:, None].repeat(3, 1, 1, 1)  # 6 x 1 x 3 x 3 (x/y per channel)
    return F.conv2d(x, k, padding=1, groups=3)


def luma(x):
    return 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def thicken(lr, amount):
    """H3 draws ~2 px ink lines at EVERY size, so a 0.5 MP render has lines ~2x too thick for 1920. Mimic that on the
    downscaled real anime: inside thin dark structures (black top-hat of luma), move each pixel toward its darkest
    3x3 neighbour by `amount` (B x 1 x 1 x 1). The target keeps the real thin lines, so the net learns to thin."""
    y = luma(lr)
    closing = -F.max_pool2d(-F.max_pool2d(y, 5, 1, 2), 5, 1, 2)
    m = ((closing - y - 0.05) / 0.12).clamp(0, 1)
    B, C, H, W = lr.shape
    best = F.unfold(y, 3, padding=1).argmin(1, keepdim=True)
    iu = F.unfold(lr, 3, padding=1).view(B, C, 9, H * W)
    dark = iu.gather(2, best[:, None].expand(B, C, 1, H * W)).view(B, C, H, W)
    return lr + amount * m * (dark - lr)


def blur(x, sigma):
    r = 2
    t = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, 1, -1).repeat(x.shape[1], 1, 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k, groups=x.shape[1])
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.transpose(2, 3), groups=x.shape[1])


def h3ify(lr, rng_amount=None):
    """line thickening scaled by how far below 0.75x the batch is (0.5 MP takes need the most)"""
    s = lr.shape[-1] / 384
    f = max(0.0, min(1.0, (0.75 - s) / 0.28))
    if f == 0:
        return lr
    if rng_amount is None:
        if random.random() > THICK_P:
            return lr
        a = torch.empty(lr.shape[0], 1, 1, 1, device=lr.device).uniform_(0.3, 0.8) * f
    else:
        a = torch.full((lr.shape[0], 1, 1, 1), rng_amount * f, device=lr.device)
    out = thicken(lr, a)
    if rng_amount is None and random.random() < 0.5:
        out = blur(out, random.uniform(0.3, 0.6))
    return out.clamp(0, 1)


def batch():
    r = random.choices(sizes, weights)[0]
    lr_all, hr_all = data["train"][r]
    idx = torch.randint(0, lr_all.shape[0], (BATCH,))
    lr = (lr_all[idx].to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255).contiguous(memory_format=torch.channels_last)
    hr = (hr_all[idx].to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255).contiguous(memory_format=torch.channels_last)
    k = random.randrange(4)
    if k:
        lr, hr = torch.rot90(lr, k, (2, 3)), torch.rot90(hr, k, (2, 3))
    if random.random() < 0.5:
        lr, hr = lr.flip(3), hr.flip(3)
    return h3ify(lr), hr


@torch.no_grad()
def validate(amount=None):
    model.eval()
    se_m, se_b, n = 0.0, 0.0, 0
    for r, (lr_all, hr_all) in data["val"].items():
        for i in range(0, lr_all.shape[0], 32):
            lr = lr_all[i:i + 32].to(dev).permute(0, 3, 1, 2).float() / 255
            hr = hr_all[i:i + 32].to(dev).permute(0, 3, 1, 2).float() / 255
            if amount is not None:
                lr = h3ify(lr, amount)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = model(lr, hr.shape[-2:])
            b = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
            se_m += ((y.float().clamp(0, 1) - hr) ** 2).mean(dim=(1, 2, 3)).log10().sum().item()
            se_b += ((b.clamp(0, 1) - hr) ** 2).mean(dim=(1, 2, 3)).log10().sum().item()
            n += lr.shape[0]
    model.train()
    return -10 * se_m / n, -10 * se_b / n  # mean per-image PSNR


def save(name, step):
    sd = {k: v.detach().float().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(OUT, name), metadata={**{k: str(v) for k, v in model.cfg.items()}, "step": str(step)})


best = -1.0
t0 = time.time()
model.train()
for it in range(1, ITERS + 1):
    lr, hr = batch()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = model(lr, hr.shape[-2:])
    y = y.float()
    l1 = torch.sqrt((y - hr) ** 2 + 1e-6).mean()
    lg = (grads(y) - grads(hr)).abs().mean()
    loss = l1 + GRAD_W * lg
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if it % 200 == 0:
        log(f"it {it} l1 {l1.item():.4f} grad {lg.item():.4f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / it:.3f}s/it")
    if it % 2000 == 0 or it == ITERS:
        pm, pb = validate()
        tm, tb = validate(0.55)
        log(f"VAL it {it}: plain model {pm:.2f} / bicubic {pb:.2f} dB | H3-thick lines model {tm:.2f} / bicubic {tb:.2f} dB")
        pm = tm  # the use case: H3's thick low-res lines
        if pm > best:
            best = pm
            save("animesr_best.safetensors", it)
        if it % 10000 == 0 or it == ITERS:
            save(f"animesr_{it}.safetensors", it)

# full val frames -> eval pngs (model vs bicubic, 1:1) for viewing
model.eval()
ev = os.path.join(OUT, "eval")
os.makedirs(ev, exist_ok=True)
for k in [k for k in sorted(full) if k.endswith("_hr")][:8]:
    sid = k[:-3]
    hr = full[k]
    for w, h in [(960, 544), (1344, 768)]:
        lrk = f"{sid}_{w}x{h}"
        if lrk not in full:
            continue
        x = torch.from_numpy(full[lrk]).to(dev).permute(2, 0, 1)[None].float() / 255
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(x, (1080, 1920)).float().clamp(0, 1)
        b = F.interpolate(x, size=(1080, 1920), mode="bicubic", align_corners=False).clamp(0, 1)
        to8 = lambda t: (t[0].permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
        cv2.imwrite(os.path.join(ev, f"{sid}_{w}x{h}_sr.png"), cv2.cvtColor(to8(y), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(ev, f"{sid}_{w}x{h}_bicubic.png"), cv2.cvtColor(to8(b), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(ev, f"{sid}_hr.png"), cv2.cvtColor(hr, cv2.COLOR_RGB2BGR))
log(f"done, best val {best:.2f} dB, {time.time() - t0:.0f}s")
