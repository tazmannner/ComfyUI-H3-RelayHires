"""AnimeSR: arbitrary-scale anime upscaler for H3 pass-1 frames (Relay Hires, 2026-10-22). Pure PyTorch, no deps.

Design (from scratch, sized for video: ~1.3 M params, ~2 s for 124 frames 960x544 -> 1920x1088 on a 5090):
- body: residual conv blocks at the INPUT resolution. All the heavy lifting happens on the small picture.
- head: a few convs at the OUTPUT resolution. Input = body features (bilinear to the output grid) + the bicubic
  upscale + where each output pixel sits inside its input pixel (dy, dx in [-0.5, 0.5)) + the scale. That lets one
  model serve every pass-1 size and put a line edge at a sub-pixel position instead of smearing it.
- output = bicubic upscale + predicted residual (the residual conv starts at zero, so training starts from bicubic).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + 0.2 * self.c2(F.silu(self.c1(x)))


def subpixel_grid(h, w, H, W, device, dtype):
    """offset of every output pixel centre from the centre of the input pixel it falls in (input-pixel units)"""
    ys = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) * (h / H) - 0.5
    xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) * (w / W) - 0.5
    dy = (ys - ys.round())[:, None].expand(H, W)
    dx = (xs - xs.round())[None, :].expand(H, W)
    sc = torch.full((H, W), 0.5 * (h / H + w / W), device=device)  # input/output ratio, ~0.5-0.72
    return torch.stack([dy, dx, sc])[None].to(dtype)


class AnimeSR(nn.Module):
    def __init__(self, ch=64, blocks=16, hch=48, head=4):
        super().__init__()
        self.cfg = dict(ch=ch, blocks=blocks, hch=hch, head=head)
        self.inp = nn.Conv2d(3, ch, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.mid = nn.Conv2d(ch, ch, 3, padding=1)
        self.head_in = nn.Conv2d(ch + 3 + 3, hch, 3, padding=1)
        self.head = nn.ModuleList([nn.Conv2d(hch, hch, 3, padding=1) for _ in range(head - 1)])
        self.out = nn.Conv2d(hch, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, lr, size):
        """lr: B x 3 x h x w in [0, 1]; size: (H, W) -> B x 3 x H x W in [0, 1] (not clamped)"""
        x = lr - 0.5
        f = self.inp(x)
        f = f + self.mid(self.body(f))
        H, W = size
        fu = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
        base = F.interpolate(x, size=(H, W), mode="bicubic", align_corners=False)
        pos = subpixel_grid(x.shape[-2], x.shape[-1], H, W, x.device, x.dtype).expand(x.shape[0], -1, -1, -1)
        g = self.head_in(torch.cat([fu, base, pos], 1))
        for c in self.head:
            g = c(F.silu(g))
        return base + self.out(F.silu(g)) + 0.5


def load(path, device="cpu"):
    from safetensors.torch import load_file
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        meta = f.metadata() or {}
    cfg = {k: int(v) for k, v in meta.items() if k in ("ch", "blocks", "hch", "head")}
    m = AnimeSR(**cfg)
    m.load_state_dict(load_file(path))
    return m.to(device).eval()


@torch.inference_mode()
def upscale_frames(model, frames, size, batch=8, dtype=torch.float16):
    """frames: N x h x w x 3 (ComfyUI IMAGE, [0,1]) -> N x H x W x 3 on the frames' device"""
    dev = next(model.parameters()).device
    out = []
    for i in range(0, frames.shape[0], batch):
        x = frames[i:i + batch].to(dev, dtype).permute(0, 3, 1, 2)
        with torch.autocast(dev.type, dtype=dtype, enabled=dev.type == "cuda"):
            y = model(x, size)
        out.append(y.float().clamp(0, 1).permute(0, 2, 3, 1).to(frames.device))
    return torch.cat(out)
