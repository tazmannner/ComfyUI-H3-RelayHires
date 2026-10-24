"""H3 Relay Hires: render the take small, hand it to 1920, let H3 redraw only the detail there.

H3RelayHires            the method in one node: AnimeSR upscale -> encode -> one DMAD step from sigma 0.60 at the
                        target size, pass-1 audio pinned outright
H3RelayUpscale          AnimeSR alone (arbitrary-scale anime upscaler, trained from scratch): frames -> target size
H3FastTailSampler       pass-1 sampler: today's euler steps where H3 decides the take, a short 2nd-order tail after
                        (20 -> 16 steps, same take, same detail and audio; fast_tail.py)
H3RelayWindowAttention  temporal window attention (research/speed option). Keep anchor frames on: without a shared
                        reference each chunk redraws still lines slightly differently and they boil
H3RelayAnchor           research: pins the low frequencies of the denoised video to a reference latent every step
H3RelayLineThin         research: morphological ink-line thinning (AnimeSR does this properly)
"""
import logging
import math
import os

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.patcher_extension

from . import animesr, fast_tail

log = logging.getLogger("H3RelayHires")
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _model_list():
    if not os.path.isdir(MODELS):
        return []
    return sorted(f for f in os.listdir(MODELS) if f.endswith(".safetensors"))


def _streams(latent):
    s = latent["samples"]
    return list(s.unbind()) if getattr(s, "is_nested", False) else [s]


class H3RelayUpscale:
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "pass-1 frames (any size, e.g. 960x544 / 1120x640 / 1344x768)"}),
            "width": ("INT", {"default": 1920, "min": 64, "max": 8192, "step": 16}),
            "height": ("INT", {"default": 1088, "min": 64, "max": 8192, "step": 16}),
            "model_name": (_model_list() or ["(put AnimeSR .safetensors in models/)"],),
        }, "optional": {
            "batch": ("INT", {"default": 8, "min": 1, "max": 64}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "H3RelayHires"
    DESCRIPTION = "AnimeSR: arbitrary-scale anime upscaler trained for H3 pass-1 frames (VAE-decoded, 0.47-0.72x scale)."

    def run(self, images, width, height, model_name, batch=8):
        dev = mm.get_torch_device()
        key = (model_name, str(dev))
        if key not in self._cache:
            self._cache.clear()
            self._cache[key] = animesr.load(os.path.join(MODELS, model_name), dev)
        out = animesr.upscale_frames(self._cache[key], images[..., :3], (height, width), batch=batch)
        mm.soft_empty_cache()
        return (out,)


def _gauss(sigma, device):
    r = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum(), r


def lowpass(v, mode, amount):
    """low frequencies of a [..., h, w] latent per frame/channel: gaussian (amount = sigma in latent px) or
    box (amount = scale of the coarse grid, e.g. 0.5 = the 0.5 MP pass-1 grid; area down, bilinear up)"""
    shp = v.shape
    x = v.reshape(-1, 1, shp[-2], shp[-1]).float()
    if mode == "gaussian":
        k, r = _gauss(amount, x.device)
        x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
        x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    else:
        hs, ws = max(1, round(shp[-2] * amount)), max(1, round(shp[-1] * amount))
        x = F.interpolate(F.interpolate(x, size=(hs, ws), mode="area"), size=shp[-2:], mode="bilinear",
                          align_corners=False)
    return x.reshape(shp)


class H3RelayAnchor:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "reference": ("LATENT", {"tooltip": "the detail pass's start latent at the target size (its video stream)"}),
            "mode": (["gaussian", "box"], {"tooltip": "gaussian: blur radius in latent px (1 px = 16 image px). "
                                                      "box: the coarse grid of the take (0.5 = a 0.5 MP pass 1 at 1920)"}),
            "amount": ("FLOAT", {"default": 2.0, "min": 0.05, "max": 16.0, "step": 0.05}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "sigma_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "pin while sigma <= this ..."}),
            "sigma_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "... and sigma >= this"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "H3RelayHires"
    DESCRIPTION = ("Keeps the take while H3 redraws detail: at every step the low frequencies of the denoised video are "
                   "replaced by the reference's, so composition, colours and poses stay and only finer detail is free.")

    def patch(self, model, reference, mode, amount, strength, sigma_start, sigma_end):
        ref = _streams(reference)[0].float()
        shape = tuple(ref.shape)
        n_vid = math.prod(shape[1:])
        ref_lp = lowpass(ref, mode, amount)
        cache = {}

        def post_cfg(args):
            den = args["denoised"]
            sigma = float(args["sigma"].flatten()[0])
            if strength <= 0 or not (sigma_end <= sigma <= sigma_start):
                return den
            B = den.shape[0]
            flat = den.reshape(B, -1)
            if flat.shape[1] < n_vid:
                log.warning("H3RelayAnchor: latent smaller than the reference; skipped")
                return den
            key = (str(den.device), den.dtype)
            if key not in cache:
                cache.clear()
                cache[key] = ref_lp.to(den.device)
            v = flat[:, :n_vid].reshape(B, *shape[1:])
            d = cache[key] - lowpass(v, mode, amount)
            out = flat.clone()
            out[:, :n_vid] = (v.float() + strength * d).to(den.dtype).reshape(B, -1)
            return out.reshape(den.shape)

        m = model.clone()
        m.set_model_sampler_post_cfg_function(post_cfg)
        return (m,)


class _Window:
    def __init__(self, window, chunk, anchors, sigma_hi, sigma_lo):
        self.window, self.chunk, self.anchors = window, chunk, anchors
        self.sigma_hi, self.sigma_lo = sigma_hi, sigma_lo
        self.plans, self.installed, self.logged = {}, set(), False

    def plan(self, layout):
        key = (layout.seq_len, tuple(layout.segments))
        if key in self.plans:
            return self.plans[key]
        va, vb = next((a, b) for a, b, kind in layout.segments if kind == "video")
        T = int(layout.signature[1])
        p = None
        if vb == layout.seq_len and T > 0 and (vb - va) % T == 0:  # video must be the last segment
            Fr = (vb - va) // T
            chunks = []
            for f0 in range(0, T, self.chunk):
                f1 = min(T, f0 + self.chunk)
                lo, hi = max(0, f0 - self.window), min(T, f1 + self.window)
                extra = [f for f in range(0, T, self.anchors) if f < lo or f >= hi] if self.anchors > 0 else []
                chunks.append((va + f0 * Fr, va + f1 * Fr, va + lo * Fr, va + hi * Fr,
                               [(va + f * Fr, va + (f + 1) * Fr) for f in extra]))
            p = {"va": va, "T": T, "F": Fr, "chunks": chunks}
        while len(self.plans) >= 4:
            del self.plans[next(iter(self.plans))]
        self.plans[key] = p
        return p


def _window_override(w, previous):
    def override(func, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False,
                 **kwargs):
        to = kwargs.get("transformer_options") or {}

        def attend(qq, kk, vv):
            kw = dict(mask=None, attn_precision=attn_precision, skip_reshape=True, skip_output_reshape=False, **kwargs)
            return func(qq, kk, vv, heads, **kw) if previous is None else previous(func, qq, kk, vv, heads, **kw)

        layout = to.get("minimax_h3_layout")
        sig = to.get("sigmas")
        p = None
        if (mask is None and skip_reshape and not skip_output_reshape and layout is not None
                and q.shape[2] == layout.seq_len and q.shape == k.shape
                and (sig is None or w.sigma_lo <= float(sig.flatten()[0]) <= w.sigma_hi)):
            p = w.plan(layout)
        if p is None:
            kw = dict(mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape,
                      skip_output_reshape=skip_output_reshape, **kwargs)
            return func(q, k, v, heads, **kw) if previous is None else previous(func, q, k, v, heads, **kw)
        va = p["va"]
        outs = [attend(q[:, :, :va], k, v)] if va > 0 else []
        for q0, q1, k0, k1, extra in p["chunks"]:
            ks = [k[:, :, :va], k[:, :, k0:k1]] + [k[:, :, a:b] for a, b in extra]
            vs = [v[:, :, :va], v[:, :, k0:k1]] + [v[:, :, a:b] for a, b in extra]
            outs.append(attend(q[:, :, q0:q1], torch.cat(ks, 2), torch.cat(vs, 2)))
        if not w.logged:
            w.logged = True
            n = sum(va + (c[3] - c[2]) + sum(b - a for a, b in c[4]) for c in p["chunks"]) / len(p["chunks"])
            log.info(f"RelayWindowAttention: {p['T']} latent frames x {p['F']} rows, {len(p['chunks'])} chunks, "
                     f"~{int(n)} of {q.shape[2]} keys per video query ({n / q.shape[2]:.0%})")
        return torch.cat(outs, dim=1)

    return override


def _install(w, transformer_options):
    cur = transformer_options.get("optimized_attention_override")
    if cur in w.installed:
        return
    ov = _window_override(w, cur)
    w.installed.add(ov)
    transformer_options["optimized_attention_override"] = ov


class H3RelayWindowAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "window": ("INT", {"default": 2, "min": 1, "max": 64,
                               "tooltip": "latent frames on each side (1 latent frame = 4 video frames)"}),
            "chunk": ("INT", {"default": 2, "min": 1, "max": 64, "tooltip": "query latent frames per attention call"}),
            "anchor_stride": ("INT", {"default": 8, "min": 0, "max": 128,
                                      "tooltip": "also attend every Nth latent frame (0 = off). Keep it on: without a "
                                                 "shared reference still lines boil"}),
            "sigma_max": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "dense above this sigma"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "H3RelayHires"
    DESCRIPTION = ("Temporal window attention for the 1920 detail pass: each video frame attends to the frames within "
                   "+-window latent frames and to every text/audio row. Dense above sigma_max.")

    def patch(self, model, window, chunk, anchor_stride, sigma_max):
        w = _Window(window, chunk, anchor_stride, sigma_max, 0.0)
        m = model.clone()
        _install(w, m.model_options["transformer_options"])
        m.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_PREPARE_STATE, "h3_relay_window_attention",
                                lambda mp, timestep, model_options: _install(w, model_options["transformer_options"]))
        return (m,)


def thin_lines(img, radius=1, tophat=7, t0=0.06, t1=0.18):
    """img: B x 3 x H x W in [0,1]. Inside thin dark ink lines (black top-hat of luma > t0..t1), replace each pixel
    by its brightest neighbour within `radius` (whole RGB of that neighbour, no channel mixing): a line loses
    `radius` px on each side, flat fills and large dark areas are untouched."""
    y = (0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2])[:, None]
    p = tophat // 2
    closing = -F.max_pool2d(-F.max_pool2d(y, tophat, 1, p), tophat, 1, p)
    m = ((closing - y - t0) / (t1 - t0)).clamp(0, 1)
    k = 2 * radius + 1
    B, C, H, W = img.shape
    yu = F.unfold(y, k, padding=radius)                     # B x k*k x H*W
    best = yu.argmax(1, keepdim=True)                       # B x 1 x H*W
    iu = F.unfold(img, k, padding=radius).view(B, C, k * k, H * W)
    bright = iu.gather(2, best[:, None].expand(B, C, 1, H * W)).view(B, C, H, W)
    return img + m * (bright - img)


class H3RelayLineThin:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "upscaled frames at the target size"}),
            "radius": ("INT", {"default": 1, "min": 0, "max": 4,
                               "tooltip": "px removed from each side of an ink line (1: 4 px -> 2 px, for a 0.5 MP take "
                                          "at 1920; H3 draws ~2 px lines at every size)"}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "blend toward the thinned frame (0.5 ~ half a pixel per side, soft edges)"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "H3RelayHires"

    def run(self, images, radius, strength=0.5):
        if radius == 0 or strength == 0:
            return (images,)
        dev = mm.get_torch_device()
        out = []
        for i in range(0, images.shape[0], 4):
            x = images[i:i + 4, :, :, :3].to(dev).permute(0, 3, 1, 2)
            out.append((x + strength * (thin_lines(x, radius) - x)).permute(0, 2, 3, 1).cpu())
        return (torch.cat(out),)


def _fit_keyframes(positive, size):
    """image-to-video conditioning carries its start/end frame as latents sized for pass 1, which H3 cannot pack at
    the target size; the refined frames already contain them, so drop the ones that don't fit (same as H3HiresFix)"""
    out, dropped = [], 0
    for c, d in positive:
        kfs = d.get("minimax_keyframes")
        if kfs:
            keep = [k for k in kfs if k.get("latent") is None or tuple(k["latent"].shape[-2:]) == tuple(size)]
            dropped += len(kfs) - len(keep)
            d = dict(d)
            if keep:
                d["minimax_keyframes"] = keep
            else:
                d.pop("minimax_keyframes")
        out.append([c, d])
    return out, dropped


class H3RelayHires:
    _sr = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "the pass-1 model (same LoRAs + shift) with the DMAD turbo LoRA at 0.8"}),
            "positive": ("CONDITIONING", {"tooltip": "the pass-1 conditioning"}),
            "latent": ("LATENT", {"tooltip": "pass-1 AV latent (sampler output); its audio is pinned outright"}),
            "images": ("IMAGE", {"tooltip": "pass-1 frames (decoded, or after De-rope / FaceDetail / Cadence)"}),
            "vae": ("VAE", {"tooltip": "encodes the upscaled frames: taeh3 is enough (the detail pass redraws)"}),
            "width": ("INT", {"default": 1920, "min": 64, "max": 8192, "step": 32}),
            "height": ("INT", {"default": 1088, "min": 64, "max": 8192, "step": 32}),
            "upscaler": (_model_list() + ["lanczos"],),
            "sigma_start": ("FLOAT", {"default": 0.60, "min": 0.05, "max": 0.95, "step": 0.01,
                                      "tooltip": "0.60 for a 0.5 MP take (keeps the take, no re-smear); 0.45 from 1 MP"}),
            "steps": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": "DMAD steps (1 looks like 2 at sigma 0.60)"}),
            "window": ("INT", {"default": 0, "min": 0, "max": 16,
                               "tooltip": "temporal attention window in latent frames (0 = dense, safest). Windows need "
                                          "anchor frames: without them still lines boil"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }, "optional": {
            "anchor_stride": ("INT", {"default": 8, "min": 1, "max": 64,
                                      "tooltip": "with a window: also attend every Nth latent frame (a shared reference "
                                                 "that keeps still lines from boiling)"}),
        }}

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "upscaled")
    FUNCTION = "run"
    CATEGORY = "H3RelayHires"
    DESCRIPTION = ("Relay Hires: a small pass 1 (0.5 MP) decides the take and audio; AnimeSR hands it to the target "
                   "size; one DMAD step at sigma 0.60 redraws the detail there. Audio pinned.")

    def run(self, model, positive, latent, images, vae, width, height, upscaler, sigma_start, steps, window, seed,
            anchor_stride=8):
        import comfy.nested_tensor, comfy.sample, comfy.samplers, comfy.utils, latent_preview
        from comfy_extras.nodes_custom_sampler import Guider_Basic
        frames = images[..., :3]
        if upscaler == "lanczos":
            up = comfy.utils.common_upscale(frames.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
        else:
            dev = mm.get_torch_device()
            key = (upscaler, str(dev))
            if key not in self._sr:
                self._sr.clear()
                self._sr[key] = animesr.load(os.path.join(MODELS, upscaler), dev)
            up = animesr.upscale_frames(self._sr[key], frames, (height, width))
        video = vae.encode(up).float()
        xs = _streams(latent)
        streams, masks = [video], [torch.ones_like(video[:1, :1])]
        if len(xs) > 1:
            audio = xs[1].float()
            streams.append(audio)
            masks.append(torch.zeros_like(audio[:1]))
        samples = comfy.nested_tensor.NestedTensor(streams) if len(streams) > 1 else streams[0]
        noise_mask = comfy.nested_tensor.NestedTensor(masks) if len(masks) > 1 else None
        m = model
        if window > 0:
            m = model.clone()
            w = _Window(window, window, anchor_stride, 1.0, 0.0)
            _install(w, m.model_options["transformer_options"])
            m.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_PREPARE_STATE, "h3_relay_window_attention",
                                    lambda mp, timestep, model_options: _install(w, model_options["transformer_options"]))
        sigmas = torch.linspace(sigma_start, 0.0, steps + 1)
        positive, dropped = _fit_keyframes(positive, tuple(video.shape[-2:]))
        log.info(f"H3RelayHires: {tuple(frames.shape[1:3])} -> {(height, width)} via {upscaler}, sigmas "
                 f"{[round(float(s), 3) for s in sigmas]}, window {window}" + (f", dropped {dropped} keyframe(s)" if dropped else ""))
        guider = Guider_Basic(m)
        guider.set_conds(positive)
        noise = comfy.sample.prepare_noise(samples, seed)
        callback = latent_preview.prepare_callback(m, sigmas.shape[-1] - 1, {})
        out = guider.sample(noise, samples, comfy.samplers.sampler_object("euler"), sigmas, denoise_mask=noise_mask,
                            callback=callback, disable_pbar=False, seed=seed)
        return ({"samples": out.to(mm.intermediate_device())}, up)


NODE_CLASS_MAPPINGS = {"H3RelayUpscale": H3RelayUpscale, "H3RelayAnchor": H3RelayAnchor,
                       "H3RelayWindowAttention": H3RelayWindowAttention, "H3RelayLineThin": H3RelayLineThin,
                       "H3RelayHires": H3RelayHires}
NODE_DISPLAY_NAME_MAPPINGS = {"H3RelayUpscale": "H3 Relay Upscale (AnimeSR)",
                              "H3RelayAnchor": "H3 Relay Anchor (keep the take's coarse picture)",
                              "H3RelayWindowAttention": "H3 Relay Window Attention (detail pass)",
                              "H3RelayLineThin": "H3 Relay Line Thin (ink lines to H3's native width)",
                              "H3RelayHires": "H3 Relay Hires (0.5 MP take -> 1920, audio pinned)"}
NODE_CLASS_MAPPINGS.update(fast_tail.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(fast_tail.NODE_DISPLAY_NAME_MAPPINGS)
