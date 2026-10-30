"""H3 Relay Finish (low RAM): an alternative to the three-node end of the full pipeline (VAEDecode -> H3 Face Atlas Stitch
Chunked -> H3 Hold Transfer) that keeps ONE full-size copy of the clip instead of three.

At 15 s and 1920x1088 every full-size float32 copy is ~9 GB, and the three nodes each return a new one (the chunked stitch
briefly holds two while it concatenates), which together with the models and the cached 0.5 MP stages runs a 92 GB machine
out of RAM. This node decodes the whole clip once (same decode, no temporal chunks, so no seams), stitches the face tiles
back chunk by chunk INTO that tensor, then applies the holds in place, one frame at a time. The stitch and hold logic is
not copied: it calls FaceDetail's H3FaceAtlasStitch and ResCascade's hold functions, so the result is the same as the
three nodes with the same settings. The original nodes and workflows are unchanged; use whichever you prefer.
"""
import sys

import torch
import torch.nn.functional as F


def _node(name):
    import nodes as comfy_nodes
    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(f"H3RelayFinish needs the {name} node (install its pack)")
    return cls


class H3RelayFinish:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT", {"tooltip": "the relay's video latent (LTXV Separate AV Latent, video output)"}),
            "vae": ("VAE",),
        }, "optional": {
            "refined": ("IMAGE", {"tooltip": "FaceDetail's decoded face tiles (as for H3 Face Atlas Stitch)"}),
            "atlas": ("H3_FACE_ATLAS", {"tooltip": "the face atlas scaled to the hires size (H3 Face Atlas Rescale)"}),
            "chunk": ("INT", {"default": 128, "min": 8, "max": 4096, "tooltip": "frames per face-stitch chunk"}),
            "mask_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
            "feather": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 32.0, "step": 0.5}),
            "color_match": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "geometry_lock": ("BOOLEAN", {"default": True}),
            "reference": ("IMAGE", {"tooltip": "the clip whose holds are the truth (Anime Cadence's output)"}),
            "source": ("IMAGE", {"tooltip": "Anime Cadence's input (as for H3 Hold Transfer)"}),
            "hold_threshold": ("FLOAT", {"default": 0.004, "min": 0.0, "max": 0.1, "step": 0.001}),
            "hold_erode": ("INT", {"default": 2, "min": 0, "max": 16}),
            "hold_feather": ("INT", {"default": 4, "min": 0, "max": 64}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "H3RelayHires"

    def run(self, samples, vae, refined=None, atlas=None, chunk=128, mask_scale=1.0, feather=3.0, color_match=1.0,
            blend=1.0, geometry_lock=True, reference=None, source=None, hold_threshold=0.004, hold_erode=2,
            hold_feather=4):
        images = _node("VAEDecode")().decode(vae, samples)[0]

        if atlas is not None and refined is not None and atlas.get("nslots", 0) > 0:
            stitch = _node("H3FaceAtlasStitch")()
            n = images.shape[0]
            for a in range(0, n, chunk):  # same chunks and calls as H3 Face Atlas Stitch Chunked, written back in place
                b = min(n, a + chunk)
                sub = dict(atlas, n=b - a, slots=[slot[a:b] for slot in atlas["slots"]])
                out = stitch.run(images[a:b], refined[a:b], sub, mask_scale, feather, color_match, blend,
                                 geometry_lock=geometry_lock)[0]
                images[a:b] = out.to(images.device, images.dtype)
                del out

        if reference is not None:
            images = self._holds(images, reference, source, hold_threshold, hold_erode, hold_feather)
        return (images,)

    @staticmethod
    def _holds(images, reference, source, threshold, erode, feather):
        """H3 Hold Transfer's run, in place: the hold masks stay at the reference size and are upscaled per frame"""
        mod = sys.modules[_node("H3HoldTransfer").__module__]
        n = min(reference.shape[0], images.shape[0])
        if reference.shape[0] != images.shape[0]:
            print(f"[H3RelayFinish] frame counts differ ({reference.shape[0]} vs {images.shape[0]}), using {n}")
        images = images[:n]
        m = mod.hold_masks(reference[:n].float(), threshold, erode)
        H, W = images.shape[1], images.shape[2]
        idx = list(range(n))
        if source is not None:
            idx = mod.source_frames(reference[:n], source[:n])
            idx += list(range(len(idx), n))
        orig = {k: images[k].clone() for t, k in enumerate(idx) if k != t}  # frames another frame shows, before edits
        held = full = 0.0
        for i in range(n):
            if idx[i] != i:
                images[i] = orig[idx[i]]
            mi = F.interpolate(m[i:i + 1], size=(H, W), mode="bilinear", align_corners=False)
            if feather > 0:
                mi = F.avg_pool2d(mi, 2 * feather + 1, stride=1, padding=feather, count_include_pad=False)
            held += mi.mean().item()
            full += float(mi.mean().item() > 0.98)
            mi = mi[0, 0, :, :, None].to(images.dtype)
            if i > 0 and mi.max() > 0:
                images[i] = mi * images[i - 1] + (1 - mi) * images[i]
        report = f"held area {held / n:.1%} of all pixels; {int(full)} of {n} frames fully held"
        if source is not None:
            report += f"; {len([1 for t, k in enumerate(idx) if k != t])} frames show another source frame"
        print(f"[H3RelayFinish] {report}")
        return images


NODE_CLASS_MAPPINGS = {"H3RelayFinish": H3RelayFinish}
NODE_DISPLAY_NAME_MAPPINGS = {"H3RelayFinish": "H3 Relay Finish (decode + face stitch + holds, low RAM)"}
