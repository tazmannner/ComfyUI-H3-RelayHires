# H3 Relay Hires

Render a MiniMax H3 take small, finish it at 1920x1088: **about half the time of an upscaled 1 MP render and a quarter
of a native 1920 render**, keeping the take you previewed and its audio. Built for 2D anime.

![Re:Zero-style close-up: native 1920 vs today's 1 MP hires route vs Relay Hires vs its 0.5 MP preview](assets/rezero_native_today_relay.jpg)

*Top: native 1920 render, 1 MP render + latent upscale + refine. Bottom: Relay Hires, and the 0.5 MP pass it came from.
Each route renders its own take, so compare the drawing, not the pose.*

## Results

5 s clips (124 frames), 1920x1088, RTX 5090, end to end (text encode, sampling, decodes, saves):

| clip | native 1920 | 1 MP + latent upscale + refine | **Relay Hires** |
|---|---|---|---|
| Teen Titans-style punch | 492 s | 240 s | **122 s** |
| plain-H3 sword fight | 503 s | 241 s | **124 s** |
| Edgerunners-style rocket | 508 s | 245 s | **126 s** |
| Re:Zero-style mecha, close-ups | 503 s | 241 s | **124 s** |
| Re:Zero-style wide shot, small faces | 500 s | 239 s | **123 s** |

- **Same take as the 0.5 MP preview**, and its soundtrack sample for sample (the audio is pinned outright).
- **Crisp at 1:1**: the preview's melted fast-frame faces and motion smears are redrawn as proper drawings, and small
  faces in wide shots come out clean (below: 0.5 MP preview | AnimeSR alone | Relay Hires, 1:1 crops at 1920).
- **Steady**: frame-to-frame change on still content 0.13 (native 0.10, the 1 MP route 0.22).
- **Line weight** is the remaining gap to native: ink lines ~3.1-3.9 px vs ~2.2-3.0 px native.
- **Very fast action**: where the 0.5 MP take itself melts (a sword clash), the result is motion blur, not crisp
  poses. Fix those frames at 0.5 MP first with a de-rope pass (example workflow 2), then relay.

![small faces in a wide shot: 0.5 MP preview, AnimeSR alone, Relay Hires](assets/rezero_wide_small_faces.jpg)

## How it works

1. **Pass 1 at 0.5 MP** (960x544) decides layout, motion, acting and audio. That's where most of a render's time goes,
   so that's where the saving is: 60 s instead of 151 s at 1 MP.
2. **AnimeSR** upscales the decoded frames to the target size in ~2 s (see below).
3. **One DMAD step from sigma 0.60** at the target size redraws the detail: the frames are encoded (taeh3 is enough),
   re-noised to 0.60 and denoised in one turbo step with the pass-1 audio pinned, then decoded.

Why these settings (measured, see the development notes):
- **sigma 0.60**: 0.45 stays soft from a 0.5 MP take; 0.75 starts redrawing fast frames (new faces, re-smeared limbs).
- **1 step**: looks the same as 2, slightly steadier, 21 s less.
- **dense attention**: temporal window attention is ~9 s faster but makes still lines shimmer unless every chunk also
  sees shared anchor frames, and even then it's a little less steady.

## AnimeSR

H3 draws ink lines ~2 px wide at **every** resolution, so a 0.5 MP take has lines twice too thick for 1920. AnimeSR is a
small arbitrary-scale upscaler (1.3 M params, 5 MB) trained from scratch to undo that:

- **Architecture** (`animesr.py`): residual conv blocks at the input resolution, a 4-conv head at the output
  resolution that sees the features, the bicubic upscale, each output pixel's sub-pixel offset and the scale. One model
  for any input from 0.47x to 0.72x of the target; it starts training as exact bicubic.
- **Data**: ~1,700 1080p anime clips and stills -> 9,470 training pairs. Low-res inputs are made the way H3 makes them:
  downscaled 0.47-0.72x and round-tripped through H3's own video VAE (in video mode), and 80% of them get H3-thick ink
  lines (thin dark lines thickened at low res); the targets keep the real thin lines.
- **Result**: on real 0.5 MP H3 frames its lines come out at 2.7 px at 1920 vs 3.3 px for Lanczos, with clean fills and
  no ringing. Because the H3 step keeps much of the start image's line width, this carries into the result.

AnimeSR alone (no H3 step) is the cheapest 1920 path (~80 s), but it keeps the take's smears and mushy small faces.

## Nodes

| node | what it does |
|---|---|
| **H3 Relay Hires** | the whole method in one node. Inputs: the pass-1 model with the DMAD LoRA at 0.8 (same style LoRAs and shift), the pass-1 conditioning, the pass-1 AV latent (for its audio), the pass-1 frames, a VAE for the encode (taeh3). Outputs the AV latent at the target size, plus the upscaled frames. |
| H3 Relay Upscale (AnimeSR) | AnimeSR alone: frames -> any size |
| H3 Relay Window Attention | temporal window attention, with anchor frames (research / speed option) |
| H3 Relay Anchor, H3 Relay Line Thin | research tools from development, not part of the recipe |

Settings on H3 Relay Hires: `sigma_start` 0.60, `steps` 1, `window` 0. `window` 4 (with `anchor_stride` 8) is ~9 s faster
with slightly more shimmer on still lines. The `upscaler` list shows AnimeSR models in `models/` plus `lanczos`.

## Example workflows

Four workflows in `example_workflows/`, all rendering the same take (a street-market fight, plain H3 without a style
LoRA, seed 101). Each adds one stage of an anime pipeline at 0.5 MP before the relay. A blue SETTINGS group holds the
pass-1 size (aspect + megapixels, 0.5 MP), the hires scale (2.0) and one prompt box; each saves the 0.5 MP preview and
the 1920 result.

[![the four example workflows on one take: 1 relay only, 2 + Fast De-rope, 3 + FaceDetail, 4 + Anime Cadence](assets/examples_1to4.jpg)](assets/examples_1to4.mp4)

*The four workflows' 1920 results side by side: 1 relay only | 2 + Fast De-rope / 3 + FaceDetail | 4 + Anime Cadence. Click for the clip (normal speed with sound, then half speed).*

| workflow | what it adds | end to end* | packs it needs besides this one |
|---|---|---|---|
| [`1_relay_hires.json`](example_workflows/1_relay_hires.json) | relay only | 118 s | FastDerope (its taeh3 loader) |
| [`2_relay_hires_derope.json`](example_workflows/2_relay_hires_derope.json) | Fast De-rope: smeared fast motion comes back as clean drawings on 2s before the relay | 144 s | FastDerope, MAINodes |
| [`3_relay_hires_derope_facedetail.json`](example_workflows/3_relay_hires_derope_facedetail.json) | + FaceDetail: small faces redrawn at 0.5 MP; its face tiles are stitched again onto the 1920 frames, so the final keeps the preview's faces | 224 s | + FaceDetail, ResCascade |
| [`4_relay_hires_derope_facedetail_cadence.json`](example_workflows/4_relay_hires_derope_facedetail_cadence.json) | + Anime Cadence: 2s/3s holds, carried to 1920 by Hold Transfer | 228 s | + AnimeCadence |

*1920x1088, 124 frames (5 s), RTX 5090. FaceDetail's cost depends on the clip: here it redrew four face tracks of
~30 px and took ~80 s, re-stitch included.

- In workflow 4 the relay renders the frames *before* Anime Cadence (so FaceDetail's tiles stay on their frames).
  Hold Transfer then copies the 0.5 MP holds onto the 1920 frames; with its `source` input each hold shows the drawing
  Cadence picked (needs ResCascade da05159 or later). Don't run Anime Cadence again after the relay.
- Packs: [FastDerope](https://github.com/tazmannner/ComfyUI-H3-FastDerope),
  [MAINodes](https://github.com/matlowai/ComfyUI-MAINodes),
  [FaceDetail](https://github.com/tazmannner/ComfyUI-H3-FaceDetail) (with its anime face detector),
  [ResCascade](https://github.com/tazmannner/ComfyUI-H3-ResCascade),
  [AnimeCadence](https://github.com/tazmannner/ComfyUI-H3-AnimeCadence); all four workflows also use
  [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) and comfyui_essentials (SimpleMath+).

## Requirements

- MiniMax H3 in ComfyUI (the example workflows load the fp16 and int8 video VAEs and the audio VAE), the DMAD 4-step
  turbo LoRA for H3
- a taeh3 loader for the encode (e.g. ComfyUI-H3-FastDerope's)
- `models/animesr_v1.safetensors` (included)
- the example workflows also use comfyui_essentials (SimpleMath+) for the size settings, and the packs listed above

## Train AnimeSR yourself

```bash
tar -cf - my_1080p_clips/ my_1080p_stills/ | python training/prep.py > pairs.tar    # ComfyUI's Python + GPU (H3 VAE)
cat pairs.tar | ITERS=12000 OUT=run python training/train.py                       # ~22 min on an RTX 5090
```

`prep.py` keeps 1920x1080 sources only and holds out 1 in 25 for validation; `train.py` saves
`run/animesr_best.safetensors` (copy it into `models/`) and before/after PNGs of held-out frames.

## Measuring

- `tools/line_width.py video.mp4`: median ink-line width (px at 1920 scale)
- `tools/static_flicker.py video.mp4`: frame-to-frame change on the most static parts of a shot (shimmer/boil)

## License

MIT
