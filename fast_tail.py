"""H3 Fast Tail Sampler: a faster pass 1 that keeps the take.

H3 decides layout, acting and audio timing above sigma ~0.79 (12 of the 20 beta57 steps), and that part is chaotic:
any change to those steps (fewer steps, more steps, another solver) gives a different take, even 40 steps does.
Below 0.79 the trajectory is smooth. So: every step of your schedule down to `decide_above` with euler, bit for bit
as today; the 8-step tail in `tail_steps` bigger steps taken from your own schedule, 2nd order (DPM++ 2M with its
step ratio in H3's flow log-SNR; the first tail step is euler). Default 20 -> 16 steps: -20% pass-1 sampling, the
same take, and as close to the exact finish (a 32-step tail) as today's 8 euler steps in fine detail and audio.
3 tail steps (-25%) lose a few tiny particles/sparks. Details: README, "Faster pass 1".
Drop-in for KSamplerSelect(euler); keep BasicScheduler beta57 20.
"""
import logging

import torch

import comfy.k_diffusion.sampling as kds
import comfy.samplers
from comfy.utils import model_trange as trange

log = logging.getLogger("H3RelayHires")


def plan_sigmas(sigmas, decide_above, tail_steps):
    """every sigma >= decide_above, then tail_steps evenly picked from the rest of the schedule (ends at 0).
    Returns the planned sigmas, the index where the tail starts, and each planned sigma's index in the schedule."""
    sig = [float(s) for s in sigmas]
    k = max([i for i, s in enumerate(sig) if s >= decide_above - 1e-4] or [0])
    rest = len(sig) - 1 - k
    n = min(tail_steps, rest)
    idx = list(range(len(sig))) if n <= 0 else list(range(k)) + sorted({k + round(j * rest / n) for j in range(n + 1)})
    return sigmas[idx], k, idx


@torch.no_grad()
def sample_fast_tail(model, x, sigmas, extra_args=None, callback=None, disable=None, decide_above=0.79, tail_steps=4,
                     solver="dpmpp_2m_flow"):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    plan, k, idx = plan_sigmas(sigmas, decide_above, tail_steps)
    log.info(f"H3FastTailSampler: {len(sigmas) - 1} -> {len(plan) - 1} steps, euler to sigma {float(plan[k]):.4f}, "
             f"tail {[round(float(s), 3) for s in plan[k:]]} ({solver})")
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    lam = lambda sigma: ((1 - sigma) / sigma).log()  # flow (rectified) half-log-SNR, log(alpha / sigma)
    old_denoised = None
    for i in trange(len(plan) - 1, disable=disable):
        s, s_next = plan[i], plan[i + 1]
        denoised = model(x, s * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": idx[i], "sigma": s, "sigma_hat": s, "denoised": denoised})  # schedule step: UI bar ends at N/N
        if i < k or solver == "euler":  # today's euler, the same arithmetic as comfy's sample_euler (bit-identical take)
            x = x + kds.to_d(x, s, denoised) * (s_next - s)
            continue
        # the tail: comfy's sample_dpmpp_2m started at the first tail sigma (its first step is first order)
        t, t_next = t_fn(s), t_fn(s_next)
        h = t_next - t
        if old_denoised is None or s_next == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            if solver == "dpmpp_2m_flow":  # step ratio in H3's own (flow) log-SNR: the beta57 tail is ~even there
                r = (lam(s) - lam(plan[i - 1])) / (lam(s_next) - lam(s))
            else:  # comfy's dpmpp_2m: step ratio in log sigma
                r = (t - t_fn(plan[i - 1])) / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


class H3FastTailSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "decide_above": ("FLOAT", {"default": 0.79, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "every scheduled step with sigma >= this runs as today (euler): "
                                                  "that is where H3 decides the take"}),
            "tail_steps": ("INT", {"default": 4, "min": 1, "max": 50,
                                   "tooltip": "steps for the rest of the schedule, picked from it (beta57 20: 8 -> 4, "
                                              "as exact as today; 3 = -25% but loses a few tiny sparks/particles)"}),
            "tail_solver": (["dpmpp_2m_flow", "dpmpp_2m", "euler"], {
                "tooltip": "dpmpp_2m_flow (best with 4 tail steps): 2nd order, step ratio in H3's flow log-SNR. "
                           "dpmpp_2m: comfy's (log sigma; best with 3, overshoots the audio with 4). euler: 1st order"}),
        }}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "run"
    CATEGORY = "H3RelayHires"
    DESCRIPTION = ("Pass-1 speed without changing the take: today's euler steps where H3 decides the take (sigma >= "
                   "0.79), then 4 2nd-order steps for the smooth tail instead of 8: -20% sampling, same take, same "
                   "detail and audio. Use with BasicScheduler beta57 20.")

    def run(self, decide_above, tail_steps, tail_solver):
        return (comfy.samplers.KSAMPLER(sample_fast_tail, extra_options={
            "decide_above": decide_above, "tail_steps": tail_steps, "solver": tail_solver}),)


NODE_CLASS_MAPPINGS = {"H3FastTailSampler": H3FastTailSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"H3FastTailSampler": "H3 Fast Tail Sampler (pass 1 -20%, same take)"}
