import os

import torch

# MIOpen is fine on RDNA2 and older; the pathological convolution shapes show
# up on RDNA3/RDNA4 and CDNA. Same list ComfyUI gates its own disable on.
RDNA2_AND_OLDER_ARCH = [
    "gfx1030", "gfx1031", "gfx1010", "gfx1011", "gfx1012",
    "gfx906", "gfx900", "gfx803",
]
ENABLE_MIOPEN_ENV = "AI_TOOLKIT_ENABLE_MIOPEN"


def configure_rocm_backends(verbose: bool = True) -> None:
    """Turn MIOpen off for convolutions on ROCm.

    torch routes convolutions through MIOpen (it is the HIP build's "cudnn").
    On RDNA3/RDNA4 MIOpen picks catastrophically bad algorithms once the
    spatial dims get large, and it JIT-compiles a kernel per unseen shape.
    Measured on gfx1201 (ROCm 7.2, torch 2.12) encoding one image with the
    Anima/Wan 3D VAE in bf16:

        shape          MIOpen on            MIOpen off
        768x768          0.33 s               0.18 s
        1024x1024        0.60 s               0.33 s
        1536x1536       30.16 s               0.84 s     (36x)
        1248x1672       26.92 s               0.72 s     (37x)
        1672x1248       26.78 s, 240 s cold   0.72 s

    Under ~1024 it is a ~1.8x tax; past that it falls off a cliff, and the
    first call on each new shape additionally pays 20-240 s of kernel
    compilation. Latent caching walks a bucketed dataset, so it hits many
    unseen shapes and eats both costs on every one -- that is the "caching
    latents takes forever" report. Disabling MIOpen falls back to torch's
    native/composable-kernel convolutions, which are shape-agnostic and need
    no compilation.

    The tradeoff is workspace: the native path peaked at 16.0 GiB vs MIOpen's
    9.7 GiB on the 1536x1536 encode. That is comfortable for VAE work (the
    VAE encodes alone during caching) but is the thing to look at first if a
    conv-heavy model starts OOMing where it used to fit -- set
    AI_TOOLKIT_ENABLE_MIOPEN=1 to put MIOpen back.
    """
    if torch.version.hip is None or not torch.cuda.is_available():
        return
    if os.environ.get(ENABLE_MIOPEN_ENV) == "1":
        return
    if not torch.backends.cudnn.enabled:
        return  # already off
    try:
        arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]
    except Exception:
        return
    if any(a in arch for a in RDNA2_AND_OLDER_ARCH):
        return
    torch.backends.cudnn.enabled = False
    if verbose:
        print(
            f"ROCm ({arch}): disabled MIOpen for convolutions "
            f"(set {ENABLE_MIOPEN_ENV}=1 to keep it)",
            flush=True,
        )
