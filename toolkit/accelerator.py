from datetime import timedelta

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from diffusers.utils.torch_utils import is_compiled_module

global_accelerator = None


def get_accelerator() -> Accelerator:
    global global_accelerator
    if global_accelerator is None:
        # Long collective timeout, and it exists for exactly one thing:
        # the cold LATENT and TEXT-EMBEDDING caching passes.
        #
        # Those run inside accelerator.main_process_first(), so rank 0 walks
        # the whole dataset alone while every other rank blocks on a
        # distributed barrier. A barrier is a collective, and torch's default
        # collective timeout is 10 minutes -- any dataset that takes longer to
        # cache cold kills the run with DistBackendError "wait timeout after
        # 600000ms". It only ever happens on the FIRST run of a dataset, since
        # the cache is warm from then on, which makes it look like a flake
        # rather than a reproducible bug.
        #
        # 6 h is the caching budget, not a training-step budget: at ~2 s per
        # image (one VAE encode plus disk I/O) it covers roughly 10,000
        # images, and it multiplies out further for a dataset cached at
        # several resolutions. Comfortably more than any normal run needs.
        #
        # Note the cost of setting it this high: any genuine collective
        # deadlock also takes 6 h to surface instead of 10 minutes, and on
        # ROCm a blocked rank spin-waits at full power the whole time.
        kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=6))
        global_accelerator = Accelerator(kwargs_handlers=[kwargs])
    return global_accelerator

def unwrap_model(model):
    try:
        accelerator = get_accelerator()
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
    except Exception as e:
        pass
    return model
