from datetime import timedelta

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from diffusers.utils.torch_utils import is_compiled_module

global_accelerator = None


def get_accelerator() -> Accelerator:
    global global_accelerator
    if global_accelerator is None:
        # Long collective timeout: in multi-GPU runs, rank 0 can spend far
        # longer than the default 10 minutes alone inside
        # main_process_first() phases (cold latent / text-embedding caching
        # of a large dataset at several resolutions) while the other ranks
        # sit at the barrier -- with the default they DistBackendError out.
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
