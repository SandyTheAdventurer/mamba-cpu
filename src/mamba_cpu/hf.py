"""Run Hugging Face transformers Mamba models on mamba_cpu's ops.

    import mamba_cpu.hf
    mamba_cpu.hf.patch_transformers()
    model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")

Without this, transformers scans over time in a Python loop on CPU, or crashes
on CPU tensors if the CUDA packages mamba_ssm / causal_conv1d are installed,
because it uses them whenever they import. Tensors on other devices keep
transformers' own path.

Requires transformers >= 5, which resolves these ops from module globals at
call time.
"""

import functools
import importlib

from mamba_cpu.ops import (
    causal_conv1d_fn,
    causal_conv1d_update,
    selective_scan_fn,
    selective_state_update,
)

SUPPORTED_MODELS = ("mamba", "falcon_mamba", "jamba")


def _scan(
    hidden_states,
    dt,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
    **kwargs,  # use_mambapy, use_associative_scan select other fallbacks
):
    return selective_scan_fn(
        hidden_states, dt, A, B, C, D, z, delta_bias, delta_softplus, return_last_state
    )


def _state_update(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False, **kwargs):
    return selective_state_update(state, x, dt, A, B, C, D, z, dt_bias, dt_softplus)


def _conv(x, weight, bias=None, activation=None, **kwargs):
    return causal_conv1d_fn(x, weight, bias, activation=activation)


def _conv_update(x, conv_state, weight, bias=None, activation=None, **kwargs):
    return causal_conv1d_update(x, conv_state, weight, bias, activation)


def _no_fused_path(*args, **kwargs):
    return None  # the mixer takes its unfused path when mamba_inner_fn returns None


_OPS = {
    "mamba_selective_scan": _scan,
    "mamba_selective_state_update": _state_update,
    "causal_conv1d_fn": _conv,
    "causal_conv1d_update": _conv_update,
    "mamba_inner_fn": _no_fused_path,
}


def _dispatch(original, native):
    @functools.wraps(original)
    def wrapper(first, *args, **kwargs):
        fn = native if first.device.type == "cpu" else original
        return fn(first, *args, **kwargs)

    wrapper._mamba_cpu_original = original
    return wrapper


def _modeling_module(name):
    return importlib.import_module(f"transformers.models.{name}.modeling_{name}")


def patch_transformers(models=SUPPORTED_MODELS):
    """Route CPU execution of the given transformers model families through mamba_cpu.

    Safe to call repeatedly.
    """
    modules = [_modeling_module(name) for name in models]
    for module in modules:
        missing = [op for op in _OPS if not hasattr(module, op)]
        if missing:
            raise RuntimeError(
                f"{module.__name__} has no {', '.join(missing)}; mamba_cpu.hf needs transformers >= 5"
            )
    for module in modules:
        for op, native in _OPS.items():
            original = getattr(module, op)
            if not hasattr(original, "_mamba_cpu_original"):
                setattr(module, op, _dispatch(original, native))


def unpatch_transformers(models=SUPPORTED_MODELS):
    """Undo patch_transformers."""
    for name in models:
        module = _modeling_module(name)
        for op in _OPS:
            original = getattr(getattr(module, op), "_mamba_cpu_original", None)
            if original is not None:
                setattr(module, op, original)
