from importlib.metadata import PackageNotFoundError, version

from mamba_cpu.ops import (
    causal_conv1d_fn,
    causal_conv1d_update,
    selective_scan_fn,
    selective_state_update,
)
from mamba_cpu.reference import causal_conv1d_ref, selective_scan_ref

try:
    __version__ = version("mamba-cpu")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "causal_conv1d_fn",
    "causal_conv1d_ref",
    "causal_conv1d_update",
    "selective_scan_fn",
    "selective_scan_ref",
    "selective_state_update",
]
