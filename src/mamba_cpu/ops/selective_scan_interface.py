from typing import Optional

import torch

from mamba_cpu import _C

# The custom op takes kernel-ready inputs: float32, contiguous, with B and C laid
# out as (batch, ngroups, seqlen, dstate). selective_scan_fn does the conversion,
# so autograd also handles the casts and transposes.


def _ptr(t: Optional[torch.Tensor]) -> int:
    return 0 if t is None else t.data_ptr()


@torch.library.custom_op("mamba_cpu::selective_scan_fwd", mutates_args=(), device_types="cpu")
def _selective_scan_fwd(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    delta_bias: Optional[torch.Tensor],
    delta_softplus: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    for t in (u, delta, A, B, C, D, z, delta_bias):
        if t is not None and (t.dtype != torch.float32 or not t.is_contiguous()):
            raise ValueError("mamba_cpu::selective_scan_fwd expects contiguous float32 tensors")
    batch, dim, seqlen = u.shape
    ngroups, dstate = B.shape[1], B.shape[3]
    out = torch.empty_like(u)
    last_state = u.new_empty(batch, dim, dstate)
    _C.selective_scan_fwd_f32(
        _ptr(u), _ptr(delta), _ptr(A), _ptr(B), _ptr(C), _ptr(D), _ptr(z), _ptr(delta_bias),
        _ptr(out), _ptr(last_state),
        batch, dim, seqlen, dstate, ngroups, delta_softplus, torch.get_num_threads(),
    )
    return out, last_state


@_selective_scan_fwd.register_fake
def _(u, delta, A, B, C, D, z, delta_bias, delta_softplus):
    return torch.empty_like(u), u.new_empty(u.shape[0], u.shape[1], B.shape[-1])


@torch.library.custom_op("mamba_cpu::selective_scan_bwd", mutates_args=(), device_types="cpu")
def _selective_scan_bwd(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    delta_bias: Optional[torch.Tensor],
    delta_softplus: bool,
    dout: torch.Tensor,
    dlast_state: Optional[torch.Tensor],
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Returns (du, ddelta, dA, dB, dC, dD, dz, ddelta_bias). dz is empty when z is None;
    dD and ddelta_bias are always computed (they are cheap)."""
    for t in (u, delta, A, B, C, D, z, delta_bias, dout, dlast_state):
        if t is not None and (t.dtype != torch.float32 or not t.is_contiguous()):
            raise ValueError("mamba_cpu::selective_scan_bwd expects contiguous float32 tensors")
    batch, dim, seqlen = u.shape
    ngroups, dstate = B.shape[1], B.shape[3]
    du = torch.empty_like(u)
    ddelta = torch.empty_like(delta)
    dA_part = u.new_empty(batch, dim, dstate)
    dB = torch.zeros_like(B)
    dC = torch.zeros_like(C)
    dD_part = u.new_empty(batch, dim)
    dz = torch.empty_like(z) if z is not None else u.new_empty(0)
    ddelta_bias_part = u.new_empty(batch, dim)
    _C.selective_scan_bwd_f32(
        _ptr(u), _ptr(delta), _ptr(A), _ptr(B), _ptr(C), _ptr(D), _ptr(z), _ptr(delta_bias),
        _ptr(dout), _ptr(dlast_state),
        _ptr(du), _ptr(ddelta), _ptr(dA_part), _ptr(dB), _ptr(dC), _ptr(dD_part),
        _ptr(dz) if z is not None else 0, _ptr(ddelta_bias_part),
        batch, dim, seqlen, dstate, ngroups, delta_softplus, torch.get_num_threads(),
    )
    return du, ddelta, dA_part.sum(0), dB, dC, dD_part.sum(0), dz, ddelta_bias_part.sum(0)


@_selective_scan_bwd.register_fake
def _(u, delta, A, B, C, D, z, delta_bias, delta_softplus, dout, dlast_state):
    dim = u.shape[1]
    return (
        torch.empty_like(u), torch.empty_like(delta), torch.empty_like(A),
        torch.empty_like(B), torch.empty_like(C), u.new_empty(dim),
        torch.empty_like(z) if z is not None else u.new_empty(0), u.new_empty(dim),
    )


def _setup_context(ctx, inputs, output):
    u, delta, A, B, C, D, z, delta_bias, delta_softplus = inputs
    ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias)
    ctx.delta_softplus = delta_softplus


def _backward(ctx, grad_out, grad_last_state):
    u, delta, A, B, C, D, z, delta_bias = ctx.saved_tensors
    dout = torch.zeros_like(u) if grad_out is None else grad_out.float().contiguous()
    dlast = None if grad_last_state is None else grad_last_state.float().contiguous()
    du, ddelta, dA, dB, dC, dD, dz, ddelta_bias = torch.ops.mamba_cpu.selective_scan_bwd(
        u, delta, A, B, C, D, z, delta_bias, ctx.delta_softplus, dout, dlast
    )
    return (
        du, ddelta, dA, dB, dC,
        dD if D is not None else None,
        dz if z is not None else None,
        ddelta_bias if delta_bias is not None else None,
        None,
    )


torch.library.register_autograd(
    "mamba_cpu::selective_scan_fwd", _backward, setup_context=_setup_context
)


def _as_f32(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if t is None else t.float().contiguous()


def _as_kernel_bc(X: torch.Tensor) -> torch.Tensor:
    # (b, n, l) or (b, g, n, l)  ->  (b, g, l, n) contiguous
    if X.dim() == 3:
        X = X.unsqueeze(1)
    return X.float().transpose(-1, -2).contiguous()


def selective_scan_fn(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    """Drop-in for mamba_ssm.ops.selective_scan_interface.selective_scan_fn on CPU.

    u, delta, z: (batch, dim, seqlen)
    A: (dim, dstate), real-valued
    B, C: (batch, dstate, seqlen) or (batch, ngroups, dstate, seqlen)
    D, delta_bias: (dim,)
    Returns out: (batch, dim, seqlen) in u's dtype [, last_state: (batch, dim, dstate) float32]
    """
    if u.device.type != "cpu":
        raise ValueError(f"mamba_cpu runs on CPU tensors, got device {u.device}")
    if A.is_complex():
        raise NotImplementedError("complex A is not supported yet")
    if B.dim() < 3 or C.dim() < 3:
        raise NotImplementedError("input-independent (2D) B/C is not supported yet")
    dim = u.shape[1]
    for name, X in (("B", B), ("C", C)):
        ngroups = 1 if X.dim() == 3 else X.shape[1]
        if dim % ngroups != 0:
            raise ValueError(f"dim ({dim}) must be divisible by {name}'s ngroups ({ngroups})")
    if B.dim() != C.dim() or (B.dim() == 4 and B.shape[1] != C.shape[1]):
        raise NotImplementedError("B and C must have the same number of groups")

    out, last_state = torch.ops.mamba_cpu.selective_scan_fwd(
        _as_f32(u), _as_f32(delta), _as_f32(A), _as_kernel_bc(B), _as_kernel_bc(C),
        _as_f32(D), _as_f32(z), _as_f32(delta_bias), bool(delta_softplus),
    )
    out = out.to(u.dtype)
    return (out, last_state) if return_last_state else out
