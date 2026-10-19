"""Slow, readable pure-PyTorch implementations used as the correctness oracle.

These follow mamba_ssm's reference semantics (real-valued A only).
"""

import torch
import torch.nn.functional as F


def selective_scan_ref(
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
    """
    u, delta, z: (batch, dim, seqlen)
    A: (dim, dstate)
    B, C: (batch, dstate, seqlen) or (batch, ngroups, dstate, seqlen)
    D, delta_bias: (dim,)
    Returns out: (batch, dim, seqlen) [, last_state: (batch, dim, dstate)]
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, seqlen = u.shape
    A = A.float()
    B = B.float()
    C = C.float()
    if B.dim() == 3:
        B = B.unsqueeze(1)
    if C.dim() == 3:
        C = C.unsqueeze(1)
    # Channels are split into contiguous groups sharing the same B/C.
    B = B.repeat_interleave(dim // B.shape[1], dim=1)  # (b, d, n, l)
    C = C.repeat_interleave(dim // C.shape[1], dim=1)

    delta_A = torch.exp(delta.unsqueeze(-1) * A[:, None, :])  # (b, d, l, n)
    delta_B_u = (delta * u).unsqueeze(-1) * B.transpose(-1, -2)  # (b, d, l, n)

    x = u.new_zeros(batch, dim, A.shape[1])
    ys = []
    for i in range(seqlen):
        x = delta_A[:, :, i] * x + delta_B_u[:, :, i]
        ys.append((x * C[:, :, :, i]).sum(-1))
    y = torch.stack(ys, dim=2) if ys else u.new_zeros(batch, dim, 0)

    out = y if D is None else y + u * D.float()[:, None]
    if z is not None:
        out = out * F.silu(z.float())
    out = out.to(dtype_in)
    return (out, x) if return_last_state else out


def causal_conv1d_ref(x, weight, bias=None, activation=None):
    """x: (batch, dim, seqlen), weight: (dim, width). Explicit loop over taps."""
    width = weight.shape[1]
    seqlen = x.shape[-1]
    xp = F.pad(x, (width - 1, 0))
    out = sum(xp[..., k : k + seqlen] * weight[:, k, None] for k in range(width))
    if bias is not None:
        out = out + bias[:, None]
    if activation in ("silu", "swish"):
        out = F.silu(out)
    return out
