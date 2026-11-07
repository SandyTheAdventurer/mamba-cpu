import torch
import torch.nn.functional as F


def selective_state_update(
    state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False, state_batch_indices=None
):
    """Single decode step of the selective scan. Updates state in place.

    Mamba-1 shapes:
        state: (batch, dim, dstate)
        x, dt, z: (batch, dim)
        A: (dim, dstate)
        B, C: (batch, dstate)
        D, dt_bias: (dim,)
    Mamba-2 (multi-head) shapes:
        state: (batch, nheads, headdim, dstate)
        x, dt, z: (batch, nheads, headdim)
        A: (nheads, headdim, dstate)
        B, C: (batch, ngroups, dstate)
        D, dt_bias: (nheads, headdim)
    Returns out shaped like x.
    """
    if state_batch_indices is not None:
        raise NotImplementedError("state_batch_indices is not supported yet")
    has_heads = state.dim() > 3
    if not has_heads:
        state = state.unsqueeze(1)
        x = x.unsqueeze(1)
        dt = dt.unsqueeze(1)
        A = A.unsqueeze(0)
        B = B.unsqueeze(1)
        C = C.unsqueeze(1)
        D = None if D is None else D.unsqueeze(0)
        z = None if z is None else z.unsqueeze(1)
        dt_bias = None if dt_bias is None else dt_bias.unsqueeze(0)
    nheads = state.shape[1]

    dt = dt.float()
    if dt_bias is not None:
        dt = dt + dt_bias.float()
    if dt_softplus:
        dt = F.softplus(dt)
    dA = torch.exp(dt.unsqueeze(-1) * A.float())  # (b, h, p, n)
    B = B.float().repeat_interleave(nheads // B.shape[1], dim=1)  # (b, h, n)
    C = C.float().repeat_interleave(nheads // C.shape[1], dim=1)
    dBx = (dt * x.float()).unsqueeze(-1) * B.unsqueeze(2)  # (b, h, p, n)

    new_state = state.float() * dA + dBx
    state.copy_(new_state)
    out = (new_state * C.unsqueeze(2)).sum(-1)
    if D is not None:
        out = out + x.float() * D.float()
    if z is not None:
        out = out * F.silu(z.float())
    out = out.to(x.dtype)
    return out if has_heads else out.squeeze(1)
