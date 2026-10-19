import torch
import torch.nn.functional as F

# A depthwise causal conv is already well optimized by torch's CPU conv1d, so
# these are thin wrappers matching the causal_conv1d package's API.


def _activate(x, activation):
    if activation is None:
        return x
    if activation in ("silu", "swish"):
        return F.silu(x)
    raise ValueError(f"activation must be None, 'silu' or 'swish', got {activation!r}")


def causal_conv1d_fn(
    x,
    weight,
    bias=None,
    seq_idx=None,
    initial_states=None,
    return_final_states=False,
    activation=None,
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    Returns out: (batch, dim, seqlen) [, final_states: (batch, dim, width - 1)]
    """
    if seq_idx is not None:
        raise NotImplementedError("seq_idx is not supported yet")
    width = weight.shape[1]
    if initial_states is None:
        x_full = F.pad(x, (width - 1, 0))
    else:
        x_full = torch.cat([initial_states.to(x.dtype), x], dim=-1)
    out = F.conv1d(x_full, weight.unsqueeze(1), bias, groups=x.shape[1])
    out = _activate(out, activation)
    if return_final_states:
        return out, x_full[..., x_full.shape[-1] - (width - 1) :]
    return out


def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None):
    """Single decode step. Updates conv_state in place.

    x: (batch, dim)
    conv_state: (batch, dim, width)
    weight: (dim, width)
    Returns out: (batch, dim)
    """
    conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
    conv_state[:, :, -1] = x
    out = (conv_state * weight).sum(-1)
    if bias is not None:
        out = out + bias
    return _activate(out, activation).to(x.dtype)
