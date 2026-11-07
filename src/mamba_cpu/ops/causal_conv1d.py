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
    final_states_out=None,
    activation=None,
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: optional (batch, dim, width - 1) buffer for the final states
    Returns out: (batch, dim, seqlen) in x's dtype [, final_states: (batch, dim, width - 1)]
    """
    if seq_idx is not None:
        raise NotImplementedError("seq_idx is not supported yet")
    width = weight.shape[1]
    if initial_states is None:
        x_full = F.pad(x, (width - 1, 0))
    else:
        x_full = torch.cat([initial_states.to(x.dtype), x], dim=-1)
    out = F.conv1d(
        x_full.to(weight.dtype),
        weight.unsqueeze(1),
        None if bias is None else bias.to(weight.dtype),
        groups=x.shape[1],
    )
    out = _activate(out, activation).to(x.dtype)
    if not return_final_states:
        return out
    final_states = x_full[..., x_full.shape[-1] - (width - 1) :]
    if final_states_out is not None:
        final_states = final_states_out.copy_(final_states)
    return out, final_states


def causal_conv1d_update(
    x, conv_state, weight, bias=None, activation=None, cache_seqlens=None, conv_state_indices=None
):
    """Decode step(s). Updates conv_state in place.

    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), state_len >= width - 1
    weight: (dim, width)
    Returns out shaped like x.
    """
    if cache_seqlens is not None or conv_state_indices is not None:
        raise NotImplementedError("cache_seqlens and conv_state_indices are not supported yet")
    single_step = x.dim() == 2
    x_seq = x.unsqueeze(-1) if single_step else x
    seqlen = x_seq.shape[-1]
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    if state_len < width - 1:
        raise ValueError(f"conv_state length ({state_len}) must be at least width - 1 ({width - 1})")

    x_full = torch.cat([conv_state.to(x.dtype), x_seq], dim=-1)
    conv_state.copy_(x_full[..., -state_len:])
    out = F.conv1d(
        x_full[..., -(seqlen + width - 1) :].to(weight.dtype),
        weight.unsqueeze(1),
        None if bias is None else bias.to(weight.dtype),
        groups=x_seq.shape[1],
    )
    out = _activate(out, activation).to(x.dtype)
    return out.squeeze(-1) if single_step else out
