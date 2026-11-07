import pytest
import torch

from mamba_cpu import causal_conv1d_fn, causal_conv1d_ref, causal_conv1d_update


@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("has_bias", [False, True])
def test_matches_reference(width, activation, has_bias):
    torch.manual_seed(0)
    x = torch.randn(2, 8, 37)
    w = torch.randn(8, width)
    b = torch.randn(8) if has_bias else None
    out = causal_conv1d_fn(x, w, b, activation=activation)
    torch.testing.assert_close(out, causal_conv1d_ref(x, w, b, activation), rtol=1e-5, atol=1e-5)


def test_initial_and_final_states_chain():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 40)
    w = torch.randn(8, 4)
    full = causal_conv1d_fn(x, w, activation="silu")
    a, state = causal_conv1d_fn(x[..., :25], w, activation="silu", return_final_states=True)
    b = causal_conv1d_fn(x[..., 25:], w, initial_states=state, activation="silu")
    torch.testing.assert_close(torch.cat([a, b], -1), full)


def test_update_matches_full_conv():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 12)
    w = torch.randn(8, 4)
    bias = torch.randn(8)
    full = causal_conv1d_fn(x, w, bias, activation="silu")
    state = torch.zeros(2, 8, 4)
    steps = [causal_conv1d_update(x[..., t], state, w, bias, "silu") for t in range(12)]
    torch.testing.assert_close(torch.stack(steps, -1), full, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("state_len", [3, 4, 6])
@pytest.mark.parametrize("seqlen", [1, 5])
def test_update_multi_token_matches_full_conv(state_len, seqlen):
    torch.manual_seed(0)
    x = torch.randn(2, 8, 12 + seqlen)
    w = torch.randn(8, 4)
    full = causal_conv1d_fn(x, w, activation="silu")
    state = torch.zeros(2, 8, state_len)
    causal_conv1d_update(x[..., :12], state, w)
    out = causal_conv1d_update(x[..., 12:], state, w, activation="silu")
    assert out.shape == (2, 8, seqlen)
    torch.testing.assert_close(out, full[..., 12:], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(state, x[..., -state_len:])


def test_final_states_out_buffer():
    x = torch.randn(2, 8, 10)
    w = torch.randn(8, 4)
    buf = torch.empty(2, 8, 3)
    _, final = causal_conv1d_fn(x, w, return_final_states=True, final_states_out=buf)
    assert final is buf
    torch.testing.assert_close(buf, x[..., -3:])


def test_mixed_dtypes_return_input_dtype():
    x = torch.randn(2, 8, 10).bfloat16()
    w = torch.randn(8, 4)
    out = causal_conv1d_fn(x, w, torch.randn(8), activation="silu")
    assert out.dtype == torch.bfloat16
    state = torch.zeros(2, 8, 4, dtype=torch.bfloat16)
    assert causal_conv1d_update(x[..., 0], state, w).dtype == torch.bfloat16
