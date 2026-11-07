"""Parity with the real mamba_ssm / causal_conv1d CUDA kernels.

Skipped unless those packages and a GPU are available, so this doesn't run in CI.
"""

import inspect

import pytest
import torch

mamba_ssm_ops = pytest.importorskip("mamba_ssm.ops.selective_scan_interface")
upstream_conv = pytest.importorskip("causal_conv1d")
upstream_update = pytest.importorskip("mamba_ssm.ops.triton.selective_state_update")

import mamba_cpu

pytestmark = pytest.mark.usefixtures("cpu_isa")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("ours,theirs", [
    (mamba_cpu.selective_scan_fn, mamba_ssm_ops.selective_scan_fn),
    (mamba_cpu.selective_state_update, upstream_update.selective_state_update),
    (mamba_cpu.causal_conv1d_fn, upstream_conv.causal_conv1d_fn),
    (mamba_cpu.causal_conv1d_update, upstream_conv.causal_conv1d_update),
])
def test_signatures_match(ours, theirs):
    assert inspect.signature(ours) == inspect.signature(theirs)


def to_cuda(inp):
    return {
        k: None if v is None else v.detach().cuda().requires_grad_(v.requires_grad)
        for k, v in inp.items()
    }


def scan_inputs(batch, dim, seqlen, dstate, ngroups, dtype, full=True):
    g = torch.Generator().manual_seed(0)

    def rand(*shape, dt=dtype):
        return torch.randn(*shape, generator=g).to(dt).requires_grad_()

    bc = (batch, dstate, seqlen) if ngroups is None else (batch, ngroups, dstate, seqlen)
    return dict(
        u=rand(batch, dim, seqlen),
        delta=(torch.rand(batch, dim, seqlen, generator=g) * 0.5).to(dtype).requires_grad_(),
        A=(-torch.rand(dim, dstate, generator=g) * 16 - 0.5).requires_grad_(),
        B=rand(*bc),
        C=rand(*bc),
        D=rand(dim, dt=torch.float32) if full else None,
        z=rand(batch, dim, seqlen) if full else None,
        delta_bias=(torch.rand(dim, generator=g) * 0.1).requires_grad_() if full else None,
    )


@cuda
@pytest.mark.parametrize("batch,dim,seqlen,dstate", [(1, 16, 1, 4), (2, 64, 257, 16), (1, 32, 2100, 16)])
@pytest.mark.parametrize("ngroups", [None, 1, 2])
@pytest.mark.parametrize("full,softplus", [(True, True), (False, False)])
def test_selective_scan_fwd_bwd(batch, dim, seqlen, dstate, ngroups, full, softplus):
    inp = scan_inputs(batch, dim, seqlen, dstate, ngroups, torch.float32, full)
    gpu = to_cuda(inp)
    out, last = mamba_cpu.selective_scan_fn(**inp, delta_softplus=softplus, return_last_state=True)
    out_c, last_c = mamba_ssm_ops.selective_scan_fn(**gpu, delta_softplus=softplus, return_last_state=True)
    torch.testing.assert_close(out, out_c.cpu(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(last, last_c.cpu(), rtol=1e-3, atol=1e-3)

    w = torch.randn_like(out)
    (out * w).sum().backward()
    (out_c * w.cuda()).sum().backward()
    for k, v in inp.items():
        if v is not None:
            scale = max(gpu[k].grad.abs().max().item(), 1.0)
            torch.testing.assert_close(v.grad / scale, gpu[k].grad.cpu() / scale, rtol=1e-3, atol=1e-3, msg=lambda m: f"{k}: {m}")


@cuda
def test_selective_scan_bf16():
    inp = scan_inputs(2, 64, 128, 16, None, torch.bfloat16)
    out = mamba_cpu.selective_scan_fn(**inp, delta_softplus=True)
    out_c = mamba_ssm_ops.selective_scan_fn(**to_cuda(inp), delta_softplus=True)
    assert out.dtype == out_c.dtype
    torch.testing.assert_close(out, out_c.cpu(), rtol=3e-2, atol=3e-2)


@cuda
@pytest.mark.parametrize("multihead", [False, True])
def test_selective_state_update(multihead):
    torch.manual_seed(0)
    b, n = 2, 16
    lead = (4, 8) if multihead else (32,)
    state = torch.randn(b, *lead, n)
    x, dt, z = torch.randn(b, *lead), torch.rand(b, *lead), torch.randn(b, *lead)
    A = -torch.rand(*lead, n) - 0.5
    B = torch.randn(b, 2, n) if multihead else torch.randn(b, n)
    C = torch.randn_like(B)
    D, dt_bias = torch.randn(*lead), torch.rand(*lead)
    args = (x, dt, A, B, C, D, z, dt_bias, True)
    state_c = state.cuda()
    out_c = upstream_update.selective_state_update(state_c, *[t.cuda() if torch.is_tensor(t) else t for t in args])
    out = mamba_cpu.selective_state_update(state, *args)
    torch.testing.assert_close(out, out_c.cpu(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state, state_c.cpu(), rtol=1e-4, atol=1e-4)


@cuda
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("activation", [None, "silu"])
def test_causal_conv1d(width, activation):
    torch.manual_seed(0)
    x, w, bias = torch.randn(2, 64, 100), torch.randn(64, width), torch.randn(64)
    out = mamba_cpu.causal_conv1d_fn(x, w, bias, activation=activation)
    out_c = upstream_conv.causal_conv1d_fn(x.cuda(), w.cuda(), bias.cuda(), activation=activation)
    torch.testing.assert_close(out, out_c.cpu(), rtol=1e-4, atol=1e-4)


@cuda
@pytest.mark.parametrize("state_len", [3, 4, 7])
@pytest.mark.parametrize("seqlen", [None, 1, 3])
def test_causal_conv1d_update(state_len, seqlen):
    torch.manual_seed(0)
    x = torch.randn(2, 64) if seqlen is None else torch.randn(2, 64, seqlen)
    w, bias = torch.randn(64, 4), torch.randn(64)
    state = torch.randn(2, 64, state_len)
    state_c = state.cuda()
    out_c = upstream_conv.causal_conv1d_update(x.cuda(), state_c, w.cuda(), bias.cuda(), "silu")
    out = mamba_cpu.causal_conv1d_update(x, state, w, bias, "silu")
    torch.testing.assert_close(out, out_c.cpu(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state, state_c.cpu())
