import pytest
import torch

from mamba_cpu import selective_scan_fn, selective_scan_ref


def make_inputs(batch, dim, seqlen, dstate, ngroups, has_D, has_z, has_bias, dtype, requires_grad=False):
    g = torch.Generator().manual_seed(0)

    def rand(*shape, scale=1.0, dt=dtype):
        return (torch.randn(*shape, generator=g) * scale).to(dt).requires_grad_(requires_grad)

    bc_shape = (batch, dstate, seqlen) if ngroups is None else (batch, ngroups, dstate, seqlen)
    return dict(
        u=rand(batch, dim, seqlen),
        delta=(torch.rand(batch, dim, seqlen, generator=g) * 0.5).to(dtype).requires_grad_(requires_grad),
        # Negative real A, as in Mamba's A = -exp(A_log).
        A=(-torch.rand(dim, dstate, generator=g) - 0.5).requires_grad_(requires_grad),
        B=rand(*bc_shape),
        C=rand(*bc_shape),
        D=rand(dim, dt=torch.float32) if has_D else None,
        z=rand(batch, dim, seqlen) if has_z else None,
        delta_bias=(torch.rand(dim, generator=g) * 0.1).requires_grad_(requires_grad) if has_bias else None,
    )


@pytest.mark.parametrize("batch,dim,seqlen,dstate", [(1, 4, 1, 1), (2, 16, 33, 16), (3, 32, 128, 8)])
@pytest.mark.parametrize("ngroups", [None, 1, 2])
@pytest.mark.parametrize("has_D,has_z,has_bias,softplus", [
    (False, False, False, False),
    (True, True, True, True),
    (True, False, True, False),
])
def test_forward_matches_reference(batch, dim, seqlen, dstate, ngroups, has_D, has_z, has_bias, softplus):
    inp = make_inputs(batch, dim, seqlen, dstate, ngroups, has_D, has_z, has_bias, torch.float32)
    out, last = selective_scan_fn(**inp, delta_softplus=softplus, return_last_state=True)
    out_ref, last_ref = selective_scan_ref(**inp, delta_softplus=softplus, return_last_state=True)
    torch.testing.assert_close(out, out_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(last, last_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_inputs(dtype):
    inp = make_inputs(2, 16, 64, 16, None, True, True, True, dtype)
    out = selective_scan_fn(**inp, delta_softplus=True)
    out_ref = selective_scan_ref(**inp, delta_softplus=True)
    assert out.dtype == dtype
    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("ngroups", [None, 2])
def test_backward_matches_reference(ngroups):
    inp = make_inputs(2, 8, 17, 4, ngroups, True, True, True, torch.float32, requires_grad=True)
    out, last = selective_scan_fn(**inp, delta_softplus=True, return_last_state=True)
    (out.square().sum() + last.sum()).backward()
    grads = {k: v.grad.clone() for k, v in inp.items() if v is not None}
    for v in inp.values():
        if v is not None:
            v.grad = None
    out_ref, last_ref = selective_scan_ref(**inp, delta_softplus=True, return_last_state=True)
    (out_ref.square().sum() + last_ref.sum()).backward()
    for k, g in grads.items():
        torch.testing.assert_close(g, inp[k].grad, rtol=1e-4, atol=1e-4, msg=k)


def test_opcheck():
    inp = make_inputs(1, 4, 8, 2, 1, True, True, True, torch.float32, requires_grad=True)
    args = (
        inp["u"], inp["delta"], inp["A"],
        inp["B"].detach().transpose(-1, -2).contiguous().requires_grad_(),
        inp["C"].detach().transpose(-1, -2).contiguous().requires_grad_(),
        inp["D"], inp["z"], inp["delta_bias"], True,
    )
    torch.library.opcheck(torch.ops.mamba_cpu.selective_scan_fwd.default, args)


def test_rejects_non_cpu_and_complex():
    inp = make_inputs(1, 4, 8, 2, None, False, False, False, torch.float32)
    inp["A"] = inp["A"].to(torch.complex64)
    with pytest.raises(NotImplementedError):
        selective_scan_fn(**inp)
