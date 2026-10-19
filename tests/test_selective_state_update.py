import torch

from mamba_cpu import selective_scan_ref, selective_state_update


def test_steps_match_full_scan():
    torch.manual_seed(0)
    batch, dim, seqlen, dstate = 2, 16, 10, 8
    u = torch.randn(batch, dim, seqlen)
    delta = torch.rand(batch, dim, seqlen) * 0.5
    A = -torch.rand(dim, dstate) - 0.5
    B = torch.randn(batch, dstate, seqlen)
    C = torch.randn(batch, dstate, seqlen)
    D = torch.randn(dim)
    z = torch.randn(batch, dim, seqlen)
    dt_bias = torch.rand(dim) * 0.1

    full, last = selective_scan_ref(u, delta, A, B, C, D, z, dt_bias, True, return_last_state=True)

    state = torch.zeros(batch, dim, dstate)
    steps = [
        selective_state_update(
            state, u[..., t], delta[..., t], A, B[..., t], C[..., t], D, z[..., t], dt_bias, True
        )
        for t in range(seqlen)
    ]
    torch.testing.assert_close(torch.stack(steps, -1), full, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(state, last, rtol=1e-5, atol=1e-5)


def test_multihead_matches_flattened():
    # Mamba-2 layout with ngroups=1 is Mamba-1 with dim = nheads * headdim.
    torch.manual_seed(0)
    batch, nheads, headdim, dstate = 2, 4, 3, 8
    state = torch.randn(batch, nheads, headdim, dstate)
    x = torch.randn(batch, nheads, headdim)
    dt = torch.rand(batch, nheads, headdim)
    A = -torch.rand(nheads, headdim, dstate)
    B = torch.randn(batch, 1, dstate)
    C = torch.randn(batch, 1, dstate)
    D = torch.randn(nheads, headdim)

    flat_state = state.reshape(batch, -1, dstate).clone()
    out = selective_state_update(state, x, dt, A, B, C, D)
    flat_out = selective_state_update(
        flat_state, x.reshape(batch, -1), dt.reshape(batch, -1), A.reshape(-1, dstate),
        B[:, 0], C[:, 0], D.reshape(-1),
    )
    torch.testing.assert_close(out.reshape(batch, -1), flat_out)
    torch.testing.assert_close(state.reshape(batch, -1, dstate), flat_state)
