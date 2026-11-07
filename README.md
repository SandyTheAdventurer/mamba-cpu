# mamba-cpu

Multithreaded CPU kernels for [Mamba](https://arxiv.org/abs/2312.00752) selective state space models, with the same API as [`mamba_ssm`](https://github.com/state-spaces/mamba) and [`causal_conv1d`](https://github.com/Dao-AILab/causal-conv1d).

`mamba_ssm` ships fused CUDA/Triton kernels and won't import without a GPU. `mamba-cpu` gives you the same ops on CPU, so you can run Mamba and hybrid models, unit tests and CI on machines without CUDA.

```bash
pip install mamba-cpu
```

Prebuilt wheels: Linux (x86_64, aarch64), macOS (arm64), Windows (x86_64), Python 3.10+. The native extension doesn't link against libtorch, so one wheel works with any PyTorch ≥ 2.4.

## Usage

```python
import torch
from mamba_cpu import selective_scan_fn, causal_conv1d_fn, selective_state_update

batch, dim, seqlen, dstate = 1, 1536, 2048, 16
u = torch.randn(batch, dim, seqlen)
delta = torch.randn(batch, dim, seqlen)
A = -torch.rand(dim, dstate)
B = torch.randn(batch, dstate, seqlen)
C = torch.randn(batch, dstate, seqlen)
D = torch.randn(dim)

y = selective_scan_fn(u, delta, A, B, C, D, delta_softplus=True)
```

Signatures match the upstream functions:

| mamba-cpu | upstream |
|---|---|
| `mamba_cpu.selective_scan_fn` | `mamba_ssm.ops.selective_scan_interface.selective_scan_fn` |
| `mamba_cpu.selective_state_update` | `mamba_ssm.ops.triton.selective_state_update.selective_state_update` |
| `mamba_cpu.causal_conv1d_fn` | `causal_conv1d.causal_conv1d_fn` |
| `mamba_cpu.causal_conv1d_update` | `causal_conv1d.causal_conv1d_update` |

Kernels use `torch.get_num_threads()` threads, so `torch.set_num_threads(n)` controls them as well.

## Status

| Op | Forward | Backward |
|---|---|---|
| `selective_scan_fn` | native C++, multithreaded | native C++, multithreaded |
| `selective_state_update` | PyTorch | n/a (inference) |
| `causal_conv1d_fn` / `_update` | PyTorch (`F.conv1d`) | autograd |

Not supported yet: complex `A`, input-independent (2D) `B`/`C`, `seq_idx` in `causal_conv1d_fn`, Mamba-2 chunked scan (`mamba_chunk_scan_combined`).

### Roadmap

- [x] Native backward for `selective_scan_fn`
- [ ] Chunked recompute in backward to bound memory for very long sequences
- [x] SIMD: vectorized `exp`, runtime AVX2+FMA dispatch on x86 (baseline SSE2 / NEON otherwise)
- [ ] Vectorize across channels instead of `dstate` (better for small `dstate` like 16)
- [ ] Persistent thread pool (thread scaling flattens past ~4 threads)
- [ ] Chunked parallel scan along time for small batch × long sequence
- [ ] Mamba-2 SSD (`mamba_chunk_scan_combined`)
- [ ] Native `selective_state_update` for low-latency decode
- [ ] Optional backend for HF `transformers` Mamba models

## Development

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test]"
pytest
python benchmarks/bench_selective_scan.py
```

Layout:

- `csrc/` — torch-free C++ kernels and nanobind bindings
- `src/mamba_cpu/ops/` — `torch.library` custom ops, autograd and public API
- `src/mamba_cpu/reference.py` — slow pure-PyTorch reference used as the test oracle

### Releasing

Wheels are built by `.github/workflows/wheels.yml` (cibuildwheel) and published to PyPI on a `v*` tag via [trusted publishing](https://docs.pypi.org/trusted-publishers/). One-time setup: add a pending publisher on PyPI for this repo with workflow `wheels.yml` and environment `pypi`, and create a `pypi` environment in the GitHub repo settings.

## License

Apache-2.0
