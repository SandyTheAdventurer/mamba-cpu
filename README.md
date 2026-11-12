# mamba-cpu

Multithreaded CPU kernels for [Mamba](https://arxiv.org/abs/2312.00752) selective state space models, with the same function signatures as [`mamba_ssm`](https://github.com/state-spaces/mamba) and [`causal_conv1d`](https://github.com/Dao-AILab/causal-conv1d).

`mamba_ssm` ships fused CUDA/Triton kernels and needs a GPU. `mamba-cpu` provides the same ops on CPU, including the backward pass, so you can run, fine-tune and test Mamba models on machines without CUDA.

**Scope:** this release covers Mamba-1. The Mamba-2 chunked scan (`mamba_chunk_scan_combined`) isn't implemented yet, so Mamba-2 models and hybrids built on it get no speedup.

```bash
pip install mamba-cpu
```

Prebuilt wheels: Linux (x86_64, aarch64), macOS (arm64), Windows (x86_64), Python 3.10+. The native extension doesn't link against libtorch, so one wheel works across PyTorch versions (≥ 2.4).

## Hugging Face transformers

One call routes `Mamba`, `FalconMamba` and `Jamba` models through these kernels on CPU (transformers ≥ 5):

```python
import mamba_cpu.hf
from transformers import AutoModelForCausalLM, AutoTokenizer

mamba_cpu.hf.patch_transformers()

tok = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
ids = tok("Mamba is", return_tensors="pt").input_ids
print(tok.decode(model.generate(ids, max_new_tokens=32)[0]))
```

Tensors on other devices keep using transformers' own path. This also fixes the crash on CPU tensors that happens when `mamba_ssm` or `causal_conv1d` are installed, because transformers uses them whenever they import.

`state-spaces/mamba-130m-hf` on an 8-thread laptop CPU, float32:

| | prefill, 1024 tokens | training step, 256 tokens | decode |
|---|---|---|---|
| transformers CPU fallback | 6.7 s | 60.6 s | 19 tokens/s |
| `mambapy` parallel scan | 5.8 s | | |
| **mamba-cpu** | **1.4 s** | **1.2 s** | 18 tokens/s |

Decode doesn't get faster yet: single-token steps use `selective_state_update`, which is still plain PyTorch.

## Direct use

```python
import torch
from mamba_cpu import selective_scan_fn

batch, dim, seqlen, dstate = 1, 1536, 2048, 16
u = torch.randn(batch, dim, seqlen)
delta = torch.randn(batch, dim, seqlen)
A = -torch.rand(dim, dstate)
B = torch.randn(batch, dstate, seqlen)
C = torch.randn(batch, dstate, seqlen)
D = torch.randn(dim)

y = selective_scan_fn(u, delta, A, B, C, D, delta_softplus=True)
```

Signatures are identical to upstream, so porting code means changing the import:

| mamba-cpu | upstream |
|---|---|
| `mamba_cpu.selective_scan_fn` | `mamba_ssm.ops.selective_scan_interface.selective_scan_fn` |
| `mamba_cpu.selective_state_update` | `mamba_ssm.ops.triton.selective_state_update.selective_state_update` |
| `mamba_cpu.causal_conv1d_fn` | `causal_conv1d.causal_conv1d_fn` |
| `mamba_cpu.causal_conv1d_update` | `causal_conv1d.causal_conv1d_update` |

Kernels use `torch.get_num_threads()` threads, so `torch.set_num_threads(n)` controls them as well. On x86 they use AVX2+FMA when the CPU supports it, chosen at runtime.

## Status

| Op | Forward | Backward |
|---|---|---|
| `selective_scan_fn` | native C++, multithreaded, SIMD | native C++, multithreaded, SIMD |
| `selective_state_update` | PyTorch | n/a (inference) |
| `causal_conv1d_fn` / `_update` | PyTorch (`F.conv1d`) | autograd |

These arguments have the upstream names but raise `NotImplementedError`:

- `selective_scan_fn`: complex `A`, input-independent (2D) `B`/`C`, different group counts for `B` and `C`
- `causal_conv1d_fn`: `seq_idx`
- `causal_conv1d_update`: `cache_seqlens`, `conv_state_indices`
- `selective_state_update`: `state_batch_indices`

Backward memory is `seqlen × dstate × 12` bytes per thread, about 200 KB per thread at 1024 × 16.

### Verification

`tests/test_upstream.py` compares every op, including all `selective_scan_fn` gradients, with the real `mamba_ssm` and `causal_conv1d` CUDA kernels. It needs a GPU, so it doesn't run in CI. `benchmarks/check_hf_model.py` does the same for a pretrained model: on `mamba-130m-hf`, greedy output is identical to the CUDA kernels and top-1 predictions agree on all 1024 positions.

### Roadmap

- [ ] Mamba-2 SSD (`mamba_chunk_scan_combined`)
- [ ] Vectorize across channels instead of `dstate` (better for small `dstate` like 16)
- [ ] Persistent thread pool (thread scaling flattens past ~4 threads)
- [ ] Chunked parallel scan along time for small batch × long sequence
- [ ] Chunked recompute in backward to bound memory for very long sequences
- [ ] Native `selective_state_update` for low-latency decode

## Development

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test]"
pytest
python benchmarks/bench_selective_scan.py
python benchmarks/check_hf_model.py
```

Layout:

- `csrc/` — torch-free C++ kernels and nanobind bindings
- `src/mamba_cpu/ops/` — `torch.library` custom ops, autograd and public API
- `src/mamba_cpu/hf.py` — Hugging Face transformers integration
- `src/mamba_cpu/reference.py` — slow pure-PyTorch reference used as the test oracle

### Releasing

Wheels are built by `.github/workflows/wheels.yml` (cibuildwheel) and published to PyPI on a `v*` tag via [trusted publishing](https://docs.pypi.org/trusted-publishers/). One-time setup: add a pending publisher on PyPI for this repo with workflow `wheels.yml` and environment `pypi`, and create a `pypi` environment in the GitHub repo settings.

To release: set `version` in `pyproject.toml` (e.g. `0.1.0`), commit, then

```bash
git tag v0.1.0
git push origin main v0.1.0
```

The workflow refuses to publish if the tag and `pyproject.toml` version differ.

## License

Apache-2.0
