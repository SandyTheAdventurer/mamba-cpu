"""mamba_cpu.hf against transformers' own CPU fallback, on small random models."""

import importlib

import pytest
import torch

pytest.importorskip("transformers", minversion="5.0")

import transformers

import mamba_cpu.hf
from mamba_cpu import selective_scan_fn

SMALL = dict(vocab_size=64, hidden_size=32, num_hidden_layers=2, state_size=8, conv_kernel=4, expand=2)


def build(name):
    torch.manual_seed(0)
    if name == "mamba":
        return transformers.MambaForCausalLM(transformers.MambaConfig(**SMALL)).eval()
    if name == "falcon_mamba":
        return transformers.FalconMambaForCausalLM(transformers.FalconMambaConfig(**SMALL)).eval()
    config = transformers.JambaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, num_experts=1, attn_layer_period=2, attn_layer_offset=1,
        expert_layer_period=2, expert_layer_offset=1, mamba_d_state=8, mamba_dt_rank=4,
        use_mamba_kernels=False,
    )
    return transformers.JambaForCausalLM(config).eval()


@pytest.fixture
def patched(monkeypatch):
    """Runs fn on transformers' pure-torch fallbacks, then on mamba_cpu.

    The baseline is pinned to the fallbacks because transformers otherwise uses
    the CUDA packages when they are installed, and those reject CPU tensors.
    """
    mamba_cpu.hf.unpatch_transformers()
    for name in mamba_cpu.hf.SUPPORTED_MODELS:
        module = importlib.import_module(f"transformers.models.{name}.modeling_{name}")
        for op in mamba_cpu.hf._OPS:
            monkeypatch.setattr(module, op, getattr(module, op).__wrapped__)

    def run_both(fn):
        want = fn()
        mamba_cpu.hf.patch_transformers()
        try:
            return fn(), want
        finally:
            mamba_cpu.hf.unpatch_transformers()

    return run_both


@pytest.fixture
def scan_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mamba_cpu.hf, "selective_scan_fn", lambda *a: calls.append(1) or selective_scan_fn(*a)
    )
    return calls


models = pytest.mark.parametrize("name", mamba_cpu.hf.SUPPORTED_MODELS)


@models
def test_logits_match(name, patched, scan_calls):
    model = build(name)
    ids = torch.randint(0, 64, (2, 37))

    def logits():
        with torch.no_grad():
            return model(ids).logits

    got, want = patched(logits)
    assert scan_calls, "patched scan was not used"
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


@models
def test_generate_matches(name, patched, scan_calls):
    model = build(name)
    ids = torch.randint(0, 64, (2, 11))
    got, want = patched(lambda: model.generate(ids, max_new_tokens=12, do_sample=False))
    assert scan_calls
    assert torch.equal(got, want)


@models
def test_padded_batch_matches(name, patched):
    model = build(name)
    ids = torch.randint(0, 64, (2, 16))
    mask = torch.ones_like(ids)
    mask[1, :5] = 0

    def logits():
        with torch.no_grad():
            return model(ids, attention_mask=mask).logits

    got, want = patched(logits)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


@models
def test_training_gradients_match(name, patched, scan_calls):
    model = build(name).train()
    ids = torch.randint(0, 64, (2, 24))

    def grads():
        model.zero_grad()
        model(ids, labels=ids).loss.backward()
        return {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}

    got, want = patched(grads)
    assert scan_calls and got.keys() == want.keys()
    for k in want:
        torch.testing.assert_close(got[k], want[k], rtol=1e-3, atol=1e-4, msg=lambda m: f"{k}: {m}")


def test_patch_is_idempotent_and_reversible():
    module = importlib.import_module("transformers.models.mamba.modeling_mamba")
    original = module.mamba_selective_scan
    mamba_cpu.hf.patch_transformers()
    patched_fn = module.mamba_selective_scan
    mamba_cpu.hf.patch_transformers()
    assert module.mamba_selective_scan is patched_fn is not original
    mamba_cpu.hf.unpatch_transformers()
    assert module.mamba_selective_scan is original


@models
def test_patched_model_runs_as_installed(name, scan_calls):
    # No pinned fallbacks: must work even with mamba_ssm / causal_conv1d installed.
    model = build(name)
    ids = torch.randint(0, 64, (1, 9))
    mamba_cpu.hf.patch_transformers()
    try:
        model.generate(ids, max_new_tokens=3, do_sample=False)
        model.train()
        model(ids, labels=ids).loss.backward()
    finally:
        mamba_cpu.hf.unpatch_transformers()
    assert scan_calls
