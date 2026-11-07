"""End-to-end check of mamba_cpu.hf on a real pretrained model.

Compares mamba_cpu on CPU against transformers' pure-torch CPU fallback and, if
a GPU and mamba_ssm are available, the real CUDA kernels. Also times prefill,
decode and a training step. mambapy is timed too if installed.

    python benchmarks/check_hf_model.py --model state-spaces/mamba-130m-hf
"""

import argparse
import importlib
import importlib.util
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mamba_cpu.hf

PROMPT = (
    "The state space model is a mathematical description of a physical system. "
    "In machine learning, selective state space models process sequences by"
)


def use_torch_fallbacks():
    """Pin transformers to its pure-torch ops, even if the CUDA packages are installed."""
    mamba_cpu.hf.unpatch_transformers()
    module = importlib.import_module("transformers.models.mamba.modeling_mamba")
    saved = {op: getattr(module, op) for op in mamba_cpu.hf._OPS}
    for op, fn in saved.items():
        setattr(module, op, fn.__wrapped__)

    def restore():
        for op, fn in saved.items():
            setattr(module, op, fn)

    return restore


def timed(fn, iters=3):
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="state-spaces/mamba-130m-hf")
    p.add_argument("--seqlen", type=int, default=1024)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids
    long_ids = torch.randint(0, model.config.vocab_size, (1, args.seqlen))
    train_ids = long_ids[:, :256]

    def logits(m=model, x=ids):
        with torch.no_grad():
            return m(x.to(m.device)).logits.float().cpu()

    def generate(m=model):
        out = m.generate(ids.to(m.device), max_new_tokens=48, do_sample=False)
        return out.cpu()

    def train_grads():
        model.zero_grad()
        loss = model(train_ids, labels=train_ids).loss
        loss.backward()
        return loss.item(), {k: v.grad.clone() for k, v in model.named_parameters() if v.grad is not None}

    def decode_tok_s():
        n = 32
        t = timed(lambda: model.generate(ids, max_new_tokens=n, min_new_tokens=n, do_sample=False), iters=2)
        return n / t

    results, timings = {}, {}

    restore = use_torch_fallbacks()
    results["fallback"] = logits(), logits(x=long_ids), generate()
    timings["transformers fallback"] = (
        timed(lambda: logits(x=long_ids), iters=1), decode_tok_s(),
        timed(lambda: train_grads(), iters=1),
    )
    loss_fb, grads_fb = train_grads()
    restore()

    mamba_cpu.hf.patch_transformers()
    results["mamba_cpu"] = logits(), logits(x=long_ids), generate()
    timings["mamba_cpu"] = (timed(lambda: logits(x=long_ids)), decode_tok_s(), timed(lambda: train_grads()))
    loss_mc, grads_mc = train_grads()
    mamba_cpu.hf.unpatch_transformers()

    if importlib.util.find_spec("mambapy"):
        restore = use_torch_fallbacks()
        for layer in model.backbone.layers:
            layer.mixer.use_mambapy = True
        timings["mambapy pscan"] = (timed(lambda: logits(x=long_ids), iters=1), float("nan"), float("nan"))
        for layer in model.backbone.layers:
            layer.mixer.use_mambapy = False
        restore()

    if torch.cuda.is_available() and importlib.util.find_spec("mamba_ssm"):
        gpu = model.cuda()
        results["mamba_ssm CUDA"] = logits(gpu), logits(gpu, long_ids), generate(gpu)
        model.cpu()

    base = results["mamba_cpu"]
    print(f"\n{args.model}: mamba_cpu on CPU vs ...")
    for name, (short, long_, gen) in results.items():
        if name == "mamba_cpu":
            continue
        print(f"  {name:16s} max|dlogit| {(base[0] - short).abs().max():.2e} (short) "
              f"{(base[1] - long_).abs().max():.2e} (L={args.seqlen}), "
              f"top-1 agree {(base[1].argmax(-1) == long_.argmax(-1)).float().mean():.4f}, "
              f"greedy text identical: {torch.equal(base[2], gen)}")
    print(f"  logit scale      max|logit| {base[1].abs().max():.1f}")
    worst = max(
        ((grads_mc[k] - g).abs().max() / g.abs().max().clamp_min(1e-12)).item() for k, g in grads_fb.items()
    )
    print(f"  training         loss {loss_mc:.6f} vs {loss_fb:.6f}, worst per-tensor grad error {worst:.2e}")
    print("\nsample:", tok.decode(base[2][0][ids.shape[1]:]).strip()[:160])
    print(f"\n{'backend':24s}{'prefill L=' + str(args.seqlen):>18s}{'decode':>14s}{'train step L=256':>20s}")
    for name, (pre, dec, tr) in timings.items():
        print(f"{name:24s}{pre * 1e3:15.0f} ms{dec:10.1f} t/s{tr * 1e3:17.0f} ms")


if __name__ == "__main__":
    main()
