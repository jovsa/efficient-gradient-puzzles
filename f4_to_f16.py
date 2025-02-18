"""

Goal: Convert a `nf4` quantized tensor into `fp16` or `bf16` into a *single* Triton kernel The double dequant of the `absmax` and weight forming must be done in 1 Triton kernel. Must work on Tesla T4.
1. Must be faster than Unsloth's `fast_dequantize` by 1.15x or more, and not use large intermediate memory buffers.
2. Must not use `torch.compile`, but can use `trace.enabled` to help on writing Triton kernels.
3. Good material: [Unsloth `fast_dequantize` function](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/utils.py#L128), also [bitsandbytes `dequantize_blockwise`](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/86b6c37a8ad448230cedb60753f63150b603a112/bitsandbytes/functional.py#L958)
4. Use `test_dequantize_function` to test your implementation.
5. No CUDA allowed. Custom CUDA inside of the Triton is allowed.
6. Watch Tim's videos on Youtube: [8-bit Optimizers](https://www.youtube.com/watch?v=2ETNONas068)
"""

import torch
import torch.nn as nn
from transformers import set_seed
import time
from bitsandbytes.nn import Linear4bit
from transformers.activations import ACT2FN
from unsloth.kernels.utils import fast_dequantize
from utils import assert_same, _F, _C
from peft.utils.integrations import dequantize_module_weight as peft_dequantize
from triton import jit
import triton
import triton.language as tl


def unsloth_dequantize(weight):
    return fast_dequantize(weight.weight, weight.weight.quant_state)


def bnb_Linear4bit(hd, m, dtype=torch.float16):
    return Linear4bit(
        hd,
        m,
        bias=None,
        compute_dtype=dtype,
        compress_statistics=True,
        quant_type="nf4",
    )


class MLP(torch.nn.Module):
    def __init__(self, hd=4096, m=14336, dtype=torch.float16):
        super().__init__()
        self.gate_proj = bnb_Linear4bit(hd, m, dtype=dtype)
        self.up_proj = bnb_Linear4bit(hd, m, dtype=dtype)
        self.down_proj = bnb_Linear4bit(m, hd, dtype=dtype)
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def mlp_forward(X, mlp, fx):
    up = X @ fx(mlp.up_proj).t()
    gate = X @ fx(mlp.gate_proj).t()
    h = mlp.act_fn(gate) * up
    down = h @ fx(mlp.down_proj).t()
    return down


def mlp_dequantize(X, mlp, fx):
    a = fx(mlp.up_proj).t()
    torch.cuda.synchronize()
    b = fx(mlp.gate_proj).t()
    torch.cuda.synchronize()
    c = fx(mlp.down_proj).t()
    torch.cuda.synchronize()
    return a, b, c


def test_dequantize(dequantize_fx):
    elapsed = 0
    options = [
        (5, 777, 1024, 4096, 3409, torch.bfloat16),
        (3, 2048, 4096, 14336, 3408, torch.bfloat16),
        (2, 3333, 2048, 8192, 3407, torch.float16),
    ]
    for bsz, qlen, hd, m, seed, dt in options:
        set_seed(seed)
        torch.set_default_dtype(dt)
        mlp = MLP(hd=hd, m=m, dtype=dt).to("cuda")
        X = torch.randn((bsz, qlen, hd), device="cuda")
        torch.cuda.synchronize()

        # Warmup
        for _ in range(2):
            assert_same(mlp_forward(X, mlp, dequantize_fx), mlp(X), _F(_C()), dt)
            a, b, c = mlp_dequantize(X, mlp, dequantize_fx)
            A, B, C = mlp_dequantize(X, mlp, unsloth_dequantize)
            assert_same(a, A, _F(_C()), dt)
            assert_same(b, B, _F(_C()), dt)
            assert_same(c, C, _F(_C()), dt)

        # Benchmarking
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(1000):
            mlp_dequantize(X, mlp, dequantize_fx)
        elapsed += time.time() - start
    return elapsed


def unsloth_dequantize(weight):
    return fast_dequantize(weight.weight, weight.weight.quant_state)


@triton.jit
def _dequantize_nf4_kernel():
    pass


def _dequantize_nf4(weight, quant_state):
    pass


def dequantize_nf4(weight):
    return _dequantize_nf4(weight.weight.data, weight.weight.quant_state)


if __name__ == "__main__":
    res = test_dequantize(unsloth_dequantize)
    print(f"unsloth_dequantize: {res}s")
    res = test_dequantize(peft_dequantize)
    print(f"peft_dequantize: {res}s")

    ### TEST IT BELOW:
    # res = test_dequantize(dequantize_nf4)
    # print(f"dequantize_nf4: {res}s")

    ### CALCULATE SPEEDUP (hopefully 1.15x faster or more)
    # print(f"Speedup: {test_dequantize(unsloth_dequantize) / test_dequantize(dequantize_nf4)}")
