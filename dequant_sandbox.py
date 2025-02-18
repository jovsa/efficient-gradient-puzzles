import torch
import triton
import triton.language as tl

@triton.jit
def _dequantize_nf4_kernel(
    input_ptr,
    absmax_ptr,
    lut_ptr,
    output_ptr,
    blocksize,
    n_blocks,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_TYPE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    if block_idx >= n_blocks:
        return

    absmax_val = tl.load(absmax_ptr + block_idx).to(tl.float32)

    input_offset = block_idx * (BLOCK_SIZE // 8)

    k = tl.arange(0, BLOCK_SIZE)
    int32_index = k // 8
    position_in_int32 = k % 8
    shift = (28 - position_in_int32 * 4)
    mask = 0xF

    int_val = tl.load(input_ptr + input_offset + int32_index)
    index = (int_val >> shift) & mask

    dequantized = tl.load(lut_ptr + index) * absmax_val

    dequantized = dequantized.to(OUTPUT_TYPE)
    output_offset = block_idx * BLOCK_SIZE + k
    tl.store(output_ptr + output_offset, dequantized)

def _dequantize_nf4(weight, quant_state):
    device = weight.device
    absmax = quant_state[1]
    nf4_lut = quant_state[0]

    assert weight.dtype in (torch.uint8, torch.bfloat16), "weight must be uint8 or bfloat16"
    assert absmax.dtype == torch.float16, "absmax must be float16"
    assert nf4_lut.dtype == torch.float32, "nf4_lut must be float32"
    assert nf4_lut.numel() == 16, "nf4_lut must have 16 elements"

    n_blocks = absmax.shape[0]
    output = torch.empty(n_blocks * 64, dtype=torch.float16, device=device)

    input = weight.view(torch.int32)
    grid = (n_blocks,)

    _dequantize_nf4_kernel[grid](
        input, absmax, nf4_lut, output,
        blocksize=64,
        n_blocks=n_blocks,
        BLOCK_SIZE=64,
        OUTPUT_TYPE=tl.float16,
    )
    return output

def test_dequantize_function():
    nf4 = torch.tensor([
        -1.0000, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0000,
        0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0000
    ], dtype=torch.float32, device='cuda')

    num_blocks = 128
    quantized = torch.randint(0, 256, (num_blocks, 32), dtype=torch.uint8, device='cuda')
    absmax = torch.randn(num_blocks, dtype=torch.float16, device='cuda').abs()

    output_fp16 = _dequantize_nf4(quantized, (nf4, absmax))
    assert output_fp16.shape == (num_blocks * 64,)
    assert output_fp16.dtype == torch.float16

    output_bf16 = _dequantize_nf4(quantized.to(torch.bfloat16), (nf4, absmax))
    assert output_bf16.shape == (num_blocks * 64,)
    assert output_bf16.dtype == torch.float16

    print("Test passed!")

if __name__ == "__main__":
    test_dequantize_function()