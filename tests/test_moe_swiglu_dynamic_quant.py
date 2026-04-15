import time
import pytest
import torch
import statistics

import tests.register_ops as ops  # noqa: F401

def baseline_moe_swiglu(
    scatter_tokens, smooth_scale, experts_token_count, experts_token_start
):
    """Pure PyTorch baseline for MoE SwiGLU + Dynamic Quantization."""
    num_experts = experts_token_count.shape[0]
    num_scattered = scatter_tokens.shape[0]
    hidden_size = scatter_tokens.shape[1] // 2

    quant_tokens = torch.zeros((num_scattered, hidden_size), dtype=torch.int8, device=scatter_tokens.device)
    per_token_scale = torch.zeros(num_scattered, dtype=torch.float32, device=scatter_tokens.device)

    for exp in range(num_experts):
        start = experts_token_start[exp].item()
        count = experts_token_count[exp].item()
        for t in range(count):
            target_idx = start + t
            
            x1 = scatter_tokens[target_idx, :hidden_size].float()
            x2 = scatter_tokens[target_idx, hidden_size:].float()
            scale = smooth_scale[exp].float()

            x1_silu = torch.nn.functional.silu(x1)
            swiglu = x1_silu * x2
            scaled = swiglu * scale
            
            max_val = scaled.abs().max().item()
            this_token_scale = max_val / 127.0

            quantized = (scaled / (this_token_scale if this_token_scale != 0 else 1.0)).to(torch.int8)

            quant_tokens[target_idx] = quantized
            per_token_scale[target_idx] = this_token_scale

    return quant_tokens, per_token_scale

@pytest.mark.parametrize("num_scattered", [32, 512, 2048])
@pytest.mark.parametrize("hidden_size", [128, 4096])
@pytest.mark.parametrize("num_experts", [64, 256])
def test_moe_swiglu_dynamic_quant(num_scattered, hidden_size, num_experts):
    device = "xpu"
    dtype = torch.bfloat16

    # Fake routing data contiguous block assignments
    experts_token_count = torch.zeros(num_experts, dtype=torch.int32, device=device)
    experts_token_start = torch.zeros(num_experts, dtype=torch.int32, device=device)
    
    # Assign scattered tokens equally to experts roughly
    tokens_per_expert = num_scattered // num_experts
    experts_token_count[:] = tokens_per_expert
    experts_token_count[-1] = num_scattered - (tokens_per_expert * (num_experts - 1)) # Dump remainder
    
    start = 0
    for i in range(num_experts):
        experts_token_start[i] = start
        start += experts_token_count[i].item()

    scatter_tokens = torch.randn((num_scattered, hidden_size * 2), dtype=dtype, device=device)
    smooth_scale = torch.rand((num_experts, hidden_size), dtype=torch.float32, device=device)

    # 1. Run Torch Accuracy Baseline
    ref_quant_tokens, ref_per_scale = baseline_moe_swiglu(
        scatter_tokens, smooth_scale, experts_token_count, experts_token_start
    )

    # 2. Run XPU Custom Kernel
    out_quant_tokens = torch.zeros((num_scattered, hidden_size), dtype=torch.int8, device=device)
    out_per_scale = torch.zeros(num_scattered, dtype=torch.float32, device=device)

    torch.ops._moe_C.moe_swiglu_dynamic_quant(
        scatter_tokens, smooth_scale, experts_token_count, experts_token_start,
        out_quant_tokens, out_per_scale, num_experts, num_scattered
    )

    # Verify Computation
    torch.testing.assert_close(out_per_scale, ref_per_scale, atol=1e-5, rtol=1e-5)
    diff = (out_quant_tokens.int() - ref_quant_tokens.int()).abs()
    assert diff.max().item() <= 1, "SwiGLU Quantized values diverge!"

    # 3. Micro Benchmark
    torch.xpu.synchronize()
    start_t = time.perf_counter()
    for _ in range(50):
        torch.ops._moe_C.moe_swiglu_dynamic_quant(
            scatter_tokens, smooth_scale, experts_token_count, experts_token_start,
            out_quant_tokens, out_per_scale, num_experts, num_scattered
        )
    torch.xpu.synchronize()
    ms_time = ((time.perf_counter() - start_t) / 50.0) * 1000
    
    print(f"SwiGLU Quant: Scattered Tokens={num_scattered}, Hidden={hidden_size}, Experts={num_experts} | Time: {ms_time:.4f} ms")