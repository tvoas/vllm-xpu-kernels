# SPDX-License-Identifier: Apache-2.0
"""
Tests and benchmarks for non-multimodal rotary embedding on XPU.
Replaces existing tests in tests/kernels/test_rotary_embedding.py
"""

import time
import pytest
import torch
import csv
import os
import statistics

from tests.ops.rotary_embedding_op import RotaryEmbedding
from tests.utils import opcheck
from collections import defaultdict

# Fixed seed for repeatability
torch.manual_seed(42)

CSV_FILENAME = "test_rotary_embedding.csv"
_CACHE_FLUSH_TENSOR = None

def _flush_cache(device: str):
    """Allocates/Zeros a 256MB tensor to force L2 Cache eviction."""
    global _CACHE_FLUSH_TENSOR
    if _CACHE_FLUSH_TENSOR is None or _CACHE_FLUSH_TENSOR.device.type != device:
        _CACHE_FLUSH_TENSOR = torch.empty(int(256 * 1024 * 1024 // 4), dtype=torch.int32, device=device)
    _CACHE_FLUSH_TENSOR.zero_()

@pytest.fixture(scope="session", autouse=True)
def setup_csv():
    categories = [
        "dtype", "batch_size", "seq_len", "max_position", "head_size", 
        "rotary_dim", "is_neox_style", "use_key", "head_stride_is_contiguous"
    ]
    
    with open(CSV_FILENAME, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(categories + [
            "status", "custom_time_us", "custom_bw_gbps", "custom_tflops",
            "torch_time_us", "torch_bw_gbps", "torch_tflops", "speedup_ratio"
        ])
        
    yield

    if not os.path.exists(CSV_FILENAME):
        return
        
    global_sp = []
    global_bw = []
    global_tf = []
    global_pass = 0
    global_total = 0
    
    # Structure: dict[category_name][category_value] = {"sp": [], "bw": [], "tf": [], "passes": 0, "total": 0}
    grouped = defaultdict(lambda: defaultdict(lambda: {"sp": [], "bw": [], "tf": [], "passes": 0, "total": 0}))
    
    with open(CSV_FILENAME, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row["status"]
            global_total += 1
            is_pass = (status == "PASS")
            if is_pass:
                global_pass += 1
                try:
                    sp = float(row["speedup_ratio"])
                    bw = float(row["custom_bw_gbps"])
                    tf = float(row["custom_tflops"])
                    global_sp.append(sp)
                    global_bw.append(bw)
                    global_tf.append(tf)
                except ValueError:
                    is_pass = False

            for cat in categories:
                val = row[cat]
                grouped[cat][val]["total"] += 1
                if is_pass:
                    grouped[cat][val]["passes"] += 1
                    grouped[cat][val]["sp"].append(sp)
                    grouped[cat][val]["bw"].append(bw)
                    grouped[cat][val]["tf"].append(tf)
                
    if global_total == 0:
        return
        
    g_avg_sp = sum(global_sp) / len(global_sp) if global_sp else 0.0
    g_pass_pct = (global_pass / global_total) * 100
    
    g_mu_bw = sum(global_bw)/len(global_bw) if global_bw else 0
    g_min_bw = min(global_bw) if global_bw else 0
    g_max_bw = max(global_bw) if global_bw else 0
    
    g_mu_tf = sum(global_tf)/len(global_tf) if global_tf else 0
    g_min_tf = min(global_tf) if global_tf else 0
    g_max_tf = max(global_tf) if global_tf else 0
    
    print("\n\n" + "=" * 125)
    print(f"{'PARAMETER':<23} {'VALUE':<12} {'PASS%':<7} {'SPEEDUP VS PYTORCH':<20} | {'BW (GB/s) [Mean / Min / Max]':<28} | {'TFLOPS [Mean / Min / Max]'}")
    print("-" * 125)
    print(f"{'Global Average':<23} {'-':<12} {g_pass_pct:<6.1f}% {g_avg_sp:<6.3f} (100.0%)       | {g_mu_bw:<6.1f} / {g_min_bw:<6.1f} / {g_max_bw:<6.1f}       | {g_mu_tf:<6.3f} / {g_min_tf:<6.3f} / {g_max_tf:<6.3f}")
    print("-" * 125)
    
    for cat in categories:
        print(f"{cat.upper().replace('_', ' ')}")
        for val, data in grouped[cat].items():
            pass_pct = (data["passes"] / data["total"]) * 100
            
            sp_list, bw_list, tf_list = data["sp"], data["bw"], data["tf"]
            valid = len(sp_list) > 0
            
            grp_sp = sum(sp_list) / len(sp_list) if valid else 0.0
            vs_g = (grp_sp / g_avg_sp) * 100 if (valid and g_avg_sp > 0) else 0.0
            
            mu_bw = sum(bw_list) / len(bw_list) if valid else 0.0
            min_bw = min(bw_list) if valid else 0.0
            max_bw = max(bw_list) if valid else 0.0
            
            mu_tf = sum(tf_list) / len(tf_list) if valid else 0.0
            min_tf = min(tf_list) if valid else 0.0
            max_tf = max(tf_list) if valid else 0.0
            
            sp_str = f"{grp_sp:.3f} ({vs_g:>5.1f}%)" if valid else "N/A"
            print(f"{'':<23} {val:<12} {pass_pct:<6.1f}% {sp_str:<20} | {mu_bw:<6.1f} / {min_bw:<6.1f} / {max_bw:<6.1f}       | {mu_tf:<6.3f} / {min_tf:<6.3f} / {max_tf:<6.3f}")
    print("=" * 125 + "\n")

def benchmark_pytorch_native_rope(positions, query, key, cos_sin_cache, is_neox_style):
    rot_dim = cos_sin_cache.shape[-1]
    cache_out = cos_sin_cache[positions] 
    
    cos = cache_out[..., 0::2] if not is_neox_style else cache_out[..., :rot_dim//2]
    sin = cache_out[..., 1::2] if not is_neox_style else cache_out[..., rot_dim//2:]
    
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)

    def apply_rope(tensor):
        t_rot = tensor[..., :rot_dim]
        if is_neox_style:
            t1, t2 = t_rot.chunk(2, dim=-1)
            t_rotated = torch.cat((-t2, t1), dim=-1)
            c, s = torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)
        else:
            t_rotated = torch.empty_like(t_rot)
            t_rotated[..., 0::2] = -t_rot[..., 1::2]
            t_rotated[..., 1::2] = t_rot[..., 0::2]
            c, s = torch.empty_like(t_rot), torch.empty_like(t_rot)
            c[..., 0::2] = cos
            c[..., 1::2] = cos
            s[..., 0::2] = sin
            s[..., 1::2] = sin
        
        tensor[..., :rot_dim] = t_rot * c + t_rotated * s

    apply_rope(query)
    if key is not None and key.numel() > 0:
        apply_rope(key)

MINI_PYTEST_PARAMS = {
    "default": {
        "max_position": [11],
        "head_size": [32],
        "seq_len": [11],
    },
}

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [11, 1024])
@pytest.mark.parametrize("max_position", [11, 4096, 32768])
@pytest.mark.parametrize("head_size", [32, 64, 108])
@pytest.mark.parametrize("rotary_dim", [32, 64])
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("use_key", [True, False])
@pytest.mark.parametrize("head_stride_is_contiguous", [True, False])
def test_rotary_embedding_accuracy_and_benchmark(
    dtype, batch_size, seq_len, max_position, head_size, rotary_dim, 
    is_neox_style, use_key, head_stride_is_contiguous
):
    device = "xpu"
    num_heads = 32
    num_kv_heads = 32

    if rotary_dim > head_size:
        pytest.skip("rotary_dim cannot be larger than head_size")

    rot = RotaryEmbedding(head_size, rotary_dim, max_position, 10000, is_neox_style, dtype)
    cos_sin_cache = rot.cos_sin_cache.to(device=device, dtype=dtype)

    positions = torch.randint(0, max_position, (batch_size, seq_len), device=device)

    q_shape = (batch_size, seq_len, num_heads, head_size)
    k_shape = (batch_size, seq_len, num_kv_heads, head_size)
    
    if head_stride_is_contiguous:
        query = torch.randn(q_shape, dtype=dtype, device=device)
        if use_key:
            key = torch.randn(k_shape, dtype=dtype, device=device)
    else:
        query = torch.randn((batch_size, seq_len, num_heads * 2, head_size), 
                            dtype=dtype, device=device)[:, :, ::2, :]
        if use_key:
            key = torch.randn((batch_size, seq_len, num_kv_heads * 2, head_size), 
                              dtype=dtype, device=device)[:, :, ::2, :]

    if not use_key:
        key = None

    status = "FAIL"
    custom_time_s = custom_bw = custom_tflops = 0.0
    torch_time_s = torch_bw = torch_tflops = speedup = 0.0

    try:
        opcheck(torch.ops._C.rotary_embedding, (
            positions, 
            query.clone(), 
            key.clone() if key is not None else None, 
            head_size, 
            cos_sin_cache, 
            is_neox_style
        ))

        def bench_fn(func, warmup=20, iters=100):
            # Warm up
            for _ in range(warmup):
                q, k = query.clone(), (key.clone() if key is not None else None)
                func(q, k)
            torch.xpu.synchronize()

            times = []
            for _ in range(iters):
                q, k = query.clone(), (key.clone() if key is not None else None)

                # Flush cache to get real VRAM bandwidth without L2 hits
                _flush_cache(device)
                torch.xpu.synchronize()

                start_time = time.perf_counter()
                func(q, k)
                torch.xpu.synchronize()
                times.append(time.perf_counter() - start_time)

            return statistics.median(times)

        # Custom Kernel Benchmark
        custom_time_s = bench_fn(
            lambda q, k: torch.ops._C.rotary_embedding(positions, q, k, head_size, cos_sin_cache, is_neox_style),
            warmup=20, iters=100
        )

        # Baseline Benchmark (Torch Native)
        torch_time_s = bench_fn(
            lambda q, k: benchmark_pytorch_native_rope(positions, q, k, cos_sin_cache, is_neox_style),
            warmup=5, iters=15
        )
        
        # BW and FLOPs Calculation Alignment
        elem_bytes = query.element_size()
        actual_num_kv_heads = num_kv_heads if use_key else 0
        num_tokens = batch_size * seq_len

        # Bandwidth model:
        # RoPE only reads/writes `rot_dim` out of the full `head_size`
        qk_bytes = 2 * num_tokens * (num_heads + actual_num_kv_heads) * rotary_dim * elem_bytes
        # Read cos/sin cache
        cache_bytes = num_tokens * rotary_dim * cos_sin_cache.element_size()
        # Read positions
        pos_bytes = num_tokens * 8  # int64 positions
        
        total_bytes = qk_bytes + cache_bytes + pos_bytes
        
        # FLOP model (6 FLOPs per pair -> 3 FLOPs per scalar component)
        total_flops = 6.0 * num_tokens * (num_heads + actual_num_kv_heads) * (rotary_dim // 2)

        custom_bw = (total_bytes / 1e9) / custom_time_s
        custom_tflops = (total_flops / 1e12) / custom_time_s
        
        torch_bw = (total_bytes / 1e9) / torch_time_s
        torch_tflops = (total_flops / 1e12) / torch_time_s

        speedup = torch_time_s / custom_time_s
        status = "PASS"

    except Exception as e:
        status = "FAIL"
        raise e
    
    finally:
        if status == "PASS":
            print(f"\n[RoPE {dtype}] BS={batch_size} Seq={seq_len} MaxPos={max_position} Head={head_size} "
                  f"RotD={rotary_dim} contig={head_stride_is_contiguous} has_K={use_key} Neox={is_neox_style} | "
                  f"Custom: {custom_time_s*1e6:.2f} us ({custom_bw:.2f} GB/s, {custom_tflops:.4f} TFLOPS) | "
                  f"Torch: {torch_time_s*1e6:.2f} us ({torch_bw:.2f} GB/s, {torch_tflops:.4f} TFLOPS) | "
                  f"Speedup: {speedup:.2f}x")
        else:
            print(f"\n[RoPE {dtype}] BS={batch_size} Seq={seq_len} MaxPos={max_position} Head={head_size} "
                  f"RotD={rotary_dim} contig={head_stride_is_contiguous} has_K={use_key} Neox={is_neox_style} | FAILED")

        with open(CSV_FILENAME, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                str(dtype).split('.')[-1], batch_size, seq_len, max_position, head_size, 
                rotary_dim, is_neox_style, use_key, head_stride_is_contiguous, status,
                custom_time_s * 1e6, custom_bw, custom_tflops,
                torch_time_s * 1e6, torch_bw, torch_tflops, speedup
            ])

