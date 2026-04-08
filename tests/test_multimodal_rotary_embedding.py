# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Multi-Modal Rotary Embedding (M-RoPE) SYCL kernel.
"""

from typing import Optional
import csv
import time
import os
import pytest
import torch
import statistics
from collections import defaultdict

import tests.register_ops as ops  # noqa: F401 – ensure custom ops are loaded
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

CSV_FILENAME = "test_mrope.csv"
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
        "dtype", "num_tokens", "num_sections", "num_heads", "num_kv_heads",
        "head_size", "rot_dim", "is_neox_style", "use_key",
        "use_triton_as_native", "include_python_framing"
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

    print("\n\n" + "=" * 140)
    print(f"{'PARAMETER':<23} {'VALUE':<12} {'PASS%':<7} {'SPEEDUP VS PYTORCH':<24} | {'BW (GB/s) [Mean / Min / Max]':<32} | {'TFLOPS [Mean / Min / Max]'}")
    print("-" * 140)
    print(f"{'Global Average':<23} {'-':<12} {g_pass_pct:<6.1f}% {g_avg_sp:<8.3f} (100.0%)       | {g_mu_bw:<8.1f} / {g_min_bw:<8.1f} / {g_max_bw:<8.1f}   | {g_mu_tf:<8.3f} / {g_min_tf:<8.3f} / {g_max_tf:<8.3f}")
    print("-" * 140)

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

            sp_str = f"{grp_sp:<8.3f} ({vs_g:>5.1f}%)" if valid else "N/A"
            print(f"{'':<23} {val:<12} {pass_pct:<6.1f}% {sp_str:<24} | {mu_bw:<8.1f} / {min_bw:<8.1f} / {max_bw:<8.1f}   | {mu_tf:<8.3f} / {min_tf:<8.3f} / {max_tf:<8.3f}")
    print("=" * 140 + "\n")

# ─── test & benchmark ────────────────────────────────────────────────────────

MINI_PYTEST_PARAMS = {
    "default": {
        "max_position": [64],
        "head_size": [32],
        "num_tokens": [8],
    }
}

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ["xpu"])
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("use_key", [True, False])
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_size,rot_dim,mrope_section",
    [
        # Real-world M-RoPE: Qwen2-VL 7B (GQA, rot=head, VEC_SIZE=4 valid)
        (28, 4, 128, 128, [16, 24, 24]),

        # Real-world M-RoPE: Qwen2-VL 2B (MHA, rot=head, VEC_SIZE=4 valid)
        (32, 32, 128, 128, [16, 24, 24]),

        # Large Head Size: Gemma-2 / Pixtral style (GQA)
        (16, 8, 256, 256, [32, 48, 48]),

        # Standard RoPE Fallback: Single section, mapped to standard RoPE
        (32, 32, 128, 128, [64]),

        # Partial RoPE: rot_dim < head_size (e.g., rot=64, head=128)
        (16, 4, 128, 64, [8, 12, 12]),

        # Dynamic Dispatch Test: Forces VEC_SIZE=2 (because 10 % 4 != 0)
        (16, 16, 64, 64, [10, 10, 12]),

        # Massive Model: Qwen2-VL 72B (GQA, high head count)
        (64, 8, 128, 128, [16, 24, 24]),
    ],
)
@pytest.mark.parametrize("num_tokens", [
    1, 8,         # Decoding phase (extremely low token count, pure memory bandwidth bound)
    16, 128,      # Short prefill
    512, 1024,    # Medium prefill
    2048, 4096,   # Long prefill
    8192          # Extended context
])
@pytest.mark.usefixtures("default_vllm_config")
def test_multimodal_rotary_embedding(dtype, device, is_neox_style, use_key, num_heads,
                                     num_kv_heads, head_size, rot_dim, mrope_section, num_tokens):
    if not hasattr(ops, "multimodal_rotary_embedding") and not hasattr(torch.ops._C, "multimodal_rotary_embedding"):
        pytest.fail("M-RoPE Custom XPU Kernel not found! Test aborted.")

    max_position = 8192
    base = 10000.0
    num_sections = len(mrope_section)

    mrope_layer = MRotaryEmbedding(
        head_size=head_size,
        rotary_dim=rot_dim,
        max_position_embeddings=max_position,
        base=base,
        is_neox_style=is_neox_style,
        dtype=dtype,
        mrope_section=mrope_section
    )

    # Extract cos/sin cache and send to device
    cos_sin_cache = mrope_layer.cos_sin_cache.to(dtype=dtype, device=device)

    # 3D positions logic (T, H, W)
    positions = torch.stack([
        torch.randint(0, max_position, (num_tokens,), device=device)
        for _ in range(num_sections)
    ])

    # vLLM explicitly uses flattened queries and keys for the standard interface
    # Multiplied by 0.1 to avoid Inf/NaN degradation through repeated benchmark passes
    query = torch.randn(num_tokens, num_heads * head_size, dtype=dtype, device=device) * 0.1
    key = torch.randn(num_tokens, num_kv_heads * head_size, dtype=dtype, device=device) * 0.1 if use_key else None

    # Track metrics mapped by variant
    timings = {}
    status = "FAIL"

    try:
        # 1. Accuracy
        with torch.no_grad():
            dummy_k_ref = key.cpu() if key is not None else torch.zeros(num_tokens, num_kv_heads * head_size, dtype=dtype)
            ref_q, ref_k = mrope_layer.forward_native(positions.cpu(), query.cpu(), dummy_k_ref)

        xpu_q = query.clone()
        xpu_k = key.clone() if key is not None else None

        out_q, out_k = mrope_layer.forward_xpu(positions, xpu_q, xpu_k)

        if dtype == torch.float32:
            atol, rtol = 1e-5, 1e-5
        elif dtype == torch.float16:
            atol, rtol = 1e-2, 1e-2
        else:
            atol, rtol = 2e-2, 2e-2

        torch.testing.assert_close(out_q.cpu().float(), ref_q.float(), atol=atol, rtol=rtol)
        if use_key:
            torch.testing.assert_close(out_k.cpu().float(), ref_k.float(), atol=atol, rtol=rtol)

        # 2. Benchmarking helper
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

        # Custom Kernel (Framed)
        timings["custom_framed"] = bench_fn(
            lambda q, k: mrope_layer.forward_xpu(positions, q, k)
        )

        # Custom Kernel (Pure)
        timings["custom_pure"] = bench_fn(
            lambda q, k: ops.multimodal_rotary_embedding(
                positions,
                q.view(num_tokens, num_heads, head_size),
                k.view(num_tokens, num_kv_heads, head_size) if k is not None else None,
                head_size, cos_sin_cache, is_neox_style, mrope_section
            )
        )

        # Baseline Benchmark (Torch Native)
        dummy_k_ref_bench = torch.zeros(num_tokens, num_kv_heads * head_size, dtype=dtype, device=device)
        timings["torch"] = bench_fn(
            lambda q, k: mrope_layer.forward_native(positions, q, k if use_key else dummy_k_ref_bench)
        )

        # Baseline Benchmark (Triton)
        if num_sections == 3:
            timings["triton"] = bench_fn(
                lambda q, k: mrope_layer.forward_cuda(positions, q, k if use_key else dummy_k_ref_bench)
            )

        status = "PASS"
    except Exception as e:
        status = f"FAIL: {str(e)}"

    finally:
        elem_bytes = query.element_size()
        actual_num_kv_heads = num_kv_heads if use_key else 0

        # Bandwidth model:
        # RoPE only reads/writes `rot_dim` out of the full `head_size`
        qk_bytes = 2 * num_tokens * (num_heads + actual_num_kv_heads) * rot_dim * elem_bytes
        # Read cos/sin cache
        cache_bytes = num_sections * num_tokens * rot_dim * cos_sin_cache.element_size()
        # Read positions
        pos_bytes = num_sections * num_tokens * 8  # int64
        total_bytes = qk_bytes + cache_bytes + pos_bytes

        # FLOP model (6 FLOPs per pair -> 3 FLOPs per scalar component)
        total_flops = 6.0 * num_tokens * (num_heads + actual_num_kv_heads) * (rot_dim // 2)

        # Report combinations
        triton_opts = [True, False] if num_sections == 3 else [False]
        triton_opts = [True] if num_sections == 3 else []
        frame_opts = [True, False]
        frame_opts = [True]

        with open(CSV_FILENAME, "a", newline="") as f:
            writer = csv.writer(f)

            for use_triton_as_native in triton_opts:
                for include_python_framing in frame_opts:
                    c_time = timings.get("custom_framed" if include_python_framing else "custom_pure", 0.0)
                    b_time = timings.get("triton" if use_triton_as_native else "torch", 0.0)

                    if status == "PASS" and c_time > 0 and b_time > 0:
                        c_bw, c_tf = (total_bytes / 1e9) / c_time, (total_flops / 1e12) / c_time
                        b_bw, b_tf = (total_bytes / 1e9) / b_time, (total_flops / 1e12) / b_time
                        speedup = b_time / c_time

                        base_name = "Triton" if use_triton_as_native else "Torch"
                        print(f"\n[M-RoPE {dtype}] Toks={num_tokens} QHeads={num_heads} KVHeads={num_kv_heads} Head={head_size} RotD={rot_dim} Sec={num_sections} has_K={use_key} Neox={is_neox_style} Include Python Frame={include_python_framing} | "
                              f"Custom: {c_time*1e6:.2f} us ({c_bw:.2f} GB/s, {c_tf:.4f} TFLOPS) | "
                              f"{base_name}: {b_time*1e6:.2f} us ({b_bw:.2f} GB/s, {b_tf:.4f} TFLOPS) | "
                              f"Speedup: {speedup:.2f}x")

                        writer.writerow([
                            str(dtype).split('.')[-1], num_tokens, num_sections, num_heads, num_kv_heads,
                            head_size, rot_dim, is_neox_style, use_key, use_triton_as_native,
                            include_python_framing, status,
                            c_time * 1e6, c_bw, c_tf,
                            b_time * 1e6, b_bw, b_tf, speedup
                        ])
                    else:
                        print(f"\n[M-RoPE {dtype}] Toks={num_tokens} QHeads={num_heads} KVHeads={num_kv_heads} Head={head_size} RotD={rot_dim} Sec={num_sections} has_K={use_key} Neox={is_neox_style} Include Python Frame={include_python_framing} | {status}")
                        writer.writerow([
                            str(dtype).split('.')[-1], num_tokens, num_sections, num_heads, num_kv_heads,
                            head_size, rot_dim, is_neox_style, use_key, use_triton_as_native,
                            include_python_framing, "FAIL",
                            0, 0, 0, 0, 0, 0, 0
                        ])
