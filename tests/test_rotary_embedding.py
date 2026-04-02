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

from tests.ops.rotary_embedding_op import RotaryEmbedding
from tests.utils import opcheck
from collections import defaultdict

# Fixed seed for repeatability
torch.manual_seed(42)

CSV_FILENAME = "test_rotary_embedding.csv"

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

        c_warmup, c_iters = 10, 100
        n_warmup, n_iters = 10, 20

        for _ in range(c_warmup):
            torch.ops._C.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox_style)
        torch.xpu.synchronize()

        start_time = time.perf_counter()
        for _ in range(c_iters):
            torch.ops._C.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox_style)
        torch.xpu.synchronize()
        custom_time_s = (time.perf_counter() - start_time) / c_iters

        query_pt = query.clone()
        key_pt = key.clone() if key is not None else None

        for _ in range(n_warmup):
            benchmark_pytorch_native_rope(positions, query_pt, key_pt, cos_sin_cache, is_neox_style)
        torch.xpu.synchronize()

        start_time = time.perf_counter()
        for _ in range(n_iters):
            benchmark_pytorch_native_rope(positions, query_pt, key_pt, cos_sin_cache, is_neox_style)
        torch.xpu.synchronize()
        torch_time_s = (time.perf_counter() - start_time) / n_iters
        
        elements_q = query.numel()
        elements_k = key.numel() if key is not None else 0
        bytes_per_element = query.element_size()
        
        total_bytes = (2 * elements_q + 2 * elements_k) * bytes_per_element
        total_bytes += positions.numel() * positions.element_size()
        
        total_flops = 4 * (batch_size * seq_len * num_heads * rotary_dim)
        if use_key:
            total_flops += 4 * (batch_size * seq_len * num_kv_heads * rotary_dim)

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


# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
#                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg      Self XPU    Self XPU %     XPU total  XPU time avg    # of Calls
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
#                                             gemm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us        5.628s        29.37%        5.628s      88.034us         63932
#                                                aten::mm        27.78%        7.798s        34.27%        9.620s     150.442us        5.624s        29.34%        5.624s      87.943us         63945
#                                            _C::rms_norm         3.40%     953.971ms         6.10%        1.711s      53.154us        2.906s        15.16%        2.906s      90.255us         32193
# vllm::rms_norm_kernel<sycl::_V1::detail::half_impl::...         0.00%       0.000us         0.00%       0.000us       0.000us        2.868s        14.96%        2.868s      90.319us         31752
# vllm::fused_add_rms_norm_kernel<sycl::_V1::detail::h...         0.00%       0.000us         0.00%       0.000us       0.000us        2.794s        14.58%        2.794s      88.017us         31744
#                                  _C::fused_add_rms_norm         3.53%     990.904ms         6.17%        1.732s      54.550us        2.792s        14.57%        2.792s      87.922us         31752
#                                             aten::fill_         2.31%     649.433ms         4.91%        1.377s      41.604us        1.865s         9.73%        1.865s      56.355us         33102
# at::native::xpu::VectorizedElementwiseKernel<4, at::...         0.00%       0.000us         0.00%       0.000us       0.000us        1.861s         9.71%        1.861s      61.105us         30456
# vllm::rotary_embedding_kernel<sycl::_V1::detail::hal...         0.00%       0.000us         0.00%       0.000us       0.000us        1.434s         7.48%        1.434s      90.319us         15876
# vllm::reshape_and_cache_flash_kernel<c10::Half, c10:...         0.00%       0.000us         0.00%       0.000us       0.000us        1.434s         7.48%        1.434s      90.319us         15876
#                                    _C::rotary_embedding         1.26%     353.522ms         2.53%     709.006ms      44.659us        1.433s         7.48%        1.433s      90.245us         15876
#                   _C_cache_ops::reshape_and_cache_flash         1.54%     431.200ms         2.93%     822.016ms      51.777us        1.433s         7.48%        1.433s      90.245us         15876
#                                 _vllm_fa2_C::varlen_fwd         2.28%     640.833ms         8.73%        2.450s     154.303us        1.432s         7.47%        3.290s     207.251us         15876
# vllm::act_and_mul_vec_kernel<sycl::_V1::detail::half...         0.00%       0.000us         0.00%       0.000us       0.000us        1.397s         7.29%        1.397s      88.017us         15872
#                                        _C::silu_and_mul         1.22%     341.111ms         2.60%     729.571ms      45.954us        1.396s         7.28%        1.396s      87.922us         15876
# _ZTSN6compat12experimental6detail13KernelFunctorIXad...         0.00%       0.000us         0.00%       0.000us       0.000us     929.408ms         4.85%     929.408ms      61.105us         15210
# compat::experimental::detail::KernelFunctor<&(void c...         0.00%       0.000us         0.00%       0.000us       0.000us     503.387ms         2.63%     503.387ms     776.832us           648
#                                             aten::copy_         0.88%     247.017ms         1.82%     510.984ms      60.272us      79.927ms         0.42%      79.927ms       9.428us          8478
#                                              aten::sort         0.93%     260.247ms         1.78%     499.394ms       1.132ms      77.949ms         0.41%      81.236ms     184.209us           441
# at::native::xpu::UnrolledElementwiseKernel<at::nativ...         0.00%       0.000us         0.00%       0.000us       0.000us      48.444ms         0.25%      48.444ms      27.525us          1760
# vllm::rms_norm_kernel<sycl::_V1::detail::half_impl::...         0.00%       0.000us         0.00%       0.000us       0.000us      40.174ms         0.21%      40.174ms      91.097us           441
# at::native::xpu::IndexKernel<at::native::xpu::IndexK...         0.00%       0.000us         0.00%       0.000us       0.000us      40.174ms         0.21%      40.174ms      91.097us           441
#                                      aten::index_select         0.05%      13.437ms         0.10%      27.936ms      31.673us      40.119ms         0.21%      40.119ms      45.487us           882
# at::native::xpu::SegmentedRadixSortPairsDownsweepFun...         0.00%       0.000us         0.00%       0.000us       0.000us      25.984ms         0.14%      25.984ms       7.365us          3528
# at::native::xpu::SegmentedRadixSortPairsScanFunctor<...         0.00%       0.000us         0.00%       0.000us       0.000us      25.983ms         0.14%      25.983ms       7.365us          3528
# at::native::xpu::SegmentedRadixSortPairsUpsweepFunct...         0.00%       0.000us         0.00%       0.000us       0.000us      25.981ms         0.14%      25.981ms       7.364us          3528
#                                            aten::cumsum         0.25%      68.974ms         0.42%     118.995ms     269.830us      13.867ms         0.07%      13.867ms      31.444us           441
#                             Memcpy H2D (HOST -> DEVICE)         0.00%       0.000us         0.00%       0.000us       0.000us      12.570ms         0.07%      12.570ms       3.924us          3203
#                                               aten::sub         0.12%      34.593ms         0.23%      64.621ms      48.844us       8.234ms         0.04%       8.234ms       6.224us          1323
#                                              aten::div_         0.07%      19.101ms         0.14%      38.942ms      44.152us       6.536ms         0.03%       6.536ms       7.410us           882
# at::native::xpu::ElementwiseGlobalRangeKernel<at::na...         0.00%       0.000us         0.00%       0.000us       0.000us       6.496ms         0.03%       6.496ms       7.365us           882
#                                          aten::_softmax         0.17%      48.263ms         0.24%      68.568ms      77.742us       6.496ms         0.03%       6.496ms       7.365us           882
# at::native::xpu::impl::SoftmaxForwardKernelFunctor<4...         0.00%       0.000us         0.00%       0.000us       0.000us       6.496ms         0.03%       6.496ms       7.365us           882
#                                      aten::masked_fill_         0.06%      16.400ms         0.13%      35.128ms      39.828us       6.496ms         0.03%       6.496ms       7.365us           882
# at::native::xpu::VectorizedElementwiseKernel<4, at::...         0.00%       0.000us         0.00%       0.000us       0.000us       6.496ms         0.03%       6.496ms       7.365us           882
#                                             aten::index         0.19%      54.526ms         0.40%     111.348ms     124.970us       5.061ms         0.03%       8.343ms       9.363us           891
# at::native::xpu::SegmentScanKernel<at::native::xpu::...         0.00%       0.000us         0.00%       0.000us       0.000us       4.123ms         0.02%       4.123ms       4.772us           864
# at::native::xpu::VectorizedElementwiseKernel<4, at::...         0.00%       0.000us         0.00%       0.000us       0.000us       3.490ms         0.02%       3.490ms       4.039us           864
#                                               aten::add         0.08%      22.033ms         0.15%      41.422ms      31.309us       3.481ms         0.02%       3.481ms       2.631us          1323
# at::native::xpu::VectorizedElementwiseKernel<4, at::...         0.00%       0.000us         0.00%       0.000us       0.000us       3.454ms         0.02%       3.454ms       3.998us           864
# at::native::xpu::UnrolledElementwiseKernel<at::nativ...         0.00%       0.000us         0.00%       0.000us       0.000us       3.289ms         0.02%       3.289ms       7.474us           440
# at::native::xpu::ElementwiseGlobalRangeKernel<at::na...         0.00%       0.000us         0.00%       0.000us       0.000us       3.288ms         0.02%       3.288ms       7.473us           440
# at::native::xpu::VectorizedElementwiseKernel<4, at::...         0.00%       0.000us         0.00%       0.000us       0.000us       3.287ms         0.02%       3.287ms       7.454us           441
#                                          aten::scatter_         0.07%      20.429ms         0.11%      30.898ms      68.662us       3.261ms         0.02%       3.261ms       7.247us           450
# at::native::xpu::ScatterGatherElementwiseKernelFunct...         0.00%       0.000us         0.00%       0.000us       0.000us       3.261ms         0.02%       3.261ms       7.247us           450
#                                      aten::exponential_         0.04%      10.302ms         0.07%      19.915ms      45.158us       3.248ms         0.02%       3.248ms       7.365us           441
# at::native::xpu::DistributionElementwiseKernelFuncto...         0.00%       0.000us         0.00%       0.000us       0.000us       3.248ms         0.02%       3.248ms       7.365us           441
#                                                aten::lt         0.04%      11.791ms         0.08%      21.548ms      24.431us       3.248ms         0.02%       3.248ms       3.682us           882
# at::native::xpu::VectorizedElementwiseKernel<4, at::...         0.00%       0.000us         0.00%       0.000us       0.000us       3.248ms         0.02%       3.248ms       7.365us           441
#                                                aten::le         0.03%       7.451ms         0.06%      16.723ms      37.920us       3.248ms         0.02%       3.248ms       7.365us           441
#                                            aten::gather         0.21%      57.579ms         0.24%      68.480ms     155.284us       3.248ms         0.02%       3.248ms       7.365us           441
# at::native::xpu::ScatterGatherElementwiseKernelFunct...         0.00%       0.000us         0.00%       0.000us       0.000us       3.248ms         0.02%       3.248ms       7.365us           441
# at::native::xpu::VectorizedElementwiseKernel<16, at:...         0.00%       0.000us         0.00%       0.000us       0.000us       3.248ms         0.02%       3.248ms       7.365us           441
# at::native::xpu::VectorizedElementwiseKernel<2, at::...         0.00%       0.000us         0.00%       0.000us       0.000us       3.248ms         0.02%       3.248ms       7.365us           441
#                                            aten::argmax         0.08%      22.072ms         0.12%      32.712ms      74.177us       3.248ms         0.02%       3.248ms       7.365us           441
# at::native::xpu::ReduceKernel<1, at::native::xpu::Re...         0.00%       0.000us         0.00%       0.000us       0.000us       3.248ms         0.02%       3.248ms       7.365us           441
# at::native::xpu::VectorizedElementwiseKernel<4, at::...         0.00%       0.000us         0.00%       0.000us       0.000us       3.248ms         0.02%       3.248ms       7.365us           441
# at::native::xpu::UnrolledElementwiseKernel<at::nativ...         0.00%       0.000us         0.00%       0.000us       0.000us       3.246ms         0.02%       3.246ms       7.361us           441
#                                   urEnqueueKernelLaunch        21.84%        6.129s        21.84%        6.129s      24.511us       2.379ms         0.01%       2.379ms       0.010us        250056
#       at::native::xpu::VectorizedGatherKernel<16, long>         0.00%       0.000us         0.00%       0.000us       0.000us       2.100ms         0.01%       2.100ms       4.884us           430
# at::native::xpu::AccumulateCarrierKernelFunctor<at::...         0.00%       0.000us         0.00%       0.000us       0.000us       2.062ms         0.01%       2.062ms       4.772us           432
# at::native::xpu::SegmentScanKernel<at::native::xpu::...         0.00%       0.000us         0.00%       0.000us       0.000us       2.062ms         0.01%       2.062ms       4.772us           432
# at::native::xpu::ElementwiseGlobalRangeKernel<at::na...         0.00%       0.000us         0.00%       0.000us       0.000us       2.062ms         0.01%       2.062ms       4.772us           432
# at::native::xpu::AccumulateCarrierKernelFunctor<at::...         0.00%       0.000us         0.00%       0.000us       0.000us       2.062ms         0.01%       2.062ms       4.772us           432
# at::native::xpu::IndexKernelFunctor<at::native::xpu:...         0.00%       0.000us         0.00%       0.000us       0.000us       1.761ms         0.01%       1.761ms       3.993us           441
#                            _compute_slot_mapping_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       1.741ms         0.01%       1.741ms       3.947us           441
# at::native::xpu::VectorizedElementwiseKernel<2, at::...         0.00%       0.000us         0.00%       0.000us       0.000us       1.741ms         0.01%       1.741ms       3.947us           441
# at::native::xpu::VectorizedElementwiseKernel<4, at::...         0.00%       0.000us         0.00%       0.000us       0.000us       1.741ms         0.01%       1.741ms       3.947us           441
# at::native::xpu::VectorizedElementwiseKernel<2, at::...         0.00%       0.000us         0.00%       0.000us       0.000us       1.741ms         0.01%       1.741ms       3.947us           441
# at::native::xpu::UnrolledElementwiseKernel<at::nativ...         0.00%       0.000us         0.00%       0.000us       0.000us       1.739ms         0.01%       1.739ms       3.951us           440
# at::native::xpu::SegmentScanKernel<at::native::xpu::...         0.00%       0.000us         0.00%       0.000us       0.000us       1.186ms         0.01%       1.186ms     131.817us             9
# at::native::xpu::UnrolledElementwiseKernel<at::nativ...         0.00%       0.000us         0.00%       0.000us       0.000us       1.186ms         0.01%       1.186ms     131.817us             9
# at::native::xpu::IndexKernelFunctor<at::native::xpu:...         0.00%       0.000us         0.00%       0.000us       0.000us       1.186ms         0.01%       1.186ms     131.817us             9
# at::native::xpu::SegmentScanKernel<at::native::xpu::...         0.00%       0.000us         0.00%       0.000us       0.000us       1.186ms         0.01%       1.186ms     131.817us             9
# at::native::xpu::AccumulateCarrierKernelFunctor<at::...         0.00%       0.000us         0.00%       0.000us       0.000us       1.186ms         0.01%       1.186ms     131.817us             9
#                             Memcpy D2H (DEVICE -> HOST)         0.00%       0.000us         0.00%       0.000us       0.000us     702.856us         0.00%     702.856us       1.594us           441
# at::native::xpu::UnrolledElementwiseKernel<at::nativ...         0.00%       0.000us         0.00%       0.000us       0.000us      27.024us         0.00%      27.024us       1.501us            18
# sycl::_V1::detail::RoundedRangeKernel<sycl::_V1::ite...         0.00%       0.000us         0.00%       0.000us       0.000us      13.434us         0.00%      13.434us       1.493us             9
#                 execute_context_1(1408)_generation_0(0)         2.09%     587.411ms         3.79%        1.065s     118.291ms       0.000us         0.00%        1.511s     167.905ms             9
#                                             aten::slice         0.18%      51.035ms         0.24%      66.478ms       1.313us       0.000us         0.00%       0.000us       0.000us         50616
#                                        aten::as_strided         0.46%     129.451ms         0.46%     129.451ms       0.463us       0.000us         0.00%       0.000us       0.000us        279513
#                                      urEnqueueUSMMemcpy         0.51%     143.913ms         0.51%     143.913ms      39.482us       0.000us         0.00%       0.000us       0.000us          3645
#                                        aten::lift_fresh         0.00%     229.507us         0.00%     229.507us       0.255us       0.000us         0.00%       0.000us       0.000us           900
#                                           aten::flatten         0.00%     901.782us         0.01%       2.382ms       5.401us       0.000us         0.00%       0.000us       0.000us           441
#                                              aten::view         0.27%      74.429ms         0.27%      74.429ms       0.513us       0.000us         0.00%       0.000us       0.000us        145098
#                                            aten::detach         0.01%       1.801ms         0.01%       1.801ms       2.042us       0.000us         0.00%       0.000us       0.000us           882
#                                                aten::to         0.02%       6.294ms         0.76%     212.682ms      39.918us       0.000us         0.00%      55.656ms      10.446us          5328
#                                      aten::resolve_conj         0.00%     242.827us         0.00%     242.827us       0.275us       0.000us         0.00%       0.000us       0.000us           882
#                                       aten::resolve_neg         0.00%     144.873us         0.00%     144.873us       0.164us       0.000us         0.00%       0.000us       0.000us           882
#                                           aten::reshape         0.00%     843.829us         0.01%       2.362ms       2.651us       0.000us         0.00%       0.000us       0.000us           891
#                                          aten::_to_copy         0.04%      10.361ms         0.74%     206.388ms      66.470us       0.000us         0.00%      55.656ms      17.925us          3105
#                                     aten::empty_strided         0.41%     116.453ms         0.41%     116.471ms       2.257us       0.000us         0.00%       0.000us       0.000us         51615
#                                         aten::embedding         0.00%       1.049ms         0.10%      27.595ms      62.574us       0.000us         0.00%      40.119ms      90.973us           441
#                                             aten::empty         0.49%     136.168ms         0.49%     136.180ms       0.947us       0.000us         0.00%       0.000us       0.000us        143802
#                                           aten::resize_         0.07%      19.680ms         0.07%      19.680ms       0.306us       0.000us         0.00%       0.000us       0.000us         64386
#                                        aten::empty_like         0.22%      61.554ms         0.57%     158.791ms       3.273us       0.000us         0.00%       0.000us       0.000us         48510
#                                            aten::linear         0.38%     107.743ms        35.65%       10.007s     156.500us       0.000us         0.00%        5.624s      87.943us         63945
#                                                 aten::t         0.34%      94.703ms         0.73%     206.126ms       3.223us       0.000us         0.00%       0.000us       0.000us         63945
#                                         aten::transpose         0.23%      65.480ms         0.40%     111.423ms       1.742us       0.000us         0.00%       0.000us       0.000us         63945
#                                            aten::matmul         0.26%      73.509ms        34.53%        9.694s     151.591us       0.000us         0.00%        5.624s      87.943us         63945
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
# Traffic request rate: inf
# Burstiness factor: 1.0 (Poisson process)
# Maximum request concurrency: 4
# 100%|██████████| 36/36 [00:32<00:00,  1.10it/s]
# tip: install termplotlib and gnuplot to plot the metrics
# ============ Serving Benchmark Result ============
# Successful requests:                     36
# Failed requests:                         0
# Maximum request concurrency:             4
# Benchmark duration (s):                  32.74
# Total input tokens:                      50400
# Total generated tokens:                  1728
# Request throughput (req/s):              1.10
# Output token throughput (tok/s):         52.78
# Peak output token throughput (tok/s):    68.00
# Peak concurrent requests:                8.00
# Total token throughput (tok/s):          1592.18
# ---------------Time to First Token----------------
# Mean TTFT (ms):                          617.38
# Median TTFT (ms):                        664.26
# P99 TTFT (ms):                           1240.33
# -----Time per Output Token (excl. 1st token)------
# Mean TPOT (ms):                          64.26
# Median TPOT (ms):                        61.95
# P99 TPOT (ms):                           75.64
# ---------------Inter-token Latency----------------
# Mean ITL (ms):                           62.92
# Median ITL (ms):                         61.26
# P99 ITL (ms):                            147.30
# ==================================================
# =============================================================================================================================
# PARAMETER               VALUE        PASS%   SPEEDUP VS PYTORCH   | BW (GB/s) [Mean / Min / Max] | TFLOPS [Mean / Min / Max]
# -----------------------------------------------------------------------------------------------------------------------------
# Global Average          -            100.0 % 15.702 (100.0%)       | 358.7  / 6.4    / 2225.1       | 0.168  / 0.005  / 0.503
# -----------------------------------------------------------------------------------------------------------------------------
# DTYPE
#                         float32      100.0 % 17.119 (109.0%)      | 472.7  / 13.8   / 2225.1       | 0.148  / 0.006  / 0.503
#                         float16      100.0 % 15.078 ( 96.0%)      | 305.3  / 7.0    / 1354.3       | 0.181  / 0.005  / 0.490
#                         bfloat16     100.0 % 14.909 ( 95.0%)      | 298.1  / 6.4    / 1323.3       | 0.176  / 0.006  / 0.474
# BATCH SIZE
#                         1            100.0 % 16.096 (102.5%)      | 409.9  / 6.4    / 2225.1       | 0.184  / 0.005  / 0.503
#                         2            100.0 % 15.308 ( 97.5%)      | 307.6  / 9.9    / 1410.7       | 0.152  / 0.010  / 0.490
# SEQ LEN
#                         11           100.0 % 22.312 (142.1%)      | 48.3   / 6.4    / 195.6        | 0.022  / 0.005  / 0.058
#                         1024         100.0 % 9.092 ( 57.9%)       | 669.1  / 250.6  / 2225.1       | 0.315  / 0.082  / 0.503
# MAX POSITION
#                         11           100.0 % 15.775 (100.5%)      | 363.8  / 6.4    / 2223.5       | 0.170  / 0.006  / 0.503
#                         4096         100.0 % 15.637 ( 99.6%)      | 359.7  / 6.8    / 2221.4       | 0.169  / 0.006  / 0.489
#                         32768        100.0 % 15.693 ( 99.9%)      | 352.7  / 6.9    / 2225.1       | 0.166  / 0.005  / 0.483
# HEAD SIZE
#                         32           100.0 % 15.551 ( 99.0%)      | 244.8  / 6.4    / 877.8        | 0.186  / 0.006  / 0.439
#                         64           100.0 % 16.078 (102.4%)      | 359.6  / 10.2   / 1678.9       | 0.191  / 0.005  / 0.503
#                         108          100.0 % 15.401 ( 98.1%)      | 414.8  / 20.2   / 2225.1       | 0.138  / 0.006  / 0.479
# ROTARY DIM
#                         32           100.0 % 15.419 ( 98.2%)      | 394.2  / 6.4    / 2225.1       | 0.157  / 0.005  / 0.439
#                         64           100.0 % 16.126 (102.7%)      | 305.5  / 11.6   / 1418.0       | 0.185  / 0.012  / 0.503
# IS NEOX STYLE
#                         True         100.0 % 14.170 ( 90.2%)      | 356.1  / 6.4    / 2195.5       | 0.167  / 0.005  / 0.495
#                         False        100.0 % 17.233 (109.8%)      | 361.4  / 6.9    / 2225.1       | 0.170  / 0.006  / 0.503
# USE KEY
#                         True         100.0 % 18.632 (118.7%)      | 329.5  / 11.4   / 1678.9       | 0.163  / 0.010  / 0.503
#                         False        100.0 % 12.771 ( 81.3%)      | 387.9  / 6.4    / 2225.1       | 0.173  / 0.005  / 0.442
# HEAD STRIDE IS CONTIGUOUS
#                         True         100.0 % 16.122 (102.7%)      | 387.4  / 6.8    / 2202.2       | 0.182  / 0.006  / 0.503
#                         False        100.0 % 15.281 ( 97.3%)      | 330.0  / 6.4    / 2225.1       | 0.155  / 0.005  / 0.499
# =============================================================================================================================
