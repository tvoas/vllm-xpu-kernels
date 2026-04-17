import time
import pytest
import torch
import csv
import os
import statistics
from collections import defaultdict

# Attempt to register ops if using vllm-xpu-kernels
try:
    import tests.register_ops as ops  # noqa: F401
except ImportError:
    pass
try:
    import intel_extension_for_pytorch as ipex  # noqa: F401
except ImportError:
    pass

torch.manual_seed(42)

CSV_FILENAME = "test_moe_swiglu_quant.csv"
KERNEL_SOURCE = os.environ.get("KERNEL_SOURCE", "VLLM_XPU").upper()
_CACHE_FLUSH_TENSOR = None

def _flush_cache(device: str):
    global _CACHE_FLUSH_TENSOR
    if _CACHE_FLUSH_TENSOR is None or _CACHE_FLUSH_TENSOR.device.type != device:
        _CACHE_FLUSH_TENSOR = torch.empty(int(256 * 1024 * 1024 // 4), dtype=torch.int32, device=device)
    _CACHE_FLUSH_TENSOR.zero_()

@pytest.fixture(scope="session", autouse=True)
def setup_csv():
    categories = ["kernel_source", "dtype", "num_scattered", "hidden_size", "num_experts"]
    
    with open(CSV_FILENAME, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(categories + [
            "status", "custom_time_us", "custom_bw_gbps", "custom_tflops",
            "torch_time_us", "torch_bw_gbps", "torch_tflops", "speedup_ratio"
        ])
        
    yield

    if not os.path.exists(CSV_FILENAME):
        return
        
    global_sp, global_bw, global_tf = [], [], []
    global_c_lat, global_t_lat = [], []
    global_pass = global_total = 0
    grouped = defaultdict(lambda: defaultdict(lambda: {"sp": [], "bw": [], "tf": [], "c_lat": [], "t_lat": [], "passes": 0, "total": 0}))
    
    with open(CSV_FILENAME, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row["status"]
            global_total += 1
            is_pass = (status == "PASS")
            if is_pass:
                global_pass += 1
                try:
                    sp, bw, tf = float(row["speedup_ratio"]), float(row["custom_bw_gbps"]), float(row["custom_tflops"])
                    c_lat, t_lat = float(row["custom_time_us"]), float(row["torch_time_us"])
                    global_sp.append(sp)
                    global_bw.append(bw)
                    global_tf.append(tf)
                    global_c_lat.append(c_lat)
                    global_t_lat.append(t_lat)
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
                    grouped[cat][val]["c_lat"].append(c_lat)
                    grouped[cat][val]["t_lat"].append(t_lat)
                
    if global_total == 0: return
        
    g_avg_sp = sum(global_sp) / len(global_sp) if global_sp else 0.0
    g_pass_pct = (global_pass / global_total) * 100
    
    g_mu_bw, g_min_bw, g_max_bw = (sum(global_bw)/len(global_bw) if global_bw else 0), (min(global_bw) if global_bw else 0), (max(global_bw) if global_bw else 0)
    g_mu_tf, g_min_tf, g_max_tf = (sum(global_tf)/len(global_tf) if global_tf else 0), (min(global_tf) if global_tf else 0), (max(global_tf) if global_tf else 0)
    g_mu_c_lat = sum(global_c_lat)/len(global_c_lat) if global_c_lat else 0
    g_mu_t_lat = sum(global_t_lat)/len(global_t_lat) if global_t_lat else 0
    
    print("\n\n" + "=" * 160)
    print(f"{'PARAMETER':<23} {'VALUE':<12} {'PASS%':<7} {'LATENCY (us) [Custom / Torch]':<32} {'SPEEDUP VS PYTORCH':<20} | {'BW (GB/s) [Mean / Min / Max]':<28} | {'TFLOPS [Mean / Min / Max]'}")
    print("-" * 160)
    print(f"{'Global Average':<23} {'-':<12} {g_pass_pct:<6.1f}% {g_mu_c_lat:<12.1f} / {g_mu_t_lat:<17.1f} {g_avg_sp:<6.3f} (100.0%)       | {g_mu_bw:<6.1f} / {g_min_bw:<6.1f} / {g_max_bw:<6.1f}       | {g_mu_tf:<6.3f} / {g_min_tf:<6.3f} / {g_max_tf:<6.3f}")
    print("-" * 160)
    
    for cat in categories:
        print(f"{cat.upper().replace('_', ' ')}")
        for val, data in grouped[cat].items():
            pass_pct = (data["passes"] / data["total"]) * 100
            valid = len(data["sp"]) > 0
            
            grp_sp = sum(data["sp"]) / len(data["sp"]) if valid else 0.0
            grp_c_lat = sum(data["c_lat"]) / len(data["c_lat"]) if valid else 0.0
            grp_t_lat = sum(data["t_lat"]) / len(data["t_lat"]) if valid else 0.0
            
            vs_g = (grp_sp / g_avg_sp) * 100 if (valid and g_avg_sp > 0) else 0.0
            sp_str = f"{grp_sp:.3f} ({vs_g:>5.1f}%)" if valid else "N/A"
            lat_str = f"{grp_c_lat:<12.1f} / {grp_t_lat:<17.1f}" if valid else "N/A"
            
            print(f"{'':<23} {val:<12} {pass_pct:<6.1f}% {lat_str:<32} {sp_str:<20} | "
                  f"{sum(data['bw'])/len(data['bw']) if valid else 0:<6.1f} / {min(data['bw']) if valid else 0:<6.1f} / {max(data['bw']) if valid else 0:<6.1f}       | "
                  f"{sum(data['tf'])/len(data['tf']) if valid else 0:<6.3f} / {min(data['tf']) if valid else 0:<6.3f} / {max(data['tf']) if valid else 0:<6.3f}")
    print("=" * 160 + "\n")


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

            x1_silu = x1 / (1.0 + torch.exp(-x1))
            swiglu = x1_silu * x2
            scaled = swiglu * scale
            
            max_val = scaled.abs().max().item()
            this_token_scale = max_val / 127.0
            scale_divisor = this_token_scale if this_token_scale != 0.0 else 1.0

            quantized = torch.round(scaled / scale_divisor).to(torch.int8)

            quant_tokens[target_idx] = quantized
            per_token_scale[target_idx] = this_token_scale

    return quant_tokens, per_token_scale

@pytest.mark.parametrize("num_scattered", [16, 40, 80, 128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("hidden_size", [64, 128, 256, 512, 1024, 2048, 4096, 7168])
@pytest.mark.parametrize("num_experts", [64, 128, 256])
def test_moe_swiglu_dynamic_quant(num_scattered, hidden_size, num_experts):
    device = "xpu"
    dtype = torch.bfloat16

    if num_scattered % num_experts != 0:
        pytest.skip(f"num_scattered ({num_scattered}) must be a multiple of num_experts ({num_experts}) to avoid IPEX C++ FPE crashes.")
    if KERNEL_SOURCE == "IPEX" and hidden_size < 128:
        pytest.skip("IPEX kernel does not support hidden_size < 128.")

    if KERNEL_SOURCE == "IPEX":
        if not hasattr(torch.ops, "torch_ipex"):
            pytest.skip("IPEX not found.")
        if not hasattr(torch.ops.torch_ipex, "moe_swiglu_dynamic_quant"):
            pytest.skip("IPEX kernel not found.")
        custom_op = torch.ops.torch_ipex.moe_swiglu_dynamic_quant
    else:
        if not hasattr(torch.ops, "_moe_C") or not hasattr(torch.ops._moe_C, "moe_swiglu_dynamic_quant"):
            pytest.skip("vllm-xpu-kernels not found.")
        custom_op = torch.ops._moe_C.moe_swiglu_dynamic_quant

    experts_token_count = torch.zeros(num_experts, dtype=torch.int32, device=device)
    experts_token_start = torch.zeros(num_experts, dtype=torch.int32, device=device)
    
    tokens_per_expert = num_scattered // num_experts
    experts_token_count[:] = tokens_per_expert
    experts_token_count[-1] = num_scattered - (tokens_per_expert * (num_experts - 1))
    
    start = 0
    max_token_num = 0
    for i in range(num_experts):
        experts_token_start[i] = start
        start += experts_token_count[i].item()
        if experts_token_count[i].item() > max_token_num:
            max_token_num = experts_token_count[i].item()

    scatter_tokens = torch.randn((num_scattered, hidden_size * 2), dtype=dtype, device=device)
    smooth_scale = torch.rand((num_experts, hidden_size), dtype=torch.float32, device=device)

    out_quant_tokens = torch.zeros((num_scattered, hidden_size), dtype=torch.int8, device=device)
    out_per_scale = torch.zeros(num_scattered, dtype=torch.float32, device=device)

    status = "FAIL"
    custom_time_s = custom_bw = custom_tflops = 0.0
    torch_time_s = torch_bw = torch_tflops = speedup = 0.0

    try:
        # 1. Run Torch Accuracy Baseline
        ref_quant_tokens, ref_per_scale = baseline_moe_swiglu(
            scatter_tokens, smooth_scale, experts_token_count, experts_token_start
        )

        out_quant_tokens.zero_()
        out_per_scale.zero_()
        custom_op(
            scatter_tokens, smooth_scale, experts_token_count, experts_token_start,
            out_quant_tokens, out_per_scale, num_experts, max_token_num
        )
        torch.xpu.synchronize()

        # Output offsets perfectly match in SwiGLU, so no complex sorting is required here.
        torch.testing.assert_close(out_per_scale, ref_per_scale, atol=1e-4, rtol=1e-4)
        
        diff = (out_quant_tokens.int() - ref_quant_tokens.int()).abs()
        assert diff.max().item() <= 2, f"Quantized values diverge completely! Max diff: {diff.max().item()}"

        # 3. Benchmarks
        def bench_fn(is_custom, warmup=25, iters=1000):
            for _ in range(warmup): 
                if is_custom:
                    custom_op(
                        scatter_tokens, smooth_scale, experts_token_count, experts_token_start,
                        out_quant_tokens, out_per_scale, num_experts, max_token_num
                    )
                else:
                    baseline_moe_swiglu(
                        scatter_tokens, smooth_scale, experts_token_count, experts_token_start
                    )
            torch.xpu.synchronize()

            times = []
            for _ in range(iters):
                _flush_cache(device)
                torch.xpu.synchronize()

                start_time = time.perf_counter()
                if is_custom:
                    custom_op(
                        scatter_tokens, smooth_scale, experts_token_count, experts_token_start,
                        out_quant_tokens, out_per_scale, num_experts, max_token_num
                    )
                else:
                    baseline_moe_swiglu(
                        scatter_tokens, smooth_scale, experts_token_count, experts_token_start
                    )
                torch.xpu.synchronize()
                
                times.append(time.perf_counter() - start_time)
            return statistics.median(times)

        custom_time_s = bench_fn(is_custom=True)
        # PyTorch baseline uses raw python for loop per token, drastically reduce torch bench iters to 5
        torch_time_s = bench_fn(is_custom=False, warmup=2, iters=5)

        # 4. Metrics
        # Bytes = Read Scatter 2*H (bf16), Read Scale H (f32), Write Quant H (int8), Write per_token 1 (f32)
        total_bytes = (num_scattered * hidden_size * 2 * 2) + \
                      (num_scattered * hidden_size * 4) + \
                      (num_scattered * hidden_size * 1) + \
                      (num_scattered * 4)
                      
        # Flops approx: silu + mul + scale + quant
        # conservatively rating at 6 flops per element
        total_flops = num_scattered * hidden_size * 6

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
        print(f"\n[SwiGLU Quant] Src={KERNEL_SOURCE} N={num_scattered} Hid={hidden_size} Exp={num_experts} | "
              f"{status} | "
              f"Custom: {custom_time_s*1e6:.2f} us ({custom_bw:.2f} GB/s) | "
              f"Torch: {torch_time_s*1e6:.2f} us ({torch_bw:.2f} GB/s) | "
              f"Speedup: {speedup:.2f}x")

        with open(CSV_FILENAME, "a", newline="") as f:
            csv.writer(f).writerow([
                KERNEL_SOURCE, "bfloat16", num_scattered, hidden_size, num_experts, status,
                custom_time_s * 1e6, custom_bw, custom_tflops,
                torch_time_s * 1e6, torch_bw, torch_tflops, speedup
            ])