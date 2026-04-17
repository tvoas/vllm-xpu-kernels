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

CSV_FILENAME = "test_moe_scatter_quant.csv"
KERNEL_SOURCE = os.environ.get("KERNEL_SOURCE", "VLLM_XPU").upper()
_CACHE_FLUSH_TENSOR = None

def _flush_cache(device: str):
    global _CACHE_FLUSH_TENSOR
    if _CACHE_FLUSH_TENSOR is None or _CACHE_FLUSH_TENSOR.device.type != device:
        _CACHE_FLUSH_TENSOR = torch.empty(int(256 * 1024 * 1024 // 4), dtype=torch.int32, device=device)
    _CACHE_FLUSH_TENSOR.zero_()

@pytest.fixture(scope="session", autouse=True)
def setup_csv():
    categories = ["kernel_source", "dtype", "num_tokens", "hidden_size", "topk", "num_experts"]
    
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

def baseline_moe_scatter(
    selected_experts, moe_weights, hidden_states, experts_smooth_scale, total_experts
):
    """Pure PyTorch baseline for MoE Scatter + Dynamic Quantization."""
    num_tokens, topk = selected_experts.shape
    hidden_size = hidden_states.shape[1]
    
    # Pre-allocate outputs
    out_tokens_count = torch.zeros(total_experts, dtype=torch.int32, device=hidden_states.device)
    out_token_to_scatter_offset = torch.zeros_like(selected_experts, dtype=torch.int32)
    
    # Pass 1: Count and offsets
    for t in range(num_tokens):
        for k in range(topk):
            exp = selected_experts[t, k].item()
            out_token_to_scatter_offset[t, k] = out_tokens_count[exp].item()
            out_tokens_count[exp] += 1

    # Pass 2: Prefix sum
    out_tokens_start = torch.zeros_like(out_tokens_count)
    start = 0
    for i in range(total_experts):
        out_tokens_start[i] = start
        start += out_tokens_count[i].item()

    # Pass 3: Scatter & Dynamic Quant
    scatter_tokens = torch.zeros((num_tokens * topk, hidden_size), dtype=torch.int8, device=hidden_states.device)
    per_token_scale = torch.zeros(num_tokens * topk, dtype=torch.float32, device=hidden_states.device)
    tokens_offset = torch.zeros(num_tokens * topk, dtype=torch.int32, device=hidden_states.device)

    for t in range(num_tokens):
        for k in range(topk):
            exp = selected_experts[t, k].item()
            offset = out_token_to_scatter_offset[t, k].item()
            target_idx = out_tokens_start[exp].item() + offset
            
            w = moe_weights[t, k].float()
            h = hidden_states[t].float()
            s = experts_smooth_scale[exp].float()
            
            smoothed = h * s * w
            max_val = smoothed.abs().max().item()
            scale = max_val / 127.0
            
            # C++ casts implicitly, truncating to zero. We replicate this.
            quantized = torch.round(smoothed / (scale if scale != 0 else 1.0)).to(torch.int8)
            
            scatter_tokens[target_idx] = quantized
            per_token_scale[target_idx] = scale
            tokens_offset[target_idx] = t

    return out_token_to_scatter_offset, out_tokens_count, out_tokens_start, scatter_tokens, per_token_scale, tokens_offset

@pytest.mark.parametrize("num_tokens", [16, 40, 80, 128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("hidden_size", [64, 128, 256, 512, 1024, 2048, 4096, 7168])
@pytest.mark.parametrize("topk", [2, 5, 8])
@pytest.mark.parametrize("num_experts", [64, 128, 256])
def test_moe_scatter_dynamic_quant(num_tokens, hidden_size, topk, num_experts):
    device = "xpu"
    dtype = torch.bfloat16
    shared_experts_num = 0

    if KERNEL_SOURCE == "IPEX":
        if not hasattr(torch.ops, "torch_ipex"):
            pytest.skip("IPEX not found.")
        if not hasattr(torch.ops.torch_ipex, "moe_scatter_dynamic_quant"):
            pytest.skip("IPEX kernel not found.")
        custom_op = torch.ops.torch_ipex.moe_scatter_dynamic_quant
    else:
        if not hasattr(torch.ops, "_moe_C") or not hasattr(torch.ops._moe_C, "moe_scatter_dynamic_quant"):
            pytest.skip("vllm-xpu-kernels not found.")
        custom_op = torch.ops._moe_C.moe_scatter_dynamic_quant

    selected_experts = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device=device)
    moe_weights = torch.rand((num_tokens, topk), dtype=torch.float32, device=device)
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device=device)
    experts_smooth_scale = torch.rand((num_experts, hidden_size), dtype=torch.float32, device=device)

    out_t_offset = torch.zeros_like(selected_experts)
    out_t_count = torch.zeros(num_experts, dtype=torch.int32, device=device)
    out_t_start = torch.zeros(num_experts, dtype=torch.int32, device=device)
    out_scatter_tokens = torch.zeros((num_tokens * topk, hidden_size), dtype=torch.int8, device=device)
    out_per_scale = torch.zeros(num_tokens * topk, dtype=torch.float32, device=device)
    out_tokens_offset = torch.zeros(num_tokens * topk, dtype=torch.int32, device=device)

    status = "FAIL"
    custom_time_s = custom_bw = custom_tflops = 0.0
    torch_time_s = torch_bw = torch_tflops = speedup = 0.0

    try:
        # 1. Run Torch Accuracy Baseline
        (ref_t_offset, ref_t_count, ref_t_start, ref_scatter_tokens, 
         ref_per_scale, ref_tokens_offset) = baseline_moe_scatter(
             selected_experts, moe_weights, hidden_states, experts_smooth_scale, num_experts
        )

        custom_op(
            selected_experts, moe_weights, out_t_offset, out_t_count, out_t_start,
            hidden_states, experts_smooth_scale, out_scatter_tokens, out_per_scale, 
            out_tokens_offset, shared_experts_num
        )

        torch.testing.assert_close(out_t_count, ref_t_count)
        torch.testing.assert_close(out_t_start, ref_t_start)

        # Build an explicit array representing which Expert ID owns which output row
        expert_ids = torch.zeros(num_tokens * topk, dtype=torch.int64, device=device)
        for i in range(num_experts):
            start = ref_t_start[i].item()
            count = ref_t_count[i].item()
            if count > 0:
                expert_ids[start:start+count] = i

        # Sort strictly by (Expert ID, Original Token Index)
        # Using num_tokens as the multiplier guarantees no overlap since offsets < num_tokens
        sort_key_custom = (expert_ids * num_tokens) + out_tokens_offset
        sort_key_ref = (expert_ids * num_tokens) + ref_tokens_offset

        sort_idx_custom = torch.argsort(sort_key_custom)
        sort_idx_ref = torch.argsort(sort_key_ref)
        if KERNEL_SOURCE != "IPEX":
            torch.testing.assert_close(out_tokens_offset[sort_idx_custom], ref_tokens_offset[sort_idx_ref])
        torch.testing.assert_close(out_per_scale[sort_idx_custom], ref_per_scale[sort_idx_ref], atol=1e-5, rtol=1e-5)
        
        diff = (out_scatter_tokens[sort_idx_custom].int() - ref_scatter_tokens[sort_idx_ref].int()).abs()
        assert diff.max().item() <= 1, "Quantized values diverge completely!"

        # 2. Benchmarks
        def bench_fn(is_custom, warmup=25, iters=1000):
            # Warm up
            for _ in range(warmup): 
                if is_custom:
                    out_t_count.zero_()
                    custom_op(
                        selected_experts, moe_weights, out_t_offset, out_t_count, out_t_start,
                        hidden_states, experts_smooth_scale, out_scatter_tokens, out_per_scale, 
                        out_tokens_offset, shared_experts_num
                    )
                else:
                    baseline_moe_scatter(selected_experts, moe_weights, hidden_states, experts_smooth_scale, num_experts)
            torch.xpu.synchronize()

            times = []
            for _ in range(iters):
                # Flush cache to get real VRAM bandwidth without L2 hits
                _flush_cache(device)
                torch.xpu.synchronize()

                start_time = time.perf_counter()
                if is_custom:
                    out_t_count.zero_()
                    custom_op(
                        selected_experts, moe_weights, out_t_offset, out_t_count, out_t_start,
                        hidden_states, experts_smooth_scale, out_scatter_tokens, out_per_scale, 
                        out_tokens_offset, shared_experts_num
                    )
                else:
                    baseline_moe_scatter(selected_experts, moe_weights, hidden_states, experts_smooth_scale, num_experts)
                torch.xpu.synchronize()
                
                times.append(time.perf_counter() - start_time)
            return statistics.median(times)

        custom_time_s = bench_fn(is_custom=True)
        torch_time_s = bench_fn(is_custom=False, warmup=2, iters=5)

        # 3. Metrics
        # Bytes: hidden_states (bf16), smooth_scale (fp32), moe_weights (fp32), scatter_tokens (int8)
        total_bytes = (num_tokens * hidden_size * 2) + \
                      (num_experts * hidden_size * 4) + \
                      (num_tokens * topk * 4) + \
                      (num_tokens * topk * hidden_size * 1)
        total_flops = num_tokens * topk * hidden_size * 2

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
        print(f"\n[Scatter Quant] Src={KERNEL_SOURCE} Tok={num_tokens} Hid={hidden_size} TopK={topk} Exp={num_experts} | "
              f"{status} | "
              f"Custom: {custom_time_s*1e6:.2f} us ({custom_bw:.2f} GB/s) | "
              f"Torch: {torch_time_s*1e6:.2f} us ({torch_bw:.2f} GB/s) | "
              f"Speedup: {speedup:.2f}x")

        with open(CSV_FILENAME, "a", newline="") as f:
            csv.writer(f).writerow([
                KERNEL_SOURCE, "bfloat16", num_tokens, hidden_size, topk, num_experts, status,
                custom_time_s * 1e6, custom_bw, custom_tflops,
                torch_time_s * 1e6, torch_bw, torch_tflops, speedup
            ])