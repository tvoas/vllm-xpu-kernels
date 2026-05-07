#include "moe_ops.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>

#include "../utils.h"
#include "../dispatch_utils.h"
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <c10/core/DeviceGuard.h>

using namespace sycl::ext::intel::esimd;
using bf16 = sycl::ext::oneapi::bfloat16;
using fp16 = sycl::half;

template <typename decl_tag> struct QuantMax;
template <> struct QuantMax<int8_t> { static constexpr float value = 127.0f; };
template <> struct QuantMax<uint8_t> { static constexpr float value = 448.0f; }; // e4m3fn max value

// Vectorized software emulation for float32 -> float8_e4m3fn conversion
template <int N>
inline simd<uint8_t, N> fast_cvt_float_to_e4m3fn(simd<float, N> x) {
    simd<uint32_t, N> bits = x.template bit_cast_view<uint32_t>();
    simd<uint32_t, N> sign = (bits >> 24) & 0x80;
    simd<uint32_t, N> abs_bits = bits & 0x7FFFFFFF;

    // Rounding tie-to-even approximation (add half of the 20-bit shifted fractional part)
    simd<uint32_t, N> rounded = abs_bits + 0x00080000;
    
    simd<int32_t, N> exp = (rounded >> 23) - 127 + 7;
    simd<uint32_t, N> mantissa = (rounded & 0x7FFFFF) >> 20;

    simd<uint8_t, N> res = 0;
    
    auto is_normal = (exp > 0) & (exp < 16);
    auto is_overflow = exp >= 16;
    auto is_underflow = exp <= 0;

    res.merge(sign | (exp << 3) | mantissa, is_normal);
    res.merge(sign | 0x7E, is_overflow); // 0x7E is 448.0 (Maximum e4m3fn value)
    res.merge(sign, is_underflow);       // Fast fallback: flush subnormals to 0

    return res;
}


template <typename T_in, typename T_out>
void moe_swiglu_dynamic_quant_impl(
    torch::Tensor& scatter_tokens,
    torch::Tensor& smooth_scale,
    torch::Tensor& experts_token_count,
    torch::Tensor& experts_token_start,
    torch::Tensor& quant_tokens,
    torch::Tensor& per_token_scale,
    int64_t total_experts_num,
    int64_t max_token_num) {

    if (max_token_num <= 0 || total_experts_num <= 0) {
        return;
    }

    auto& queue = vllm::xpu::vllmGetQueue();

    auto scatter_tokens_ptr = reinterpret_cast<T_in*>(scatter_tokens.data_ptr());
    auto smooth_scale_ptr = smooth_scale.data_ptr<float>();
    auto experts_token_count_ptr = experts_token_count.data_ptr<int32_t>();
    auto experts_token_start_ptr = experts_token_start.data_ptr<int32_t>();
    auto quant_tokens_ptr = reinterpret_cast<T_out*>(quant_tokens.data_ptr());
    auto per_token_scale_ptr = per_token_scale.data_ptr<float>();

    int hidden_size = scatter_tokens.size(1) / 2;
    int num_scattered = scatter_tokens.size(0);
    constexpr float quant_max = QuantMax<T_out>::value;

    auto launch_swiglu = [&](auto unroll_tag, auto slm_tag) {
        constexpr int UNROLL = decltype(unroll_tag)::value;
        constexpr uint32_t SLM_BYTES = decltype(slm_tag)::value;
        constexpr int CHUNK = 64;
        constexpr int BS = CHUNK * UNROLL;

        int num_blocks = hidden_size / BS;
        int wg_size = std::min(num_blocks, 64);

        sycl::range<2> GlobalRange(num_scattered, wg_size);
        sycl::range<2> LocalRange(1, wg_size);

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<2>(GlobalRange, LocalRange), [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL [[intel::kernel_args_restrict]] {
                slm_init(SLM_BYTES); // Constexpr literal statically passed to compiler

                const int loc_id = item.get_local_id(1);
                const int flat_idx = item.get_group(0);

                int left = 0;
                int right = total_experts_num - 1;
                int expert_idx = 0;
                
                while (left <= right) {
                    int mid = left + (right - left) / 2;
                    int start = experts_token_start_ptr[mid];
                    int count = experts_token_count_ptr[mid];
                    
                    if (flat_idx >= start && flat_idx < start + count) {
                        expert_idx = mid;
                        break;
                    } else if (flat_idx < start) {
                        right = mid - 1;
                    } else {
                        left = mid + 1;
                    }
                }

                T_in* scatter_token_base = scatter_tokens_ptr + flat_idx * 2 * hidden_size;
                T_out* output_base = quant_tokens_ptr + flat_idx * hidden_size;
                uint32_t reduction_base = hidden_size * sizeof(float); // Safely offset reduction array

                simd<float, CHUNK> thread_max_vec = 0.0f;

                // Pass 1: Read, compute, track extrema, and CACHE to SLM 
                for (int bid = loc_id; bid < num_blocks; bid += wg_size) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        simd<T_in, CHUNK> x1 = block_load<T_in, CHUNK>(
                            scatter_token_base + bid * BS + u * CHUNK);
                        simd<T_in, CHUNK> x2 = block_load<T_in, CHUNK>(
                            scatter_token_base + hidden_size + bid * BS + u * CHUNK);
                        simd<float, CHUNK> scale = block_load<float, CHUNK>(
                            smooth_scale_ptr + expert_idx * hidden_size + bid * BS + u * CHUNK);

                        simd<float, CHUNK> sigmoid = sycl::ext::intel::esimd::inv(1.0f + sycl::ext::intel::esimd::exp(-simd<float, CHUNK>(x1)));
                        simd<float, CHUNK> scaled_swiglu_tokens = (simd<float, CHUNK>(x1) * sigmoid) * simd<float, CHUNK>(x2) * scale;
                        
                        thread_max_vec = sycl::ext::intel::esimd::max(thread_max_vec, sycl::ext::intel::esimd::abs(scaled_swiglu_tokens));

                        // Write to SLM (Fallback 16-stride loop guarantees successful JIT on all formats)
                        uint32_t base_offset = (bid * BS + u * CHUNK) * 4;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            slm_block_store<float, 16>(base_offset + i * 64, scaled_swiglu_tokens.template select<16, 1>(i * 16));
                        }
                    }
                }

                float thread_max = hmax<float, float, CHUNK>(thread_max_vec);

                slm_block_store<float, 4>(reduction_base + loc_id * 16, simd<float, 4>(thread_max));
                barrier();

                float this_token_scale = 1.0f;

                if (loc_id == 0) {
                    float max_value_final = 0.0f;
                    for (int i = 0; i < wg_size; i++) {
                        simd<float, 4> val = slm_block_load<float, 4>(reduction_base + i * 16);
                        if (val[0] > max_value_final) max_value_final = val[0];
                    }

                    float raw_token_scale = max_value_final / quant_max;
                    this_token_scale = raw_token_scale == 0.0f ? 1.0f : raw_token_scale;
                    
                    slm_block_store<float, 4>(reduction_base + 1024, simd<float, 4>(this_token_scale));
                }
                barrier();

                this_token_scale = slm_block_load<float, 4>(reduction_base + 1024)[0];
                float recip_scale = 1.0f / this_token_scale;

                // Pass 2: ZERO Global Reads, ZERO Arithmetic functions! Fast-fetch SLM output.
                for (int bid = loc_id; bid < num_blocks; bid += wg_size) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        uint32_t base_offset = (bid * BS + u * CHUNK) * 4;
                        simd<float, CHUNK> cached_tokens;
                        
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            cached_tokens.template select<16, 1>(i * 16) = slm_block_load<float, 16>(base_offset + i * 64);
                        }

                        simd<T_out, CHUNK> quantized_out;
                        if constexpr (std::is_same_v<T_out, int8_t>) {
                            quantized_out = rnde<float>(cached_tokens * recip_scale);
                        } else {
                            quantized_out = fast_cvt_float_to_e4m3fn<CHUNK>(cached_tokens * recip_scale);
                        }

                        block_store<T_out, CHUNK>(output_base + bid * BS + u * CHUNK, quantized_out);
                    }
                }

                if (loc_id == 0) {
                    block_store<float, 1>(per_token_scale_ptr + flat_idx, this_token_scale);
                }
            });
        });
    };

    auto dispatch_slm = [&](auto unroll_tag) {
        if (hidden_size <=   128) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,   2560>{}); //   128 * 4 + 2048
        if (hidden_size <=   256) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,   3072>{}); //   256 * 4 + 2048
        if (hidden_size <=   512) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,   4096>{}); //   512 * 4 + 2048
        if (hidden_size <=  1024) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,   6144>{}); //  1024 * 4 + 2048
        if (hidden_size <=  2048) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  10240>{}); //  2048 * 4 + 2048
        if (hidden_size <=  3072) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  14336>{}); //  3072 * 4 + 2048
        if (hidden_size <=  4096) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  18432>{}); //  4096 * 4 + 2048
        if (hidden_size <=  5120) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  22528>{}); //  5120 * 4 + 2048
        if (hidden_size <=  6144) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  26624>{}); //  6144 * 4 + 2048
        if (hidden_size <=  7168) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  30720>{}); //  7168 * 4 + 2048
        if (hidden_size <=  8192) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  34816>{}); //  8192 * 4 + 2048
        if (hidden_size <= 10240) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  43008>{}); // 10240 * 4 + 2048
        if (hidden_size <= 12288) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  51200>{}); // 12288 * 4 + 2048
        if (hidden_size <= 14336) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  59392>{}); // 14336 * 4 + 2048
        if (hidden_size <= 16384) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  67584>{}); // 16384 * 4 + 2048
        if (hidden_size <= 20480) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t,  83968>{}); // 20480 * 4 + 2048
        if (hidden_size <= 24576) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t, 100352>{}); // 24576 * 4 + 2048
        if (hidden_size <= 28672) return launch_swiglu(unroll_tag, std::integral_constant<uint32_t, 116736>{}); // 28672 * 4 + 2048
                                  return launch_swiglu(unroll_tag, std::integral_constant<uint32_t, 131072>{}); // 32256 * 4 + 2048
    };

    int total_swiglu_items = num_scattered;
    
    // The "inflection" threshold: Minimum total threads required to properly saturate the GPU.
    //int target_total_threads = 1024;
    //int target_total_threads = 2048;
    //int target_total_threads = 3072;
    //int target_total_threads = 4096;
    //int target_total_threads = 6144;
    //int target_total_threads = 8192;
    //int target_total_threads = 10240;
    //int target_total_threads = 12288;
    //int target_total_threads = 16384;
    int target_total_threads = 24576;
    //int target_total_threads = 32768;

    auto is_valid_unroll = [&](int unroll) {
        int bs = unroll * 64;
        int max_wg_size = std::min(hidden_size / bs, 64);
        return (hidden_size % bs == 0) && ((total_swiglu_items * max_wg_size) >= target_total_threads);
    };

    if (is_valid_unroll(64)) return dispatch_slm(std::integral_constant<int, 64>{});
    if (is_valid_unroll(32)) return dispatch_slm(std::integral_constant<int, 32>{});
    if (is_valid_unroll(16)) return dispatch_slm(std::integral_constant<int, 16>{});
    if (is_valid_unroll(8))  return dispatch_slm(std::integral_constant<int, 8>{});
    if (is_valid_unroll(4))  return dispatch_slm(std::integral_constant<int, 4>{});
    if (is_valid_unroll(2))  return dispatch_slm(std::integral_constant<int, 2>{});
                             return dispatch_slm(std::integral_constant<int, 1>{});
}

template <typename T_in, typename T_out>
void moe_scatter_dynamic_quant_impl(
    torch::Tensor& selected_experts,
    torch::Tensor& moe_weights,
    torch::Tensor& token_to_scatter_offset,
    torch::Tensor& experts_token_count,
    torch::Tensor& experts_token_start,
    torch::Tensor& hidden_states,
    torch::Tensor& experts_smooth_scale,
    torch::Tensor& scatter_tokens,
    torch::Tensor& scatter_per_token_scale,
    torch::Tensor& scatter_tokens_offset,
    int64_t shared_experts_num) {

    int n_tokens = selected_experts.size(0);
    if (n_tokens <= 0) {
        return;
    }

    // Python frontend concatenates shared experts into selected_experts, 
    // meaning the XPU kernel processes them uniformly via the topk dimension.
    (void)shared_experts_num;

    auto& queue = vllm::xpu::vllmGetQueue();

    int topk = selected_experts.size(1);
    int n_expert = experts_token_count.size(0);
    int hd_size = hidden_states.size(1);
    int n_expert_total = n_expert;
    
    constexpr float quant_max = QuantMax<T_out>::value;

    auto selected_experts_ptr = selected_experts.data_ptr<int32_t>();
    auto ext_tokens_cnt_ptr = experts_token_count.data_ptr<int32_t>();
    auto ext_tokens_start_ptr = experts_token_start.data_ptr<int32_t>();
    auto token_to_scatter_offset_ptr = token_to_scatter_offset.data_ptr<int32_t>();

    // UNIFIED ROUTING PASS: Replaces Pass 0, Pass 1, and Pass 2, eliminating two 
    // kernel launch overheads (~15us+ saved) and all global atomics.
    auto routing_event = queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int32_t, 1> local_expert_counts(n_expert_total, cgh);
        
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)), 
        [=](sycl::nd_item<1> item) [[intel::kernel_args_restrict]] {
            int lid = item.get_local_id(0);
            
            // 1. Zero out SLM histogram
            for (int i = lid; i < n_expert_total; i += 256) {
                local_expert_counts[i] = 0;
            }
            item.barrier(sycl::access::fence_space::local_space);

            // 2. Count tokens using rapid SLM atomics
            int total_items = n_tokens * topk;
            for (int i = lid; i < total_items; i += 256) {
                int expert_id = selected_experts_ptr[i];
                if (expert_id >= 0 && expert_id < n_expert_total) {
                    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>
                        atomic_cnt(local_expert_counts[expert_id]);
                    // Offset uniquely maps to bin density
                    token_to_scatter_offset_ptr[i] = atomic_cnt.fetch_add(1);
                } else {
                    token_to_scatter_offset_ptr[i] = 0;
                }
            }
            item.barrier(sycl::access::fence_space::local_space);

            // 3. Prefix sum & Export to Global Memory
            // (A sequential loop inside a single thread is dramatically faster 
            // than launching a separate kernel for small expert counts)
            if (lid == 0) {
                int32_t sum = 0;
                for (int i = 0; i < n_expert_total; ++i) {
                    int count = local_expert_counts[i];
                    ext_tokens_cnt_ptr[i] = count;
                    ext_tokens_start_ptr[i] = sum;
                    sum += count;
                }
            }
        });
    });

    auto hidden_states_ptr = reinterpret_cast<T_in*>(hidden_states.data_ptr());
    auto smooth_scale_ptr = experts_smooth_scale.data_ptr<float>();
    auto moe_weights_ptr = moe_weights.data_ptr<float>();
    auto scatter_tokens_ptr = reinterpret_cast<T_out*>(scatter_tokens.data_ptr());
    auto scatter_per_token_scale_ptr = scatter_per_token_scale.data_ptr<float>();
    auto scatter_tokens_offset_ptr = scatter_tokens_offset.data_ptr<int32_t>();

    // Pass 3: Gather, quantize, and scatter
    auto launch_scatter = [&](auto unroll_tag) {
        constexpr int UNROLL = decltype(unroll_tag)::value;
        constexpr int CHUNK = 64;
        constexpr int BS = CHUNK * UNROLL;

        int num_blocks = hd_size / BS;
        int wg_size = std::min(num_blocks, 64);

        queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(routing_event);
            cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_tokens * topk, wg_size), sycl::range<2>(1, wg_size)),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL [[intel::kernel_args_restrict]] {
                
                const int token_k_idx = item.get_group(0);
                const int expert_id = selected_experts_ptr[token_k_idx];
                
                if (expert_id < 0 || expert_id >= n_expert_total) {
                    return;
                }

                // Increase SLM to handle OWORD block alignment: 1056B
                slm_init(1056);

                const int loc_id = item.get_local_id(1);
                
                const int token_idx = token_k_idx / topk;

                const int offset = token_to_scatter_offset_ptr[token_k_idx];
                const int expert_start = ext_tokens_start_ptr[expert_id];

                const int target_idx = expert_start + offset;
                const float weight = moe_weights_ptr[token_k_idx];

                simd<float, CHUNK> thread_max_vec = 0.0f;

                // Pass 1 (Removed Cache Hints)
                for (int hd_bid = loc_id; hd_bid < num_blocks; hd_bid += wg_size) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        simd<T_in, CHUNK> hidden = block_load<T_in, CHUNK>(
                            hidden_states_ptr + token_idx * hd_size + hd_bid * BS + u * CHUNK);
                        simd<float, CHUNK> scale = block_load<float, CHUNK>(
                            smooth_scale_ptr + expert_id * hd_size + hd_bid * BS + u * CHUNK);

                        simd<float, CHUNK> smoothed = simd<float, CHUNK>(hidden) * scale * weight;
                        simd<float, CHUNK> smoothed_abs = sycl::ext::intel::esimd::abs(smoothed);

                        thread_max_vec = sycl::ext::intel::esimd::max(thread_max_vec, smoothed_abs);
                    }
                }

                float thread_max = hmax<float, float, CHUNK>(thread_max_vec);

                // Use SLM Block Stores array spacing for max bandwidth (16-byte aligned natively)
                slm_block_store<float, 4>(loc_id * 16, simd<float, 4>(thread_max));
                barrier();

                float this_token_scale = 1.0f;
                
                if (loc_id == 0) {
                    float max_value_final = 0.0f;
                    
                    for (int i = 0; i < wg_size; i++) {
                        simd<float, 4> val = slm_block_load<float, 4>(i * 16);
                        if (val[0] > max_value_final) max_value_final = val[0];
                    }

                    float raw_token_scale = max_value_final / quant_max;
                    this_token_scale = raw_token_scale == 0.0f ? 1.0f : raw_token_scale;
                    
                    slm_block_store<float, 4>(1024, simd<float, 4>(this_token_scale));
                }
                barrier();

                this_token_scale = slm_block_load<float, 4>(1024)[0];
                float recip_scale = 1.0f / this_token_scale;

                // Pass 2 (Removed Cache Hints)
                for (int hd_bid = loc_id; hd_bid < num_blocks; hd_bid += wg_size) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        simd<T_in, CHUNK> hidden = block_load<T_in, CHUNK>(
                            hidden_states_ptr + token_idx * hd_size + hd_bid * BS + u * CHUNK);
                        simd<float, CHUNK> scale = block_load<float, CHUNK>(
                            smooth_scale_ptr + expert_id * hd_size + hd_bid * BS + u * CHUNK);

                        simd<float, CHUNK> smoothed = simd<float, CHUNK>(hidden) * scale * weight;

                        simd<T_out, CHUNK> quantized;
                        if constexpr (std::is_same_v<T_out, int8_t>) {
                            quantized = rnde<float>(smoothed * recip_scale);
                        } else {
                            quantized = fast_cvt_float_to_e4m3fn<CHUNK>(smoothed * recip_scale);
                        }

                        block_store<T_out, CHUNK>(
                            scatter_tokens_ptr + target_idx * hd_size + hd_bid * BS + u * CHUNK, quantized);
                    }
                }

                if (loc_id == 0) {
                    block_store<float, 1>(scatter_per_token_scale_ptr + target_idx, this_token_scale);
                    block_store<int32_t, 1>(scatter_tokens_offset_ptr + target_idx, token_idx);
                }
            });
        });
    };

    int total_scatter_items = n_tokens * topk;
    
    // The "inflection" threshold: Minimum total threads required to properly saturate the GPU.
    //int target_total_threads = 1024;
    //int target_total_threads = 2048;
    //int target_total_threads = 3072;
    //int target_total_threads = 4096;
    //int target_total_threads = 6144;
    //int target_total_threads = 8192;
    //int target_total_threads = 10240;
    //int target_total_threads = 12288;
    //int target_total_threads = 16384;
    int target_total_threads = 24576;
    //int target_total_threads = 32768;

    auto is_valid_unroll = [&](int unroll) {
        int bs = unroll * 64;
        int max_wg_size = std::min(hd_size / bs, 64);
        return (hd_size % bs == 0) && ((total_scatter_items * max_wg_size) >= target_total_threads);
    };

    if (is_valid_unroll(64)) return launch_scatter(std::integral_constant<int, 64>{});
    if (is_valid_unroll(32)) return launch_scatter(std::integral_constant<int, 32>{});
    if (is_valid_unroll(16)) return launch_scatter(std::integral_constant<int, 16>{});
    if (is_valid_unroll(8))  return launch_scatter(std::integral_constant<int, 8>{});
    if (is_valid_unroll(4))  return launch_scatter(std::integral_constant<int, 4>{});
    if (is_valid_unroll(2))  return launch_scatter(std::integral_constant<int, 2>{});
                             return launch_scatter(std::integral_constant<int, 1>{});
}

// Outer dispatch macros to select implementation
#define DISPATCH_MOE_QUANT_IMPL(FUNC_NAME, ...) \
    if (in_dtype == at::ScalarType::BFloat16 && out_dtype == at::ScalarType::Char) { \
        FUNC_NAME<bf16, int8_t>(__VA_ARGS__); \
    } else if (in_dtype == at::ScalarType::Half && out_dtype == at::ScalarType::Char) { \
        FUNC_NAME<fp16, int8_t>(__VA_ARGS__); \
    } else if (in_dtype == at::ScalarType::BFloat16 && out_dtype == at::ScalarType::Float8_e4m3fn) { \
        FUNC_NAME<bf16, uint8_t>(__VA_ARGS__); \
    } else if (in_dtype == at::ScalarType::Half && out_dtype == at::ScalarType::Float8_e4m3fn) { \
        FUNC_NAME<fp16, uint8_t>(__VA_ARGS__); \
    } else { \
        TORCH_CHECK(false, "Unsupported in_dtype/out_dtype combination for dynamic quant."); \
    }

void moe_swiglu_dynamic_quant(
    torch::Tensor& scatter_tokens,
    torch::Tensor& smooth_scale,
    torch::Tensor& experts_token_count,
    torch::Tensor& experts_token_start,
    torch::Tensor& quant_tokens,
    torch::Tensor& per_token_scale,
    int64_t total_experts_num,
    int64_t max_token_num) {

    at::DeviceGuard guard(scatter_tokens.device());

    // Contiguity checks
    TORCH_CHECK(scatter_tokens.is_contiguous(), "scatter_tokens must be contiguous");
    TORCH_CHECK(smooth_scale.is_contiguous(), "smooth_scale must be contiguous");
    TORCH_CHECK(experts_token_count.is_contiguous(), "experts_token_count must be contiguous");
    TORCH_CHECK(experts_token_start.is_contiguous(), "experts_token_start must be contiguous");
    TORCH_CHECK(quant_tokens.is_contiguous(), "quant_tokens must be contiguous");
    TORCH_CHECK(per_token_scale.is_contiguous(), "per_token_scale must be contiguous");

    // Dtype checks
    TORCH_CHECK(smooth_scale.scalar_type() == at::ScalarType::Float, "smooth_scale must be Float32");
    TORCH_CHECK(per_token_scale.scalar_type() == at::ScalarType::Float, "per_token_scale must be Float32");
    TORCH_CHECK(experts_token_count.scalar_type() == at::ScalarType::Int, "experts_token_count must be Int32");
    TORCH_CHECK(experts_token_start.scalar_type() == at::ScalarType::Int, "experts_token_start must be Int32");

    // Shape checks
    int64_t num_scattered = scatter_tokens.size(0);
    int64_t hidden_size2 = scatter_tokens.size(1);
    TORCH_CHECK(hidden_size2 % 2 == 0, "scatter_tokens hidden dimension must be divisible by 2");
    int64_t hidden_size = hidden_size2 / 2;

    // Block size alignment check for ESIMD vectorized loads
    TORCH_CHECK(hidden_size >= 64 && hidden_size % 64 == 0, 
                "hidden_size must be a positive multiple of 64 for vectorized XPU block loads, got ", hidden_size);

    TORCH_CHECK(quant_tokens.size(0) == num_scattered && quant_tokens.size(1) == hidden_size, 
                "quant_tokens shape mismatch");
    TORCH_CHECK(smooth_scale.size(0) == total_experts_num && smooth_scale.size(1) == hidden_size, 
                "smooth_scale shape mismatch");
    TORCH_CHECK(experts_token_count.size(0) == total_experts_num, "experts_token_count size mismatch");
    TORCH_CHECK(experts_token_start.size(0) == total_experts_num, "experts_token_start size mismatch");
    TORCH_CHECK(per_token_scale.size(0) == num_scattered, "per_token_scale size mismatch");

    auto in_dtype = scatter_tokens.scalar_type();
    auto out_dtype = quant_tokens.scalar_type();

    DISPATCH_MOE_QUANT_IMPL(moe_swiglu_dynamic_quant_impl, 
                            scatter_tokens, smooth_scale, experts_token_count, 
                            experts_token_start, quant_tokens, per_token_scale, 
                            total_experts_num, max_token_num);
}

void moe_scatter_dynamic_quant(
    torch::Tensor& selected_experts,
    torch::Tensor& moe_weights,
    torch::Tensor& token_to_scatter_offset,
    torch::Tensor& experts_token_count,
    torch::Tensor& experts_token_start,
    torch::Tensor& hidden_states,
    torch::Tensor& experts_smooth_scale,
    torch::Tensor& scatter_tokens,
    torch::Tensor& scatter_per_token_scale,
    torch::Tensor& scatter_tokens_offset,
    int64_t shared_experts_num) {

    at::DeviceGuard guard(hidden_states.device());

    // Contiguity checks
    TORCH_CHECK(selected_experts.is_contiguous(), "selected_experts must be contiguous");
    TORCH_CHECK(moe_weights.is_contiguous(), "moe_weights must be contiguous");
    TORCH_CHECK(token_to_scatter_offset.is_contiguous(), "token_to_scatter_offset must be contiguous");
    TORCH_CHECK(experts_token_count.is_contiguous(), "experts_token_count must be contiguous");
    TORCH_CHECK(experts_token_start.is_contiguous(), "experts_token_start must be contiguous");
    TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");
    TORCH_CHECK(experts_smooth_scale.is_contiguous(), "experts_smooth_scale must be contiguous");
    TORCH_CHECK(scatter_tokens.is_contiguous(), "scatter_tokens must be contiguous");
    TORCH_CHECK(scatter_per_token_scale.is_contiguous(), "scatter_per_token_scale must be contiguous");
    TORCH_CHECK(scatter_tokens_offset.is_contiguous(), "scatter_tokens_offset must be contiguous");

    // Dtype checks (Int32)
    TORCH_CHECK(selected_experts.scalar_type() == at::ScalarType::Int, "selected_experts must be Int32");
    TORCH_CHECK(token_to_scatter_offset.scalar_type() == at::ScalarType::Int, "token_to_scatter_offset must be Int32");
    TORCH_CHECK(experts_token_count.scalar_type() == at::ScalarType::Int, "experts_token_count must be Int32");
    TORCH_CHECK(experts_token_start.scalar_type() == at::ScalarType::Int, "experts_token_start must be Int32");
    TORCH_CHECK(scatter_tokens_offset.scalar_type() == at::ScalarType::Int, "scatter_tokens_offset must be Int32");

    // Dtype checks (Float32)
    TORCH_CHECK(moe_weights.scalar_type() == at::ScalarType::Float, "moe_weights must be Float32");
    TORCH_CHECK(experts_smooth_scale.scalar_type() == at::ScalarType::Float, "experts_smooth_scale must be Float32");
    TORCH_CHECK(scatter_per_token_scale.scalar_type() == at::ScalarType::Float, "scatter_per_token_scale must be Float32");

    // Logical shape checks
    TORCH_CHECK(experts_token_count.size(0) == experts_token_start.size(0), "Token count and start tensors must match in size");

    // Block size alignment check for ESIMD vectorized loads
    int64_t hd_size = hidden_states.size(1);
    TORCH_CHECK(hd_size >= 64 && hd_size % 64 == 0, 
                "hidden_states inner dimension must be a positive multiple of 64 for XPU block loads, got ", hd_size);

    auto in_dtype = hidden_states.scalar_type();
    auto out_dtype = scatter_tokens.scalar_type();

    DISPATCH_MOE_QUANT_IMPL(moe_scatter_dynamic_quant_impl,
                            selected_experts, moe_weights, token_to_scatter_offset, 
                            experts_token_count, experts_token_start, hidden_states, 
                            experts_smooth_scale, scatter_tokens, scatter_per_token_scale, 
                            scatter_tokens_offset, shared_experts_num);
}
