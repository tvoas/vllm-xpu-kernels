#include "moe_ops.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>

#include "../utils.h"
#include "../dispatch_utils.h"
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <c10/core/DeviceGuard.h>
#include <string>
#include <algorithm>

using namespace sycl::ext::intel::esimd;
using bf16 = sycl::ext::oneapi::bfloat16;
using fp16 = sycl::half;

template <typename decl_tag> struct QuantMax;
template <> struct QuantMax<int8_t> { static constexpr float value = 127.0f; };
template <> struct QuantMax<uint8_t> { static constexpr float value = 448.0f; }; 

template <int N>
inline simd<uint8_t, N> fast_cvt_float_to_e4m3fn(simd<float, N> x) {
    simd<uint32_t, N> bits = x.template bit_cast_view<uint32_t>();
    simd<uint32_t, N> sign = (bits >> 24) & 0x80;
    simd<uint32_t, N> abs_bits = bits & 0x7FFFFFFF;
    simd<uint32_t, N> rounded = abs_bits + 0x00080000;
    simd<int32_t, N> exp = (rounded >> 23) - 127 + 7;
    simd<uint32_t, N> mantissa = (rounded & 0x7FFFFF) >> 20;

    simd<uint8_t, N> res = 0;
    auto is_normal = (exp > 0) & (exp < 16);
    auto is_overflow = exp >= 16;
    auto is_underflow = exp <= 0;

    res.merge(sign | (exp << 3) | mantissa, is_normal);
    res.merge(sign | 0x7E, is_overflow);
    res.merge(sign, is_underflow);
    return res;
}

// ----------------------------------------------------------------------------
// SCATTER IMPLEMENTATIONS
// ----------------------------------------------------------------------------
template <typename T_in, typename T_out>
void moe_scatter_dynamic_quant_impl(
    torch::Tensor& selected_experts, torch::Tensor& moe_weights,
    torch::Tensor& token_to_scatter_offset, torch::Tensor& experts_token_count,
    torch::Tensor& experts_token_start, torch::Tensor& hidden_states,
    torch::Tensor& experts_smooth_scale, torch::Tensor& scatter_tokens,
    torch::Tensor& scatter_per_token_scale, torch::Tensor& scatter_tokens_offset,
    int64_t shared_experts_num, std::string chosen_case) {

    int n_tokens = selected_experts.size(0);
    if (n_tokens <= 0) return;
    (void)shared_experts_num;

    auto& queue = vllm::xpu::vllmGetQueue();
    int topk = selected_experts.size(1);
    int n_expert_total = experts_token_count.size(0);
    int hd_size = hidden_states.size(1);
    constexpr float quant_max = QuantMax<T_out>::value;

    auto selected_experts_ptr = selected_experts.data_ptr<int32_t>();
    auto ext_tokens_cnt_ptr = experts_token_count.data_ptr<int32_t>();
    auto ext_tokens_start_ptr = experts_token_start.data_ptr<int32_t>();
    auto token_to_scatter_offset_ptr = token_to_scatter_offset.data_ptr<int32_t>();
    auto hidden_states_ptr = reinterpret_cast<T_in*>(hidden_states.data_ptr());
    auto smooth_scale_ptr = experts_smooth_scale.data_ptr<float>();
    auto moe_weights_ptr = moe_weights.data_ptr<float>();
    auto scatter_tokens_ptr = reinterpret_cast<T_out*>(scatter_tokens.data_ptr());
    auto scatter_per_token_scale_ptr = scatter_per_token_scale.data_ptr<float>();
    auto scatter_tokens_offset_ptr = scatter_tokens_offset.data_ptr<int32_t>();

    int total_items = n_tokens * topk;

    int sub_group_snap = 32;
    int items_per_thread = 32;
    bool use_v20_routing = false;

    if (chosen_case.find("2032") != std::string::npos) {
        use_v20_routing = true;
        if (chosen_case == "reports_test_203222") { sub_group_snap = 8; items_per_thread = 1; }
        else if (chosen_case == "reports_test_203217") { sub_group_snap = 8; items_per_thread = 32; }
        else if (chosen_case == "reports_test_203215") { sub_group_snap = 16; items_per_thread = 2; }
        else if (chosen_case == "reports_test_203214") { sub_group_snap = 16; items_per_thread = 4; }
        else if (chosen_case == "reports_test_203212") { sub_group_snap = 16; items_per_thread = 8; }
        else if (chosen_case == "reports_test_203207") { sub_group_snap = 32; items_per_thread = 2; }
    }

    int routing_wg = 256;
    if (use_v20_routing) {
        int target_threads = (total_items + items_per_thread - 1) / items_per_thread;
        routing_wg = ((target_threads + sub_group_snap - 1) / sub_group_snap) * sub_group_snap;
        routing_wg = std::min(256, std::max(sub_group_snap, routing_wg));
    }

    auto routing_event = queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int32_t, 1> local_expert_counts(n_expert_total, cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(routing_wg), sycl::range<1>(routing_wg)), 
        [=](sycl::nd_item<1> item) [[intel::kernel_args_restrict]] {
            int lid = item.get_local_id(0);
            for (int i = lid; i < n_expert_total; i += routing_wg) local_expert_counts[i] = 0;
            item.barrier(sycl::access::fence_space::local_space);

            for (int i = lid; i < total_items; i += routing_wg) {
                int expert_id = selected_experts_ptr[i];
                if (expert_id >= 0 && expert_id < n_expert_total) {
                    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>
                        atomic_cnt(local_expert_counts[expert_id]);
                    token_to_scatter_offset_ptr[i] = atomic_cnt.fetch_add(1);
                } else {
                    token_to_scatter_offset_ptr[i] = 0;
                }
            }
            item.barrier(sycl::access::fence_space::local_space);

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

    auto launch_scatter_global = [&](auto unroll_tag) {
        constexpr int UNROLL = decltype(unroll_tag)::value;
        constexpr int CHUNK = 64; constexpr int BS = CHUNK * UNROLL;
        int num_blocks = hd_size / BS; int wg_size = std::min(num_blocks, 64);
        queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(routing_event);
            cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(total_items, wg_size), sycl::range<2>(1, wg_size)),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL [[intel::kernel_args_restrict]] {
                const int token_k_idx = item.get_group(0);
                const int expert_id = selected_experts_ptr[token_k_idx];
                if (expert_id < 0 || expert_id >= n_expert_total) return;
                slm_init(1056);
                const int loc_id = item.get_local_id(1);
                const int token_idx = token_k_idx / topk;
                const int target_idx = ext_tokens_start_ptr[expert_id] + token_to_scatter_offset_ptr[token_k_idx];
                const float weight = moe_weights_ptr[token_k_idx];

                simd<float, CHUNK> thread_max_vec = 0.0f;
                for (int hd_bid = loc_id; hd_bid < num_blocks; hd_bid += wg_size) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        simd<T_in, CHUNK> hidden = block_load<T_in, CHUNK>(hidden_states_ptr + token_idx * hd_size + hd_bid * BS + u * CHUNK);
                        simd<float, CHUNK> scale = block_load<float, CHUNK>(smooth_scale_ptr + expert_id * hd_size + hd_bid * BS + u * CHUNK);
                        simd<float, CHUNK> smoothed = simd<float, CHUNK>(hidden) * scale * weight;
                        simd<float, CHUNK> smoothed_abs = sycl::ext::intel::esimd::abs(smoothed);
                        thread_max_vec = sycl::ext::intel::esimd::max(thread_max_vec, smoothed_abs);
                    }
                }
                slm_block_store<float, 4>(loc_id * 16, simd<float, 4>(hmax<float, float, CHUNK>(thread_max_vec)));
                barrier();

                float this_token_scale = 1.0f;
                if (loc_id == 0) {
                    float max_value_final = 0.0f;
                    for (int i = 0; i < wg_size; i++) {
                        simd<float, 4> val = slm_block_load<float, 4>(i * 16);
                        if (val[0] > max_value_final) max_value_final = val[0];
                    }
                    float raw = max_value_final / quant_max;
                    this_token_scale = raw == 0.0f ? 1.0f : raw;
                    slm_block_store<float, 4>(1024, simd<float, 4>(this_token_scale));
                }
                barrier();

                this_token_scale = slm_block_load<float, 4>(1024)[0];
                float recip_scale = 1.0f / this_token_scale;

                for (int hd_bid = loc_id; hd_bid < num_blocks; hd_bid += wg_size) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        simd<T_in, CHUNK> hidden = block_load<T_in, CHUNK>(hidden_states_ptr + token_idx * hd_size + hd_bid * BS + u * CHUNK);
                        simd<float, CHUNK> scale = block_load<float, CHUNK>(smooth_scale_ptr + expert_id * hd_size + hd_bid * BS + u * CHUNK);
                        simd<float, CHUNK> smoothed = simd<float, CHUNK>(hidden) * scale * weight;
                        simd<T_out, CHUNK> quantized;
                        if constexpr (std::is_same_v<T_out, int8_t>) {
                            quantized = rnde<float>(smoothed * recip_scale);
                        } else {
                            quantized = fast_cvt_float_to_e4m3fn<CHUNK>(smoothed * recip_scale);
                        }
                        block_store<T_out, CHUNK>(scatter_tokens_ptr + target_idx * hd_size + hd_bid * BS + u * CHUNK, quantized);
                    }
                }
                if (loc_id == 0) {
                    block_store<float, 1>(scatter_per_token_scale_ptr + target_idx, this_token_scale);
                    block_store<int32_t, 1>(scatter_tokens_offset_ptr + target_idx, token_idx);
                }
            });
        });
    };

    auto launch_scatter_slm = [&](auto unroll_tag, auto slm_tag) {
        constexpr int UNROLL = decltype(unroll_tag)::value;
        constexpr uint32_t SLM_BYTES = decltype(slm_tag)::value;
        constexpr int CHUNK = 64; constexpr int BS = CHUNK * UNROLL;
        int num_blocks = hd_size / BS; int wg_size = std::min(num_blocks, 64);
        queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(routing_event);
            cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(total_items, wg_size), sycl::range<2>(1, wg_size)),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL [[intel::kernel_args_restrict]] {
                const int token_k_idx = item.get_group(0);
                const int expert_id = selected_experts_ptr[token_k_idx];
                if (expert_id < 0 || expert_id >= n_expert_total) return;
                slm_init(SLM_BYTES);
                const int loc_id = item.get_local_id(1);
                const int token_idx = token_k_idx / topk;
                const int target_idx = ext_tokens_start_ptr[expert_id] + token_to_scatter_offset_ptr[token_k_idx];
                const float weight = moe_weights_ptr[token_k_idx];
                uint32_t reduction_base = hd_size * sizeof(float);

                simd<float, CHUNK> thread_max_vec = 0.0f;
                for (int hd_bid = loc_id; hd_bid < num_blocks; hd_bid += wg_size) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        simd<T_in, CHUNK> hidden = block_load<T_in, CHUNK>(hidden_states_ptr + token_idx * hd_size + hd_bid * BS + u * CHUNK);
                        simd<float, CHUNK> scale = block_load<float, CHUNK>(smooth_scale_ptr + expert_id * hd_size + hd_bid * BS + u * CHUNK);
                        simd<float, CHUNK> smoothed = simd<float, CHUNK>(hidden) * scale * weight;
                        thread_max_vec = sycl::ext::intel::esimd::max(thread_max_vec, sycl::ext::intel::esimd::abs(smoothed));
                        uint32_t base_offset = (hd_bid * BS + u * CHUNK) * 4;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            slm_block_store<float, 16>(base_offset + i * 64, smoothed.template select<16, 1>(i * 16));
                        }
                    }
                }
                slm_block_store<float, 4>(reduction_base + loc_id * 16, simd<float, 4>(hmax<float, float, CHUNK>(thread_max_vec)));
                barrier();

                float this_token_scale = 1.0f;
                if (loc_id == 0) {
                    float max_val = 0.0f;
                    for (int i = 0; i < wg_size; i++) {
                        simd<float, 4> val = slm_block_load<float, 4>(reduction_base + i * 16);
                        if (val[0] > max_val) max_val = val[0];
                    }
                    float raw = max_val / quant_max;
                    this_token_scale = raw == 0.0f ? 1.0f : raw;
                    slm_block_store<float, 4>(reduction_base + 1024, simd<float, 4>(this_token_scale));
                }
                barrier();

                this_token_scale = slm_block_load<float, 4>(reduction_base + 1024)[0];
                float recip_scale = 1.0f / this_token_scale;

                for (int hd_bid = loc_id; hd_bid < num_blocks; hd_bid += wg_size) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        uint32_t base_offset = (hd_bid * BS + u * CHUNK) * 4;
                        simd<float, CHUNK> cached_smoothed;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            cached_smoothed.template select<16, 1>(i * 16) = slm_block_load<float, 16>(base_offset + i * 64);
                        }
                        simd<T_out, CHUNK> quantized;
                        if constexpr (std::is_same_v<T_out, int8_t>) {
                            quantized = rnde<float>(cached_smoothed * recip_scale);
                        } else {
                            quantized = fast_cvt_float_to_e4m3fn<CHUNK>(cached_smoothed * recip_scale);
                        }
                        block_store<T_out, CHUNK>(scatter_tokens_ptr + target_idx * hd_size + hd_bid * BS + u * CHUNK, quantized);
                    }
                }
                if (loc_id == 0) {
                    block_store<float, 1>(scatter_per_token_scale_ptr + target_idx, this_token_scale);
                    block_store<int32_t, 1>(scatter_tokens_offset_ptr + target_idx, token_idx);
                }
            });
        });
    };

    auto dispatch_scatter_slm = [&](auto unroll_tag) {
        if (hd_size <=   128) return launch_scatter_slm(unroll_tag, std::integral_constant<uint32_t,   2560>{});
        if (hd_size <=   256) return launch_scatter_slm(unroll_tag, std::integral_constant<uint32_t,   3072>{});
        if (hd_size <=   512) return launch_scatter_slm(unroll_tag, std::integral_constant<uint32_t,   4096>{});
        if (hd_size <=  1024) return launch_scatter_slm(unroll_tag, std::integral_constant<uint32_t,   6144>{});
        if (hd_size <=  2048) return launch_scatter_slm(unroll_tag, std::integral_constant<uint32_t,  10240>{});
        if (hd_size <=  3072) return launch_scatter_slm(unroll_tag, std::integral_constant<uint32_t,  14336>{});
        if (hd_size <=  4096) return launch_scatter_slm(unroll_tag, std::integral_constant<uint32_t,  18432>{});
        if (hd_size <=  5120) return launch_scatter_slm(unroll_tag, std::integral_constant<uint32_t,  22528>{});
        if (hd_size <=  6144) return launch_scatter_slm(unroll_tag, std::integral_constant<uint32_t,  26624>{});
        if (hd_size <=  7168) return launch_scatter_slm(unroll_tag, std::integral_constant<uint32_t,  30720>{});
        if (hd_size <=  8192) return launch_scatter_slm(unroll_tag, std::integral_constant<uint32_t,  34816>{});
        if (hd_size <= 10240) return launch_scatter_slm(unroll_tag, std::integral_constant<uint32_t,  43008>{});
        if (hd_size <= 12288) return launch_scatter_slm(unroll_tag, std::integral_constant<uint32_t,  51200>{});
        if (hd_size <= 14336) return launch_scatter_slm(unroll_tag, std::integral_constant<uint32_t,  59392>{});
        if (hd_size <= 16384) return launch_scatter_slm(unroll_tag, std::integral_constant<uint32_t,  67584>{});
        if (hd_size <= 20480) return launch_scatter_slm(unroll_tag, std::integral_constant<uint32_t,  83968>{});
        if (hd_size <= 24576) return launch_scatter_slm(unroll_tag, std::integral_constant<uint32_t, 100352>{});
        if (hd_size <= 28672) return launch_scatter_slm(unroll_tag, std::integral_constant<uint32_t, 116736>{});
                              return launch_scatter_slm(unroll_tag, std::integral_constant<uint32_t, 131072>{});
    };

    int best_unroll = 1;
    bool is_slm = false;
    
    if (chosen_case.find("14") != std::string::npos || chosen_case.find("18") != std::string::npos) {
        is_slm = true;
    }

    if (chosen_case.find("32") != std::string::npos && !use_v20_routing) best_unroll = 32;
    else if (chosen_case.find("16") != std::string::npos && !use_v20_routing) best_unroll = 16;
    else if (chosen_case.find("08") != std::string::npos) best_unroll = 8;
    else if (chosen_case.find("04") != std::string::npos) best_unroll = 4;
    else if (chosen_case.find("02") != std::string::npos) best_unroll = 2;
    else best_unroll = 1;

    while (best_unroll > 1 && (hd_size / (64 * best_unroll)) <= 0) {
        best_unroll /= 2;
    }

    if (is_slm) {
        switch (best_unroll) {
            case 16: return dispatch_scatter_slm(std::integral_constant<int, 16>{});
            case  8: return dispatch_scatter_slm(std::integral_constant<int, 8>{});
            case  4: return dispatch_scatter_slm(std::integral_constant<int, 4>{});
            case  2: return dispatch_scatter_slm(std::integral_constant<int, 2>{});
            default: return dispatch_scatter_slm(std::integral_constant<int, 1>{});
        }
    } else {
        switch (best_unroll) {
            case 32: return launch_scatter_global(std::integral_constant<int, 32>{});
            case 16: return launch_scatter_global(std::integral_constant<int, 16>{});
            case  8: return launch_scatter_global(std::integral_constant<int, 8>{});
            case  4: return launch_scatter_global(std::integral_constant<int, 4>{});
            case  2: return launch_scatter_global(std::integral_constant<int, 2>{});
            default: return launch_scatter_global(std::integral_constant<int, 1>{});
        }
    }
}

// ----------------------------------------------------------------------------
// SWIGLU IMPLEMENTATIONS
// ----------------------------------------------------------------------------
template <typename T_in, typename T_out>
void moe_swiglu_dynamic_quant_impl(
    torch::Tensor& scatter_tokens, torch::Tensor& smooth_scale,
    torch::Tensor& experts_token_count, torch::Tensor& experts_token_start,
    torch::Tensor& quant_tokens, torch::Tensor& per_token_scale,
    int64_t total_experts_num, int64_t max_token_num, std::string chosen_case) {

    if (max_token_num <= 0 || total_experts_num <= 0) return;

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

    auto launch_swiglu_global = [&](auto unroll_tag) {
        constexpr int UNROLL = decltype(unroll_tag)::value;
        constexpr int CHUNK = 64; constexpr int BS = CHUNK * UNROLL;
        int num_blocks = hidden_size / BS; int wg_size = std::min(num_blocks, 64);
        sycl::range<2> GlobalRange(num_scattered, wg_size);
        sycl::range<2> LocalRange(1, wg_size);

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<2>(GlobalRange, LocalRange), [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL [[intel::kernel_args_restrict]] {
                slm_init(1056);
                const int loc_id = item.get_local_id(1);
                const int flat_idx = item.get_group(0);

                int left = 0; int right = total_experts_num - 1; int expert_idx = 0;
                while (left <= right) {
                    int mid = left + (right - left) / 2;
                    int start = experts_token_start_ptr[mid];
                    int count = experts_token_count_ptr[mid];
                    if (flat_idx >= start && flat_idx < start + count) { expert_idx = mid; break; }
                    else if (flat_idx < start) right = mid - 1;
                    else left = mid + 1;
                }

                T_in* scatter_token_base = scatter_tokens_ptr + flat_idx * 2 * hidden_size;
                T_out* output_base = quant_tokens_ptr + flat_idx * hidden_size;
                simd<float, CHUNK> thread_max_vec = 0.0f;

                for (int bid = loc_id; bid < num_blocks; bid += wg_size) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        simd<T_in, CHUNK> x1 = block_load<T_in, CHUNK>(scatter_token_base + bid * BS + u * CHUNK);
                        simd<T_in, CHUNK> x2 = block_load<T_in, CHUNK>(scatter_token_base + hidden_size + bid * BS + u * CHUNK);
                        simd<float, CHUNK> scale = block_load<float, CHUNK>(smooth_scale_ptr + expert_idx * hidden_size + bid * BS + u * CHUNK);

                        simd<float, CHUNK> sigmoid = sycl::ext::intel::esimd::inv(1.0f + sycl::ext::intel::esimd::exp(-simd<float, CHUNK>(x1)));
                        simd<float, CHUNK> swiglu_tokens = (simd<float, CHUNK>(x1) * sigmoid) * simd<float, CHUNK>(x2);
                        simd<float, CHUNK> scaled_swiglu_tokens = swiglu_tokens * scale;
                        thread_max_vec = sycl::ext::intel::esimd::max(thread_max_vec, sycl::ext::intel::esimd::abs(scaled_swiglu_tokens));
                    }
                }
                slm_block_store<float, 4>(loc_id * 16, simd<float, 4>(hmax<float, float, CHUNK>(thread_max_vec)));
                barrier();

                float this_token_scale = 1.0f;
                if (loc_id == 0) {
                    float max_val = 0.0f;
                    for (int i = 0; i < wg_size; i++) {
                        simd<float, 4> val = slm_block_load<float, 4>(i * 16);
                        if (val[0] > max_val) max_val = val[0];
                    }
                    float raw = max_val / quant_max;
                    this_token_scale = raw == 0.0f ? 1.0f : raw;
                    slm_block_store<float, 4>(1024, simd<float, 4>(this_token_scale));
                }
                barrier();

                this_token_scale = slm_block_load<float, 4>(1024)[0];
                float recip_scale = 1.0f / this_token_scale;

                for (int bid = loc_id; bid < num_blocks; bid += wg_size) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        simd<T_in, CHUNK> x1 = block_load<T_in, CHUNK>(scatter_token_base + bid * BS + u * CHUNK);
                        simd<T_in, CHUNK> x2 = block_load<T_in, CHUNK>(scatter_token_base + hidden_size + bid * BS + u * CHUNK);
                        simd<float, CHUNK> scale = block_load<float, CHUNK>(smooth_scale_ptr + expert_idx * hidden_size + bid * BS + u * CHUNK);

                        simd<float, CHUNK> sigmoid = sycl::ext::intel::esimd::inv(1.0f + sycl::ext::intel::esimd::exp(-simd<float, CHUNK>(x1)));
                        simd<float, CHUNK> scaled_swiglu_tokens = (simd<float, CHUNK>(x1) * sigmoid) * simd<float, CHUNK>(x2) * scale;
                        
                        simd<T_out, CHUNK> quantized;
                        if constexpr (std::is_same_v<T_out, int8_t>) {
                            quantized = rnde<float>(scaled_swiglu_tokens * recip_scale);
                        } else {
                            quantized = fast_cvt_float_to_e4m3fn<CHUNK>(scaled_swiglu_tokens * recip_scale);
                        }
                        block_store<T_out, CHUNK>(output_base + bid * BS + u * CHUNK, quantized);
                    }
                }
                if (loc_id == 0) block_store<float, 1>(per_token_scale_ptr + flat_idx, this_token_scale);
            });
        });
    };

    auto launch_swiglu_slm = [&](auto unroll_tag, auto slm_tag) {
        constexpr int UNROLL = decltype(unroll_tag)::value;
        constexpr uint32_t SLM_BYTES = decltype(slm_tag)::value;
        constexpr int CHUNK = 64; constexpr int BS = CHUNK * UNROLL;
        int num_blocks = hidden_size / BS; int wg_size = std::min(num_blocks, 64);
        sycl::range<2> GlobalRange(num_scattered, wg_size);
        sycl::range<2> LocalRange(1, wg_size);

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<2>(GlobalRange, LocalRange), [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL [[intel::kernel_args_restrict]] {
                slm_init(SLM_BYTES);
                const int loc_id = item.get_local_id(1);
                const int flat_idx = item.get_group(0);

                int left = 0; int right = total_experts_num - 1; int expert_idx = 0;
                while (left <= right) {
                    int mid = left + (right - left) / 2;
                    int start = experts_token_start_ptr[mid];
                    int count = experts_token_count_ptr[mid];
                    if (flat_idx >= start && flat_idx < start + count) { expert_idx = mid; break; }
                    else if (flat_idx < start) right = mid - 1;
                    else left = mid + 1;
                }

                T_in* scatter_token_base = scatter_tokens_ptr + flat_idx * 2 * hidden_size;
                T_out* output_base = quant_tokens_ptr + flat_idx * hidden_size;
                uint32_t reduction_base = hidden_size * sizeof(float);
                simd<float, CHUNK> thread_max_vec = 0.0f;

                for (int bid = loc_id; bid < num_blocks; bid += wg_size) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        simd<T_in, CHUNK> x1 = block_load<T_in, CHUNK>(scatter_token_base + bid * BS + u * CHUNK);
                        simd<T_in, CHUNK> x2 = block_load<T_in, CHUNK>(scatter_token_base + hidden_size + bid * BS + u * CHUNK);
                        simd<float, CHUNK> scale = block_load<float, CHUNK>(smooth_scale_ptr + expert_idx * hidden_size + bid * BS + u * CHUNK);
                        simd<float, CHUNK> sigmoid = sycl::ext::intel::esimd::inv(1.0f + sycl::ext::intel::esimd::exp(-simd<float, CHUNK>(x1)));
                        simd<float, CHUNK> scaled_swiglu_tokens = (simd<float, CHUNK>(x1) * sigmoid) * simd<float, CHUNK>(x2) * scale;
                        thread_max_vec = sycl::ext::intel::esimd::max(thread_max_vec, sycl::ext::intel::esimd::abs(scaled_swiglu_tokens));
                        uint32_t base_offset = (bid * BS + u * CHUNK) * 4;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            slm_block_store<float, 16>(base_offset + i * 64, scaled_swiglu_tokens.template select<16, 1>(i * 16));
                        }
                    }
                }
                slm_block_store<float, 4>(reduction_base + loc_id * 16, simd<float, 4>(hmax<float, float, CHUNK>(thread_max_vec)));
                barrier();

                float this_token_scale = 1.0f;
                if (loc_id == 0) {
                    float max_val = 0.0f;
                    for (int i = 0; i < wg_size; i++) {
                        simd<float, 4> val = slm_block_load<float, 4>(reduction_base + i * 16);
                        if (val[0] > max_val) max_val = val[0];
                    }
                    float raw = max_val / quant_max;
                    this_token_scale = raw == 0.0f ? 1.0f : raw;
                    slm_block_store<float, 4>(reduction_base + 1024, simd<float, 4>(this_token_scale));
                }
                barrier();

                this_token_scale = slm_block_load<float, 4>(reduction_base + 1024)[0];
                float recip_scale = 1.0f / this_token_scale;

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
                if (loc_id == 0) block_store<float, 1>(per_token_scale_ptr + flat_idx, this_token_scale);
            });
        });
    };

    auto dispatch_swiglu_slm = [&](auto unroll_tag) {
        if (hidden_size <=   128) return launch_swiglu_slm(unroll_tag, std::integral_constant<uint32_t,   2560>{});
        if (hidden_size <=   256) return launch_swiglu_slm(unroll_tag, std::integral_constant<uint32_t,   3072>{});
        if (hidden_size <=   512) return launch_swiglu_slm(unroll_tag, std::integral_constant<uint32_t,   4096>{});
        if (hidden_size <=  1024) return launch_swiglu_slm(unroll_tag, std::integral_constant<uint32_t,   6144>{});
        if (hidden_size <=  2048) return launch_swiglu_slm(unroll_tag, std::integral_constant<uint32_t,  10240>{});
        if (hidden_size <=  3072) return launch_swiglu_slm(unroll_tag, std::integral_constant<uint32_t,  14336>{});
        if (hidden_size <=  4096) return launch_swiglu_slm(unroll_tag, std::integral_constant<uint32_t,  18432>{});
        if (hidden_size <=  5120) return launch_swiglu_slm(unroll_tag, std::integral_constant<uint32_t,  22528>{});
        if (hidden_size <=  6144) return launch_swiglu_slm(unroll_tag, std::integral_constant<uint32_t,  26624>{});
        if (hidden_size <=  7168) return launch_swiglu_slm(unroll_tag, std::integral_constant<uint32_t,  30720>{});
        if (hidden_size <=  8192) return launch_swiglu_slm(unroll_tag, std::integral_constant<uint32_t,  34816>{});
        if (hidden_size <= 10240) return launch_swiglu_slm(unroll_tag, std::integral_constant<uint32_t,  43008>{});
        if (hidden_size <= 12288) return launch_swiglu_slm(unroll_tag, std::integral_constant<uint32_t,  51200>{});
        if (hidden_size <= 14336) return launch_swiglu_slm(unroll_tag, std::integral_constant<uint32_t,  59392>{});
        if (hidden_size <= 16384) return launch_swiglu_slm(unroll_tag, std::integral_constant<uint32_t,  67584>{});
        if (hidden_size <= 20480) return launch_swiglu_slm(unroll_tag, std::integral_constant<uint32_t,  83968>{});
        if (hidden_size <= 24576) return launch_swiglu_slm(unroll_tag, std::integral_constant<uint32_t, 100352>{});
        if (hidden_size <= 28672) return launch_swiglu_slm(unroll_tag, std::integral_constant<uint32_t, 116736>{});
                                  return launch_swiglu_slm(unroll_tag, std::integral_constant<uint32_t, 131072>{});
    };

    int best_unroll = 1;
    bool is_slm = false;
    
    if (chosen_case.find("18") != std::string::npos) is_slm = true;

    if (chosen_case.find("32") != std::string::npos) best_unroll = 32;
    else if (chosen_case.find("16") != std::string::npos) best_unroll = 16;
    else if (chosen_case.find("08") != std::string::npos) best_unroll = 8;
    else if (chosen_case.find("04") != std::string::npos) best_unroll = 4;
    else if (chosen_case.find("02") != std::string::npos) best_unroll = 2;
    else best_unroll = 1;

    while (best_unroll > 1 && (hidden_size / (64 * best_unroll)) <= 0) {
        best_unroll /= 2;
    }

    if (is_slm) {
        switch (best_unroll) {
            case 32: return dispatch_swiglu_slm(std::integral_constant<int, 32>{});
            case 16: return dispatch_swiglu_slm(std::integral_constant<int, 16>{});
            case  8: return dispatch_swiglu_slm(std::integral_constant<int, 8>{});
            case  4: return dispatch_swiglu_slm(std::integral_constant<int, 4>{});
            case  2: return dispatch_swiglu_slm(std::integral_constant<int, 2>{});
            default: return dispatch_swiglu_slm(std::integral_constant<int, 1>{});
        }
    } else {
        switch (best_unroll) {
            case 32: return launch_swiglu_global(std::integral_constant<int, 32>{});
            case 16: return launch_swiglu_global(std::integral_constant<int, 16>{});
            case  8: return launch_swiglu_global(std::integral_constant<int, 8>{});
            case  4: return launch_swiglu_global(std::integral_constant<int, 4>{});
            case  2: return launch_swiglu_global(std::integral_constant<int, 2>{});
            default: return launch_swiglu_global(std::integral_constant<int, 1>{});
        }
    }
}

// ----------------------------------------------------------------------------
// DISPATCH MACROS & API SHIMS 
// ----------------------------------------------------------------------------
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
    torch::Tensor& scatter_tokens, torch::Tensor& smooth_scale,
    torch::Tensor& experts_token_count, torch::Tensor& experts_token_start,
    torch::Tensor& quant_tokens, torch::Tensor& per_token_scale,
    int64_t total_experts_num, int64_t max_token_num) {

    at::DeviceGuard guard(scatter_tokens.device());
    int64_t num_scattered = scatter_tokens.size(0);
    int64_t hidden_size = scatter_tokens.size(1) / 2;

    int num_tokens = num_scattered; 

    std::string chosen_case = "reports_test_170801"; 
    if ((hidden_size >= 1365 && hidden_size < 7646) && (num_tokens >= 341 && num_tokens < 1365)) {
        chosen_case = "reports_test_180802";
    }
    else if ((hidden_size >= 2731 && hidden_size < 5213) && (num_tokens < 341)) {
        chosen_case = "reports_test_180802";
    }
    else if ((hidden_size >= 5213) && (num_tokens >= 53 && num_tokens < 341)) {
        chosen_case = "reports_test_180808";
    }
    else if ((hidden_size < 2731) && (num_tokens < 98)) {
        chosen_case = "reports_test_181604";
    }
    else if ((hidden_size < 1365) && (num_tokens >= 683)) {
        chosen_case = "reports_test_170801";
    }
    else if ((hidden_size >= 1365 && hidden_size < 2731) && (num_tokens >= 98 && num_tokens < 341)) {
        chosen_case = "reports_test_180401";
    }
    else if ((hidden_size >= 1365 && hidden_size < 7646) && (num_tokens >= 1365)) {
        chosen_case = "reports_test_173204";
    }
    else if ((hidden_size >= 7646) && (num_tokens < 53)) {
        chosen_case = "reports_test_173216";
    }
    else if ((hidden_size < 1365) && (num_tokens >= 98 && num_tokens < 683)) {
        chosen_case = "reports_test_180404";
    }
    else if ((hidden_size >= 7646) && (num_tokens >= 341)) {
        chosen_case = "reports_test_171601";
    }
    else if ((hidden_size >= 5213 && hidden_size < 7646) && (num_tokens < 53)) {
        chosen_case = "reports_test_183208";
    }

    auto in_dtype = scatter_tokens.scalar_type();
    auto out_dtype = quant_tokens.scalar_type();

    DISPATCH_MOE_QUANT_IMPL(moe_swiglu_dynamic_quant_impl, 
                            scatter_tokens, smooth_scale, experts_token_count, 
                            experts_token_start, quant_tokens, per_token_scale, 
                            total_experts_num, max_token_num, chosen_case);
}

void moe_scatter_dynamic_quant(
    torch::Tensor& selected_experts, torch::Tensor& moe_weights,
    torch::Tensor& token_to_scatter_offset, torch::Tensor& experts_token_count,
    torch::Tensor& experts_token_start, torch::Tensor& hidden_states,
    torch::Tensor& experts_smooth_scale, torch::Tensor& scatter_tokens,
    torch::Tensor& scatter_per_token_scale, torch::Tensor& scatter_tokens_offset,
    int64_t shared_experts_num) {

    at::DeviceGuard guard(hidden_states.device());
    int hd_size = hidden_states.size(1);
    int n_tokens = selected_experts.size(0);
    int topk = selected_experts.size(1);

    int hidden_size = hd_size;
    int num_tokens = n_tokens;

    std::string chosen_case = "reports_test_161601";
    if ((hidden_size >= 5213 && hidden_size < 7646) && (num_tokens >= 53 && num_tokens < 1365) && (topk < 3)) {
        chosen_case = "reports_test_141606";
    }
    else if ((hidden_size >= 5213 && hidden_size < 7646) && (num_tokens >= 53 && num_tokens < 2731) && (topk >= 3 && topk < 6)) {
        chosen_case = "reports_test_181602";
    }
    else if ((hidden_size >= 5213 && hidden_size < 7646) && (num_tokens >= 1365) && (topk < 3)) {
        chosen_case = "reports_test_141604";
    }
    else if ((hidden_size >= 2731) && (num_tokens >= 2731) && (topk >= 3 && topk < 6)) {
        chosen_case = "reports_test_203217";
    }
    else if ((hidden_size < 1365) && (num_tokens < 341) && (topk < 3)) {
        chosen_case = "reports_test_140803";
    }
    else if ((hidden_size < 1365) && (num_tokens < 53) && (topk >= 3)) {
        chosen_case = "reports_test_140803";
    }
    else if ((hidden_size >= 2731) && (num_tokens < 53) && (topk < 3)) {
        chosen_case = "reports_test_161604";
    }
    else if ((hidden_size >= 1365 && hidden_size < 2731) && (num_tokens >= 171 && num_tokens < 683) && (topk >= 3)) {
        chosen_case = "reports_test_161604";
    }
    else if ((hidden_size >= 2731) && (num_tokens >= 53 && num_tokens < 98) && (topk >= 6)) {
        chosen_case = "reports_test_203207";
    }
    else if ((hidden_size >= 2731 && hidden_size < 5213) && (num_tokens >= 98 && num_tokens < 2731) && (topk >= 3 && topk < 6)) {
        chosen_case = "reports_test_203207";
    }
    else if ((hidden_size < 1365) && (num_tokens >= 53 && num_tokens < 2731) && (topk >= 6)) {
        chosen_case = "reports_test_203214";
    }
    else if ((hidden_size < 1365) && (num_tokens >= 341) && (topk < 3)) {
        chosen_case = "reports_test_163202";
    }
    else if ((hidden_size < 1365) && (num_tokens >= 171 && num_tokens < 2731) && (topk >= 3 && topk < 6)) {
        chosen_case = "reports_test_203215";
    }
    else if ((hidden_size < 2731) && (num_tokens >= 2731) && (topk >= 3)) {
        chosen_case = "reports_test_163201";
    }
    else if ((hidden_size >= 1365 && hidden_size < 2731) && (num_tokens >= 98 && num_tokens < 171) && (topk >= 6)) {
        chosen_case = "reports_test_163201";
    }
    else if ((hidden_size >= 1365 && hidden_size < 2731) && (num_tokens >= 683 && num_tokens < 1365) && (topk >= 6)) {
        chosen_case = "reports_test_163201";
    }
    else if ((hidden_size >= 1365 && hidden_size < 2731) && (num_tokens >= 341) && (topk < 3)) {
        chosen_case = "reports_test_170804";
    }
    else if ((hidden_size >= 1365 && hidden_size < 2731) && (num_tokens >= 683 && num_tokens < 1365) && (topk >= 3 && topk < 6)) {
        chosen_case = "reports_test_160802";
    }
    else if ((hidden_size >= 2731 && hidden_size < 5213) && (num_tokens < 53) && (topk >= 3)) {
        chosen_case = "reports_test_171604";
    }
    else if ((hidden_size >= 2731 && hidden_size < 5213) && (num_tokens >= 98) && (topk < 3)) {
        chosen_case = "reports_test_170808";
    }
    else if ((hidden_size >= 5213 && hidden_size < 7646) && (num_tokens >= 98) && (topk >= 6)) {
        chosen_case = "reports_test_161601";
    }
    else if ((hidden_size >= 5213) && (num_tokens < 53) && (topk >= 3 && topk < 6)) {
        chosen_case = "reports_test_170802";
    }
    else if ((hidden_size >= 5213) && (num_tokens < 53) && (topk >= 6)) {
        chosen_case = "reports_test_203222";
    }
    else if ((hidden_size >= 7646) && (num_tokens >= 98 && num_tokens < 171)) {
        chosen_case = "reports_test_173204";
    }
    else if ((hidden_size >= 7646) && (num_tokens >= 171 && num_tokens < 2731) && (topk >= 3 && topk < 6)) {
        chosen_case = "reports_test_163204";
    }
    else if ((hidden_size >= 1365 && hidden_size < 2731) && (num_tokens >= 98 && num_tokens < 171) && (topk >= 3 && topk < 6)) {
        chosen_case = "reports_test_163204";
    }
    else if ((hidden_size >= 7646) && (num_tokens >= 171) && (topk >= 6)) {
        chosen_case = "reports_test_173201";
    }
    else if ((hidden_size >= 1365 && hidden_size < 2731) && (num_tokens >= 1365 && num_tokens < 2731) && (topk >= 3)) {
        chosen_case = "reports_test_173201";
    }
    else if ((hidden_size >= 2731 && hidden_size < 5213) && (num_tokens >= 1365) && (topk >= 6)) {
        chosen_case = "reports_test_173201";
    }
    else if ((hidden_size >= 7646) && (num_tokens >= 683 && num_tokens < 2731) && (topk < 3)) {
        chosen_case = "reports_test_161603";
    }
    else if ((hidden_size >= 1365 && hidden_size < 2731) && (num_tokens >= 53 && num_tokens < 98) && (topk >= 3)) {
        chosen_case = "reports_test_161603";
    }
    else if ((hidden_size >= 7646) && (num_tokens >= 2731) && (topk < 3)) {
        chosen_case = "reports_test_173202";
    }
    else if ((hidden_size >= 1365 && hidden_size < 2731) && (num_tokens >= 53 && num_tokens < 341) && (topk < 3)) {
        chosen_case = "reports_test_160804";
    }
    else if ((hidden_size < 1365) && (num_tokens >= 53 && num_tokens < 171) && (topk >= 3 && topk < 6)) {
        chosen_case = "reports_test_160803";
    }
    else if ((hidden_size >= 1365 && hidden_size < 2731) && (num_tokens < 53)) {
        chosen_case = "reports_test_161605";
    }
    else if ((hidden_size >= 2731 && hidden_size < 5213) && (num_tokens >= 53 && num_tokens < 98) && (topk < 6)) {
        chosen_case = "reports_test_161605";
    }
    else if ((hidden_size >= 2731 && hidden_size < 5213) && (num_tokens >= 98 && num_tokens < 341) && (topk >= 6)) {
        chosen_case = "reports_test_173208";
    }
    else if ((hidden_size >= 7646) && (num_tokens >= 53 && num_tokens < 98) && (topk < 6)) {
        chosen_case = "reports_test_163212";
    }
    else if ((hidden_size >= 7646) && (num_tokens >= 171 && num_tokens < 683) && (topk < 3)) {
        chosen_case = "reports_test_161610";
    }
    else if ((hidden_size >= 2731 && hidden_size < 5213) && (num_tokens >= 341 && num_tokens < 1365) && (topk >= 6)) {
        chosen_case = "reports_test_203212";
    }

    auto in_dtype = hidden_states.scalar_type();
    auto out_dtype = scatter_tokens.scalar_type();

    DISPATCH_MOE_QUANT_IMPL(moe_scatter_dynamic_quant_impl,
                            selected_experts, moe_weights, token_to_scatter_offset, 
                            experts_token_count, experts_token_start, hidden_states, 
                            experts_smooth_scale, scatter_tokens, scatter_per_token_scale, 
                            scatter_tokens_offset, shared_experts_num, chosen_case);
}
