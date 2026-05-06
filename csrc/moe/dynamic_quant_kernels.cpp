#include "moe_ops.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <string>

#include "../utils.h"
#include "../dispatch_utils.h"
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <c10/core/DeviceGuard.h>

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

// -------------------------------------------------------------
// SWIGLU
// -------------------------------------------------------------
template <typename T_in, typename T_out>
void moe_swiglu_dynamic_quant_impl(
    torch::Tensor& scatter_tokens, torch::Tensor& smooth_scale,
    torch::Tensor& experts_token_count, torch::Tensor& experts_token_start,
    torch::Tensor& quant_tokens, torch::Tensor& per_token_scale,
    int64_t total_experts_num, int64_t max_token_num) {

    if (max_token_num <= 0 || total_experts_num <= 0) return;

    auto& queue = vllm::xpu::vllmGetQueue();
    auto scatter_tokens_ptr = reinterpret_cast<T_in*>(scatter_tokens.data_ptr());
    auto smooth_scale_ptr = smooth_scale.data_ptr<float>();
    auto experts_token_count_ptr = experts_token_count.data_ptr<int32_t>();
    auto experts_token_start_ptr = experts_token_start.data_ptr<int32_t>();
    auto quant_tokens_ptr = reinterpret_cast<T_out*>(quant_tokens.data_ptr());
    auto per_token_scale_ptr = per_token_scale.data_ptr<float>();

    int hidden_size = scatter_tokens.size(1) / 2;
    int num_tokens = scatter_tokens.size(0);
    constexpr float quant_max = QuantMax<T_out>::value;

    std::string chosen_case = "";
    if ((hidden_size >= 1365 && hidden_size < 7646) && (num_tokens >= 341 && num_tokens < 1365)) chosen_case = "180802";
    else if ((hidden_size >= 2731 && hidden_size < 5213) && (num_tokens < 341)) chosen_case = "180802";
    else if ((hidden_size >= 5213) && (num_tokens >= 53 && num_tokens < 341)) chosen_case = "180808";
    else if ((hidden_size < 2731) && (num_tokens < 98)) chosen_case = "181604";
    else if ((hidden_size < 1365) && (num_tokens >= 683)) chosen_case = "170801";
    else if ((hidden_size >= 1365 && hidden_size < 2731) && (num_tokens >= 98 && num_tokens < 341)) chosen_case = "180401";
    else if ((hidden_size >= 1365 && hidden_size < 7646) && (num_tokens >= 1365)) chosen_case = "173204";
    else if ((hidden_size >= 7646) && (num_tokens < 53)) chosen_case = "173216";
    else if ((hidden_size < 1365) && (num_tokens >= 98 && num_tokens < 683)) chosen_case = "180404";
    else if ((hidden_size >= 7646) && (num_tokens >= 341)) chosen_case = "171601";
    else if ((hidden_size >= 5213 && hidden_size < 7646) && (num_tokens < 53)) chosen_case = "183208";
    else chosen_case = "173204"; // Fallback

    int target_wg = std::stoi(chosen_case.substr(4, 2));
    int max_unroll = std::stoi(chosen_case.substr(2, 2));

    auto launch_swiglu = [&](auto unroll_tag) {
        constexpr int UNROLL = decltype(unroll_tag)::value;
        constexpr int CHUNK = 64;
        constexpr int BS = CHUNK * UNROLL;
        int num_blocks = hidden_size / BS;
        int wg_size = std::min(num_blocks, 64);

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(num_tokens, wg_size), sycl::range<2>(1, wg_size)), 
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL [[intel::kernel_args_restrict]] {
                slm_init(1056);
                const int loc_id = item.get_local_id(1);
                const int flat_idx = item.get_group(0);

                int left = 0, right = total_experts_num - 1, expert_idx = 0;
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
                        simd<float, CHUNK> scaled = (simd<float, CHUNK>(x1) * sigmoid) * simd<float, CHUNK>(x2) * scale;
                        thread_max_vec = sycl::ext::intel::esimd::max(thread_max_vec, sycl::ext::intel::esimd::abs(scaled));
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
                    this_token_scale = (max_val / quant_max) == 0.0f ? 1.0f : (max_val / quant_max);
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
                        simd<float, CHUNK> scaled = (simd<float, CHUNK>(x1) * sigmoid) * simd<float, CHUNK>(x2) * scale;
                        
                        simd<T_out, CHUNK> q;
                        if constexpr (std::is_same_v<T_out, int8_t>) q = rnde<float>(scaled * recip_scale);
                        else q = fast_cvt_float_to_e4m3fn<CHUNK>(scaled * recip_scale);
                        block_store<T_out, CHUNK>(output_base + bid * BS + u * CHUNK, q);
                    }
                }
                if (loc_id == 0) block_store<float, 1>(per_token_scale_ptr + flat_idx, this_token_scale);
            });
        });
    };

    int best_unroll = 1;
    int num_chunks = hidden_size / 64;
    for (int u : {32, 16, 8, 4, 2}) {
        if (u <= max_unroll && num_chunks % u == 0 && (num_chunks / u) >= target_wg) {
            best_unroll = u; break;
        }
    }

    switch (best_unroll) {
        case 32: return launch_swiglu(std::integral_constant<int, 32>{});
        case 16: return launch_swiglu(std::integral_constant<int, 16>{});
        case  8: return launch_swiglu(std::integral_constant<int, 8>{});
        case  4: return launch_swiglu(std::integral_constant<int, 4>{});
        case  2: return launch_swiglu(std::integral_constant<int, 2>{});
        default: return launch_swiglu(std::integral_constant<int, 1>{});
    }
}

// -------------------------------------------------------------
// SCATTER
// -------------------------------------------------------------
template <typename T_in, typename T_out>
void moe_scatter_dynamic_quant_impl(
    torch::Tensor& selected_experts, torch::Tensor& moe_weights,
    torch::Tensor& token_to_scatter_offset, torch::Tensor& experts_token_count,
    torch::Tensor& experts_token_start, torch::Tensor& hidden_states,
    torch::Tensor& experts_smooth_scale, torch::Tensor& scatter_tokens,
    torch::Tensor& scatter_per_token_scale, torch::Tensor& scatter_tokens_offset,
    int64_t shared_experts_num) {

    int num_tokens = selected_experts.size(0);
    if (num_tokens <= 0) return;
    
    (void)shared_experts_num;
    auto& queue = vllm::xpu::vllmGetQueue();

    int topk = selected_experts.size(1);
    int n_expert_total = experts_token_count.size(0);
    int hidden_size = hidden_states.size(1);
    constexpr float quant_max = QuantMax<T_out>::value;

    std::string chosen_case = "";
    if ((hidden_size >= 5213 && hidden_size < 7646) && (num_tokens >= 53 && num_tokens < 1365) && (topk < 3)) chosen_case = "141606";
    else if ((hidden_size >= 5213 && hidden_size < 7646) && (num_tokens >= 53 && num_tokens < 2731) && (topk >= 3 && topk < 6)) chosen_case = "181602";
    else if ((hidden_size >= 5213 && hidden_size < 7646) && (num_tokens >= 1365) && (topk < 3)) chosen_case = "141604";
    else if ((hidden_size >= 2731) && (num_tokens >= 2731) && (topk >= 3 && topk < 6)) chosen_case = "203217";
    else if ((hidden_size < 1365) && (num_tokens < 341) && (topk < 3)) chosen_case = "140803";
    else if ((hidden_size < 1365) && (num_tokens < 53) && (topk >= 3)) chosen_case = "140803";
    else if ((hidden_size >= 2731) && (num_tokens < 53) && (topk < 3)) chosen_case = "161604";
    else if ((hidden_size >= 1365 && hidden_size < 2731) && (num_tokens >= 171 && num_tokens < 683) && (topk >= 3)) chosen_case = "161604";
    else if ((hidden_size >= 2731) && (num_tokens >= 53 && num_tokens < 98) && (topk >= 6)) chosen_case = "203207";
    else if ((hidden_size >= 2731 && hidden_size < 5213) && (num_tokens >= 98 && num_tokens < 2731) && (topk >= 3 && topk < 6)) chosen_case = "203207";
    else if ((hidden_size < 1365) && (num_tokens >= 53 && num_tokens < 2731) && (topk >= 6)) chosen_case = "203214";
    else if ((hidden_size < 1365) && (num_tokens >= 341) && (topk < 3)) chosen_case = "163202";
    else if ((hidden_size < 1365) && (num_tokens >= 171 && num_tokens < 2731) && (topk >= 3 && topk < 6)) chosen_case = "203215";
    else if ((hidden_size < 2731) && (num_tokens >= 2731) && (topk >= 3)) chosen_case = "163201";
    else if ((hidden_size >= 1365 && hidden_size < 2731) && (num_tokens >= 98 && num_tokens < 171) && (topk >= 6)) chosen_case = "163201";
    else if ((hidden_size >= 1365 && hidden_size < 2731) && (num_tokens >= 683 && num_tokens < 1365) && (topk >= 6)) chosen_case = "163201";
    else if ((hidden_size >= 1365 && hidden_size < 2731) && (num_tokens >= 341) && (topk < 3)) chosen_case = "170804";
    else if ((hidden_size >= 1365 && hidden_size < 2731) && (num_tokens >= 683 && num_tokens < 1365) && (topk >= 3 && topk < 6)) chosen_case = "160802";
    else if ((hidden_size >= 2731 && hidden_size < 5213) && (num_tokens < 53) && (topk >= 3)) chosen_case = "171604";
    else if ((hidden_size >= 2731 && hidden_size < 5213) && (num_tokens >= 98) && (topk < 3)) chosen_case = "170808";
    else if ((hidden_size >= 5213 && hidden_size < 7646) && (num_tokens >= 98) && (topk >= 6)) chosen_case = "161601";
    else if ((hidden_size >= 5213) && (num_tokens < 53) && (topk >= 3 && topk < 6)) chosen_case = "170802";
    else if ((hidden_size >= 5213) && (num_tokens < 53) && (topk >= 6)) chosen_case = "203222";
    else if ((hidden_size >= 7646) && (num_tokens >= 98 && num_tokens < 171)) chosen_case = "173204";
    else if ((hidden_size >= 7646) && (num_tokens >= 171 && num_tokens < 2731) && (topk >= 3 && topk < 6)) chosen_case = "163204";
    else if ((hidden_size >= 1365 && hidden_size < 2731) && (num_tokens >= 98 && num_tokens < 171) && (topk >= 3 && topk < 6)) chosen_case = "163204";
    else if ((hidden_size >= 7646) && (num_tokens >= 171) && (topk >= 6)) chosen_case = "173201";
    else if ((hidden_size >= 1365 && hidden_size < 2731) && (num_tokens >= 1365 && num_tokens < 2731) && (topk >= 3)) chosen_case = "173201";
    else if ((hidden_size >= 2731 && hidden_size < 5213) && (num_tokens >= 1365) && (topk >= 6)) chosen_case = "173201";
    else if ((hidden_size >= 7646) && (num_tokens >= 683 && num_tokens < 2731) && (topk < 3)) chosen_case = "161603";
    else if ((hidden_size >= 1365 && hidden_size < 2731) && (num_tokens >= 53 && num_tokens < 98) && (topk >= 3)) chosen_case = "161603";
    else if ((hidden_size >= 7646) && (num_tokens >= 2731) && (topk < 3)) chosen_case = "173202";
    else if ((hidden_size >= 1365 && hidden_size < 2731) && (num_tokens >= 53 && num_tokens < 341) && (topk < 3)) chosen_case = "160804";
    else if ((hidden_size < 1365) && (num_tokens >= 53 && num_tokens < 171) && (topk >= 3 && topk < 6)) chosen_case = "160803";
    else if ((hidden_size >= 1365 && hidden_size < 2731) && (num_tokens < 53)) chosen_case = "161605";
    else if ((hidden_size >= 2731 && hidden_size < 5213) && (num_tokens >= 53 && num_tokens < 98) && (topk < 6)) chosen_case = "161605";
    else if ((hidden_size >= 2731 && hidden_size < 5213) && (num_tokens >= 98 && num_tokens < 341) && (topk >= 6)) chosen_case = "173208";
    else if ((hidden_size >= 7646) && (num_tokens >= 53 && num_tokens < 98) && (topk < 6)) chosen_case = "163212";
    else if ((hidden_size >= 7646) && (num_tokens >= 171 && num_tokens < 683) && (topk < 3)) chosen_case = "161610";
    else if ((hidden_size >= 2731 && hidden_size < 5213) && (num_tokens >= 341 && num_tokens < 1365) && (topk >= 6)) chosen_case = "203212";
    else chosen_case = "163204"; // Fallback

    int version = std::stoi(chosen_case.substr(0, 2));
    int max_unroll = std::stoi(chosen_case.substr(2, 2));
    int target_wg = std::stoi(chosen_case.substr(4, 2));

    // Dynamic routing config based on version mapping
    int sub_group_snap = 32;
    int items_per_thread = 8;
    if (version == 20) {
        if (chosen_case == "203222") { sub_group_snap = 8; items_per_thread = 1; }
        else if (chosen_case == "203217") { sub_group_snap = 8; items_per_thread = 32; }
        else if (chosen_case == "203215") { sub_group_snap = 16; items_per_thread = 2; }
        else if (chosen_case == "203214") { sub_group_snap = 16; items_per_thread = 4; }
        else if (chosen_case == "203212") { sub_group_snap = 16; items_per_thread = 8; }
        else if (chosen_case == "203207") { sub_group_snap = 32; items_per_thread = 2; }
    }

    auto selected_experts_ptr = selected_experts.data_ptr<int32_t>();
    auto ext_tokens_cnt_ptr = experts_token_count.data_ptr<int32_t>();
    auto ext_tokens_start_ptr = experts_token_start.data_ptr<int32_t>();
    auto token_to_scatter_offset_ptr = token_to_scatter_offset.data_ptr<int32_t>();

    int total_items = num_tokens * topk;
    int target_threads = (total_items + items_per_thread - 1) / items_per_thread;
    int routing_wg = ((target_threads + sub_group_snap - 1) / sub_group_snap) * sub_group_snap;
    routing_wg = std::min(256, std::max(sub_group_snap, routing_wg));

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

    auto hidden_states_ptr = reinterpret_cast<T_in*>(hidden_states.data_ptr());
    auto smooth_scale_ptr = experts_smooth_scale.data_ptr<float>();
    auto moe_weights_ptr = moe_weights.data_ptr<float>();
    auto scatter_tokens_ptr = reinterpret_cast<T_out*>(scatter_tokens.data_ptr());
    auto scatter_per_token_scale_ptr = scatter_per_token_scale.data_ptr<float>();
    auto scatter_tokens_offset_ptr = scatter_tokens_offset.data_ptr<int32_t>();

    auto launch_scatter = [&](auto unroll_tag) {
        constexpr int UNROLL = decltype(unroll_tag)::value;
        constexpr int CHUNK = 64;
        constexpr int BS = CHUNK * UNROLL;
        constexpr int MAX_TOPK = 16; 

        int num_blocks = hidden_size / BS;
        int wg_size = std::min(num_blocks, 64);

        queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(routing_event);
            cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(num_tokens, wg_size), sycl::range<2>(1, wg_size)),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL [[intel::kernel_args_restrict]] {
                slm_init(32768);
                const int token_idx = item.get_group(0);
                const int loc_id = item.get_local_id(1);
                int local_topk = std::min(topk, MAX_TOPK);

                simd<float, CHUNK> thread_max_vec[MAX_TOPK];
                for (int k = 0; k < MAX_TOPK; ++k) thread_max_vec[k] = 0.0f;

                for (int hd_bid = loc_id; hd_bid < num_blocks; hd_bid += wg_size) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        simd<T_in, CHUNK> hidden = block_load<T_in, CHUNK>(hidden_states_ptr + token_idx * hidden_size + hd_bid * BS + u * CHUNK);
                        for (int k = 0; k < local_topk; ++k) {
                            int token_k_idx = token_idx * topk + k;
                            int expert_id = selected_experts_ptr[token_k_idx];
                            if (expert_id < 0 || expert_id >= n_expert_total) continue;

                            float weight = moe_weights_ptr[token_k_idx];
                            simd<float, CHUNK> scale = block_load<float, CHUNK>(smooth_scale_ptr + expert_id * hidden_size + hd_bid * BS + u * CHUNK);
                            simd<float, CHUNK> smoothed = simd<float, CHUNK>(hidden) * scale * weight;
                            thread_max_vec[k] = sycl::ext::intel::esimd::max(thread_max_vec[k], sycl::ext::intel::esimd::abs(smoothed));
                        }
                    }
                }

                float thread_max[MAX_TOPK];
                for (int k = 0; k < local_topk; ++k) {
                    thread_max[k] = hmax<float, float, CHUNK>(thread_max_vec[k]);
                    slm_block_store<float, 4>((k * wg_size + loc_id) * 16, simd<float, 4>(thread_max[k]));
                }
                barrier();

                float this_token_scale[MAX_TOPK];
                if (loc_id == 0) {
                    for (int k = 0; k < local_topk; ++k) {
                        float max_val = 0.0f;
                        for (int i = 0; i < wg_size; i++) {
                            simd<float, 4> val = slm_block_load<float, 4>((k * wg_size + i) * 16);
                            if (val[0] > max_val) max_val = val[0];
                        }
                        float scale_val = (max_val / quant_max) == 0.0f ? 1.0f : (max_val / quant_max);
                        slm_block_store<float, 4>(16384 + k * 16, simd<float, 4>(scale_val));
                    }
                }
                barrier();

                for (int k = 0; k < local_topk; ++k) {
                    this_token_scale[k] = slm_block_load<float, 4>(16384 + k * 16)[0];
                }

                for (int hd_bid = loc_id; hd_bid < num_blocks; hd_bid += wg_size) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        simd<T_in, CHUNK> hidden = block_load<T_in, CHUNK>(hidden_states_ptr + token_idx * hidden_size + hd_bid * BS + u * CHUNK);
                        for (int k = 0; k < local_topk; ++k) {
                            int token_k_idx = token_idx * topk + k;
                            int expert_id = selected_experts_ptr[token_k_idx];
                            if (expert_id < 0 || expert_id >= n_expert_total) continue;

                            int target_idx = ext_tokens_start_ptr[expert_id] + token_to_scatter_offset_ptr[token_k_idx];
                            float weight = moe_weights_ptr[token_k_idx];
                            float recip_scale = 1.0f / this_token_scale[k];

                            simd<float, CHUNK> scale = block_load<float, CHUNK>(smooth_scale_ptr + expert_id * hidden_size + hd_bid * BS + u * CHUNK);
                            simd<float, CHUNK> smoothed = simd<float, CHUNK>(hidden) * scale * weight;

                            simd<T_out, CHUNK> q;
                            if constexpr (std::is_same_v<T_out, int8_t>) q = rnde<float>(smoothed * recip_scale);
                            else q = fast_cvt_float_to_e4m3fn<CHUNK>(smoothed * recip_scale);

                            block_store<T_out, CHUNK>(scatter_tokens_ptr + target_idx * hidden_size + hd_bid * BS + u * CHUNK, q);
                        }
                    }
                }

                if (loc_id == 0) {
                    for (int k = 0; k < local_topk; ++k) {
                        int token_k_idx = token_idx * topk + k;
                        int expert_id = selected_experts_ptr[token_k_idx];
                        if (expert_id < 0 || expert_id >= n_expert_total) continue;

                        int target_idx = ext_tokens_start_ptr[expert_id] + token_to_scatter_offset_ptr[token_k_idx];
                        block_store<float, 1>(scatter_per_token_scale_ptr + target_idx, this_token_scale[k]);
                        block_store<int32_t, 1>(scatter_tokens_offset_ptr + target_idx, token_idx);
                    }
                }
            });
        });
    };

    int best_unroll = 1;
    int num_chunks = hidden_size / 64;
    for (int u : {32, 16, 8, 4, 2}) {
        if (u <= max_unroll && num_chunks % u == 0 && (num_chunks / u) >= target_wg) {
            best_unroll = u; break;
        }
    }

    switch (best_unroll) {
        case 32: return launch_scatter(std::integral_constant<int, 32>{});
        case 16: return launch_scatter(std::integral_constant<int, 16>{});
        case  8: return launch_scatter(std::integral_constant<int, 8>{});
        case  4: return launch_scatter(std::integral_constant<int, 4>{});
        case  2: return launch_scatter(std::integral_constant<int, 2>{});
        default: return launch_scatter(std::integral_constant<int, 1>{});
    }
}

// -------------------------------------------------------------
// DISPATCH MACROS & BINDINGS (unchanged)
// -------------------------------------------------------------
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
    int64_t hidden_size2 = scatter_tokens.size(1);
    int64_t hidden_size = hidden_size2 / 2;
    auto in_dtype = scatter_tokens.scalar_type();
    auto out_dtype = quant_tokens.scalar_type();

    DISPATCH_MOE_QUANT_IMPL(moe_swiglu_dynamic_quant_impl, 
                            scatter_tokens, smooth_scale, experts_token_count, 
                            experts_token_start, quant_tokens, per_token_scale, 
                            total_experts_num, max_token_num);
}

void moe_scatter_dynamic_quant(
    torch::Tensor& selected_experts, torch::Tensor& moe_weights,
    torch::Tensor& token_to_scatter_offset, torch::Tensor& experts_token_count,
    torch::Tensor& experts_token_start, torch::Tensor& hidden_states,
    torch::Tensor& experts_smooth_scale, torch::Tensor& scatter_tokens,
    torch::Tensor& scatter_per_token_scale, torch::Tensor& scatter_tokens_offset,
    int64_t shared_experts_num) {

    at::DeviceGuard guard(hidden_states.device());
    auto in_dtype = hidden_states.scalar_type();
    auto out_dtype = scatter_tokens.scalar_type();

    DISPATCH_MOE_QUANT_IMPL(moe_scatter_dynamic_quant_impl,
                            selected_experts, moe_weights, token_to_scatter_offset, 
                            experts_token_count, experts_token_start, hidden_states, 
                            experts_smooth_scale, scatter_tokens, scatter_per_token_scale, 
                            scatter_tokens_offset, shared_experts_num);
}