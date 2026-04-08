#include <sycl/sycl.hpp>
#include <algorithm>
#include "utils.h"
#include "dispatch_utils.h"
#include "rotary_embedding.hpp"
#include <cmath>
#include <c10/macros/Macros.h>

namespace vllm {

// Maximum number of M-RoPE sections supported (e.g. 3 for Qwen2-VL:
// temporal / height / width).
constexpr int MROPE_MAX_SECTIONS = 4;
// Maximum rot_dim supported for the merged cache buffer (cos + sin).
constexpr int MROPE_MAX_ROT_DIM = 512;

template <typename scalar_t, bool IS_NEOX>
class multimodal_rotary_embedding_kernel {
 public:
  multimodal_rotary_embedding_kernel(
      const int64_t* __restrict__ positions_,
      scalar_t* __restrict__ query_,
      scalar_t* __restrict__ key_,
      const scalar_t* __restrict__ cos_sin_cache_,
      const int* mrope_section_data,
      const int num_mrope_sections_,
      const int num_tokens_,
      const int rot_dim_,
      const int64_t query_stride_,
      const int64_t key_stride_,
      const int64_t head_stride_,
      const int num_heads_,
      const int num_kv_heads_,
      const int head_size_,
      const int tokens_per_block_) // Passed directly from launch
      : positions(positions_),
        query(query_),
        key(key_),
        cos_sin_cache(cos_sin_cache_),
        num_mrope_sections(num_mrope_sections_),
        num_tokens(num_tokens_),
        rot_dim(rot_dim_),
        query_stride(query_stride_),
        key_stride(key_stride_),
        head_stride(head_stride_),
        num_heads(num_heads_),
        num_kv_heads(num_kv_heads_),
        head_size(head_size_),
        tokens_per_block(tokens_per_block_) {
    for (int s = 0; s < num_mrope_sections_; ++s)
      mrope_section[s] = mrope_section_data[s];
  }

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {

    constexpr int VEC_SIZE = 2;
    using vec_t = sycl::vec<scalar_t, VEC_SIZE>;

    const int embed_dim = rot_dim / 2;
    const int num_vec_dims = embed_dim / VEC_SIZE;

    const int token_group = item_ct1.get_group(2);
    const int local_id = item_ct1.get_local_id(2);

    // Map thread to a specific token chunk and dimension
    const int t_offset = local_id / num_vec_dims;
    const int vec_dim_idx = local_id % num_vec_dims;

    const int token_idx = token_group * tokens_per_block + t_offset;
    if (token_idx >= num_tokens) return;

    const int dim_idx = vec_dim_idx * VEC_SIZE;

    // 1. Calculate exactly which M-RoPE section this dim falls into
    int section_idx = 0;
    int section_offset = dim_idx;
    int cumsum = 0;
    for (int s = 0; s < num_mrope_sections; ++s) {
        if (dim_idx < cumsum + mrope_section[s]) {
            section_idx = s;
            section_offset = dim_idx - cumsum;
            break;
        }
        cumsum += mrope_section[s];
    }

    const int64_t pos = positions[section_idx * num_tokens + token_idx];

    // 2. Fetch the cos and sin components
    const vec_t cos_vec = *reinterpret_cast<const vec_t*>(&cos_sin_cache[pos * rot_dim + cumsum + section_offset]);
    const vec_t sin_vec = *reinterpret_cast<const vec_t*>(&cos_sin_cache[pos * rot_dim + embed_dim + cumsum + section_offset]);

    // 3. Loop over heads
    for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        const int64_t q_offset = token_idx * query_stride + head_idx * head_stride;

        if constexpr (IS_NEOX) {
            vec_t q1 = *reinterpret_cast<const vec_t*>(&query[q_offset + dim_idx]);
            vec_t q2 = *reinterpret_cast<const vec_t*>(&query[q_offset + embed_dim + dim_idx]);

            vec_t out_q1, out_q2;
            #pragma unroll
            for (int v = 0; v < VEC_SIZE; ++v) {
                out_q1[v] = q1[v] * cos_vec[v] - q2[v] * sin_vec[v];
                out_q2[v] = q2[v] * cos_vec[v] + q1[v] * sin_vec[v];
            }
            *reinterpret_cast<vec_t*>(&query[q_offset + dim_idx]) = out_q1;
            *reinterpret_cast<vec_t*>(&query[q_offset + embed_dim + dim_idx]) = out_q2;
        } else {
            #pragma unroll
            for (int v = 0; v < VEC_SIZE; ++v) {
                vec_t q_pair = *reinterpret_cast<const vec_t*>(&query[q_offset + (dim_idx + v) * 2]);
                vec_t out_q;

                out_q[0] = q_pair[0] * cos_vec[v] - q_pair[1] * sin_vec[v];
                out_q[1] = q_pair[1] * cos_vec[v] + q_pair[0] * sin_vec[v];

                *reinterpret_cast<vec_t*>(&query[q_offset + (dim_idx + v) * 2]) = out_q;
            }
        }
    }

    if (key != nullptr) {
        for (int head_idx = 0; head_idx < num_kv_heads; ++head_idx) {
            const int64_t k_offset = token_idx * key_stride + head_idx * head_stride;

            if constexpr (IS_NEOX) {
                vec_t k1 = *reinterpret_cast<const vec_t*>(&key[k_offset + dim_idx]);
                vec_t k2 = *reinterpret_cast<const vec_t*>(&key[k_offset + embed_dim + dim_idx]);

                vec_t out_k1, out_k2;
                #pragma unroll
                for (int v = 0; v < VEC_SIZE; ++v) {
                    out_k1[v] = k1[v] * cos_vec[v] - k2[v] * sin_vec[v];
                    out_k2[v] = k2[v] * cos_vec[v] + k1[v] * sin_vec[v];
                }
                *reinterpret_cast<vec_t*>(&key[k_offset + dim_idx]) = out_k1;
                *reinterpret_cast<vec_t*>(&key[k_offset + embed_dim + dim_idx]) = out_k2;
            } else {
                #pragma unroll
                for (int v = 0; v < VEC_SIZE; ++v) {
                    vec_t k_pair = *reinterpret_cast<const vec_t*>(&key[k_offset + (dim_idx + v) * 2]);
                    vec_t out_k;

                    out_k[0] = k_pair[0] * cos_vec[v] - k_pair[1] * sin_vec[v];
                    out_k[1] = k_pair[1] * cos_vec[v] + k_pair[0] * sin_vec[v];

                    *reinterpret_cast<vec_t*>(&key[k_offset + (dim_idx + v) * 2]) = out_k;
                }
            }
        }
    }
  }

 private:
  const int64_t* __restrict__ positions;
  scalar_t* __restrict__ query;
  scalar_t* __restrict__ key;
  const scalar_t* __restrict__ cos_sin_cache;
  int mrope_section[MROPE_MAX_SECTIONS];
  const int num_mrope_sections;
  const int num_tokens;
  const int rot_dim;
  const int64_t query_stride;
  const int64_t key_stride;
  const int64_t head_stride;
  const int num_heads;
  const int num_kv_heads;
  const int head_size;
  const int tokens_per_block;
};

}  // namespace vllm

// ── Multi-Modal Rotary Embedding (M-RoPE) ──────────────────────────────────
// Used by models such as Qwen2-VL that need per-section position encoding.
//
// positions      : [num_mrope_sections, num_tokens]  int64, on device
// query          : [num_tokens, num_heads * head_size] or
//                  [num_tokens, num_heads, head_size]
// key            : same shapes as query but kv_heads, or nullopt
// cos_sin_cache  : [max_position, rot_dim]
// mrope_section  : [num_mrope_sections]  int32, on device;
//                  values in embed_dim units summing to rot_dim / 2

template <typename scalar_t>
void call_multimodal_rotary_embedding_kernel(
    torch::Tensor& positions,
    torch::Tensor& query,
    std::optional<torch::Tensor> key,
    int64_t head_size,
    torch::Tensor& cos_sin_cache,
    bool is_neox,
    const std::vector<int64_t>& mrope_section) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  TORCH_CHECK(
      positions.dim() == 2,
      "positions must have shape [num_mrope_sections, num_tokens]");
  const int num_mrope_sections = positions.size(0);
  const int64_t num_tokens = positions.size(1);

  TORCH_CHECK(
      (int)mrope_section.size() == num_mrope_sections,
      "mrope_section length must equal positions.size(0)");
  TORCH_CHECK(
      num_mrope_sections <= vllm::MROPE_MAX_SECTIONS,
      "num_mrope_sections exceeds MROPE_MAX_SECTIONS=",
      vllm::MROPE_MAX_SECTIONS);

  const int query_hidden_size = query.numel() / num_tokens;
  const int key_hidden_size = key.has_value() ? key->numel() / num_tokens : 0;
  TORCH_CHECK(query_hidden_size % head_size == 0);
  TORCH_CHECK(key_hidden_size % head_size == 0);

  const int num_heads = query_hidden_size / head_size;
  const int num_kv_heads =
      key.has_value() ? key_hidden_size / head_size : num_heads;
  TORCH_CHECK(num_heads % num_kv_heads == 0);

  const int rot_dim = cos_sin_cache.size(1);
  TORCH_CHECK(
      rot_dim <= vllm::MROPE_MAX_ROT_DIM,
      "rot_dim exceeds MROPE_MAX_ROT_DIM=",
      vllm::MROPE_MAX_ROT_DIM,
      ", got rot_dim=",
      rot_dim);

  // query is always [num_tokens, ...] in the M-RoPE path.
  const int64_t query_stride = query.stride(0);
  const int64_t key_stride = key.has_value() ? key->stride(0) : 0;
  const int query_ndim = query.dim();
  // For [num_tokens, num_heads, head_size] use stride(-2), else head_size.
  const int64_t head_stride = (query_ndim == 3) ? query.stride(-2) : head_size;

  // Ensure positions is contiguous so that raw pointer arithmetic
  // s * num_tokens + t correctly addresses positions[s, t].
  at::Tensor positions_contig = positions.contiguous();
  auto positions_ptr = positions_contig.data_ptr<int64_t>();
  auto query_ptr = query.data_ptr<scalar_t>();
  auto key_ptr = key.has_value() ? key->data_ptr<scalar_t>() : nullptr;
  auto cos_sin_cache_ptr = cos_sin_cache.data_ptr<scalar_t>();

  // Convert int64 list to int array for the kernel and verify sections
  // sum to embed_dim (= rot_dim / 2).
  int mrope_section_arr[vllm::MROPE_MAX_SECTIONS] = {};
  int section_sum = 0;
  for (int s = 0; s < num_mrope_sections; ++s) {
    mrope_section_arr[s] = static_cast<int>(mrope_section[s]);
    section_sum += mrope_section_arr[s];
  }
  TORCH_CHECK(
      section_sum == rot_dim / 2,
      "mrope_section values must sum to rot_dim / 2 (embed_dim=",
      rot_dim / 2,
      "), but got ",
      section_sum);

  // Chunking tokens to saturate SIMD occupancy and reduce dispatch overhead
  const int VEC_SIZE = 2;
  const int num_vec_dims = (rot_dim / 2) / VEC_SIZE;
  const int target_threads_per_block = 256;
  int tokens_per_block = std::max(1, target_threads_per_block / num_vec_dims);

  sycl::range<3> block(1, 1, tokens_per_block * num_vec_dims);
  sycl::range<3> grid(1, 1, (num_tokens + tokens_per_block - 1) / tokens_per_block);

  at::DeviceGuard device_guard(query.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  if (is_neox) {
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          vllm::multimodal_rotary_embedding_kernel<sycl_t, true>(
              positions_ptr,
              (sycl_t*)query_ptr,
              (sycl_t*)key_ptr,
              (sycl_t*)cos_sin_cache_ptr,
              mrope_section_arr,
              num_mrope_sections,
              num_tokens,
              rot_dim,
              query_stride,
              key_stride,
              head_stride,
              num_heads,
              num_kv_heads,
              head_size,
              tokens_per_block));
    });
  } else {
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          vllm::multimodal_rotary_embedding_kernel<sycl_t, false>(
              positions_ptr,
              (sycl_t*)query_ptr,
              (sycl_t*)key_ptr,
              (sycl_t*)cos_sin_cache_ptr,
              mrope_section_arr,
              num_mrope_sections,
              num_tokens,
              rot_dim,
              query_stride,
              key_stride,
              head_stride,
              num_heads,
              num_kv_heads,
              head_size,
              tokens_per_block));
    });
  }
}

void multimodal_rotary_embedding(
    torch::Tensor& positions,  // [num_mrope_sections, num_tokens]
    torch::Tensor& query,
    std::optional<torch::Tensor> key,
    int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox,
    std::vector<int64_t> mrope_section)  // host int list [num_mrope_sections]
{
  VLLM_DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "multimodal_rotary_embedding", [&] {
        call_multimodal_rotary_embedding_kernel<scalar_t>(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_neox,
            mrope_section);
      });
}