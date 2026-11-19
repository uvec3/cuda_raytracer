#include "BVHAS.cuh"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "imgui.h"
#include "../../cuda_common.hpp"

namespace bvh_detail
{
    static constexpr int SAH_BINS = 16;

    struct NodeBoundsAgg
    {
        glm::vec3 bmin;
        glm::vec3 bmax;
        glm::vec3 cmin;
        glm::vec3 cmax;
    };

    struct PrimToAggFunctor
    {
        const glm::vec3* prim_min;
        const glm::vec3* prim_max;
        const glm::vec3* prim_centroid;

        __host__ __device__ NodeBoundsAgg operator()(uint32_t prim_idx) const
        {
            const glm::vec3 c = prim_centroid[prim_idx];
            return NodeBoundsAgg{prim_min[prim_idx], prim_max[prim_idx], c, c};
        }
    };

    struct CombineBoundsAggFunctor
    {
        __host__ __device__ NodeBoundsAgg operator()(const NodeBoundsAgg& a, const NodeBoundsAgg& b) const
        {
            return NodeBoundsAgg{
                glm::min(a.bmin, b.bmin),
                glm::max(a.bmax, b.bmax),
                glm::min(a.cmin, b.cmin),
                glm::max(a.cmax, b.cmax)
            };
        }
    };

    struct BVHSahBinGpu
    {
        uint32_t count;
        int32_t bmin_i[3];
        int32_t bmax_i[3];
        uint32_t pad;
    };
    static_assert(sizeof(BVHSahBinGpu) == 32);

    struct BVHSahBinHost
    {
        uint32_t count = 0;
        glm::vec3 bmin{FLT_MAX};
        glm::vec3 bmax{-FLT_MAX};
    };

    __host__ __device__ __forceinline__ int32_t float_to_ordered_int(float f)
    {
        const int32_t i =
#ifdef __CUDA_ARCH__
            __float_as_int(f);
#else
            static_cast<int32_t>(std::bit_cast<int32_t>(f));
#endif
        return (i >= 0) ? i : (i ^ 0x7FFFFFFF);
    }

    __host__ __device__ __forceinline__ float ordered_int_to_float(int32_t i)
    {
        const int32_t raw = (i >= 0) ? i : (i ^ 0x7FFFFFFF);
#ifdef __CUDA_ARCH__
        return __int_as_float(raw);
#else
        return std::bit_cast<float>(raw);
#endif
    }

    __host__ __device__ __forceinline__ uint32_t float_to_ordered_uint(float f)
    {
#ifdef __CUDA_ARCH__
        const uint32_t u = __float_as_uint(f);
#else
        const uint32_t u = std::bit_cast<uint32_t>(f);
#endif
        return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    }

    __host__ __device__ __forceinline__ uint32_t longest_axis(const glm::vec3& ext)
    {
        uint32_t axis = 0;
        if (ext.y > ext.x && ext.y >= ext.z)
            axis = 1;
        else if (ext.z > ext.x && ext.z > ext.y)
            axis = 2;
        return axis;
    }

    __host__ __device__ __forceinline__ float half_surface_area(const glm::vec3& bmin, const glm::vec3& bmax)
    {
        const glm::vec3 e = glm::max(bmax - bmin, glm::vec3(0.0f));
        return e.x * e.y + e.y * e.z + e.z * e.x;
    }

    __host__ __device__ inline void get_primitive_aabb_and_centroid(
        soa::soa_view<PrimitiveChunk16> primitives,
        soa::soa_view<Vertex> vertices,
        uint32_t prim_slot_id,
        glm::vec3& out_min,
        glm::vec3& out_max,
        glm::vec3& out_centroid)
    {
        PrimitiveChunk16 chunk0 = primitives[prim_slot_id];
        PrimitiveType ptype = chunk0.type();
        glm::vec3 v0(0.0f), v1(0.0f), v2(0.0f);
        glm::vec3 n0(0.0f), n1(0.0f), n2(0.0f);

        if (ptype == PrimitiveType::Triangle)
        {
            Triangle tri = chunk0.as<Triangle>();
            v0 = glm::vec3(vertices[tri.v0].position);
            v1 = glm::vec3(vertices[tri.v1].position);
            v2 = glm::vec3(vertices[tri.v2].position);
        }
        else if (ptype == PrimitiveType::FullTriangle)
        {
            InlinedTriangle ftri = load_primitive<InlinedTriangle>(primitives, prim_slot_id, chunk0);
            v0 = ftri.v0;
            v1 = ftri.v1;
            v2 = ftri.v2;
        }
        else if (ptype == PrimitiveType::NagataPatch)
        {
            NagataPatchPrimitive patch = load_primitive<NagataPatchPrimitive>(primitives, prim_slot_id, chunk0);
            v0 = glm::vec3(vertices[patch.v0].position);
            v1 = glm::vec3(vertices[patch.v1].position);
            v2 = glm::vec3(vertices[patch.v2].position);
            n0 = glm::vec3(vertices[patch.v0].normal);
            n1 = glm::vec3(vertices[patch.v1].normal);
            n2 = glm::vec3(vertices[patch.v2].normal);
        }
        else if (ptype == PrimitiveType::Subspace)
        {
            SubspacePrimitive sub = load_primitive<SubspacePrimitive>(primitives, prim_slot_id, chunk0);
            v0 = sub.bounds_min;
            v1 = sub.bounds_max;
            v2 = 0.5f * (sub.bounds_min + sub.bounds_max);
        }

        if (ptype == PrimitiveType::NagataPatch)
        {
            NagataPatch::fromTriangle(v0, v1, v2, n0, n1, n2).computeBoundingBox(out_min, out_max);
        }
        else
        {
            out_min = glm::min(glm::min(v0, v1), v2);
            out_max = glm::max(glm::max(v0, v1), v2);
        }

        out_centroid = 0.5f * (out_min + out_max);
        // Prevent zero-thickness slabs on planar triangles, scaling with coordinate magnitude
        // so padding survives FP32 ULP rounding and BVH16 quantization.
        const glm::vec3 abs_max = glm::max(glm::abs(out_min), glm::abs(out_max));
        const glm::vec3 pad = glm::max(glm::vec3(1e-5f), abs_max * 1e-5f);
        out_min -= pad;
        out_max += pad;
    }

    __host__ __device__ __forceinline__ int compute_sah_bin(float c, float cmin, float cext)
    {
        const float scale = static_cast<float>(SAH_BINS) / cext;
        int bin = static_cast<int>(floorf((c - cmin) * scale));
        return bin < 0 ? 0 : (bin >= SAH_BINS ? SAH_BINS - 1 : bin);
    }

    template <typename BinAccessor>
    __host__ __device__ inline void evaluate_bvh_sah_split(
        const glm::vec3& node_bmin,
        const glm::vec3& node_bmax,
        const glm::vec3& cent_bmin,
        const glm::vec3& cent_bmax,
        uint32_t N,
        float traversal_cost,
        float intersection_cost,
        BinAccessor&& get_bin,
        int& out_best_axis,
        float& out_best_split_pos,
        float& out_best_cost)
    {
        const float parent_area = half_surface_area(node_bmin, node_bmax);
        const float inv_parent_area = (parent_area > 1e-12f) ? (1.0f / parent_area) : 0.0f;

        out_best_axis = -1;
        out_best_split_pos = 0.0f;
        out_best_cost = FLT_MAX;

        for (int a = 0; a < 3; ++a)
        {
            const float cext = cent_bmax[a] - cent_bmin[a];
            if (cext < 1e-6f)
                continue;

            uint32_t right_count[SAH_BINS];
            glm::vec3 right_bmin[SAH_BINS];
            glm::vec3 right_bmax[SAH_BINS];

            uint32_t r_cnt = 0;
            glm::vec3 r_min(FLT_MAX);
            glm::vec3 r_max(-FLT_MAX);

            for (int b = SAH_BINS - 1; b >= 0; --b)
            {
                uint32_t cnt = 0;
                glm::vec3 bmin(FLT_MAX), bmax(-FLT_MAX);
                get_bin(a, b, cnt, bmin, bmax);
                if (cnt > 0)
                {
                    r_cnt += cnt;
                    r_min = glm::min(r_min, bmin);
                    r_max = glm::max(r_max, bmax);
                }
                right_count[b] = r_cnt;
                right_bmin[b] = r_min;
                right_bmax[b] = r_max;
            }

            uint32_t l_cnt = 0;
            glm::vec3 l_min(FLT_MAX);
            glm::vec3 l_max(-FLT_MAX);

            for (int k = 1; k < SAH_BINS; ++k)
            {
                uint32_t cnt = 0;
                glm::vec3 bmin(FLT_MAX), bmax(-FLT_MAX);
                get_bin(a, k - 1, cnt, bmin, bmax);
                if (cnt > 0)
                {
                    l_cnt += cnt;
                    l_min = glm::min(l_min, bmin);
                    l_max = glm::max(l_max, bmax);
                }

                const uint32_t n_left = l_cnt;
                const uint32_t n_right = right_count[k];
                if (n_left == 0 || n_right == 0)
                    continue;

                const float area_left = half_surface_area(l_min, l_max);
                const float area_right = half_surface_area(right_bmin[k], right_bmax[k]);
                const float cost = traversal_cost + intersection_cost * inv_parent_area *
                    (area_left * static_cast<float>(n_left) + area_right * static_cast<float>(n_right));

                if (cost < out_best_cost)
                {
                    out_best_cost = cost;
                    out_best_axis = a;
                    out_best_split_pos = cent_bmin[a] + (static_cast<float>(k) / static_cast<float>(SAH_BINS)) * cext;
                }
            }
        }
    }

    struct CpuBuildContext
    {
        const glm::vec3* prim_min = nullptr;
        const glm::vec3* prim_max = nullptr;
        const glm::vec3* prim_centroid = nullptr;
        const uint32_t* prim_slot_ids = nullptr;
        std::vector<uint32_t>* prim_indices = nullptr;
        BVHSplitMode split_mode = BVHSplitMode::SAH;
        int max_depth = 24;
        int max_leaf_primitives = 4;
        float traversal_cost = 1.0f;
        float intersection_cost = 1.5f;
        std::vector<BVHNode>* nodes = nullptr;
        std::vector<std::pair<uint32_t, uint32_t>>* leaf_pairs = nullptr;
        uint32_t total_leaves = 0;
    };

    inline void build_cpu_recursive(
        CpuBuildContext& ctx,
        uint32_t node_idx,
        uint32_t start,
        uint32_t end,
        int depth)
    {
        const uint32_t count = end - start;
        glm::vec3 node_bmin(FLT_MAX), node_bmax(-FLT_MAX);
        glm::vec3 cent_bmin(FLT_MAX), cent_bmax(-FLT_MAX);

        auto& indices = *ctx.prim_indices;
        for (uint32_t i = start; i < end; ++i)
        {
            const uint32_t p = indices[i];
            node_bmin = glm::min(node_bmin, ctx.prim_min[p]);
            node_bmax = glm::max(node_bmax, ctx.prim_max[p]);
            const glm::vec3 c = ctx.prim_centroid[p];
            cent_bmin = glm::min(cent_bmin, c);
            cent_bmax = glm::max(cent_bmax, c);
        }

        auto make_leaf = [&]()
        {
            (*ctx.nodes)[node_idx] = BVHNode::make_leaf(node_bmin, node_bmax, 0u, count);
            ctx.leaf_pairs->reserve(ctx.leaf_pairs->size() + count);
            for (uint32_t i = start; i < end; ++i)
            {
                ctx.leaf_pairs->emplace_back(node_idx, ctx.prim_slot_ids[indices[i]]);
            }
            ++ctx.total_leaves;
        };

        if (count <= static_cast<uint32_t>(ctx.max_leaf_primitives) || depth >= ctx.max_depth)
        {
            make_leaf();
            return;
        }

        const glm::vec3 cent_ext = cent_bmax - cent_bmin;
        uint32_t split_axis = longest_axis(cent_ext);
        uint32_t mid = start + count / 2u;
        bool use_median = false;

        if (ctx.split_mode == BVHSplitMode::SAH)
        {
            BVHSahBinHost bins[3 * SAH_BINS]{};
            for (uint32_t i = start; i < end; ++i)
            {
                const uint32_t p = indices[i];
                const glm::vec3 c = ctx.prim_centroid[p];
                const glm::vec3 pmin = ctx.prim_min[p];
                const glm::vec3 pmax = ctx.prim_max[p];
                for (int a = 0; a < 3; ++a)
                {
                    if (cent_ext[a] < 1e-6f)
                        continue;
                    const int b = compute_sah_bin(c[a], cent_bmin[a], cent_ext[a]);
                    BVHSahBinHost& bin = bins[a * SAH_BINS + b];
                    ++bin.count;
                    bin.bmin = glm::min(bin.bmin, pmin);
                    bin.bmax = glm::max(bin.bmax, pmax);
                }
            }

            int best_axis = -1;
            float best_pos = 0.0f;
            float best_cost = FLT_MAX;
            evaluate_bvh_sah_split(
                node_bmin, node_bmax, cent_bmin, cent_bmax, count,
                ctx.traversal_cost, ctx.intersection_cost,
                [&](int a, int b, uint32_t& cnt, glm::vec3& bmin, glm::vec3& bmax)
                {
                    const BVHSahBinHost& bin = bins[a * SAH_BINS + b];
                    cnt = bin.count;
                    bmin = bin.bmin;
                    bmax = bin.bmax;
                },
                best_axis, best_pos, best_cost);

            const float leaf_cost = ctx.intersection_cost * static_cast<float>(count);
            if (best_axis >= 0 && best_cost >= leaf_cost && count <= static_cast<uint32_t>(ctx.max_leaf_primitives))
            {
                make_leaf();
                return;
            }

            if (best_axis >= 0)
            {
                split_axis = static_cast<uint32_t>(best_axis);
                auto mid_it = std::partition(
                    indices.begin() + start,
                    indices.begin() + end,
                    [&](uint32_t p) { return ctx.prim_centroid[p][split_axis] < best_pos; });
                mid = static_cast<uint32_t>(mid_it - indices.begin());
                if (mid == start || mid == end)
                    use_median = true;
            }
            else
            {
                if (cent_ext[split_axis] < 1e-6f)
                    split_axis = longest_axis(node_bmax - node_bmin);
                use_median = true;
            }
        }
        else if (ctx.split_mode == BVHSplitMode::SpatialMedian)
        {
            if (cent_ext[split_axis] >= 1e-6f)
            {
                const float split_pos = 0.5f * (cent_bmin[split_axis] + cent_bmax[split_axis]);
                auto mid_it = std::partition(
                    indices.begin() + start,
                    indices.begin() + end,
                    [&](uint32_t p) { return ctx.prim_centroid[p][split_axis] < split_pos; });
                mid = static_cast<uint32_t>(mid_it - indices.begin());
                if (mid == start || mid == end)
                    use_median = true;
            }
            else
            {
                split_axis = longest_axis(node_bmax - node_bmin);
                use_median = true;
            }
        }
        else // ObjectMedian
        {
            if (cent_ext[split_axis] < 1e-6f)
                split_axis = longest_axis(node_bmax - node_bmin);
            use_median = true;
        }

        if (use_median)
        {
            mid = start + count / 2u;
            std::nth_element(
                indices.begin() + start,
                indices.begin() + mid,
                indices.begin() + end,
                [&](uint32_t a, uint32_t b)
                {
                    return ctx.prim_centroid[a][split_axis] < ctx.prim_centroid[b][split_axis];
                });
        }

        const uint32_t left_child_idx = static_cast<uint32_t>(ctx.nodes->size());
        ctx.nodes->resize(left_child_idx + 2u);
        (*ctx.nodes)[node_idx] = BVHNode::make_interior(node_bmin, node_bmax, split_axis, left_child_idx, count);

        build_cpu_recursive(ctx, left_child_idx, start, mid, depth + 1);
        build_cpu_recursive(ctx, left_child_idx + 1u, mid, end, depth + 1);
    }
} // namespace bvh_detail

__global__ void compute_bvh_prim_aabbs_kernel(
    soa::soa_view<PrimitiveChunk16> primitives,
    soa::soa_view<Vertex> vertices,
    uint32_t primitives_start,
    uint32_t number_of_triangles,
    uint32_t occupied_slots,
    glm::vec3* prim_min,
    glm::vec3* prim_max,
    glm::vec3* prim_centroid,
    uint32_t* prim_slot_ids)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= number_of_triangles)
        return;

    const uint32_t slot_id = primitives_start + i * occupied_slots;
    glm::vec3 pmin, pmax, pcent;
    bvh_detail::get_primitive_aabb_and_centroid(primitives, vertices, slot_id, pmin, pmax, pcent);

    prim_min[i] = pmin;
    prim_max[i] = pmax;
    prim_centroid[i] = pcent;
    prim_slot_ids[i] = slot_id;
}

__global__ void init_bvh_bins_kernel(
    uint32_t total_bins,
    int32_t init_min_i,
    int32_t init_max_i,
    bvh_detail::BVHSahBinGpu* bins)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_bins)
        return;

    bvh_detail::BVHSahBinGpu bin{};
    bin.count = 0u;
    bin.bmin_i[0] = init_min_i;
    bin.bmin_i[1] = init_min_i;
    bin.bmin_i[2] = init_min_i;
    bin.bmax_i[0] = init_max_i;
    bin.bmax_i[1] = init_max_i;
    bin.bmax_i[2] = init_max_i;
    bin.pad = 0u;
    bins[idx] = bin;
}

__global__ void bin_bvh_prims_kernel(
    uint32_t num_active_prims,
    const uint32_t* prim_ids,
    const uint32_t* prim_node,
    const uint32_t* node_bin_slot,
    const bvh_detail::NodeBoundsAgg* node_bounds,
    const glm::vec3* prim_min,
    const glm::vec3* prim_max,
    const glm::vec3* prim_centroid,
    bvh_detail::BVHSahBinGpu* bins)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_active_prims)
        return;

    const uint32_t u = prim_node[i];
    const uint32_t bin_slot = node_bin_slot[u];
    if (bin_slot == UINT32_MAX)
        return;

    const bvh_detail::NodeBoundsAgg nb = node_bounds[u];
    const glm::vec3 cent_ext = nb.cmax - nb.cmin;

    const uint32_t p = prim_ids[i];
    const glm::vec3 c = prim_centroid[p];
    const glm::vec3 pmin = prim_min[p];
    const glm::vec3 pmax = prim_max[p];

    const int32_t pmin_ix = bvh_detail::float_to_ordered_int(pmin.x);
    const int32_t pmin_iy = bvh_detail::float_to_ordered_int(pmin.y);
    const int32_t pmin_iz = bvh_detail::float_to_ordered_int(pmin.z);
    const int32_t pmax_ix = bvh_detail::float_to_ordered_int(pmax.x);
    const int32_t pmax_iy = bvh_detail::float_to_ordered_int(pmax.y);
    const int32_t pmax_iz = bvh_detail::float_to_ordered_int(pmax.z);

    const uint32_t base = bin_slot * (3u * bvh_detail::SAH_BINS);
    for (int a = 0; a < 3; ++a)
    {
        if (cent_ext[a] < 1e-6f)
            continue;

        const int b = bvh_detail::compute_sah_bin(c[a], nb.cmin[a], cent_ext[a]);
        bvh_detail::BVHSahBinGpu* dst = &bins[base + static_cast<uint32_t>(a * bvh_detail::SAH_BINS + b)];
        atomicAdd(&dst->count, 1u);
        atomicMin(&dst->bmin_i[0], pmin_ix);
        atomicMin(&dst->bmin_i[1], pmin_iy);
        atomicMin(&dst->bmin_i[2], pmin_iz);
        atomicMax(&dst->bmax_i[0], pmax_ix);
        atomicMax(&dst->bmax_i[1], pmax_iy);
        atomicMax(&dst->bmax_i[2], pmax_iz);
    }
}

__global__ void decide_bvh_splits_kernel(
    uint32_t num_active,
    int depth,
    BVHSplitMode split_mode,
    int max_depth,
    int max_leaf_primitives,
    float traversal_cost,
    float intersection_cost,
    const bvh_detail::NodeBoundsAgg* node_bounds,
    const uint32_t* node_counts,
    const uint32_t* node_bin_slot,
    const bvh_detail::BVHSahBinGpu* bins,
    uint8_t* is_split,
    uint32_t* split_axis,
    float* split_pos)
{
    const uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_active)
        return;

    const uint32_t count = node_counts[u];
    if (depth >= max_depth || count <= static_cast<uint32_t>(max_leaf_primitives))
    {
        is_split[u] = 0;
        split_axis[u] = 0;
        split_pos[u] = 0.0f;
        return;
    }

    const bvh_detail::NodeBoundsAgg nb = node_bounds[u];
    const glm::vec3 cent_ext = nb.cmax - nb.cmin;
    const glm::vec3 node_ext = nb.bmax - nb.bmin;
    uint32_t axis = bvh_detail::longest_axis(cent_ext);

    if (split_mode == BVHSplitMode::SAH && bins != nullptr && node_bin_slot[u] != UINT32_MAX)
    {
        const bvh_detail::BVHSahBinGpu* node_bins = bins + node_bin_slot[u] * (3u * bvh_detail::SAH_BINS);
        int best_axis = -1;
        float best_pos = 0.0f;
        float best_cost = FLT_MAX;
        bvh_detail::evaluate_bvh_sah_split(
            nb.bmin, nb.bmax, nb.cmin, nb.cmax, count,
            traversal_cost, intersection_cost,
            [&](int a, int b, uint32_t& cnt, glm::vec3& bmin, glm::vec3& bmax)
            {
                const bvh_detail::BVHSahBinGpu& bin = node_bins[a * bvh_detail::SAH_BINS + b];
                cnt = bin.count;
                if (cnt > 0)
                {
                    bmin = glm::vec3(
                        bvh_detail::ordered_int_to_float(bin.bmin_i[0]),
                        bvh_detail::ordered_int_to_float(bin.bmin_i[1]),
                        bvh_detail::ordered_int_to_float(bin.bmin_i[2]));
                    bmax = glm::vec3(
                        bvh_detail::ordered_int_to_float(bin.bmax_i[0]),
                        bvh_detail::ordered_int_to_float(bin.bmax_i[1]),
                        bvh_detail::ordered_int_to_float(bin.bmax_i[2]));
                }
            },
            best_axis, best_pos, best_cost);

        const float leaf_cost = intersection_cost * static_cast<float>(count);
        if (best_axis >= 0 && best_cost >= leaf_cost && count <= static_cast<uint32_t>(max_leaf_primitives))
        {
            is_split[u] = 0;
            split_axis[u] = 0;
            split_pos[u] = 0.0f;
            return;
        }

        if (best_axis >= 0)
        {
            is_split[u] = 1;
            split_axis[u] = static_cast<uint32_t>(best_axis);
            split_pos[u] = best_pos;
            return;
        }

        if (cent_ext[axis] < 1e-6f)
            axis = bvh_detail::longest_axis(node_ext);
        is_split[u] = 1;
        split_axis[u] = axis;
        split_pos[u] = FLT_MAX; // Trigger object-median fallback
        return;
    }

    if (split_mode == BVHSplitMode::SpatialMedian)
    {
        if (cent_ext[axis] >= 1e-6f)
        {
            is_split[u] = 1;
            split_axis[u] = axis;
            split_pos[u] = 0.5f * (nb.cmin[axis] + nb.cmax[axis]);
        }
        else
        {
            is_split[u] = 1;
            split_axis[u] = bvh_detail::longest_axis(node_ext);
            split_pos[u] = FLT_MAX; // Trigger object-median fallback
        }
        return;
    }

    // ObjectMedian
    if (cent_ext[axis] < 1e-6f)
        axis = bvh_detail::longest_axis(node_ext);
    is_split[u] = 1;
    split_axis[u] = axis;
    split_pos[u] = 0.0f;
}

__global__ void mark_bvh_leaf_prims_kernel(
    uint32_t num_active_prims,
    const uint32_t* prim_node,
    const uint8_t* is_split,
    uint32_t* leaf_flags)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_active_prims)
        return;

    leaf_flags[i] = (is_split[prim_node[i]] == 0) ? 1u : 0u;
}

__global__ void scatter_bvh_leaf_prims_kernel(
    uint32_t num_active_prims,
    const uint32_t* prim_ids,
    const uint32_t* prim_node,
    const uint32_t* active_node_idx,
    const uint32_t* prim_slot_ids,
    const uint32_t* leaf_flags,
    const uint32_t* leaf_offsets,
    uint32_t* out_leaf_nodes,
    uint32_t* out_leaf_prims)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_active_prims || leaf_flags[i] == 0u)
        return;

    const uint32_t dst = leaf_offsets[i];
    out_leaf_nodes[dst] = active_node_idx[prim_node[i]];
    out_leaf_prims[dst] = prim_slot_ids[prim_ids[i]];
}

__global__ void compute_bvh_split_sides_kernel(
    uint32_t num_active_prims,
    const uint32_t* prim_ids,
    const uint32_t* prim_node,
    const glm::vec3* prim_centroid,
    const uint8_t* is_split,
    const uint32_t* split_axis,
    const float* split_pos,
    uint8_t* out_side,
    uint32_t* out_left_counts)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_active_prims)
        return;

    const uint32_t u = prim_node[i];
    if (is_split[u] == 0)
    {
        out_side[i] = 2u;
        return;
    }

    const uint32_t axis = split_axis[u];
    const float c = prim_centroid[prim_ids[i]][axis];
    const uint8_t s = (c < split_pos[u]) ? 0u : 1u;
    out_side[i] = s;
    if (s == 0u)
    {
        atomicAdd(&out_left_counts[u], 1u);
    }
}

__global__ void mark_bvh_median_prims_kernel(
    uint32_t num_active_prims,
    const uint32_t* prim_node,
    const uint8_t* needs_median,
    uint32_t* out_flags)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_active_prims)
        return;

    out_flags[i] = (needs_median[prim_node[i]] != 0) ? 1u : 0u;
}

__global__ void build_bvh_median_keys_kernel(
    uint32_t num_active_prims,
    const uint32_t* prim_ids,
    const uint32_t* prim_node,
    const glm::vec3* prim_centroid,
    const uint32_t* split_axis,
    const uint32_t* med_flags,
    const uint32_t* med_offsets,
    uint64_t* out_keys,
    uint32_t* out_prims)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_active_prims || med_flags[i] == 0u)
        return;

    const uint32_t u = prim_node[i];
    const uint32_t p = prim_ids[i];
    const float c = prim_centroid[p][split_axis[u]];
    const uint32_t dst = med_offsets[i];

    out_keys[dst] = (static_cast<uint64_t>(u) << 32) | static_cast<uint64_t>(bvh_detail::float_to_ordered_uint(c));
    out_prims[dst] = p;
}

__global__ void apply_bvh_median_split_kernel(
    uint32_t num_median_prims,
    const uint64_t* sorted_keys,
    const uint32_t* sorted_prims,
    const uint32_t* node_starts,
    const uint32_t* node_counts,
    const uint32_t* med_node_offsets,
    uint32_t* prim_ids,
    uint8_t* sides)
{
    const uint32_t m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= num_median_prims)
        return;

    const uint32_t u = static_cast<uint32_t>(sorted_keys[m] >> 32);
    const uint32_t idx_in_node = m - med_node_offsets[u];
    const uint32_t dst = node_starts[u] + idx_in_node;

    prim_ids[dst] = sorted_prims[m];
    sides[dst] = (idx_in_node < (node_counts[u] / 2u)) ? 0u : 1u;
}

__global__ void scatter_bvh_child_prims_kernel(
    uint32_t num_active_prims,
    const uint32_t* prim_ids,
    const uint32_t* prim_node,
    const uint8_t* is_split,
    const uint32_t* split_rank,
    const uint8_t* sides,
    uint32_t* child_cursors,
    uint32_t* next_prim_ids,
    uint32_t* next_prim_node)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_active_prims)
        return;

    const uint32_t u = prim_node[i];
    if (is_split[u] == 0)
        return;

    const uint32_t c = 2u * split_rank[u] + static_cast<uint32_t>(sides[i]);
    const uint32_t dst = atomicAdd(&child_cursors[c], 1u);
    next_prim_ids[dst] = prim_ids[i];
    next_prim_node[dst] = c;
}

BVHAS::BVHAS(GeometryContext* geometry_context)
    : m_geometry_context(geometry_context)
{
    // Reserve element 0 in pools as sentinel (0 = empty)
    as.alloc(1);
    as[0] = BVHNode{};
    as16.alloc(1);
    as16[0] = BVHNode16{};
    box_content.alloc(1);
    sync_to_device();
}

void BVHAS::config_ui()
{
    const char* split_mode_names[] = {"SAH (16 bins)", "Spatial Median", "Object Median"};
    int split_idx = static_cast<int>(config.split_mode);
    if (ImGui::Combo("Split mode", &split_idx, split_mode_names, 3))
    {
        config.split_mode = static_cast<BVHSplitMode>(split_idx);
    }

    ImGui::DragInt("Max depth", &config.max_depth, 0.25f, 1, 30);
    if (config.max_depth < 1) config.max_depth = 1;
    if (config.max_depth > 30) config.max_depth = 30;

    ImGui::DragInt("Max leaf primitives", &config.max_leaf_primitives, 0.5f, 1, 128);
    if (config.max_leaf_primitives < 1) config.max_leaf_primitives = 1;

    if (config.split_mode == BVHSplitMode::SAH)
    {
        ImGui::DragFloat("Traversal cost", &config.traversal_cost, 0.05f, 0.01f, 100.0f);
        ImGui::DragFloat("Intersection cost", &config.intersection_cost, 0.05f, 0.01f, 100.0f);
    }

    ImGui::Checkbox("Build on GPU", &config.use_gpu);
    ImGui::Checkbox("Use 16B quantized nodes (BVH16)", &config.use_bvh16);
}

void BVHAS::info_ui(uint32_t space_id, Space space)
{
    auto it = m_build_times_ms.find(space_id);
    if (it != m_build_times_ms.end())
    {
        ImGui::Text("  Build Time: %.2f ms", it->second);
    }

    uint32_t num_leaves = 0;
    bool is_16b = false;
    if (space.root_box >= 0)
    {
        auto tree = trees[static_cast<size_t>(space.root_box)];
        num_leaves = tree.num_leaves;
        is_16b = (tree.use_bvh16 != 0u);
        ImGui::Text("  Format:     %s", is_16b ? "BVH16 (16B Quantized)" : "BVH32 (32B FP32)");
        ImGui::Text("  Bounds Min: (%.3f, %.3f, %.3f)", tree.bounds_min.x, tree.bounds_min.y, tree.bounds_min.z);
        ImGui::Text("  Bounds Max: (%.3f, %.3f, %.3f)", tree.bounds_max.x, tree.bounds_max.y, tree.bounds_max.z);
    }

    const size_t node_stride = is_16b ? sizeof(BVHNode16) : sizeof(BVHNode);
    const size_t nodes_bytes = static_cast<size_t>(space.as_size) * node_stride;
    const size_t content_bytes = static_cast<size_t>(space.box_content_size) * sizeof(BoxContent);
    const size_t tree_meta_bytes = (space.root_box >= 0) ? (sizeof(glm::vec3) * 2 + sizeof(uint32_t) * 3) : 0;
    const size_t as_bytes = nodes_bytes + content_bytes + tree_meta_bytes;
    const size_t geom_bytes = space_geometry_bytes(space);
    const size_t total_bytes = as_bytes + geom_bytes;

    ImGui::Text("  Nodes:      %u (%u leaves)", static_cast<uint32_t>(space.as_size), num_leaves);
    ImGui::Text("  Contents:   %u", static_cast<uint32_t>(space.box_content_size));
    ImGui::Text("  Primitives: %u slots", static_cast<uint32_t>(space.primitives_size));
    ImGui::Text("  Vertices:   %u", static_cast<uint32_t>(space.vertices_size));
    ImGui::Text("  AS Memory:  %s", format_bytes(as_bytes).c_str());
    ImGui::Text("    Nodes:    %s", format_bytes(nodes_bytes).c_str());
    ImGui::Text("    Contents: %s", format_bytes(content_bytes).c_str());
    ImGui::Text("  Geo Memory: %s", format_bytes(geom_bytes).c_str());
    ImGui::Text("  Total Size: %s", format_bytes(total_bytes).c_str());
}

void BVHAS::free_space(uint32_t space_id, Space space)
{
    m_build_times_ms.erase(space_id);
    if (space.as_type != ASType::BVH)
        return;

    bool is_16b = false;
    if (space.root_box >= 0)
    {
        is_16b = (trees[static_cast<size_t>(space.root_box)].use_bvh16 != 0u);
        trees.free(static_cast<size_t>(space.root_box));
    }
    if (space.as_size > 0)
    {
        if (is_16b)
        {
            as16.free(space.as_start);
        }
        else
        {
            as.free(space.as_start);
        }
    }
    if (space.box_content_size > 0)
    {
        box_content.free(space.box_content_start);
    }
    sync_to_device();
}

void BVHAS::finalize_tree_pools(
    Space space,
    glm::vec3 bounds_min,
    glm::vec3 bounds_max,
    std::vector<BVHNode>& host_nodes,
    std::vector<std::pair<uint32_t, uint32_t>>& leaf_pairs,
    uint32_t total_leaves)
{
    std::sort(leaf_pairs.begin(), leaf_pairs.end());

    const uint32_t tree_id = static_cast<uint32_t>(trees.alloc_element());
    const uint32_t total_nodes = static_cast<uint32_t>(host_nodes.size());
    const bool use_16b = config.use_bvh16;
    // Keep every BVH16 tree allocation an even number of nodes (starting after the 1-node sentinel)
    // so every tree root begins at an odd index and all sibling pairs align to 32-byte boundaries.
    const uint32_t as16_alloc_count = (total_nodes + 1u) & ~1u;
    const uint32_t as_start = use_16b
        ? static_cast<uint32_t>(as16.alloc(as16_alloc_count))
        : static_cast<uint32_t>(as.alloc(total_nodes));
    assert(!use_16b || ((as_start & 1u) == 1u));
    const uint32_t total_pairs = static_cast<uint32_t>(leaf_pairs.size());
    const uint32_t bc_start = static_cast<uint32_t>(box_content.alloc(total_pairs + 1u));

    for (size_t i = 0; i < host_nodes.size(); ++i)
    {
        if (!host_nodes[i].is_leaf())
        {
            host_nodes[i] = BVHNode::make_interior(
                host_nodes[i].bounds_min,
                host_nodes[i].bounds_max,
                host_nodes[i].axis(),
                as_start + host_nodes[i].left_child(),
                host_nodes[i].prim_count);
        }
    }

    box_content[bc_start] = BoxContent{};
    for (uint32_t i = 0; i < total_pairs; ++i)
    {
        const uint32_t ln = leaf_pairs[i].first;
        const uint32_t pid = leaf_pairs[i].second;
        const bool is_first = (i == 0) || (leaf_pairs[i - 1].first != ln);
        const bool is_last = (i + 1u == total_pairs) || (leaf_pairs[i + 1u].first != ln);
        const uint32_t bc_idx = bc_start + 1u + i;
        box_content[bc_idx] = BoxContent(pid, is_last);
        if (is_first)
        {
            host_nodes[ln] = BVHNode::make_leaf(
                host_nodes[ln].bounds_min,
                host_nodes[ln].bounds_max,
                bc_idx,
                host_nodes[ln].prim_count);
        }
    }

    if (use_16b)
    {
        for (uint32_t i = 0; i < total_nodes; ++i)
        {
            as16[as_start + i] = BVHNode16::from_node(host_nodes[i], bounds_min, bounds_max);
        }
        if (as16_alloc_count > total_nodes)
        {
            as16[as_start + total_nodes] = BVHNode16{};
        }
    }
    else
    {
        for (uint32_t i = 0; i < total_nodes; ++i)
        {
            as[as_start + i] = host_nodes[i];
        }
    }

    trees[tree_id].bounds_min = bounds_min;
    trees[tree_id].bounds_max = bounds_max;
    trees[tree_id].as_start = as_start;
    trees[tree_id].num_leaves = total_leaves;
    trees[tree_id].use_bvh16 = use_16b ? 1u : 0u;

    space.root_box = static_cast<int>(tree_id);
    space.as_start = as_start;
    space.as_size = total_nodes;
    space.box_content_start = bc_start;
    space.box_content_size = total_pairs + 1u;

    sync_to_device();
}

void BVHAS::build_gpu(uint32_t /*space_id*/, Space space, uint32_t number_of_triangles, uint32_t occupied_slots)
{
    if (number_of_triangles == 0)
        return;

    constexpr uint32_t block_size = 256;
    const uint32_t tri_blocks = (number_of_triangles - 1u) / block_size + 1u;

    thrust::device_vector<glm::vec3> d_prim_min(number_of_triangles);
    thrust::device_vector<glm::vec3> d_prim_max(number_of_triangles);
    thrust::device_vector<glm::vec3> d_prim_centroid(number_of_triangles);
    thrust::device_vector<uint32_t> d_prim_slot_ids(number_of_triangles);

    compute_bvh_prim_aabbs_kernel<<<tri_blocks, block_size>>>(
        m_geometry_context->primitives.view(),
        m_geometry_context->vertices.view(),
        space.primitives_start,
        number_of_triangles,
        occupied_slots,
        d_prim_min.data().get(),
        d_prim_max.data().get(),
        d_prim_centroid.data().get(),
        d_prim_slot_ids.data().get());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<BVHNode> host_nodes(1);
    std::vector<std::pair<uint32_t, uint32_t>> leaf_pairs;
    leaf_pairs.reserve(number_of_triangles);
    uint32_t total_leaves = 0;

    thrust::device_vector<uint32_t> d_prim_ids(number_of_triangles);
    thrust::sequence(thrust::device, d_prim_ids.begin(), d_prim_ids.end(), 0u);
    thrust::device_vector<uint32_t> d_prim_node(number_of_triangles, 0u);

    std::vector<uint32_t> h_active_node_idx(1, 0u);
    std::vector<uint32_t> h_node_starts(1, 0u);
    std::vector<uint32_t> h_node_counts(1, number_of_triangles);

    const int max_depth = config.max_depth < 1 ? 1 : (config.max_depth > 30 ? 30 : config.max_depth);
    const int max_leaf_prims = std::max(config.max_leaf_primitives, 1);

    glm::vec3 root_min(0.0f);
    glm::vec3 root_max(0.0f);

    const int32_t init_min_i = bvh_detail::float_to_ordered_int(FLT_MAX);
    const int32_t init_max_i = bvh_detail::float_to_ordered_int(-FLT_MAX);

    for (int depth = 0; depth <= max_depth; ++depth)
    {
        const uint32_t num_active = static_cast<uint32_t>(h_active_node_idx.size());
        const uint32_t num_active_prims = static_cast<uint32_t>(d_prim_ids.size());
        if (num_active == 0 || num_active_prims == 0)
            break;

        const thrust::device_vector<uint32_t> d_active_node_idx(h_active_node_idx.begin(), h_active_node_idx.end());
        const thrust::device_vector<uint32_t> d_node_starts(h_node_starts.begin(), h_node_starts.end());
        const thrust::device_vector<uint32_t> d_node_counts(h_node_counts.begin(), h_node_counts.end());

        // Compute per-node tight AABBs and centroid AABBs in one segmented reduction
        thrust::device_vector<uint32_t> d_out_keys(num_active);
        thrust::device_vector<bvh_detail::NodeBoundsAgg> d_node_bounds(num_active);
        auto agg_begin = thrust::make_transform_iterator(
            d_prim_ids.begin(),
            bvh_detail::PrimToAggFunctor{
                d_prim_min.data().get(),
                d_prim_max.data().get(),
                d_prim_centroid.data().get()
            });
        thrust::reduce_by_key(
            thrust::device,
            d_prim_node.begin(),
            d_prim_node.end(),
            agg_begin,
            d_out_keys.begin(),
            d_node_bounds.begin(),
            thrust::equal_to<uint32_t>{},
            bvh_detail::CombineBoundsAggFunctor{});

        std::vector<bvh_detail::NodeBoundsAgg> h_node_bounds(num_active);
        thrust::copy(d_node_bounds.begin(), d_node_bounds.end(), h_node_bounds.begin());
        if (depth == 0)
        {
            root_min = h_node_bounds[0].bmin;
            root_max = h_node_bounds[0].bmax;
        }

        // Assign compact SAH bin slots only to active nodes that are eligible to split
        std::vector<uint32_t> h_node_bin_slot(num_active, UINT32_MAX);
        uint32_t num_bin_nodes = 0;
        if (depth < max_depth && config.split_mode == BVHSplitMode::SAH)
        {
            for (uint32_t u = 0; u < num_active; ++u)
            {
                if (h_node_counts[u] > static_cast<uint32_t>(max_leaf_prims))
                {
                    h_node_bin_slot[u] = num_bin_nodes++;
                }
            }
        }

        const thrust::device_vector<uint32_t> d_node_bin_slot(h_node_bin_slot.begin(), h_node_bin_slot.end());
        thrust::device_vector<bvh_detail::BVHSahBinGpu> d_bins;
        const uint32_t active_prim_blocks = (num_active_prims - 1u) / block_size + 1u;

        if (num_bin_nodes > 0)
        {
            const uint32_t total_bins = num_bin_nodes * (3u * bvh_detail::SAH_BINS);
            d_bins.resize(total_bins);
            const uint32_t bin_blocks = (total_bins - 1u) / block_size + 1u;
            init_bvh_bins_kernel<<<bin_blocks, block_size>>>(
                total_bins, init_min_i, init_max_i, d_bins.data().get());

            bin_bvh_prims_kernel<<<active_prim_blocks, block_size>>>(
                num_active_prims,
                d_prim_ids.data().get(),
                d_prim_node.data().get(),
                d_node_bin_slot.data().get(),
                d_node_bounds.data().get(),
                d_prim_min.data().get(),
                d_prim_max.data().get(),
                d_prim_centroid.data().get(),
                d_bins.data().get());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        thrust::device_vector<uint8_t> d_is_split(num_active, 0);
        thrust::device_vector<uint32_t> d_split_axis(num_active, 0);
        thrust::device_vector<float> d_split_pos(num_active, 0.0f);

        const uint32_t node_blocks = (num_active - 1u) / block_size + 1u;
        decide_bvh_splits_kernel<<<node_blocks, block_size>>>(
            num_active,
            depth,
            config.split_mode,
            max_depth,
            max_leaf_prims,
            config.traversal_cost,
            config.intersection_cost,
            d_node_bounds.data().get(),
            d_node_counts.data().get(),
            d_node_bin_slot.data().get(),
            d_bins.empty() ? nullptr : d_bins.data().get(),
            d_is_split.data().get(),
            d_split_axis.data().get(),
            d_split_pos.data().get());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<uint8_t> h_is_split(num_active);
        std::vector<uint32_t> h_split_axis(num_active);
        std::vector<float> h_split_pos(num_active);
        thrust::copy(d_is_split.begin(), d_is_split.end(), h_is_split.begin());
        thrust::copy(d_split_axis.begin(), d_split_axis.end(), h_split_axis.begin());
        thrust::copy(d_split_pos.begin(), d_split_pos.end(), h_split_pos.begin());

        const uint32_t next_node_offset = static_cast<uint32_t>(host_nodes.size());
        std::vector<uint32_t> h_split_rank(num_active, 0u);
        uint32_t num_splits = 0;

        for (uint32_t u = 0; u < num_active; ++u)
        {
            const uint32_t node_idx = h_active_node_idx[u];
            const glm::vec3 bmin = h_node_bounds[u].bmin;
            const glm::vec3 bmax = h_node_bounds[u].bmax;
            const uint32_t cnt = h_node_counts[u];

            if (h_is_split[u] != 0)
            {
                h_split_rank[u] = num_splits++;
                host_nodes[node_idx] = BVHNode::make_interior(
                    bmin, bmax, h_split_axis[u], next_node_offset + 2u * h_split_rank[u], cnt);
            }
            else
            {
                ++total_leaves;
                host_nodes[node_idx] = BVHNode::make_leaf(bmin, bmax, 0u, cnt);
            }
        }

        if (num_splits > 0)
        {
            host_nodes.resize(next_node_offset + 2u * num_splits);
        }

        // Collect finished leaf primitive pairs at this level
        if (num_splits < num_active)
        {
            thrust::device_vector<uint32_t> d_leaf_flags(num_active_prims, 0u);
            thrust::device_vector<uint32_t> d_leaf_offsets(num_active_prims, 0u);
            mark_bvh_leaf_prims_kernel<<<active_prim_blocks, block_size>>>(
                num_active_prims,
                d_prim_node.data().get(),
                d_is_split.data().get(),
                d_leaf_flags.data().get());
            CUDA_CHECK(cudaDeviceSynchronize());

            thrust::exclusive_scan(d_leaf_flags.begin(), d_leaf_flags.end(), d_leaf_offsets.begin());
            const uint32_t last_flag = d_leaf_flags[num_active_prims - 1u];
            const uint32_t last_off = d_leaf_offsets[num_active_prims - 1u];
            const uint32_t level_leaf_prims = last_off + last_flag;

            if (level_leaf_prims > 0)
            {
                thrust::device_vector<uint32_t> d_level_leaf_nodes(level_leaf_prims);
                thrust::device_vector<uint32_t> d_level_leaf_prims(level_leaf_prims);
                scatter_bvh_leaf_prims_kernel<<<active_prim_blocks, block_size>>>(
                    num_active_prims,
                    d_prim_ids.data().get(),
                    d_prim_node.data().get(),
                    d_active_node_idx.data().get(),
                    d_prim_slot_ids.data().get(),
                    d_leaf_flags.data().get(),
                    d_leaf_offsets.data().get(),
                    d_level_leaf_nodes.data().get(),
                    d_level_leaf_prims.data().get());
                CUDA_CHECK(cudaDeviceSynchronize());

                std::vector<uint32_t> h_leaf_nodes(level_leaf_prims);
                std::vector<uint32_t> h_leaf_prims(level_leaf_prims);
                thrust::copy(d_level_leaf_nodes.begin(), d_level_leaf_nodes.end(), h_leaf_nodes.begin());
                thrust::copy(d_level_leaf_prims.begin(), d_level_leaf_prims.end(), h_leaf_prims.begin());

                for (uint32_t k = 0; k < level_leaf_prims; ++k)
                {
                    leaf_pairs.emplace_back(h_leaf_nodes[k], h_leaf_prims[k]);
                }
            }
        }

        if (num_splits == 0)
            break;

        // Partition splitting nodes into left / right sides
        thrust::device_vector<uint8_t> d_sides(num_active_prims, 0u);
        std::vector<uint32_t> h_left_counts(num_active, 0u);
        std::vector<uint8_t> h_needs_median(num_active, 0u);
        uint32_t num_median_prims = 0;

        if (config.split_mode == BVHSplitMode::ObjectMedian)
        {
            for (uint32_t u = 0; u < num_active; ++u)
            {
                if (h_is_split[u] != 0)
                {
                    h_needs_median[u] = 1u;
                    h_left_counts[u] = h_node_counts[u] / 2u;
                    num_median_prims += h_node_counts[u];
                }
            }
        }
        else
        {
            thrust::device_vector<uint32_t> d_left_counts(num_active, 0u);
            compute_bvh_split_sides_kernel<<<active_prim_blocks, block_size>>>(
                num_active_prims,
                d_prim_ids.data().get(),
                d_prim_node.data().get(),
                d_prim_centroid.data().get(),
                d_is_split.data().get(),
                d_split_axis.data().get(),
                d_split_pos.data().get(),
                d_sides.data().get(),
                d_left_counts.data().get());
            CUDA_CHECK(cudaDeviceSynchronize());

            thrust::copy(d_left_counts.begin(), d_left_counts.end(), h_left_counts.begin());
            for (uint32_t u = 0; u < num_active; ++u)
            {
                if (h_is_split[u] != 0 &&
                    (h_left_counts[u] == 0u || h_left_counts[u] == h_node_counts[u]))
                {
                    h_needs_median[u] = 1u;
                    h_left_counts[u] = h_node_counts[u] / 2u;
                    num_median_prims += h_node_counts[u];
                }
            }
        }

        if (num_median_prims > 0)
        {
            std::vector<uint32_t> h_med_node_offsets(num_active, 0u);
            uint32_t running_med = 0u;
            for (uint32_t u = 0; u < num_active; ++u)
            {
                h_med_node_offsets[u] = running_med;
                if (h_needs_median[u] != 0)
                {
                    running_med += h_node_counts[u];
                }
            }

            const thrust::device_vector<uint8_t> d_needs_median(h_needs_median.begin(), h_needs_median.end());
            const thrust::device_vector<uint32_t> d_med_node_offsets(h_med_node_offsets.begin(), h_med_node_offsets.end());
            thrust::device_vector<uint32_t> d_med_flags(num_active_prims, 0u);
            thrust::device_vector<uint32_t> d_med_offsets(num_active_prims, 0u);

            mark_bvh_median_prims_kernel<<<active_prim_blocks, block_size>>>(
                num_active_prims,
                d_prim_node.data().get(),
                d_needs_median.data().get(),
                d_med_flags.data().get());
            CUDA_CHECK(cudaDeviceSynchronize());

            thrust::exclusive_scan(d_med_flags.begin(), d_med_flags.end(), d_med_offsets.begin());

            thrust::device_vector<uint64_t> d_med_keys(num_median_prims);
            thrust::device_vector<uint32_t> d_med_prims(num_median_prims);
            build_bvh_median_keys_kernel<<<active_prim_blocks, block_size>>>(
                num_active_prims,
                d_prim_ids.data().get(),
                d_prim_node.data().get(),
                d_prim_centroid.data().get(),
                d_split_axis.data().get(),
                d_med_flags.data().get(),
                d_med_offsets.data().get(),
                d_med_keys.data().get(),
                d_med_prims.data().get());
            CUDA_CHECK(cudaDeviceSynchronize());

            thrust::sort_by_key(thrust::device, d_med_keys.begin(), d_med_keys.end(), d_med_prims.begin());

            const uint32_t med_blocks = (num_median_prims - 1u) / block_size + 1u;
            apply_bvh_median_split_kernel<<<med_blocks, block_size>>>(
                num_median_prims,
                d_med_keys.data().get(),
                d_med_prims.data().get(),
                d_node_starts.data().get(),
                d_node_counts.data().get(),
                d_med_node_offsets.data().get(),
                d_prim_ids.data().get(),
                d_sides.data().get());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Build contiguous child slices for next level
        const uint32_t num_children = 2u * num_splits;
        std::vector<uint32_t> h_next_node_idx(num_children);
        std::vector<uint32_t> h_next_starts(num_children);
        std::vector<uint32_t> h_next_counts(num_children);

        uint32_t total_next_prims = 0u;
        for (uint32_t u = 0; u < num_active; ++u)
        {
            if (h_is_split[u] == 0)
                continue;

            const uint32_t left_c = 2u * h_split_rank[u];
            const uint32_t right_c = left_c + 1u;
            const uint32_t l_cnt = h_left_counts[u];
            const uint32_t r_cnt = h_node_counts[u] - l_cnt;

            h_next_node_idx[left_c] = next_node_offset + left_c;
            h_next_counts[left_c] = l_cnt;
            h_next_starts[left_c] = total_next_prims;
            total_next_prims += l_cnt;

            h_next_node_idx[right_c] = next_node_offset + right_c;
            h_next_counts[right_c] = r_cnt;
            h_next_starts[right_c] = total_next_prims;
            total_next_prims += r_cnt;
        }

        const thrust::device_vector<uint32_t> d_split_rank(h_split_rank.begin(), h_split_rank.end());
        thrust::device_vector<uint32_t> d_child_cursors(h_next_starts.begin(), h_next_starts.end());
        thrust::device_vector<uint32_t> d_next_prim_ids(total_next_prims);
        thrust::device_vector<uint32_t> d_next_prim_node(total_next_prims);

        scatter_bvh_child_prims_kernel<<<active_prim_blocks, block_size>>>(
            num_active_prims,
            d_prim_ids.data().get(),
            d_prim_node.data().get(),
            d_is_split.data().get(),
            d_split_rank.data().get(),
            d_sides.data().get(),
            d_child_cursors.data().get(),
            d_next_prim_ids.data().get(),
            d_next_prim_node.data().get());
        CUDA_CHECK(cudaDeviceSynchronize());

        d_prim_ids = std::move(d_next_prim_ids);
        d_prim_node = std::move(d_next_prim_node);
        h_active_node_idx = std::move(h_next_node_idx);
        h_node_starts = std::move(h_next_starts);
        h_node_counts = std::move(h_next_counts);
    }

    finalize_tree_pools(space, root_min, root_max, host_nodes, leaf_pairs, total_leaves);
}

void BVHAS::build_cpu(uint32_t /*space_id*/, Space space, uint32_t number_of_triangles, uint32_t occupied_slots)
{
    if (number_of_triangles == 0)
        return;

    soa::soa_vector<PrimitiveChunk16> host_primitives = m_geometry_context->primitives;
    soa::soa_vector<Vertex> host_vertices = m_geometry_context->vertices;
    const auto prim_view = host_primitives.view();
    const auto vert_view = host_vertices.view();

    std::vector<glm::vec3> prim_min(number_of_triangles);
    std::vector<glm::vec3> prim_max(number_of_triangles);
    std::vector<glm::vec3> prim_centroid(number_of_triangles);
    std::vector<uint32_t> prim_slot_ids(number_of_triangles);

    glm::vec3 root_min(FLT_MAX);
    glm::vec3 root_max(-FLT_MAX);
    for (uint32_t i = 0; i < number_of_triangles; ++i)
    {
        const uint32_t slot_id = space.primitives_start + i * occupied_slots;
        prim_slot_ids[i] = slot_id;

        bvh_detail::get_primitive_aabb_and_centroid(
            prim_view, vert_view, slot_id, prim_min[i], prim_max[i], prim_centroid[i]);

        root_min = glm::min(root_min, prim_min[i]);
        root_max = glm::max(root_max, prim_max[i]);
    }

    std::vector<uint32_t> prim_indices(number_of_triangles);
    std::iota(prim_indices.begin(), prim_indices.end(), 0u);

    std::vector<BVHNode> host_nodes(1);
    std::vector<std::pair<uint32_t, uint32_t>> leaf_pairs;
    leaf_pairs.reserve(number_of_triangles);

    bvh_detail::CpuBuildContext ctx{
        .prim_min = prim_min.data(),
        .prim_max = prim_max.data(),
        .prim_centroid = prim_centroid.data(),
        .prim_slot_ids = prim_slot_ids.data(),
        .prim_indices = &prim_indices,
        .split_mode = config.split_mode,
        .max_depth = config.max_depth < 1 ? 1 : (config.max_depth > 30 ? 30 : config.max_depth),
        .max_leaf_primitives = std::max(config.max_leaf_primitives, 1),
        .traversal_cost = config.traversal_cost,
        .intersection_cost = config.intersection_cost,
        .nodes = &host_nodes,
        .leaf_pairs = &leaf_pairs,
        .total_leaves = 0
    };

    bvh_detail::build_cpu_recursive(ctx, 0u, 0u, number_of_triangles, 0);
    finalize_tree_pools(space, root_min, root_max, host_nodes, leaf_pairs, ctx.total_leaves);
}

void BVHAS::build(uint32_t space_id, Space space, PrimitiveType primitive_type)
{
    assert(m_geometry_context != nullptr);
    auto t0 = std::chrono::high_resolution_clock::now();

    uint32_t occupied_slots = Triangle::occupied_slots;
    if (primitive_type == PrimitiveType::FullTriangle)
        occupied_slots = InlinedTriangle::occupied_slots;
    else if (primitive_type == PrimitiveType::NagataPatch)
        occupied_slots = NagataPatchPrimitive::occupied_slots;
    else if (primitive_type == PrimitiveType::Subspace)
        occupied_slots = SubspacePrimitive::occupied_slots;

    const uint32_t number_of_triangles = space.primitives_size / occupied_slots;
    if (number_of_triangles == 0)
        return;

    if (config.use_gpu)
    {
        build_gpu(space_id, space, number_of_triangles, occupied_slots);
    }
    else
    {
        build_cpu(space_id, space, number_of_triangles, occupied_slots);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    m_build_times_ms[space_id] = std::chrono::duration<double, std::milli>(t1 - t0).count();
}
