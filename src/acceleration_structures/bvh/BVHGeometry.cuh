#pragma once

#include "../../types.hpp"
#include "../../geometry_common.cuh"
#include "../../raytracer/rt_shaders.cuh"

struct BVHTreeInfo
{
    glm::vec3& bounds_min;
    glm::vec3& bounds_max;
    uint32_t& as_start;
    uint32_t& num_leaves;
    uint32_t& use_bvh16;
};

struct alignas(16) BVHNode
{
    glm::vec3 bounds_min{0.0f};
    uint32_t data = 0x80000000u; // bit 31 = is_leaf; interior: bits 29..30 = axis, bits 0..28 = left_child; leaf: bits 0..30 = box_content_idx
    glm::vec3 bounds_max{0.0f};
    uint32_t prim_count = 0;

    __host__ __device__ constexpr BVHNode() = default;
    __host__ __device__ constexpr BVHNode(glm::vec3 bmin, glm::vec3 bmax, uint32_t packed_data, uint32_t count)
        : bounds_min(bmin), data(packed_data), bounds_max(bmax), prim_count(count) {}

    [[nodiscard]] __host__ __device__ constexpr bool is_leaf() const { return (data & 0x80000000u) != 0u; }
    [[nodiscard]] __host__ __device__ constexpr uint32_t axis() const { return (data >> 29) & 0x3u; }
    [[nodiscard]] __host__ __device__ constexpr uint32_t left_child() const { return data & 0x1FFFFFFFu; }
    [[nodiscard]] __host__ __device__ constexpr uint32_t content() const { return data & 0x7FFFFFFFu; }

    __host__ __device__ static constexpr BVHNode make_leaf(glm::vec3 bmin, glm::vec3 bmax, uint32_t content_idx, uint32_t count)
    {
        return BVHNode(bmin, bmax, 0x80000000u | (content_idx & 0x7FFFFFFFu), count);
    }

    __host__ __device__ static constexpr BVHNode make_interior(glm::vec3 bmin, glm::vec3 bmax, uint32_t split_axis, uint32_t left_child_idx, uint32_t count = 0)
    {
        return BVHNode(bmin, bmax, ((split_axis & 0x3u) << 29) | (left_child_idx & 0x1FFFFFFFu), count);
    }
};
static_assert(sizeof(BVHNode) == 32);

struct alignas(16) BVHNode16
{
    uint32_t bounds[3]{0xFFFF0000u, 0xFFFF0000u, 0xFFFF0000u}; // per axis a: bits 0..15 = qmin, bits 16..31 = qmax
    uint32_t data = 0x80000000u; // bit 31 = is_leaf; interior: bits 29..30 = axis, bits 0..28 = left_child; leaf: bits 0..30 = box_content_idx

    __host__ __device__ constexpr BVHNode16() = default;
    __host__ __device__ constexpr BVHNode16(uint32_t bx, uint32_t by, uint32_t bz, uint32_t packed_data)
        : bounds{bx, by, bz}, data(packed_data) {}

    [[nodiscard]] __host__ __device__ constexpr uint16_t bmin(int a) const { return static_cast<uint16_t>(bounds[a] & 0xFFFFu); }
    [[nodiscard]] __host__ __device__ constexpr uint16_t bmax(int a) const { return static_cast<uint16_t>(bounds[a] >> 16); }
    [[nodiscard]] __host__ __device__ constexpr bool is_leaf() const { return (data & 0x80000000u) != 0u; }
    [[nodiscard]] __host__ __device__ constexpr uint32_t axis() const { return (data >> 29) & 0x3u; }
    [[nodiscard]] __host__ __device__ constexpr uint32_t left_child() const { return data & 0x1FFFFFFFu; }
    [[nodiscard]] __host__ __device__ constexpr uint32_t content() const { return data & 0x7FFFFFFFu; }

    __host__ __device__ static inline BVHNode16 from_node(
        const BVHNode& node,
        const glm::vec3& tree_min,
        const glm::vec3& tree_max)
    {
        const glm::vec3 ext = glm::max(tree_max - tree_min, glm::vec3(1e-12f));
        const glm::vec3 inv_ext = glm::vec3(1.0f) / ext;
        const glm::vec3 step = ext * (1.0f / 65535.0f);

        BVHNode16 out{};
        out.data = node.data;
        for (int a = 0; a < 3; ++a)
        {
            const float umin = glm::clamp((node.bounds_min[a] - tree_min[a]) * inv_ext[a], 0.0f, 1.0f);
            const float umax = glm::clamp((node.bounds_max[a] - tree_min[a]) * inv_ext[a], 0.0f, 1.0f);

            uint32_t qmin = static_cast<uint32_t>(glm::clamp(floorf(umin * 65535.0f), 0.0f, 65535.0f));
            uint32_t qmax = static_cast<uint32_t>(glm::clamp(ceilf(umax * 65535.0f), 0.0f, 65535.0f));

            if (qmin > 0u && (tree_min[a] + static_cast<float>(qmin) * step[a] > node.bounds_min[a]))
                --qmin;
            if (qmax < 65535u && (tree_min[a] + static_cast<float>(qmax) * step[a] < node.bounds_max[a]))
                ++qmax;
            if (qmin >= qmax)
            {
                if (qmax < 65535u)
                    ++qmax;
                else if (qmin > 0u)
                    --qmin;
            }

            out.bounds[a] = (qmax << 16) | (qmin & 0xFFFFu);
        }
        return out;
    }
};
static_assert(sizeof(BVHNode16) == 16);

struct BVHASDeviceView
{
    soa::soa_view<BVHTreeInfo> trees;
    soa::soa_view<BVHNode> as;
    soa::soa_view<BVHNode16> as16;
    soa::soa_view<BoxContent> box_content;

    template <bool HasNagata, bool UseBVH16, typename RenderParams>
    __host__ __device__ inline glm::vec4 first_hit(const RenderParams& p, glm::vec3 source, glm::vec3 direction) const;

    template <bool HasNagata = true, typename RenderParams>
    __host__ __device__ inline glm::vec4 first_hit(const RenderParams& p, glm::vec3 source, glm::vec3 direction) const;
};

struct alignas(8) BVHStackEntry
{
    int32_t node_id; // > 0: interior node's left_child index; < 0: ~leaf_node_index
    float t_min;
};
static_assert(sizeof(BVHStackEntry) == 8);

struct CurrentBVHNode
{
    static constexpr int SHORT_STACK_SIZE = 16;

    int32_t current_left = 0; // > 0: left_child index of active interior node; 0: pop from stack
    uint32_t stack_size : 8 = 0;
    uint32_t initialized : 1 = 0;
    uint32_t exited : 1 = 0;
    float t_min = 0.0f;
    float t_max = 0.0f;
};
static_assert(sizeof(CurrentBVHNode) == 16);

__host__ __device__ __forceinline__
glm::vec3 bvh_safe_inv_dir(glm::vec3 dir)
{
    return glm::vec3(
        1.0f / (glm::abs(dir.x) > 1e-20f ? dir.x : (dir.x >= 0.0f ? 1e-20f : -1e-20f)),
        1.0f / (glm::abs(dir.y) > 1e-20f ? dir.y : (dir.y >= 0.0f ? 1e-20f : -1e-20f)),
        1.0f / (glm::abs(dir.z) > 1e-20f ? dir.z : (dir.z >= 0.0f ? 1e-20f : -1e-20f)));
}

__host__ __device__ __forceinline__
bool bvh_finish_aabb_interval(
    float tmin,
    float box_tmax,
    float ray_t_max,
    float& out_tmin,
    float& out_tmax)
{
    // Conservative slab exit tolerance (Ize 2013) to cover FP32/-use_fast_math/quantization rounding
    // on planar surfaces and box edges without risking false misses.
    const float safe_tmax = fmaf(box_tmax, 1.000002f, 1e-6f);
    const float tmax = fminf(ray_t_max, safe_tmax);
    out_tmin = tmin;
    out_tmax = tmax;
    return tmin <= tmax;
}

__host__ __device__ __forceinline__
bool bvh_intersect_aabb(
    glm::vec3 bmin,
    glm::vec3 bmax,
    glm::vec3 orig,
    glm::vec3 inv_dir,
    float ray_t_min,
    float ray_t_max,
    float& out_tmin,
    float& out_tmax)
{
    const glm::vec3 t0 = (bmin - orig) * inv_dir;
    const glm::vec3 t1 = (bmax - orig) * inv_dir;
    const glm::vec3 tsmaller = glm::min(t0, t1);
    const glm::vec3 tbigger = glm::max(t0, t1);

    const float tmin = fmaxf(ray_t_min, fmaxf(tsmaller.x, fmaxf(tsmaller.y, tsmaller.z)));
    const float box_tmax = fminf(tbigger.x, fminf(tbigger.y, tbigger.z));
    return bvh_finish_aabb_interval(tmin, box_tmax, ray_t_max, out_tmin, out_tmax);
}

template <uint32_t StackStride = 1, uint32_t MaxStackSize = CurrentBVHNode::SHORT_STACK_SIZE>
__host__ __device__ inline uint32_t traverse_bvh_impl(
    const BVHNode* as,
    uint32_t as_start,
    CurrentBVHNode& state,
    BVHStackEntry* stack_base,
    glm::vec3 orig,
    glm::vec3 dir,
    int& iterations,
    float max_t = 1e30f,
    float ray_t_min = 0.0f)
{
    if (state.exited)
        return 0;

    const glm::vec3 inv_dir = bvh_safe_inv_dir(dir);

    if (!state.initialized)
    {
        state.initialized = 1;
        state.stack_size = 0;
        const float enter_min = fmaxf(ray_t_min, state.t_min);
        const BVHNode root = as[as_start];
        float root_tmin = 0.0f;
        float root_tmax = 0.0f;
        if (!bvh_intersect_aabb(root.bounds_min, root.bounds_max, orig, inv_dir, enter_min, max_t, root_tmin, root_tmax))
        {
            state.exited = 1;
            return 0;
        }
        if (root.is_leaf())
        {
            state.current_left = 0;
            state.exited = 1;
            state.t_min = root_tmin;
            state.t_max = root_tmax;
            ++iterations;
            return root.content();
        }
        state.current_left = static_cast<int32_t>(root.left_child());
    }

    constexpr int max_iterations = 500;
    while (iterations < max_iterations)
    {
        if (state.current_left == 0)
        {
            while (state.stack_size > 0)
            {
                state.stack_size = state.stack_size - 1;
                const BVHStackEntry entry = stack_base[static_cast<uint32_t>(state.stack_size) * StackStride];
                if (entry.t_min >= max_t)
                    continue;

                if (entry.node_id < 0)
                {
                    ++iterations;
                    const BVHNode leaf = as[static_cast<uint32_t>(~entry.node_id)];
                    float l_tmin = 0.0f;
                    float l_tmax = 0.0f;
                    if (bvh_intersect_aabb(leaf.bounds_min, leaf.bounds_max, orig, inv_dir, ray_t_min, max_t, l_tmin, l_tmax))
                    {
                        const uint32_t content = leaf.content();
                        if (content > 0)
                        {
                            state.t_min = l_tmin;
                            state.t_max = l_tmax;
                            if (state.stack_size == 0)
                                state.exited = 1;
                            return content;
                        }
                    }
                    if (iterations >= max_iterations)
                        break;
                }
                else
                {
                    state.current_left = entry.node_id;
                    break;
                }
            }

            if (state.current_left == 0)
            {
                state.exited = 1;
                return 0;
            }
            if (iterations >= max_iterations)
                break;
        }

        ++iterations;
        const uint32_t left_idx = static_cast<uint32_t>(state.current_left);
        const uint32_t right_idx = left_idx + 1u;
        const BVHNode child0 = as[left_idx];
        const BVHNode child1 = as[right_idx];

        float tmin0 = 0.0f, tmax0 = 0.0f;
        float tmin1 = 0.0f, tmax1 = 0.0f;
        const bool hit0 = bvh_intersect_aabb(child0.bounds_min, child0.bounds_max, orig, inv_dir, ray_t_min, max_t, tmin0, tmax0);
        const bool hit1 = bvh_intersect_aabb(child1.bounds_min, child1.bounds_max, orig, inv_dir, ray_t_min, max_t, tmin1, tmax1);

        if (!hit0 && !hit1)
        {
            state.current_left = 0;
            if (state.stack_size == 0)
            {
                state.exited = 1;
                return 0;
            }
            continue;
        }

        const bool pick_left = hit0 && (!hit1 || tmin0 <= tmin1);
        if (hit0 && hit1 && state.stack_size < MaxStackSize)
        {
            const int32_t far_entry_id = pick_left
                ? (child1.is_leaf() ? ~static_cast<int32_t>(right_idx) : static_cast<int32_t>(child1.left_child()))
                : (child0.is_leaf() ? ~static_cast<int32_t>(left_idx) : static_cast<int32_t>(child0.left_child()));
            const float far_tmin = pick_left ? tmin1 : tmin0;
            stack_base[static_cast<uint32_t>(state.stack_size) * StackStride] = BVHStackEntry{far_entry_id, far_tmin};
            state.stack_size = state.stack_size + 1;
        }

        const BVHNode near_node = pick_left ? child0 : child1;
        const float near_tmin = pick_left ? tmin0 : tmin1;
        const float near_tmax = pick_left ? tmax0 : tmax1;

        if (near_node.is_leaf())
        {
            state.current_left = 0;
            const uint32_t content = near_node.content();
            if (content > 0)
            {
                state.t_min = near_tmin;
                state.t_max = near_tmax;
                if (state.stack_size == 0)
                    state.exited = 1;
                return content;
            }
            if (state.stack_size == 0)
            {
                state.exited = 1;
                return 0;
            }
        }
        else
        {
            state.current_left = static_cast<int32_t>(near_node.left_child());
        }
    }

    state.exited = 1;
    return 0;
}

template <uint32_t StackStride = 1, uint32_t MaxStackSize = CurrentBVHNode::SHORT_STACK_SIZE>
__host__ __device__ inline uint32_t traverse_bvh(
    const BVHASDeviceView& view,
    int tree_id,
    CurrentBVHNode& state,
    BVHStackEntry* stack_base,
    glm::vec3 orig,
    glm::vec3 dir,
    int& iterations,
    float max_t = 1e30f,
    float ray_t_min = 0.0f)
{
    if (state.exited || tree_id < 0)
    {
        state.exited = 1;
        return 0;
    }

    const uint32_t as_start = view.trees[static_cast<uint32_t>(tree_id)].as_start;
    return traverse_bvh_impl<StackStride, MaxStackSize>(
        &view.as[0],
        as_start,
        state,
        stack_base,
        orig,
        dir,
        iterations,
        max_t,
        ray_t_min);
}

__host__ __device__ __forceinline__
uint4 bvh16_load_raw(const BVHNode16* __restrict__ as16, uint32_t idx)
{
#if defined(__CUDA_ARCH__)
    return __ldg(reinterpret_cast<const uint4*>(&as16[idx]));
#else
    const BVHNode16& n = as16[idx];
    return uint4{n.bounds[0], n.bounds[1], n.bounds[2], n.data};
#endif
}

__host__ __device__ __forceinline__
void bvh16_decode_axis(uint32_t packed, float scale, float bias, float& out_t0, float& out_t1)
{
    const float q0 = static_cast<float>(packed & 0xFFFFu);
    const float q1 = static_cast<float>(packed >> 16);
    out_t0 = fmaf(q0, scale, bias);
    out_t1 = fmaf(q1, scale, bias);
}

__host__ __device__ __forceinline__
void bvh16_compute_ray_transform(
    glm::vec3 tree_bmin,
    glm::vec3 tree_bmax,
    glm::vec3 orig,
    glm::vec3 inv_dir,
    glm::vec3& q_scale,
    glm::vec3& q_bias)
{
    const glm::vec3 ext = glm::max(tree_bmax - tree_bmin, glm::vec3(1e-12f));
    q_scale = (ext * (1.0f / 65535.0f)) * inv_dir;
    q_bias = (tree_bmin - orig) * inv_dir;
#if defined(__CUDA_ARCH__)
    // Pin materialized (q_scale, q_bias) so NVCC does not rematerialize them inside traversal loops.
    asm volatile(""
        : "+f"(q_scale.x), "+f"(q_scale.y), "+f"(q_scale.z),
          "+f"(q_bias.x), "+f"(q_bias.y), "+f"(q_bias.z));
#endif
}

__host__ __device__ __forceinline__
bool bvh16_intersect_aabb(
    uint4 raw,
    glm::vec3 q_scale,
    glm::vec3 q_bias,
    float ray_t_min,
    float ray_t_max,
    float& out_tmin,
    float& out_tmax)
{
    float tx0, tx1, ty0, ty1, tz0, tz1;
    bvh16_decode_axis(raw.x, q_scale.x, q_bias.x, tx0, tx1);
    bvh16_decode_axis(raw.y, q_scale.y, q_bias.y, ty0, ty1);
    bvh16_decode_axis(raw.z, q_scale.z, q_bias.z, tz0, tz1);

    const float tmin = fmaxf(ray_t_min, fmaxf(fminf(tx0, tx1), fmaxf(fminf(ty0, ty1), fminf(tz0, tz1))));
    const float box_tmax = fminf(fmaxf(tx0, tx1), fminf(fmaxf(ty0, ty1), fmaxf(tz0, tz1)));
    return bvh_finish_aabb_interval(tmin, box_tmax, ray_t_max, out_tmin, out_tmax);
}

template <uint32_t StackStride = 1, uint32_t MaxStackSize = CurrentBVHNode::SHORT_STACK_SIZE>
__host__ __device__ inline uint32_t traverse_bvh16_precomputed(
    const BVHNode16* __restrict__ as16,
    glm::vec3 q_scale,
    glm::vec3 q_bias,
    CurrentBVHNode& state,
    BVHStackEntry* stack_base,
    int& iterations,
    float max_t = 1e30f,
    float ray_t_min = 0.0f)
{
    if (state.exited)
        return 0;

    const float finite_max_t = fminf(max_t, 1e30f);

    constexpr int max_iterations = 500;
    while (iterations < max_iterations)
    {
        if (state.current_left == 0)
        {
            while (state.stack_size > 0)
            {
                state.stack_size = state.stack_size - 1;
                const BVHStackEntry entry = stack_base[static_cast<uint32_t>(state.stack_size) * StackStride];
                if (entry.t_min >= finite_max_t)
                    continue;

                if (entry.node_id < 0)
                {
                    ++iterations;
                    const uint32_t content = static_cast<uint32_t>(~entry.node_id);
                    if (content > 0)
                    {
                        state.t_min = entry.t_min;
                        state.t_max = finite_max_t;
                        if (state.stack_size == 0)
                            state.exited = 1;
                        return content;
                    }
                    if (iterations >= max_iterations)
                        break;
                }
                else
                {
                    state.current_left = entry.node_id;
                    break;
                }
            }

            if (state.current_left == 0)
            {
                state.exited = 1;
                return 0;
            }
            if (iterations >= max_iterations)
                break;
        }

        ++iterations;
        const uint32_t left_idx = static_cast<uint32_t>(state.current_left);
        const uint4 raw0 = bvh16_load_raw(as16, left_idx);
        const uint4 raw1 = bvh16_load_raw(as16, left_idx + 1u);

        float tmin0 = 0.0f, tmax0 = 0.0f;
        float tmin1 = 0.0f, tmax1 = 0.0f;
        const bool hit0 = bvh16_intersect_aabb(raw0, q_scale, q_bias, ray_t_min, finite_max_t, tmin0, tmax0);
        const bool hit1 = bvh16_intersect_aabb(raw1, q_scale, q_bias, ray_t_min, finite_max_t, tmin1, tmax1);

        if (!hit0 && !hit1)
        {
            state.current_left = 0;
            if (state.stack_size == 0)
            {
                state.exited = 1;
                return 0;
            }
            continue;
        }

        const bool pick_left = hit0 && (!hit1 || tmin0 <= tmin1);
        if (hit0 && hit1 && state.stack_size < MaxStackSize)
        {
            const uint32_t far_data = pick_left ? raw1.w : raw0.w;
            const bool far_is_leaf = (far_data & 0x80000000u) != 0u;
            const int32_t far_entry_id = far_is_leaf
                ? ~static_cast<int32_t>(far_data & 0x7FFFFFFFu)
                : static_cast<int32_t>(far_data & 0x1FFFFFFFu);
            const float far_tmin = pick_left ? tmin1 : tmin0;
            stack_base[static_cast<uint32_t>(state.stack_size) * StackStride] = BVHStackEntry{far_entry_id, far_tmin};
            state.stack_size = state.stack_size + 1;
        }

        const uint32_t near_data = pick_left ? raw0.w : raw1.w;
        const float near_tmin = pick_left ? tmin0 : tmin1;
        const float near_tmax = pick_left ? tmax0 : tmax1;

        if ((near_data & 0x80000000u) != 0u)
        {
            state.current_left = 0;
            const uint32_t content = near_data & 0x7FFFFFFFu;
            if (content > 0)
            {
                state.t_min = near_tmin;
                state.t_max = near_tmax;
                if (state.stack_size == 0)
                    state.exited = 1;
                return content;
            }
            if (state.stack_size == 0)
            {
                state.exited = 1;
                return 0;
            }
        }
        else
        {
            state.current_left = static_cast<int32_t>(near_data & 0x1FFFFFFFu);
        }
    }

    state.exited = 1;
    return 0;
}

template <uint32_t StackStride = 1, uint32_t MaxStackSize = CurrentBVHNode::SHORT_STACK_SIZE>
__host__ __device__ inline uint32_t traverse_bvh16_impl(
    const BVHNode16* as16,
    uint32_t as_start,
    glm::vec3 tree_bmin,
    glm::vec3 tree_bmax,
    CurrentBVHNode& state,
    BVHStackEntry* stack_base,
    glm::vec3 orig,
    glm::vec3 dir,
    int& iterations,
    float max_t = 1e30f,
    float ray_t_min = 0.0f)
{
    if (state.exited)
        return 0;

    const glm::vec3 inv_dir = bvh_safe_inv_dir(dir);
    const float finite_max_t = fminf(max_t, 1e30f);

    if (!state.initialized)
    {
        state.initialized = 1;
        state.stack_size = 0;
        const float enter_min = fmaxf(ray_t_min, state.t_min);
        float root_tmin = 0.0f;
        float root_tmax = 0.0f;
        if (!bvh_intersect_aabb(tree_bmin, tree_bmax, orig, inv_dir, enter_min, finite_max_t, root_tmin, root_tmax))
        {
            state.exited = 1;
            return 0;
        }
        const uint4 root_raw = bvh16_load_raw(as16, as_start);
        if ((root_raw.w & 0x80000000u) != 0u)
        {
            state.current_left = 0;
            state.exited = 1;
            state.t_min = root_tmin;
            state.t_max = root_tmax;
            ++iterations;
            return root_raw.w & 0x7FFFFFFFu;
        }
        state.current_left = static_cast<int32_t>(root_raw.w & 0x1FFFFFFFu);
    }

    glm::vec3 q_scale{};
    glm::vec3 q_bias{};
    bvh16_compute_ray_transform(tree_bmin, tree_bmax, orig, inv_dir, q_scale, q_bias);

    return traverse_bvh16_precomputed<StackStride, MaxStackSize>(
        as16, q_scale, q_bias, state, stack_base, iterations, finite_max_t, ray_t_min);
}

template <uint32_t StackStride = 1, uint32_t MaxStackSize = CurrentBVHNode::SHORT_STACK_SIZE>
__host__ __device__ inline uint32_t traverse_bvh16(
    const BVHASDeviceView& view,
    int tree_id,
    CurrentBVHNode& state,
    BVHStackEntry* stack_base,
    glm::vec3 orig,
    glm::vec3 dir,
    int& iterations,
    float max_t = 1e30f,
    float ray_t_min = 0.0f)
{
    if (state.exited || tree_id < 0)
    {
        state.exited = 1;
        return 0;
    }

    const auto tree = view.trees[static_cast<uint32_t>(tree_id)];
    return traverse_bvh16_impl<StackStride, MaxStackSize>(
        &view.as16[0],
        tree.as_start,
        tree.bounds_min,
        tree.bounds_max,
        state,
        stack_base,
        orig,
        dir,
        iterations,
        max_t,
        ray_t_min);
}

template <bool HasNagata, bool UseBVH16, typename RenderParams>
__host__ __device__ inline
glm::vec4 BVHASDeviceView::first_hit(const RenderParams& p, glm::vec3 source, glm::vec3 direction) const
{
    if (p.root_box < 0)
        return miss_shader(source, direction, p);

    int max_iterations = 2000;

    int box_hits = 0;
    glm::vec4 hit_primitive_color{0};

#if defined(__CUDA_ARCH__)
    __shared__ BVHStackEntry shared_stack[CurrentBVHNode::SHORT_STACK_SIZE][RENDER_BLOCK_THREADS];
    const uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
    BVHStackEntry* stack_base = &shared_stack[0][tid];
    constexpr uint32_t stack_stride = RENDER_BLOCK_THREADS;
#else
    BVHStackEntry host_stack[CurrentBVHNode::SHORT_STACK_SIZE];
    BVHStackEntry* stack_base = host_stack;
    constexpr uint32_t stack_stride = 1;
#endif

    const auto tree = trees[static_cast<uint32_t>(p.root_box)];
    const uint32_t as_start = tree.as_start;
    const glm::vec3 tree_bmin = UseBVH16 ? tree.bounds_min : glm::vec3(0.0f);
    const glm::vec3 tree_bmax = UseBVH16 ? tree.bounds_max : glm::vec3(0.0f);

    int it = 0;
    bool continue_traversal = true;
    bool missed = false;
    bool alpha_terminated = false;

    for (int bounce = 0; continue_traversal && it < max_iterations; ++bounce)
    {
        const float ray_t_min = (bounce == 0) ? 0.0f : 1e-4f;
        uint32_t closest_triangle = UINT32_MAX;
        glm::vec3 closest_tuv{INFINITY, 0.0f, 0.0f};

        if constexpr (UseBVH16)
        {
            const glm::vec3 inv_dir = bvh_safe_inv_dir(direction);
            float root_tmin = 0.0f;
            float root_tmax = 0.0f;
            if (bvh_intersect_aabb(tree_bmin, tree_bmax, source, inv_dir, ray_t_min, 1e30f, root_tmin, root_tmax))
            {
                const uint4 root_raw = bvh16_load_raw(&as16[0], as_start);
                if ((root_raw.w & 0x80000000u) != 0u)
                {
                    ++it;
                    const uint32_t container_id = root_raw.w & 0x7FFFFFFFu;
                    if (container_id > 0)
                    {
                        ++box_hits;
                        if (p.show_triangles)
                        {
                            intersect_box_primitives<HasNagata, false>(
                                box_content, container_id, p, source, direction,
                                ray_t_min, fminf(closest_tuv.x, 1e30f), closest_triangle, closest_tuv);
                        }
                    }
                }
                else
                {
                    glm::vec3 q_scale{};
                    glm::vec3 q_bias{};
                    bvh16_compute_ray_transform(tree_bmin, tree_bmax, source, inv_dir, q_scale, q_bias);

                    CurrentBVHNode current_node{};
                    current_node.current_left = static_cast<int32_t>(root_raw.w & 0x1FFFFFFFu);
                    current_node.initialized = 1;
                    current_node.t_min = ray_t_min;

                    while (!current_node.exited && it < max_iterations)
                    {
                        const uint32_t container_id = traverse_bvh16_precomputed<stack_stride>(
                            &as16[0], q_scale, q_bias, current_node, stack_base, it, closest_tuv.x, ray_t_min);

                        if (container_id > 0)
                        {
                            ++box_hits;
                            if (p.show_triangles)
                            {
                                intersect_box_primitives<HasNagata, false>(
                                    box_content,
                                    container_id,
                                    p,
                                    source,
                                    direction,
                                    ray_t_min,
                                    fminf(closest_tuv.x, 1e30f),
                                    closest_triangle,
                                    closest_tuv);
                            }
                        }
                    }
                }
            }
        }
        else
        {
            CurrentBVHNode current_node{.t_min = ray_t_min};

            while (!current_node.exited && it < max_iterations)
            {
                const uint32_t container_id = traverse_bvh_impl<stack_stride>(
                    &as[0], as_start, current_node, stack_base, source, direction, it, closest_tuv.x, ray_t_min);

                if (container_id > 0)
                {
                    ++box_hits;
                    if (p.show_triangles)
                    {
                        intersect_box_primitives<HasNagata, false>(
                            box_content,
                            container_id,
                            p,
                            source,
                            direction,
                            ray_t_min,
                            fminf(closest_tuv.x, 1e30f),
                            closest_triangle,
                            closest_tuv);
                    }
                }
            }
        }

        if (it >= max_iterations)
            break;

        if (closest_triangle != UINT32_MAX)
        {
            auto res = fragment_hit_shader<HasNagata>(closest_triangle, source, direction, closest_tuv, p, continue_traversal);
            hit_primitive_color = blend_colors(hit_primitive_color, res);
            if (continue_traversal && hit_primitive_color.a > p.alpha_threshold)
            {
                continue_traversal = false;
                alpha_terminated = true;
            }
        }
        else
        {
            missed = true;
            continue_traversal = false;
        }
    }

    if (it >= max_iterations)
        return glm::vec4{1.f};

    if (!p.show_iterations && !p.show_boxes)
    {
        if (missed || alpha_terminated)
            hit_primitive_color = blend_colors(hit_primitive_color, miss_shader(source, direction, p));
        return hit_primitive_color;
    }

    return compose_debug_overlay(p, hit_primitive_color, it, box_hits);
}

template <bool HasNagata, typename RenderParams>
__host__ __device__ inline
glm::vec4 BVHASDeviceView::first_hit(const RenderParams& p, glm::vec3 source, glm::vec3 direction) const
{
    if (p.root_box < 0)
        return miss_shader(source, direction, p);
    const bool use_bvh16 = (trees[static_cast<uint32_t>(p.root_box)].use_bvh16 != 0u);
    if (use_bvh16)
        return first_hit<HasNagata, true>(p, source, direction);
    return first_hit<HasNagata, false>(p, source, direction);
}
