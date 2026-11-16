#pragma once

#include "../../types.hpp"
#include "../../geometry_common.cuh"
#include "../../raytracer/rt_shaders.cuh"

struct KDTreeInfo
{
    glm::vec3& bounds_min;
    glm::vec3& bounds_max;
    uint32_t& as_start;
    uint32_t& num_leaves;
};

struct alignas(8) KDNode
{
    float split_pos = 0.0f;
    uint32_t data = 0x80000000u;

    __host__ __device__ constexpr KDNode() = default;
    __host__ __device__ constexpr KDNode(float split, uint32_t packed_data)
        : split_pos(split), data(packed_data) {}

    [[nodiscard]] __host__ __device__ constexpr bool is_leaf() const { return (data & 0x80000000u) != 0u; }
    [[nodiscard]] __host__ __device__ constexpr uint32_t axis() const { return (data >> 29) & 0x3u; }
    [[nodiscard]] __host__ __device__ constexpr uint32_t left_child() const { return data & 0x1FFFFFFFu; }
    [[nodiscard]] __host__ __device__ constexpr uint32_t content() const { return data & 0x7FFFFFFFu; }

    __host__ __device__ static constexpr KDNode make_leaf(uint32_t content_idx)
    {
        return KDNode(0.0f, 0x80000000u | (content_idx & 0x7FFFFFFFu));
    }

    __host__ __device__ static constexpr KDNode make_interior(uint32_t split_axis, float split_coord, uint32_t left_child_idx)
    {
        return KDNode(split_coord, ((split_axis & 0x3u) << 29) | (left_child_idx & 0x1FFFFFFFu));
    }
};
static_assert(sizeof(KDNode) == 8);

struct KDTreeASDeviceView
{
    soa::soa_view<KDTreeInfo> trees;
    soa::soa_view<KDNode> as;
    soa::soa_view<BoxContent> box_content;

    template <bool HasNagata = true, typename RenderParams>
    __host__ __device__ inline glm::vec4 first_hit(const RenderParams& p, glm::vec3 source, glm::vec3 direction) const;
};

struct alignas(8) KDStackEntry
{
    int32_t node_id;
    float t_max;
};
static_assert(sizeof(KDStackEntry) == 8);

struct CurrentKDNode
{
    static constexpr int SHORT_STACK_SIZE = 12;

    uint32_t stack_head : 8 = 0;
    uint32_t stack_count : 8 = 0;
    uint32_t initialized : 1 = 0;
    uint32_t exited : 1 = 0;
    float t_min = 0.0f;
    float t_max = 0.0f;
    float ray_t_max = 0.0f;
};
static_assert(sizeof(CurrentKDNode) == 16);

__host__ __device__ __forceinline__
glm::vec3 kdtree_safe_inv_dir(glm::vec3 dir)
{
    return glm::vec3(
        1.0f / (glm::abs(dir.x) > 1e-20f ? dir.x : (dir.x >= 0.0f ? 1e-20f : -1e-20f)),
        1.0f / (glm::abs(dir.y) > 1e-20f ? dir.y : (dir.y >= 0.0f ? 1e-20f : -1e-20f)),
        1.0f / (glm::abs(dir.z) > 1e-20f ? dir.z : (dir.z >= 0.0f ? 1e-20f : -1e-20f)));
}

__host__ __device__ __forceinline__
bool kdtree_ray_aabb(
    glm::vec3 orig,
    glm::vec3 inv_dir,
    glm::vec3 bounds_min,
    glm::vec3 bounds_max,
    float& t_min,
    float& t_max)
{
    const glm::vec3 t0 = (bounds_min - orig) * inv_dir;
    const glm::vec3 t1 = (bounds_max - orig) * inv_dir;
    const glm::vec3 tsmaller = glm::min(t0, t1);
    const glm::vec3 tbigger = glm::max(t0, t1);

    t_min = glm::max(t_min, glm::max(tsmaller.x, glm::max(tsmaller.y, tsmaller.z)));
    t_max = glm::min(t_max, glm::min(tbigger.x, glm::min(tbigger.y, tbigger.z)));
    return t_min <= t_max && t_max >= 0.0f;
}

template <uint32_t StackStride = 1, uint32_t MaxStackSize = CurrentKDNode::SHORT_STACK_SIZE, typename GetBoundsFn>
__host__ __device__ inline uint32_t traverse_kdtree_impl(
    const KDNode* as,
    uint32_t as_start,
    GetBoundsFn&& get_bounds,
    CurrentKDNode& state,
    KDStackEntry* stack_base,
    glm::vec3 orig,
    glm::vec3 dir,
    int& iterations)
{
    if (state.exited)
        return 0;

    const glm::vec3 inv_dir = kdtree_safe_inv_dir(dir);
    float seg_t_min = 0.0f;
    float seg_t_max = 0.0f;
    int32_t node_idx = -1;

    if (!state.initialized)
    {
        glm::vec3 bounds_min, bounds_max;
        get_bounds(bounds_min, bounds_max);

        float t_enter = glm::max(0.0f, state.t_max);
        float t_leave = 1e30f;
        if (!kdtree_ray_aabb(orig, inv_dir, bounds_min, bounds_max, t_enter, t_leave))
        {
            state.exited = 1;
            return 0;
        }

        node_idx = static_cast<int32_t>(as_start);
        state.ray_t_max = t_leave;
        state.stack_head = 0;
        state.stack_count = 0;
        state.initialized = 1;
        seg_t_min = t_enter;
        seg_t_max = t_leave;
    }

    constexpr int max_iterations = 500;
    while (iterations < max_iterations)
    {
        if (node_idx < 0)
        {
            if (state.stack_count > 0)
            {
                state.stack_count = state.stack_count - 1;
                state.stack_head = (state.stack_head == 0)
                    ? static_cast<uint32_t>(MaxStackSize - 1)
                    : (state.stack_head - 1);
                const KDStackEntry entry = stack_base[static_cast<uint32_t>(state.stack_head) * StackStride];
                node_idx = entry.node_id;
                seg_t_min = state.t_max;
                seg_t_max = entry.t_max;
            }
            else if (state.t_max < state.ray_t_max)
            {
                node_idx = static_cast<int32_t>(as_start);
                seg_t_min = state.t_max;
                seg_t_max = state.ray_t_max;
            }
            else
            {
                state.exited = 1;
                return 0;
            }
        }

        while (!as[node_idx].is_leaf() && iterations < max_iterations)
        {
            ++iterations;
            const KDNode node = as[node_idx];
            const uint32_t axis = node.axis();
            const float split = node.split_pos;
            const int32_t left_idx = static_cast<int32_t>(node.left_child());
            const int32_t right_idx = left_idx + 1;

            const float orig_a = orig[axis];
            const float inv_d = inv_dir[axis];
            const float t_split = (split - orig_a) * inv_d;

            const int32_t near_idx = (inv_d >= 0.0f) ? left_idx : right_idx;
            const int32_t far_idx = (inv_d >= 0.0f) ? right_idx : left_idx;

            if (t_split >= seg_t_max)
            {
                node_idx = near_idx;
            }
            else if (t_split <= seg_t_min)
            {
                node_idx = far_idx;
            }
            else
            {
                stack_base[static_cast<uint32_t>(state.stack_head) * StackStride] = KDStackEntry{far_idx, seg_t_max};
                state.stack_head = (state.stack_head + 1 == MaxStackSize)
                    ? 0u
                    : (state.stack_head + 1);
                if (state.stack_count < MaxStackSize)
                {
                    state.stack_count = state.stack_count + 1;
                }
                node_idx = near_idx;
                seg_t_max = t_split;
            }
        }

        if (iterations >= max_iterations)
        {
            state.exited = 1;
            return 0;
        }

        ++iterations;
        state.t_min = seg_t_min;
        state.t_max = seg_t_max;
        const uint32_t content = as[node_idx].content();
        node_idx = -1;

        if (content > 0)
        {
            if (state.stack_count == 0 && state.t_max >= state.ray_t_max)
            {
                state.exited = 1;
            }
            return content;
        }
    }

    state.exited = 1;
    return 0;
}

template <uint32_t StackStride = 1, uint32_t MaxStackSize = CurrentKDNode::SHORT_STACK_SIZE>
__host__ __device__ inline uint32_t traverse_kdtree(
    const KDNode* as,
    glm::vec3 bounds_min,
    glm::vec3 bounds_max,
    uint32_t as_start,
    CurrentKDNode& state,
    KDStackEntry* stack_base,
    glm::vec3 orig,
    glm::vec3 dir,
    int& iterations)
{
    return traverse_kdtree_impl<StackStride, MaxStackSize>(
        as,
        as_start,
        [&](glm::vec3& bmin, glm::vec3& bmax)
        {
            bmin = bounds_min;
            bmax = bounds_max;
        },
        state,
        stack_base,
        orig,
        dir,
        iterations);
}

template <uint32_t StackStride = 1, uint32_t MaxStackSize = CurrentKDNode::SHORT_STACK_SIZE>
__host__ __device__ inline uint32_t traverse_kdtree(
    const KDTreeASDeviceView& view,
    int tree_id,
    CurrentKDNode& state,
    KDStackEntry* stack_base,
    glm::vec3 orig,
    glm::vec3 dir,
    int& iterations)
{
    if (state.exited || tree_id < 0)
    {
        state.exited = 1;
        return 0;
    }

    const uint32_t u_tree_id = static_cast<uint32_t>(tree_id);
    const uint32_t as_start = view.trees[u_tree_id].as_start;
    return traverse_kdtree_impl<StackStride, MaxStackSize>(
        &view.as[0],
        as_start,
        [&](glm::vec3& bmin, glm::vec3& bmax)
        {
            const auto tree = view.trees[u_tree_id];
            bmin = tree.bounds_min;
            bmax = tree.bounds_max;
        },
        state,
        stack_base,
        orig,
        dir,
        iterations);
}

template <bool HasNagata, typename RenderParams>
__host__ __device__ inline
glm::vec4 KDTreeASDeviceView::first_hit(const RenderParams& p, glm::vec3 source, glm::vec3 direction) const
{
    if (p.root_box < 0)
        return miss_shader(source, direction, p);

    constexpr int max_iterations = 500;

    int box_hits = 0;
    glm::vec4 hit_primitive_color{0};

    uint32_t container_id = UINT32_MAX;
    CurrentKDNode current_node{};

#if defined(__CUDA_ARCH__)
    __shared__ KDStackEntry shared_stack[CurrentKDNode::SHORT_STACK_SIZE][256];
    const uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
    KDStackEntry* stack_base = &shared_stack[0][tid];
    constexpr uint32_t stack_stride = 256;
#else
    KDStackEntry host_stack[CurrentKDNode::SHORT_STACK_SIZE];
    KDStackEntry* stack_base = host_stack;
    constexpr uint32_t stack_stride = 1;
#endif

    int it = 0;
    int bounce = 0;
    bool continue_traversal = true;

    while (container_id != 0 && continue_traversal && it < max_iterations)
    {
        container_id = traverse_kdtree<stack_stride>(
            *this, p.root_box, current_node, stack_base, source, direction, it);

        if (container_id > 0)
        {
            uint32_t closest_triangle = UINT32_MAX;
            glm::vec3 closest_tuv{INFINITY, 0, 0};
            ++box_hits;

            if (p.show_triangles)
            {
                intersect_box_primitives<HasNagata>(
                    box_content,
                    container_id,
                    p,
                    source,
                    direction,
                    current_node.t_min,
                    current_node.t_max,
                    closest_triangle,
                    closest_tuv);
            }

            if (closest_triangle != UINT32_MAX)
            {
                auto res = fragment_hit_shader<HasNagata>(closest_triangle, source, direction, closest_tuv, p, continue_traversal);
                hit_primitive_color = blend_colors(hit_primitive_color, res);
                if (continue_traversal)
                {
                    if (++bounce < 8)
                    {
                        current_node = CurrentKDNode{.t_min = 1e-4f, .t_max = 1e-4f};
                    }
                    else
                    {
                        continue_traversal = false;
                    }
                }
            }
        }

        if (current_node.exited)
        {
            break;
        }
    }

    if (it >= max_iterations)
        return glm::vec4{1.f};

    if (!p.show_iterations && !p.show_boxes)
    {
        if (current_node.exited)
            hit_primitive_color = blend_colors(hit_primitive_color, miss_shader(source, direction, p));
        return hit_primitive_color;
    }

    return compose_debug_overlay(p, hit_primitive_color, it, box_hits);
}
