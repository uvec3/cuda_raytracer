#pragma once

#include "../../types.hpp"
#include "../../geometry_common.cuh"
#include "../../raytracer/rt_shaders.cuh"
#include "../octree/OctreeGeometry.cuh"
#include "../uniform_grid/UniformGridGeometry.cuh"
#include "../kdtree/KDTreeGeometry.cuh"
#include "../bvh/BVHGeometry.cuh"

struct HeterogeneousTreeInfo
{
    int32_t& tlas_space_id;
    int32_t& tlas_root_box;
    uint32_t& tlas_as_type;
    uint32_t& blas_offset;
    uint32_t& blas_count;
};

struct HeterogeneousInstanceInfo
{
    glm::vec4& s2l_row0;
    glm::vec4& s2l_row1;
    glm::vec4& s2l_row2;
    int32_t& space_id;
};

struct HeterogeneousASDeviceView
{
    static constexpr uint32_t TLAS_STACK_SIZE = 12;
    static constexpr uint32_t BLAS_STACK_SIZE = 20;
    static constexpr uint32_t TOTAL_STACK_SIZE = TLAS_STACK_SIZE + BLAS_STACK_SIZE; // 32
    static constexpr uint32_t BLOCK_THREADS = RENDER_BLOCK_THREADS; // up to 32 * 64 * 8B = 16 KB shared memory
    static constexpr uint32_t PROBE_BLOCK_THREADS = RENDER_BLOCK_THREADS; // 64 threads/block matching draw_first_hit_kernel and standalone first_hit shared-memory stacks

    soa::soa_view<HeterogeneousTreeInfo> trees;
    soa::soa_view<HeterogeneousInstanceInfo> instances;

    template <bool HasNagata = true, ASType TlasType = ASType::None, typename RenderParams>
    __host__ __device__ inline glm::vec4 first_hit(const RenderParams& p, glm::vec3 source, glm::vec3 direction) const;
};

namespace heterogeneous_detail
{
    __host__ __device__ inline glm::vec3 transform_point_3x4(
        glm::vec4 r0, glm::vec4 r1, glm::vec4 r2, glm::vec3 p)
    {
        return glm::vec3(
            r0.x * p.x + r0.y * p.y + r0.z * p.z + r0.w,
            r1.x * p.x + r1.y * p.y + r1.z * p.z + r1.w,
            r2.x * p.x + r2.y * p.y + r2.z * p.z + r2.w);
    }

    __host__ __device__ inline glm::vec3 transform_dir_3x4(
        glm::vec4 r0, glm::vec4 r1, glm::vec4 r2, glm::vec3 d)
    {
        return glm::vec3(
            r0.x * d.x + r0.y * d.y + r0.z * d.z,
            r1.x * d.x + r1.y * d.y + r1.z * d.z,
            r2.x * d.x + r2.y * d.y + r2.z * d.z);
    }

    struct InstanceMailbox
    {
        uint32_t entries[4] = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX };
        uint32_t next_slot = 0;

        __host__ __device__ __forceinline__ bool contains(uint32_t inst_idx) const
        {
            return entries[0] == inst_idx || entries[1] == inst_idx ||
                   entries[2] == inst_idx || entries[3] == inst_idx;
        }

        __host__ __device__ __forceinline__ void insert(uint32_t inst_idx)
        {
            entries[next_slot & 3u] = inst_idx;
            ++next_slot;
        }
    };

    template <ASType AsType, typename RenderParams>
    __host__ __device__ inline const auto& get_as_view(const RenderParams& p)
    {
        if constexpr (AsType == ASType::BVH)
            return p.as_bvh;
        else if constexpr (AsType == ASType::KDTree)
            return p.as_kdtree;
        else if constexpr (AsType == ASType::UniformGrid)
            return p.as_uniform_grid;
        else
            return p.as_octree;
    }

    template <ASType AsType, uint32_t StackStride, uint32_t StackSize, bool AllowBVH32, typename ASView, typename OnContainer>
    __host__ __device__ inline void traverse_as(
        const ASView& as_view,
        int32_t root_box,
        void* stack_base,
        glm::vec3 orig,
        glm::vec3 dir,
        float ray_t_min,
        float& closest_t,
        int& total_iterations,
        int max_local_iterations,
        int max_total_iterations,
        OnContainer&& on_container)
    {
        if (root_box < 0)
            return;

        int local_it = 0;

        if constexpr (AsType == ASType::BVH)
        {
            const auto tree = as_view.trees[static_cast<uint32_t>(root_box)];
            const bool use_bvh16 = !AllowBVH32 || (tree.use_bvh16 != 0u);
            auto* bvh_stack = reinterpret_cast<BVHStackEntry*>(stack_base);

            if (use_bvh16)
            {
                const glm::vec3 inv_dir = bvh_safe_inv_dir(dir);
                const float finite_max_t = fminf(closest_t, 1e30f);
                float root_tmin = 0.0f;
                float root_tmax = 0.0f;
                if (!bvh_intersect_aabb(tree.bounds_min, tree.bounds_max, orig, inv_dir, ray_t_min, finite_max_t, root_tmin, root_tmax))
                    return;

                const uint4 root_raw = bvh16_load_raw(&as_view.as16[0], tree.as_start);
                if ((root_raw.w & 0x80000000u) != 0u)
                {
                    ++total_iterations;
                    const uint32_t container_id = root_raw.w & 0x7FFFFFFFu;
                    if (container_id > 0)
                    {
                        on_container(container_id, orig, ray_t_min, fminf(closest_t, 1e30f), true);
                    }
                    return;
                }

                glm::vec3 q_scale{};
                glm::vec3 q_bias{};
                bvh16_compute_ray_transform(tree.bounds_min, tree.bounds_max, orig, inv_dir, q_scale, q_bias);

                CurrentBVHNode state{};
                state.current_left = static_cast<int32_t>(root_raw.w & 0x1FFFFFFFu);
                state.initialized = 1;
                state.t_min = ray_t_min;

                while (!state.exited && local_it < max_local_iterations && total_iterations < max_total_iterations)
                {
                    const int prev_it = local_it;
                    const uint32_t container_id = traverse_bvh16_precomputed<StackStride, StackSize>(
                        &as_view.as16[0],
                        q_scale,
                        q_bias,
                        state,
                        bvh_stack,
                        local_it,
                        closest_t,
                        ray_t_min);
                    total_iterations += (local_it - prev_it);

                    if (container_id > 0)
                    {
                        on_container(container_id, orig, ray_t_min, fminf(closest_t, 1e30f), state.exited != 0);
                    }
                }
            }
            else if constexpr (AllowBVH32)
            {
                CurrentBVHNode state{.t_min = ray_t_min};
                while (!state.exited && local_it < max_local_iterations && total_iterations < max_total_iterations)
                {
                    const int prev_it = local_it;
                    const uint32_t container_id = traverse_bvh_impl<StackStride, StackSize>(
                        &as_view.as[0],
                        tree.as_start,
                        state,
                        bvh_stack,
                        orig,
                        dir,
                        local_it,
                        closest_t,
                        ray_t_min);
                    total_iterations += (local_it - prev_it);

                    if (container_id > 0)
                    {
                        on_container(container_id, orig, ray_t_min, fminf(closest_t, 1e30f), state.exited != 0);
                    }
                }
            }
        }
        else if constexpr (AsType == ASType::KDTree)
        {
            CurrentKDNode state{};
            state.t_max = ray_t_min;
            auto* kd_stack = reinterpret_cast<KDStackEntry*>(stack_base);

            while (!state.exited && local_it < max_local_iterations && total_iterations < max_total_iterations)
            {
                const int prev_it = local_it;
                const uint32_t container_id = traverse_kdtree<StackStride, StackSize>(
                    as_view,
                    root_box,
                    state,
                    kd_stack,
                    orig,
                    dir,
                    local_it);
                total_iterations += (local_it - prev_it);

                if (container_id > 0)
                {
                    if (state.t_min > closest_t)
                        break;

                    if (on_container(container_id, orig, fmaxf(state.t_min, ray_t_min), fminf(state.t_max, closest_t), state.exited != 0))
                        break;

                    if (closest_t <= state.t_max)
                        break;
                }
            }
        }
        else if constexpr (AsType == ASType::UniformGrid)
        {
            CurrentGridCell state{};
            state.t_max = ray_t_min;

            while (!state.exited && local_it < max_local_iterations && total_iterations < max_total_iterations)
            {
                const int prev_it = local_it;
                const uint32_t container_id = traverse_uniform_grid(
                    as_view,
                    root_box,
                    state,
                    orig,
                    dir,
                    local_it);
                total_iterations += (local_it - prev_it);

                if (container_id > 0)
                {
                    if (state.t_min > closest_t)
                        break;

                    if (on_container(container_id, orig, fmaxf(state.t_min, ray_t_min), fminf(state.t_max, closest_t), state.exited != 0))
                        break;

                    if (closest_t <= state.t_max)
                        break;
                }
            }
        }
        else if constexpr (AsType == ASType::Octree)
        {
            const glm::vec3 oct_orig = (ray_t_min > 0.0f) ? (orig + dir * ray_t_min) : orig;
            CurrentBox state{.id = root_box};

            while (local_it < max_local_iterations && total_iterations < max_total_iterations)
            {
                const int prev_it = local_it;
                const TraversalResult res = traverse_stackless(&as_view.as[0], state, oct_orig, dir, local_it);
                total_iterations += (local_it - prev_it);

                const uint32_t container_id = res.content;
                const bool is_last_cell = (state.stack.depth() == 0);
                if (container_id > 0)
                {
                    if (ray_t_min + state.t_min > closest_t)
                        break;

                    const float oct_closest_t = isinf(closest_t) ? INFINITY : fmaxf(0.0f, closest_t - ray_t_min);
                    const float cell_tmax = (state.t_max > state.t_min) ? fminf(state.t_max, oct_closest_t) : oct_closest_t;

                    if (on_container(container_id, oct_orig, state.t_min, cell_tmax, is_last_cell) && state.t_max > state.t_min)
                        break;

                    if (!is_last_cell && closest_t <= ray_t_min + state.t_max)
                        break;
                }

                if (is_last_cell)
                    break;
            }
        }
    }

    template <ASType AsType, bool HasNagata, bool TrackDebug, typename ASView, typename RenderParams>
    struct LeafPrimitivesVisitor
    {
        const ASView& as_view;
        const RenderParams& p;
        glm::vec3 dir;
        float ray_t_min;
        uint32_t& closest_tri;
        glm::vec3& closest_tuv;
        int* box_hits;

        __host__ __device__ inline bool operator()(
            uint32_t container_id,
            glm::vec3 leaf_orig,
            float cell_tmin,
            float cell_tmax,
            bool /*is_last_cell*/) const
        {
            if constexpr (TrackDebug)
            {
                ++(*box_hits);
                if (!p.show_triangles)
                    return false;
            }

            if constexpr (AsType == ASType::Octree)
            {
                glm::vec3 oct_tuv{
                    isinf(closest_tuv.x) ? INFINITY : fmaxf(0.0f, closest_tuv.x - ray_t_min),
                    closest_tuv.y,
                    closest_tuv.z};
                uint32_t oct_tri = UINT32_MAX;
                intersect_box_primitives<HasNagata, true>(
                    as_view.box_content,
                    container_id,
                    p,
                    leaf_orig,
                    dir,
                    cell_tmin,
                    cell_tmax,
                    oct_tri,
                    oct_tuv);
                if (oct_tri != UINT32_MAX)
                {
                    closest_tri = oct_tri;
                    closest_tuv = glm::vec3(ray_t_min + oct_tuv.x, oct_tuv.y, oct_tuv.z);
                    return true;
                }
                return false;
            }
            else
            {
                intersect_box_primitives<HasNagata, AsType != ASType::BVH>(
                    as_view.box_content,
                    container_id,
                    p,
                    leaf_orig,
                    dir,
                    cell_tmin,
                    cell_tmax,
                    closest_tri,
                    closest_tuv);
                if constexpr (AsType != ASType::BVH)
                    return closest_tri != UINT32_MAX;
                else
                    return false;
            }
        }
    };

    template <ASType AsType, bool HasNagata, uint32_t StackStride, uint32_t StackSize, bool AllowBVH32, bool TrackDebug, typename ASView, typename RenderParams>
    __host__ __device__ inline void traverse_as_primitives(
        const ASView& as_view,
        int32_t root_box,
        void* stack_base,
        const RenderParams& p,
        glm::vec3 orig,
        glm::vec3 dir,
        float ray_t_min,
        glm::vec3& closest_tuv,
        uint32_t& closest_tri,
        int& total_iterations,
        int* box_hits,
        int max_local_iterations = 2000,
        int max_total_iterations = 1000000)
    {
        LeafPrimitivesVisitor<AsType, HasNagata, TrackDebug, ASView, RenderParams> visitor{
            as_view, p, dir, ray_t_min, closest_tri, closest_tuv, box_hits};
        traverse_as<AsType, StackStride, StackSize, AllowBVH32>(
            as_view,
            root_box,
            stack_base,
            orig,
            dir,
            ray_t_min,
            closest_tuv.x,
            total_iterations,
            max_local_iterations,
            max_total_iterations,
            visitor);
    }

    template <bool HasNagata, uint32_t StackStride, uint32_t BlasStackSize, bool ClipToCell = false, typename RenderParams>
    __host__ __device__ inline void intersect_blas_subspace(
        const RenderParams& p,
        uint32_t tlas_blas_offset,
        const SubspacePrimitive& sub,
        glm::vec3 scene_orig,
        glm::vec3 scene_dir,
        float scene_tmin,
        float& inout_closest_scene_t,
        glm::vec3& inout_closest_tuv,
        uint32_t& inout_closest_tri,
        uint32_t& inout_closest_inst,
        uint32_t& inout_closest_space,
        void* blas_stack_base,
        int& iterations,
        int& box_hits,
        float cell_tmin = 0.0f,
        float cell_tmax = INFINITY,
        InstanceMailbox* mailbox = nullptr)
    {
        const uint32_t inst_idx = tlas_blas_offset + sub.blas_info_id;
        if constexpr (ClipToCell)
        {
            if (mailbox != nullptr && mailbox->contains(inst_idx))
                return;
        }

        float box_tmin = 0.0f;
        float box_tmax = 0.0f;
        const glm::vec3 scene_inv_dir = bvh_safe_inv_dir(scene_dir);
        if (!bvh_intersect_aabb(
                sub.bounds_min, sub.bounds_max,
                scene_orig, scene_inv_dir,
                scene_tmin, inout_closest_scene_t,
                box_tmin, box_tmax) ||
            box_tmin >= inout_closest_scene_t)
        {
            return;
        }

        if constexpr (ClipToCell)
        {
            if (!isinf(cell_tmax))
            {
                const float cell_eps = fmaxf(1e-4f, (cell_tmax - cell_tmin) * 1e-3f);
                if (box_tmin > cell_tmax + cell_eps)
                    return;
            }
            if (mailbox != nullptr)
                mailbox->insert(inst_idx);
        }

        glm::vec4 r0(1.0f, 0.0f, 0.0f, 0.0f);
        glm::vec4 r1(0.0f, 1.0f, 0.0f, 0.0f);
        glm::vec4 r2(0.0f, 0.0f, 1.0f, 0.0f);
        int32_t active_space_id = sub.space_id;

        if (inst_idx < p.as_heterogeneous.instances.size())
        {
            const auto inst = p.as_heterogeneous.instances[inst_idx];
            r0 = inst.s2l_row0;
            r1 = inst.s2l_row1;
            r2 = inst.s2l_row2;
            if (inst.space_id >= 0)
                active_space_id = inst.space_id;
        }

        if (active_space_id < 0 || static_cast<uint32_t>(active_space_id) >= p.spaces.size())
            return;

        const auto blas_space = p.spaces[static_cast<uint32_t>(active_space_id)];
        if (blas_space.root_box < 0 || blas_space.as_type == ASType::None)
            return;

        const glm::vec3 local_swapped_orig = transform_point_3x4(r0, r1, r2, scene_orig);
        const glm::vec3 local_swapped_dir = transform_dir_3x4(r0, r1, r2, scene_dir);
        const glm::vec3 trans = blas_space.normalization_translate;
        const glm::vec3 scale_vec = blas_space.normalization_scale;

        const glm::vec3 local_orig = (local_swapped_orig + trans) * scale_vec;
        const glm::vec3 local_dir_unnorm = local_swapped_dir * scale_vec;
        const float dir_scale2 = glm::dot(local_dir_unnorm, local_dir_unnorm);
        if (dir_scale2 <= 1e-30f)
            return;

        const float dir_scale = sqrtf(dir_scale2);
        const float inv_dir_scale = 1.0f / dir_scale;
        const glm::vec3 local_dir = local_dir_unnorm * inv_dir_scale;
        const float local_ray_tmin = scene_tmin * dir_scale;
        const float local_closest_t = isinf(inout_closest_scene_t) ? INFINITY : (inout_closest_scene_t * dir_scale);

        glm::vec3 local_tuv{local_closest_t, inout_closest_tuv.y, inout_closest_tuv.z};
        uint32_t local_best_tri = UINT32_MAX;
        int blas_it = 0;

        switch (blas_space.as_type)
        {
        case ASType::BVH:
            traverse_as_primitives<ASType::BVH, HasNagata, StackStride, BlasStackSize, true, true>(
                p.as_bvh, blas_space.root_box, blas_stack_base, p, local_orig, local_dir, local_ray_tmin, local_tuv, local_best_tri, blas_it, &box_hits);
            break;
        case ASType::KDTree:
            traverse_as_primitives<ASType::KDTree, HasNagata, StackStride, BlasStackSize, true, true>(
                p.as_kdtree, blas_space.root_box, blas_stack_base, p, local_orig, local_dir, local_ray_tmin, local_tuv, local_best_tri, blas_it, &box_hits);
            break;
        case ASType::UniformGrid:
            traverse_as_primitives<ASType::UniformGrid, HasNagata, StackStride, BlasStackSize, true, true>(
                p.as_uniform_grid, blas_space.root_box, blas_stack_base, p, local_orig, local_dir, local_ray_tmin, local_tuv, local_best_tri, blas_it, &box_hits);
            break;
        case ASType::Octree:
            traverse_as_primitives<ASType::Octree, HasNagata, StackStride, BlasStackSize, true, true>(
                p.as_octree, blas_space.root_box, blas_stack_base, p, local_orig, local_dir, local_ray_tmin, local_tuv, local_best_tri, blas_it, &box_hits);
            break;
        default:
            break;
        }

        iterations += blas_it;

        if (local_best_tri != UINT32_MAX)
        {
            inout_closest_scene_t = local_tuv.x * inv_dir_scale;
            inout_closest_tuv = glm::vec3(inout_closest_scene_t, local_tuv.y, local_tuv.z);
            inout_closest_tri = local_best_tri;
            inout_closest_inst = inst_idx;
            inout_closest_space = static_cast<uint32_t>(active_space_id);
        }
    }

    template <bool HasNagata, uint32_t StackStride, uint32_t BlasStackSize, bool ClipToCell = false, typename RenderParams>
    __host__ __device__ inline void intersect_tlas_subspaces(
        soa::soa_view<BoxContent> tlas_box_content,
        uint32_t container_id,
        const RenderParams& p,
        uint32_t tlas_blas_offset,
        glm::vec3 scene_orig,
        glm::vec3 scene_dir,
        float ray_t_min,
        glm::vec3& closest_tuv,
        uint32_t& closest_tri,
        uint32_t& closest_inst,
        uint32_t& closest_space,
        void* blas_stack_base,
        int& iterations,
        int& box_hits,
        float cell_tmin = 0.0f,
        float cell_tmax = INFINITY,
        InstanceMailbox* mailbox = nullptr)
    {
        for (;; ++container_id)
        {
            const auto primitive = tlas_box_content[container_id];
            const uint32_t prim_id = primitive.id();
            const PrimitiveChunk16 chunk0 = p.primitives[prim_id];
            if (chunk0.type() == PrimitiveType::Subspace)
            {
                const SubspacePrimitive sub = load_primitive<SubspacePrimitive>(p.primitives, prim_id, chunk0);
                intersect_blas_subspace<HasNagata, StackStride, BlasStackSize, ClipToCell>(
                    p,
                    tlas_blas_offset,
                    sub,
                    scene_orig,
                    scene_dir,
                    ray_t_min,
                    closest_tuv.x,
                    closest_tuv,
                    closest_tri,
                    closest_inst,
                    closest_space,
                    blas_stack_base,
                    iterations,
                    box_hits,
                    cell_tmin,
                    cell_tmax,
                    mailbox);
            }
            if (primitive.is_last())
                break;
        }
    }

    template <ASType TlasType, bool HasNagata, uint32_t StackStride, uint32_t BlasStackSize, typename RenderParams>
    struct TlasSubspaceVisitor
    {
        static constexpr bool clip_to_cell = (TlasType != ASType::BVH);

        soa::soa_view<BoxContent> box_content;
        const RenderParams& p;
        uint32_t tlas_blas_offset;
        glm::vec3 scene_orig;
        glm::vec3 scene_dir;
        float ray_t_min;
        glm::vec3& closest_tuv;
        uint32_t& closest_tri;
        uint32_t& closest_inst;
        uint32_t& closest_space;
        void* blas_stack_base;
        int& iterations;
        int& box_hits;
        mutable InstanceMailbox mailbox{};

        __host__ __device__ inline bool operator()(
            uint32_t container_id,
            glm::vec3 /*leaf_orig*/,
            float cell_tmin,
            float cell_tmax,
            bool is_last_cell) const
        {
            ++box_hits;
            const float t_offset = (TlasType == ASType::Octree) ? ray_t_min : 0.0f;
            const float scene_cell_tmin = t_offset + cell_tmin;
            const float scene_cell_tmax = is_last_cell ? INFINITY : (t_offset + cell_tmax);
            intersect_tlas_subspaces<HasNagata, StackStride, BlasStackSize, clip_to_cell>(
                box_content,
                container_id,
                p,
                tlas_blas_offset,
                scene_orig,
                scene_dir,
                ray_t_min,
                closest_tuv,
                closest_tri,
                closest_inst,
                closest_space,
                blas_stack_base,
                iterations,
                box_hits,
                scene_cell_tmin,
                scene_cell_tmax,
                clip_to_cell ? &mailbox : nullptr);
            return false;
        }
    };
} // namespace heterogeneous_detail

template <bool HasNagata, ASType TlasType, typename RenderParams>
__host__ __device__ inline glm::vec4 HeterogeneousASDeviceView::first_hit(
    const RenderParams& p, glm::vec3 source, glm::vec3 direction) const
{
    if (p.root_box < 0)
        return miss_shader(source, direction, p);

    const auto htree = trees[static_cast<uint32_t>(p.root_box)];
    const int32_t tlas_root_box = htree.tlas_root_box;
    if (tlas_root_box < 0)
        return miss_shader(source, direction, p);

    if constexpr (TlasType == ASType::None)
    {
        switch (static_cast<ASType>(htree.tlas_as_type))
        {
        case ASType::BVH:
            return first_hit<HasNagata, ASType::BVH>(p, source, direction);
        case ASType::KDTree:
            return first_hit<HasNagata, ASType::KDTree>(p, source, direction);
        case ASType::UniformGrid:
            return first_hit<HasNagata, ASType::UniformGrid>(p, source, direction);
        case ASType::Octree:
            return first_hit<HasNagata, ASType::Octree>(p, source, direction);
        default:
            return miss_shader(source, direction, p);
        }
    }
    else
    {
        constexpr int max_iterations = 2000;
        constexpr int tlas_max_local_iterations = (TlasType == ASType::Octree) ? 500 : max_iterations;
        constexpr bool tlas_needs_stack = (TlasType == ASType::BVH || TlasType == ASType::KDTree);
        constexpr uint32_t tlas_stack_slots = tlas_needs_stack ? TLAS_STACK_SIZE : 0u;
        constexpr uint32_t active_stack_size = tlas_stack_slots + BLAS_STACK_SIZE;

        int box_hits = 0;
        glm::vec4 hit_primitive_color{0};

#if defined(__CUDA_ARCH__)
        __shared__ BVHStackEntry shared_stack[active_stack_size][BLOCK_THREADS];
        const uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
        void* tlas_stack_base = tlas_needs_stack ? static_cast<void*>(&shared_stack[0][tid]) : nullptr;
        void* blas_stack_base = static_cast<void*>(&shared_stack[tlas_stack_slots][tid]);
        constexpr uint32_t stack_stride = BLOCK_THREADS;
#else
        BVHStackEntry host_stack[active_stack_size];
        void* tlas_stack_base = tlas_needs_stack ? static_cast<void*>(host_stack) : nullptr;
        void* blas_stack_base = static_cast<void*>(host_stack + tlas_stack_slots);
        constexpr uint32_t stack_stride = 1;
#endif

        const uint32_t tlas_blas_offset = htree.blas_offset;
        const auto& tlas_view = heterogeneous_detail::get_as_view<TlasType>(p);

        int it = 0;
        bool continue_traversal = true;
        bool missed = false;
        bool alpha_terminated = false;

        for (int bounce = 0; continue_traversal && it < max_iterations; ++bounce)
        {
            const float ray_t_min = (bounce == 0) ? 0.0f : 1e-4f;
            uint32_t closest_triangle = UINT32_MAX;
            uint32_t closest_inst = 0;
            uint32_t closest_space = 0;
            glm::vec3 closest_tuv{INFINITY, 0.0f, 0.0f};

            heterogeneous_detail::TlasSubspaceVisitor<TlasType, HasNagata, stack_stride, BLAS_STACK_SIZE, RenderParams> visitor{
                tlas_view.box_content,
                p,
                tlas_blas_offset,
                source,
                direction,
                ray_t_min,
                closest_tuv,
                closest_triangle,
                closest_inst,
                closest_space,
                blas_stack_base,
                it,
                box_hits};

            heterogeneous_detail::traverse_as<TlasType, stack_stride, TLAS_STACK_SIZE, false>(
                tlas_view,
                tlas_root_box,
                tlas_stack_base,
                source,
                direction,
                ray_t_min,
                closest_tuv.x,
                it,
                tlas_max_local_iterations,
                max_iterations,
                visitor);

            if (closest_triangle != UINT32_MAX)
            {
                glm::vec4 r0(1.0f, 0.0f, 0.0f, 0.0f);
                glm::vec4 r1(0.0f, 1.0f, 0.0f, 0.0f);
                glm::vec4 r2(0.0f, 0.0f, 1.0f, 0.0f);
                if (closest_inst < instances.size())
                {
                    const auto inst = instances[closest_inst];
                    r0 = inst.s2l_row0;
                    r1 = inst.s2l_row1;
                    r2 = inst.s2l_row2;
                }
                const glm::vec3 blas_scale = (closest_space < p.spaces.size())
                    ? p.spaces[closest_space].normalization_scale
                    : glm::vec3(1.0f);

                auto res = fragment_hit_shader_instanced<HasNagata>(
                    closest_triangle,
                    source,
                    direction,
                    closest_tuv,
                    p,
                    r0,
                    r1,
                    r2,
                    blas_scale,
                    continue_traversal);
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
                break;
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
}
