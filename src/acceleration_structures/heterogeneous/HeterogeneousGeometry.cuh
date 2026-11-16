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
    static constexpr uint32_t BLOCK_THREADS = 32; // 32 * 128 * 8B = 32 KB shared memory

    soa::soa_view<HeterogeneousTreeInfo> trees;
    soa::soa_view<HeterogeneousInstanceInfo> instances;

    template <bool HasNagata = true, typename RenderParams>
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

    template <bool HasNagata, uint32_t StackStride, uint32_t BlasStackSize, typename RenderParams>
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
        int& box_hits)
    {
        float box_tmin = 0.0f;
        float box_tmax = 0.0f;
        const glm::vec3 scene_inv_dir = bvh_safe_inv_dir(scene_dir);
        if (!bvh_intersect_aabb(
                sub.bounds_min, sub.bounds_max,
                scene_orig, scene_inv_dir,
                scene_tmin, inout_closest_scene_t,
                box_tmin, box_tmax))
        {
            return;
        }

        const uint32_t inst_idx = tlas_blas_offset + sub.blas_info_id;
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

        constexpr int max_blas_iterations = 500;
        int blas_it = 0;

        const ASType blas_type = blas_space.as_type;
        if (blas_type == ASType::BVH)
        {
            const bool is_bvh16 = (p.as_bvh.trees[static_cast<uint32_t>(blas_space.root_box)].use_bvh16 != 0u);
            CurrentBVHNode state{.t_min = local_ray_tmin};
            while (!state.exited && blas_it < max_blas_iterations)
            {
                const uint32_t container_id = is_bvh16
                    ? traverse_bvh16<StackStride, BlasStackSize>(
                        p.as_bvh,
                        blas_space.root_box,
                        state,
                        reinterpret_cast<BVHStackEntry*>(blas_stack_base),
                        local_orig,
                        local_dir,
                        blas_it,
                        local_tuv.x,
                        local_ray_tmin)
                    : traverse_bvh<StackStride, BlasStackSize>(
                        p.as_bvh,
                        blas_space.root_box,
                        state,
                        reinterpret_cast<BVHStackEntry*>(blas_stack_base),
                        local_orig,
                        local_dir,
                        blas_it,
                        local_tuv.x,
                        local_ray_tmin);

                if (container_id > 0)
                {
                    ++box_hits;
                    if (p.show_triangles)
                    {
                        intersect_box_primitives<HasNagata, false>(
                            p.as_bvh.box_content,
                            container_id,
                            p,
                            local_orig,
                            local_dir,
                            local_ray_tmin,
                            fminf(local_tuv.x, 1e30f),
                            local_best_tri,
                            local_tuv);
                    }
                }
            }
        }
        else if (blas_type == ASType::KDTree)
        {
            CurrentKDNode state{};
            state.t_max = local_ray_tmin;
            while (!state.exited && blas_it < max_blas_iterations)
            {
                const uint32_t container_id = traverse_kdtree<StackStride, BlasStackSize>(
                    p.as_kdtree,
                    blas_space.root_box,
                    state,
                    reinterpret_cast<KDStackEntry*>(blas_stack_base),
                    local_orig,
                    local_dir,
                    blas_it);

                if (container_id > 0)
                {
                    if (state.t_min > local_tuv.x)
                        break;

                    ++box_hits;
                    if (p.show_triangles)
                    {
                        intersect_box_primitives<HasNagata>(
                            p.as_kdtree.box_content,
                            container_id,
                            p,
                            local_orig,
                            local_dir,
                            fmaxf(state.t_min, local_ray_tmin),
                            fminf(state.t_max, local_tuv.x),
                            local_best_tri,
                            local_tuv);
                    }
                }
            }
        }
        else if (blas_type == ASType::UniformGrid)
        {
            CurrentGridCell state{};
            state.t_max = local_ray_tmin;
            while (!state.exited && blas_it < 2000)
            {
                const uint32_t container_id = traverse_uniform_grid(
                    p.as_uniform_grid,
                    blas_space.root_box,
                    state,
                    local_orig,
                    local_dir,
                    blas_it);

                if (container_id > 0)
                {
                    if (state.t_min > local_tuv.x)
                        break;

                    ++box_hits;
                    if (p.show_triangles)
                    {
                        intersect_box_primitives<HasNagata>(
                            p.as_uniform_grid.box_content,
                            container_id,
                            p,
                            local_orig,
                            local_dir,
                            fmaxf(state.t_min, local_ray_tmin),
                            fminf(state.t_max, local_tuv.x),
                            local_best_tri,
                            local_tuv);
                    }
                }
            }
        }
        else if (blas_type == ASType::Octree)
        {
            if (blas_space.root_box >= 0)
            {
                const glm::vec3 oct_orig = (local_ray_tmin > 0.0f) ? (local_orig + local_dir * local_ray_tmin) : local_orig;
                glm::vec3 oct_tuv{isinf(local_tuv.x) ? INFINITY : fmaxf(0.0f, local_tuv.x - local_ray_tmin), local_tuv.y, local_tuv.z};
                uint32_t oct_best_tri = UINT32_MAX;

                CurrentBox state{.id = blas_space.root_box};
                while (blas_it < max_blas_iterations)
                {
                    const TraversalResult res = traverse_stackless(&p.as_octree.as[0], state, oct_orig, local_dir, blas_it);
                    const uint32_t container_id = res.content;

                    if (container_id > 0)
                    {
                        if (state.t_min > oct_tuv.x)
                            break;

                        ++box_hits;
                        if (p.show_triangles)
                        {
                            const float cell_tmin = state.t_min;
                            const float cell_tmax = (state.t_max > state.t_min) ? fminf(state.t_max, oct_tuv.x) : oct_tuv.x;
                            intersect_box_primitives<HasNagata>(
                                p.as_octree.box_content,
                                container_id,
                                p,
                                oct_orig,
                                local_dir,
                                cell_tmin,
                                cell_tmax,
                                oct_best_tri,
                                oct_tuv);
                        }
                    }

                    if (state.stack.depth() == 0)
                        break;
                }

                if (oct_best_tri != UINT32_MAX)
                {
                    local_best_tri = oct_best_tri;
                    local_tuv = glm::vec3(local_ray_tmin + oct_tuv.x, oct_tuv.y, oct_tuv.z);
                }
            }
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

    template <bool HasNagata, uint32_t StackStride, uint32_t BlasStackSize, typename RenderParams>
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
        int& box_hits)
    {
        for (;; ++container_id)
        {
            const auto primitive = tlas_box_content[container_id];
            const uint32_t prim_id = primitive.id();
            const PrimitiveChunk16 chunk0 = p.primitives[prim_id];
            if (chunk0.type() == PrimitiveType::Subspace)
            {
                const SubspacePrimitive sub = load_primitive<SubspacePrimitive>(p.primitives, prim_id, chunk0);
                intersect_blas_subspace<HasNagata, StackStride, BlasStackSize>(
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
                    box_hits);
            }
            if (primitive.is_last())
                break;
        }
    }
} // namespace heterogeneous_detail

template <bool HasNagata, typename RenderParams>
__host__ __device__ inline glm::vec4 HeterogeneousASDeviceView::first_hit(
    const RenderParams& p, glm::vec3 source, glm::vec3 direction) const
{
    if (p.root_box < 0)
        return miss_shader(source, direction, p);

    constexpr int max_iterations = 2000;
    int box_hits = 0;
    glm::vec4 hit_primitive_color{0};

#if defined(__CUDA_ARCH__)
    __shared__ BVHStackEntry shared_stack[TOTAL_STACK_SIZE][BLOCK_THREADS];
    const uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
    BVHStackEntry* tlas_stack_base = &shared_stack[0][tid];
    BVHStackEntry* blas_stack_base = &shared_stack[TLAS_STACK_SIZE][tid];
    constexpr uint32_t stack_stride = BLOCK_THREADS;
#else
    BVHStackEntry host_stack[TOTAL_STACK_SIZE];
    BVHStackEntry* tlas_stack_base = host_stack;
    BVHStackEntry* blas_stack_base = host_stack + TLAS_STACK_SIZE;
    constexpr uint32_t stack_stride = 1;
#endif

    const auto htree = trees[static_cast<uint32_t>(p.root_box)];
    const int32_t tlas_root_box = htree.tlas_root_box;
    const ASType tlas_type = static_cast<ASType>(htree.tlas_as_type);
    const uint32_t tlas_blas_offset = htree.blas_offset;
    if (tlas_root_box < 0)
        return miss_shader(source, direction, p);

    int it = 0;
    bool continue_traversal = true;
    bool missed = false;

    for (int bounce = 0; bounce < 8 && continue_traversal && it < max_iterations; ++bounce)
    {
        const float ray_t_min = (bounce == 0) ? 0.0f : 1e-4f;
        uint32_t closest_triangle = UINT32_MAX;
        uint32_t closest_inst = 0;
        uint32_t closest_space = 0;
        glm::vec3 closest_tuv{INFINITY, 0.0f, 0.0f};

        if (tlas_type == ASType::BVH)
        {
            CurrentBVHNode state{.t_min = ray_t_min};
            while (!state.exited && it < max_iterations)
            {
                const uint32_t container_id = traverse_bvh16<stack_stride, TLAS_STACK_SIZE>(
                    p.as_bvh,
                    tlas_root_box,
                    state,
                    tlas_stack_base,
                    source,
                    direction,
                    it,
                    closest_tuv.x,
                    ray_t_min);

                if (container_id > 0)
                {
                    ++box_hits;
                    heterogeneous_detail::intersect_tlas_subspaces<HasNagata, stack_stride, BLAS_STACK_SIZE>(
                        p.as_bvh.box_content,
                        container_id,
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
                        box_hits);
                }
            }
        }
        else if (tlas_type == ASType::KDTree)
        {
            CurrentKDNode state{};
            state.t_max = ray_t_min;
            while (!state.exited && it < max_iterations)
            {
                const uint32_t container_id = traverse_kdtree<stack_stride, TLAS_STACK_SIZE>(
                    p.as_kdtree,
                    tlas_root_box,
                    state,
                    reinterpret_cast<KDStackEntry*>(tlas_stack_base),
                    source,
                    direction,
                    it);

                if (container_id > 0)
                {
                    if (state.t_min > closest_tuv.x)
                        break;

                    ++box_hits;
                    heterogeneous_detail::intersect_tlas_subspaces<HasNagata, stack_stride, BLAS_STACK_SIZE>(
                        p.as_kdtree.box_content,
                        container_id,
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
                        box_hits);
                }
            }
        }
        else if (tlas_type == ASType::UniformGrid)
        {
            CurrentGridCell state{};
            state.t_max = ray_t_min;
            int tlas_grid_it = 0;
            while (!state.exited && tlas_grid_it < 2000 && it < max_iterations)
            {
                const uint32_t container_id = traverse_uniform_grid(
                    p.as_uniform_grid,
                    tlas_root_box,
                    state,
                    source,
                    direction,
                    tlas_grid_it);

                if (container_id > 0)
                {
                    if (state.t_min > closest_tuv.x)
                        break;

                    ++box_hits;
                    heterogeneous_detail::intersect_tlas_subspaces<HasNagata, stack_stride, BLAS_STACK_SIZE>(
                        p.as_uniform_grid.box_content,
                        container_id,
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
                        box_hits);
                }
            }
            it += tlas_grid_it;
        }
        else if (tlas_type == ASType::Octree)
        {
            const glm::vec3 oct_orig = (ray_t_min > 0.0f) ? (source + direction * ray_t_min) : source;
            int tlas_oct_it = 0;
            CurrentBox state{.id = tlas_root_box};
            while (tlas_oct_it < 500 && it < max_iterations)
            {
                const TraversalResult res = traverse_stackless(&p.as_octree.as[0], state, oct_orig, direction, tlas_oct_it);
                const uint32_t container_id = res.content;

                if (container_id > 0)
                {
                    if (ray_t_min + state.t_min > closest_tuv.x)
                        break;

                    ++box_hits;
                    heterogeneous_detail::intersect_tlas_subspaces<HasNagata, stack_stride, BLAS_STACK_SIZE>(
                        p.as_octree.box_content,
                        container_id,
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
                        box_hits);
                }

                if (state.stack.depth() == 0)
                    break;
            }
            it += tlas_oct_it;
        }

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
        if (missed)
            hit_primitive_color = blend_colors(hit_primitive_color, miss_shader(source, direction, p));
        return hit_primitive_color;
    }

    return compose_debug_overlay(p, hit_primitive_color, it, box_hits);
}
