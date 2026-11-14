#pragma once

#include "../../types.hpp"
#include "../../geometry_common.cuh"
#include "../../raytracer/rt_shaders.cuh"

struct GridInfo
{
    glm::uvec3& dims;
    float& cell_size;
    uint32_t& as_start;
};

struct UniformGridASDeviceView
{
    soa::soa_view<GridInfo> grids;
    soa::soa_view<uint32_t> as;
    soa::soa_view<BoxContent> box_content;

    template <typename RenderParams>
    __host__ __device__ inline glm::vec4 first_hit(RenderParams p, glm::vec3 source, glm::vec3 direction) const;
};

struct CellPrimPair
{
    uint32_t& cell_id;
    uint32_t& prim_id;
};

__host__ __device__ __forceinline__
uint32_t uniform_grid_cell_index(glm::uvec3 cell, glm::uvec3 dims)
{
    return cell.x + cell.y * dims.x + cell.z * (dims.x * dims.y);
}

struct CurrentGridCell
{
    int grid_id = -1;
    glm::ivec3 cell = glm::ivec3(0);
    glm::ivec3 step = glm::ivec3(0);
    glm::vec3 t_next = glm::vec3(0.0f);
    glm::vec3 t_delta = glm::vec3(0.0f);
    float t_min = 0.0f;
    float t_max = 0.0f;
    bool initialized = false;
    bool exited = false;
};

__host__ __device__ inline uint32_t traverse_uniform_grid(
    const uint32_t* as,
    glm::uvec3 dims,
    float cell_size,
    uint32_t as_start,
    CurrentGridCell& state,
    glm::vec3 orig,
    glm::vec3 dir,
    int& iterations)
{
    if (state.exited || dims.x == 0 || dims.y == 0 || dims.z == 0 || cell_size <= 0.0f)
    {
        state.exited = true;
        return 0;
    }

    if (!state.initialized)
    {
        const glm::vec3 grid_max = glm::vec3(dims) * cell_size;

        float t_enter = state.t_max;
        float t_leave = 1e30f;

        for (int a = 0; a < 3; ++a)
        {
            if (glm::abs(dir[a]) < 1e-20f)
            {
                if (orig[a] < 0.0f || orig[a] > grid_max[a])
                {
                    state.exited = true;
                    return 0;
                }
            }
            else
            {
                const float inv_d = 1.0f / dir[a];
                float t0 = (0.0f - orig[a]) * inv_d;
                float t1 = (grid_max[a] - orig[a]) * inv_d;
                if (t0 > t1)
                {
                    const float tmp = t0;
                    t0 = t1;
                    t1 = tmp;
                }
                t_enter = glm::max(t_enter, t0);
                t_leave = glm::min(t_leave, t1);
            }
        }

        if (t_enter > t_leave || t_leave < 0.0f)
        {
            state.exited = true;
            return 0;
        }

        const glm::vec3 p = orig + dir * t_enter;
        state.cell = glm::clamp(
            glm::ivec3(glm::floor(p / cell_size)),
            glm::ivec3(0),
            glm::ivec3(dims) - 1);

        for (int a = 0; a < 3; ++a)
        {
            if (dir[a] >= 1e-20f)
            {
                state.step[a] = 1;
                const float next_boundary = static_cast<float>(state.cell[a] + 1) * cell_size;
                state.t_next[a] = (next_boundary - orig[a]) / dir[a];
                state.t_delta[a] = cell_size / dir[a];
            }
            else if (dir[a] <= -1e-20f)
            {
                state.step[a] = -1;
                const float next_boundary = static_cast<float>(state.cell[a]) * cell_size;
                state.t_next[a] = (next_boundary - orig[a]) / dir[a];
                state.t_delta[a] = -cell_size / dir[a];
            }
            else
            {
                state.step[a] = 0;
                state.t_next[a] = 1e30f;
                state.t_delta[a] = 1e30f;
            }
        }

        if (state.step.x == 0 && state.step.y == 0 && state.step.z == 0)
        {
            state.exited = true;
            return 0;
        }

        state.t_max = t_enter;
        state.initialized = true;
    }

    const int max_iterations = static_cast<int>(dims.x + dims.y + dims.z);
    while (!state.exited && iterations < max_iterations)
    {
        ++iterations;

        const uint32_t linear_idx = uniform_grid_cell_index(glm::uvec3(state.cell), dims);
        const uint32_t content = as[as_start + linear_idx];

        float exit_t;
        if (state.t_next.x <= state.t_next.y && state.t_next.x <= state.t_next.z)
        {
            exit_t = state.t_next.x;
            state.cell.x += state.step.x;
            state.t_next.x += state.t_delta.x;
            if (state.cell.x < 0 || state.cell.x >= static_cast<int>(dims.x))
            {
                state.exited = true;
            }
        }
        else if (state.t_next.y <= state.t_next.z)
        {
            exit_t = state.t_next.y;
            state.cell.y += state.step.y;
            state.t_next.y += state.t_delta.y;
            if (state.cell.y < 0 || state.cell.y >= static_cast<int>(dims.y))
            {
                state.exited = true;
            }
        }
        else
        {
            exit_t = state.t_next.z;
            state.cell.z += state.step.z;
            state.t_next.z += state.t_delta.z;
            if (state.cell.z < 0 || state.cell.z >= static_cast<int>(dims.z))
            {
                state.exited = true;
            }
        }

        state.t_min = state.t_max;
        state.t_max = glm::max(state.t_max, exit_t);

        if (content > 0)
        {
            return content;
        }
    }

    if (iterations >= max_iterations)
    {
        state.exited = true;
    }

    return 0;
}

__host__ __device__ inline uint32_t traverse_uniform_grid(
    UniformGridASDeviceView view,
    CurrentGridCell& state,
    glm::vec3 orig,
    glm::vec3 dir,
    int& iterations)
{
    if (state.exited || state.grid_id < 0)
    {
        state.exited = true;
        return 0;
    }

    const auto grid = view.grids[static_cast<uint32_t>(state.grid_id)];
    return traverse_uniform_grid(
        &view.as[0],
        grid.dims,
        grid.cell_size,
        grid.as_start,
        state,
        orig,
        dir,
        iterations);
}

template <typename RenderParams>
__host__ __device__ inline
glm::vec4 UniformGridASDeviceView::first_hit(RenderParams p, glm::vec3 source, glm::vec3 direction) const
{
    if (p.root_box < 0)
        return miss_shader(source, direction, p);

    const auto grid = grids[static_cast<uint32_t>(p.root_box)];
    const glm::uvec3 dims = grid.dims;
    const float cell_size = grid.cell_size;
    const uint32_t as_start = grid.as_start;
    const int max_iterations = static_cast<int>(dims.x + dims.y + dims.z);

    int box_hits = 0;
    glm::vec4 hit_primitive_color{0};

    uint32_t container_id = UINT32_MAX;
    CurrentGridCell current_cell{.grid_id = p.root_box};

    int it = 0;
    int total_it = 0;
    int bounce = 0;
    bool continue_traversal = true;

    while (container_id != 0 && continue_traversal && it < max_iterations)
    {
        container_id = traverse_uniform_grid(&as[0], dims, cell_size, as_start, current_cell, source, direction, it);

        if (container_id > 0)
        {
            uint32_t closest_triangle = UINT32_MAX;
            glm::vec3 closest_tuv{INFINITY, 0, 0};
            ++box_hits;

            if (p.show_triangles)
            {
                intersect_box_primitives(
                    box_content,
                    container_id,
                    p,
                    source,
                    direction,
                    current_cell.t_min,
                    current_cell.t_max,
                    closest_triangle,
                    closest_tuv);
            }

            if (closest_triangle != UINT32_MAX)
            {
                auto res = fragment_hit_shader(closest_triangle, source, direction, closest_tuv, p, continue_traversal);
                hit_primitive_color = blend_colors(hit_primitive_color, res);
                if (continue_traversal)
                {
                    if (++bounce < 8)
                    {
                        total_it += it;
                        it = 0;
                        current_cell = CurrentGridCell{.grid_id = p.root_box, .t_min = 1e-4f, .t_max = 1e-4f};
                    }
                    else
                    {
                        continue_traversal = false;
                    }
                }
            }
        }

        if (current_cell.exited)
        {
            break;
        }
    }

    total_it += it;

    if (it >= max_iterations)
        return glm::vec4{1.f};

    if (!p.show_iterations && !p.show_boxes)
    {
        if (current_cell.exited)
            hit_primitive_color = blend_colors(hit_primitive_color, miss_shader(source, direction, p));
        return hit_primitive_color;
    }

    return compose_debug_overlay(p, hit_primitive_color, total_it, box_hits);
}
