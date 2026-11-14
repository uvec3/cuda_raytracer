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

    template <bool HasNagata = true, typename RenderParams>
    __host__ __device__ inline glm::vec4 first_hit(const RenderParams& p, glm::vec3 source, glm::vec3 direction) const;
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
    uint32_t cell_x : 10 = 0;
    uint32_t cell_y : 10 = 0;
    uint32_t cell_z : 10 = 0;
    uint32_t initialized : 1 = 0;
    uint32_t exited : 1 = 0;
    float t_min = 0.0f;
    float t_max = 0.0f;
};
static_assert(sizeof(CurrentGridCell) == 12);

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
        state.exited = 1;
        return 0;
    }

    glm::ivec3 cell;
    if (!state.initialized)
    {
        const glm::vec3 grid_max = glm::vec3(dims) * cell_size;

        float t_enter = glm::max(0.0f, state.t_max);
        float t_leave = 1e30f;

        for (int a = 0; a < 3; ++a)
        {
            if (glm::abs(dir[a]) < 1e-20f)
            {
                if (orig[a] < 0.0f || orig[a] > grid_max[a])
                {
                    state.exited = 1;
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
            state.exited = 1;
            return 0;
        }

        const glm::vec3 p = orig + dir * t_enter;
        cell = glm::clamp(
            glm::ivec3(glm::floor(p / cell_size)),
            glm::ivec3(0),
            glm::ivec3(dims) - 1);

        state.cell_x = static_cast<uint32_t>(cell.x);
        state.cell_y = static_cast<uint32_t>(cell.y);
        state.cell_z = static_cast<uint32_t>(cell.z);
        state.t_max = t_enter;
        state.initialized = 1;
    }
    else
    {
        cell = glm::ivec3(
            static_cast<int>(state.cell_x),
            static_cast<int>(state.cell_y),
            static_cast<int>(state.cell_z));
    }

    glm::ivec3 step;
    glm::vec3 t_next;
    glm::vec3 t_delta;
    for (int a = 0; a < 3; ++a)
    {
        if (dir[a] >= 1e-20f)
        {
            const float inv_d = 1.0f / dir[a];
            step[a] = 1;
            t_next[a] = (static_cast<float>(cell[a] + 1) * cell_size - orig[a]) * inv_d;
            t_delta[a] = cell_size * inv_d;
        }
        else if (dir[a] <= -1e-20f)
        {
            const float inv_d = 1.0f / dir[a];
            step[a] = -1;
            t_next[a] = (static_cast<float>(cell[a]) * cell_size - orig[a]) * inv_d;
            t_delta[a] = -cell_size * inv_d;
        }
        else
        {
            step[a] = 0;
            t_next[a] = 1e30f;
            t_delta[a] = 1e30f;
        }
    }

    if (step.x == 0 && step.y == 0 && step.z == 0)
    {
        state.exited = 1;
        return 0;
    }

    const int max_iterations = static_cast<int>(dims.x + dims.y + dims.z);
    while (!state.exited && iterations < max_iterations)
    {
        ++iterations;

        const uint32_t linear_idx = uniform_grid_cell_index(glm::uvec3(cell), dims);
        const uint32_t content = as[as_start + linear_idx];

        float exit_t;
        if (t_next.x <= t_next.y && t_next.x <= t_next.z)
        {
            exit_t = t_next.x;
            cell.x += step.x;
            t_next.x += t_delta.x;
            if (cell.x < 0 || cell.x >= static_cast<int>(dims.x))
            {
                state.exited = 1;
            }
        }
        else if (t_next.y <= t_next.z)
        {
            exit_t = t_next.y;
            cell.y += step.y;
            t_next.y += t_delta.y;
            if (cell.y < 0 || cell.y >= static_cast<int>(dims.y))
            {
                state.exited = 1;
            }
        }
        else
        {
            exit_t = t_next.z;
            cell.z += step.z;
            t_next.z += t_delta.z;
            if (cell.z < 0 || cell.z >= static_cast<int>(dims.z))
            {
                state.exited = 1;
            }
        }

        state.t_min = state.t_max;
        state.t_max = glm::max(state.t_max, exit_t);

        if (content > 0)
        {
            state.cell_x = static_cast<uint32_t>(cell.x);
            state.cell_y = static_cast<uint32_t>(cell.y);
            state.cell_z = static_cast<uint32_t>(cell.z);
            return content;
        }
    }

    state.cell_x = static_cast<uint32_t>(cell.x);
    state.cell_y = static_cast<uint32_t>(cell.y);
    state.cell_z = static_cast<uint32_t>(cell.z);
    if (iterations >= max_iterations)
    {
        state.exited = 1;
    }

    return 0;
}

__host__ __device__ inline uint32_t traverse_uniform_grid(
    const UniformGridASDeviceView& view,
    int grid_id,
    CurrentGridCell& state,
    glm::vec3 orig,
    glm::vec3 dir,
    int& iterations)
{
    if (state.exited || grid_id < 0)
    {
        state.exited = 1;
        return 0;
    }

    const auto grid = view.grids[static_cast<uint32_t>(grid_id)];
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

template <bool HasNagata, typename RenderParams>
__host__ __device__ inline
glm::vec4 UniformGridASDeviceView::first_hit(const RenderParams& p, glm::vec3 source, glm::vec3 direction) const
{
    if (p.root_box < 0)
        return miss_shader(source, direction, p);

    constexpr int max_iterations = 2000;

    int box_hits = 0;
    glm::vec4 hit_primitive_color{0};

    uint32_t container_id = UINT32_MAX;
    CurrentGridCell current_cell{};

    int it = 0;
    int bounce = 0;
    bool continue_traversal = true;

    while (container_id != 0 && continue_traversal && it < max_iterations)
    {
        container_id = traverse_uniform_grid(*this, p.root_box, current_cell, source, direction, it);

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
                    current_cell.t_min,
                    current_cell.t_max,
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
                        current_cell = CurrentGridCell{.t_min = 1e-4f, .t_max = 1e-4f};
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

    if (it >= max_iterations)
        return glm::vec4{1.f};

    if (!p.show_iterations && !p.show_boxes)
    {
        if (current_cell.exited)
            hit_primitive_color = blend_colors(hit_primitive_color, miss_shader(source, direction, p));
        return hit_primitive_color;
    }

    return compose_debug_overlay(p, hit_primitive_color, it, box_hits);
}
