
#include "../types.hpp"
#include "Raytracer.cuh"

#include "Scene.hpp"
#include "../cuda_common.hpp"
#include "rt_kernels.cuh"
#include <algorithm>
#include <cmath>
#include <limits>


Raytracer::Raytracer()
{
    start_time = std::chrono::high_resolution_clock::now();
}

void Raytracer::reset_animations_state()
{
    active_animations.clear();
    animation_playing = false;
    animation_time = 0.0f;
    animation_dirty = false;
    has_last_animation_tick = false;
}

void Raytracer::step_animations(Scene& scene)
{
    float min_start = std::numeric_limits<float>::infinity();
    float max_end = -std::numeric_limits<float>::infinity();
    bool has_valid = false;
    for (uint32_t idx : active_animations)
    {
        if (idx < scene.m_animations.size())
        {
            const Animation& anim = scene.m_animations[idx];
            min_start = std::min(min_start, anim.start_time);
            max_end = std::max(max_end, anim.end_time);
            has_valid = true;
        }
    }

    if (!has_valid)
    {
        has_last_animation_tick = false;
        animation_dirty = false;
        return;
    }

    if (animation_playing)
    {
        const auto now = std::chrono::high_resolution_clock::now();
        const float dt = has_last_animation_tick
            ? std::chrono::duration<float>(now - last_animation_tick).count()
            : 0.0f;
        last_animation_tick = now;
        has_last_animation_tick = true;

        animation_time += dt * animation_speed;
        const float span = max_end - min_start;
        if (span > 1e-6f)
        {
            if (animation_loop)
            {
                float offset = std::fmod(animation_time - min_start, span);
                if (offset < 0.0f)
                    offset += span;
                animation_time = min_start + offset;
            }
            else
            {
                if (animation_speed >= 0.0f && animation_time >= max_end)
                {
                    animation_time = max_end;
                    animation_playing = false;
                    has_last_animation_tick = false;
                }
                else if (animation_speed < 0.0f && animation_time <= min_start)
                {
                    animation_time = min_start;
                    animation_playing = false;
                    has_last_animation_tick = false;
                }
                else
                {
                    animation_time = std::clamp(animation_time, min_start, max_end);
                }
            }
        }
        else
        {
            animation_time = min_start;
            if (!animation_loop)
            {
                animation_playing = false;
                has_last_animation_tick = false;
            }
        }

        if (scene.apply_animations(active_animations, animation_time, animation_loop))
        {
            scene.rebuild_active_as();
        }
        animation_dirty = false;
    }
    else
    {
        has_last_animation_tick = false;
        if (animation_dirty)
        {
            if (scene.apply_animations(active_animations, animation_time, animation_loop))
            {
                scene.rebuild_active_as();
            }
            animation_dirty = false;
        }
    }
}

void Raytracer::update_parameters(Scene& scene, uint32_t width, uint32_t height)
{
    scene.as_heterogeneous.set_probe_render_params(parameters);
    step_animations(scene);

    parameters.t= (std::chrono::high_resolution_clock::now()-start_time).count() / 1000000000.0;

    parameters.root_box=scene.get_root_box();
    parameters.width=width;
    parameters.height=height;

    uint32_t active_space = scene.get_active_space();
    parameters.active_as_type = scene.spaces.is_allocated(active_space) ? scene.spaces[active_space].as_type : ASType::Octree;
    parameters.as_octree = scene.as_octree.device_view();
    parameters.as_uniform_grid = scene.as_uniform_grid.device_view();
    parameters.as_kdtree = scene.as_kdtree.device_view();
    parameters.as_bvh = scene.as_bvh.device_view();
    parameters.as_heterogeneous = scene.as_heterogeneous.device_view();
    parameters.vertices = scene.geometry_context.vertices.view();
    parameters.materials = scene.geometry_context.materials.view();
    parameters.primitives = scene.geometry_context.primitives.view();
    parameters.spaces = scene.geometry_context.d_spaces.view();
    parameters.update_camera();
}

void Raytracer::draw_first_hit(Scene& scene, cudaSurfaceObject_t surface, uint32_t width, uint32_t height, cudaStream_t stream)
{
    uint32_t active_space = scene.get_active_space();
    if (scene.spaces.is_allocated(active_space) && scene.spaces[active_space].as_type == ASType::Optix)
    {
        scene.as_optix.render(active_space, surface, parameters, stream);
    }
    else if (scene.spaces.is_allocated(active_space))
    {
        const ASType active_type = scene.spaces[active_space].as_type;
        static_assert(RENDER_BLOCK_THREADS == 64, "threadPixelPos() assumes an 8x8 tile (64 threads/block)");
        constexpr uint32_t block_dim_x = 8u;
        constexpr uint32_t block_dim_y = 8u;
        dim3 grid_dim{(width - 1) / block_dim_x + 1, (height - 1) / block_dim_y + 1, 1};
        dim3 block_dim{RENDER_BLOCK_THREADS, 1, 1};

        const bool has_nagata = scene.spaces[active_space].has_nagata;
        auto launch = [&]<ASType Type, bool UseBVH16 = false, ASType TlasType = ASType::None>()
        {
            if (has_nagata)
                draw_first_hit_kernel<Type, true, UseBVH16, TlasType><<<grid_dim, block_dim, 0, stream>>>(surface, parameters);
            else
                draw_first_hit_kernel<Type, false, UseBVH16, TlasType><<<grid_dim, block_dim, 0, stream>>>(surface, parameters);
        };

        switch (active_type)
        {
        case ASType::Octree:
            launch.template operator()<ASType::Octree>();
            break;
        case ASType::UniformGrid:
            launch.template operator()<ASType::UniformGrid>();
            break;
        case ASType::KDTree:
            launch.template operator()<ASType::KDTree>();
            break;
        case ASType::BVH:
        {
            const int root_box = scene.spaces[active_space].root_box;
            const bool use_bvh16 = (root_box >= 0) && (scene.as_bvh.trees[static_cast<size_t>(root_box)].use_bvh16 != 0u);
            if (use_bvh16)
                launch.template operator()<ASType::BVH, true>();
            else
                launch.template operator()<ASType::BVH, false>();
            break;
        }
        case ASType::Heterogeneous:
        {
            const int root_box = scene.spaces[active_space].root_box;
            const ASType tlas_type = (root_box >= 0 && scene.as_heterogeneous.trees.is_allocated(static_cast<size_t>(root_box)))
                ? static_cast<ASType>(scene.as_heterogeneous.trees[static_cast<size_t>(root_box)].tlas_as_type)
                : ASType::BVH;
            switch (tlas_type)
            {
            case ASType::KDTree:
                launch.template operator()<ASType::Heterogeneous, false, ASType::KDTree>();
                break;
            case ASType::UniformGrid:
                launch.template operator()<ASType::Heterogeneous, false, ASType::UniformGrid>();
                break;
            case ASType::Octree:
                launch.template operator()<ASType::Heterogeneous, false, ASType::Octree>();
                break;
            case ASType::BVH:
            default:
                launch.template operator()<ASType::Heterogeneous, false, ASType::BVH>();
                break;
            }
            break;
        }
        default:
            break;
        }
    }
}