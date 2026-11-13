
#include "../types.hpp"
#include "Raytracer.cuh"

#include "Scene.hpp"
#include "../cuda_common.hpp"
#include "rt_kernels.cuh"


Raytracer::Raytracer()
{
    start_time = std::chrono::high_resolution_clock::now();
}

void Raytracer::update_parameters(Scene& scene, uint32_t width, uint32_t height)
{
    parameters.t= (std::chrono::high_resolution_clock::now()-start_time).count() / 1000000000.0;

    parameters.root_box=scene.get_root_box();
    parameters.width=width;
    parameters.height=height;

    parameters.as_octree = scene.as_octree.device_view();
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
    else if (scene.spaces.is_allocated(active_space) && scene.spaces[active_space].as_type == ASType::Octree)
    {
        uint32_t block_dim_x=16;//must be multiple of 8
        uint32_t block_dim_y=16;//must be multiple of 8
        dim3 grid_dim{(width-1)/block_dim_x+1,(height-1)/block_dim_y+1 ,1};
        draw_first_hit_kernel<<<grid_dim,{64,block_dim_x/8,block_dim_y/8}, 0, stream>>>(surface,parameters);
    }
}
