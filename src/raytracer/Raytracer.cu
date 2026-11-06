#include "Raytracer.cuh"

#include "Scene.hpp"
#include "../cuda_common.hpp"
#include "rt_kernels.cuh"


Raytracer::Raytracer()
{
    start_time=std::chrono::high_resolution_clock::now();
}

void Raytracer::update_parameters(Scene& scene, uint32_t width, uint32_t height)
{
    parameters.t= (std::chrono::high_resolution_clock::now()-start_time).count() / 1000000000.0;
    parameters.ax_c={cos(parameters.ax),sin(parameters.ax)};
    parameters.az_c={cos(parameters.az),sin(parameters.az)};
    parameters.root_box=scene.get_root_box();
    parameters.width=width;
    parameters.height=height;

    parameters.vertices=scene.as.d_vertices.view();
    parameters.triangles=scene.as.d_triangles.view();
    parameters.as=scene.as.d_as.view();
    parameters.containers=scene.as.d_box_content.view();
    parameters.primitives=scene.as.d_primitives.view();
}

void Raytracer::draw_first_hit(Scene& scene, cudaSurfaceObject_t surface, uint32_t width, uint32_t height)
{
    update_parameters(scene,width, height);
    uint32_t block_dim_x=16;//must be multiple of 8
    uint32_t block_dim_y=16;//must be multiple of 8
    dim3 grid_dim{(width-1)/block_dim_x+1,(height-1)/block_dim_y+1 ,1};
    draw_first_hit_kernel<<<grid_dim,{64,block_dim_x/8,block_dim_y/8}>>>(surface,parameters);
}