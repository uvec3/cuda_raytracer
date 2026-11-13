#pragma once
#include <chrono>

#include "../../external/soa_vector/src/soa_vector.hpp"
#include "rt_kernels.cuh"


//expose for the host-side debugging purposes
__host__ __device__
glm::vec4 trace_pixel_first_hit(uint32_t x, uint32_t y, RenderParameters p);


class Scene;

class Raytracer
{
public:
    RenderParameters parameters{};
    std::chrono::high_resolution_clock::time_point start_time;
    Raytracer();
    void update_parameters(Scene& scene, uint32_t width, uint32_t height);
    void draw_first_hit(Scene& scene, cudaSurfaceObject_t surface, uint32_t width, uint32_t height, cudaStream_t stream = 0);

    void buildAS();
};