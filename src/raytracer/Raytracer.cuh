#pragma once
#include <chrono>

#include "rt_kernels.cuh"
#include "Scene.hpp"
#include "../acceleration_structures/octree/OctreeAS.cuh"
#include "../../external/soa_vector/src/soa_vector.hpp"
#include "../types.hpp"

__host__ __device__
glm::vec4 draw_pixel(uint32_t x, uint32_t y, RenderParameters p);


class Raytracer
{
public:
    RenderParameters parameters{};
    std::chrono::high_resolution_clock::time_point start_time;
    Raytracer();
    void update_parameters(Scene& scene, uint32_t width, uint32_t height);
    void draw(Scene& scene, cudaSurfaceObject_t surface, uint32_t width, uint32_t height);
};




