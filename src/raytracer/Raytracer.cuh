#pragma once
#include <chrono>
#include <cstdint>
#include <vector>

#include "../../external/soa_vector/src/soa_vector.hpp"
#include "rt_kernels.cuh"


//expose for the host-side debugging purposes
__host__ __device__
glm::vec4 trace_pixel_first_hit(uint32_t x, uint32_t y, const RenderParameters& p);


class Scene;

class Raytracer
{
public:
    RenderParameters parameters{};
    std::chrono::high_resolution_clock::time_point start_time;

    std::vector<uint32_t> active_animations;
    bool animation_playing = false;
    bool animation_loop = true;
    float animation_speed = 1.0f;
    float animation_time = 0.0f;
    bool animation_dirty = false;
    std::chrono::high_resolution_clock::time_point last_animation_tick{};
    bool has_last_animation_tick = false;

    Raytracer();
    void update_parameters(Scene& scene, uint32_t width, uint32_t height);
    void draw_first_hit(Scene& scene, cudaSurfaceObject_t surface, uint32_t width, uint32_t height, cudaStream_t stream = 0);

    void step_animations(Scene& scene);
    void reset_animations_state();

    void buildAS();
};