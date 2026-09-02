#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>

struct RenderParameters
{
    double t;
    glm::vec2 mouse_pos;
};

void deviceAdd(float* result_h, float* a_h, float* b_h,uint64_t N);
void draw_device(cudaSurfaceObject_t surface, uint32_t width, uint32_t height, RenderParameters t);