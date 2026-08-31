#pragma once
#include <cuda_runtime.h>

void deviceAdd(float* result_h, float* a_h, float* b_h,uint64_t N);

void draw_device(cudaSurfaceObject_t surface, uint32_t width, uint32_t height, double t);