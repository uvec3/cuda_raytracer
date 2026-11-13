#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <glm/glm.hpp>
#include "../../types.hpp"
#include "../../geometry_common.cuh"
#include "../../raytracer/RenderParameters.hpp"

struct Params {
    RenderParameters       render_params;
    OptixTraversableHandle handle;
    cudaSurfaceObject_t    surface;
    uint32_t               primitives_start;
};

// Aligned SBT record header
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) EmptySbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};
