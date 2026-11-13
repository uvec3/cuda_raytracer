#include <optix.h>
#include "params.hpp"
#include "../../raytracer/rt_kernels.cuh"

extern "C" {
extern __constant__ Params params;
}

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();

    if (idx.x >= params.render_params.width || idx.y >= params.render_params.height)
        return;

    glm::vec3 source, direction;
    params.render_params.calculate_camera_ray(idx.x, idx.y, source, direction);

    // Payload values: RGB (0-255), depth
    unsigned int r = 0, g = 0, b = 0, depth = 0;

    if (params.render_params.measure_time) {
        asm volatile("" ::: "memory");
        long long start = clock64();
        asm volatile("" ::: "memory");

        optixTrace(
            params.handle,
            make_float3(source.x, source.y, source.z),
            make_float3(direction.x, direction.y, direction.z),
            1e-4f,                     // tmin
            1e16f,                     // tmax
            0.0f,                      // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0,                         // SBT offset
            1,                         // SBT stride
            0,                         // missSBTIndex
            r, g, b, depth             // Payload values
        );

        asm volatile("" ::: "memory");
        long long end = clock64();
        asm volatile("" ::: "memory");

        float time_max_val = static_cast<float>(params.render_params.time_max > 0 ? params.render_params.time_max : 1);
        float time_val = glm::clamp(static_cast<float>(end - start) / time_max_val, 0.0f, 1.0f);
        r = static_cast<unsigned int>(time_val * 255.0f);
    } else {
        optixTrace(
            params.handle,
            make_float3(source.x, source.y, source.z),
            make_float3(direction.x, direction.y, direction.z),
            1e-4f,                     // tmin
            1e16f,                     // tmax
            0.0f,                      // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0,                         // SBT offset
            1,                         // SBT stride
            0,                         // missSBTIndex
            r, g, b, depth             // Payload values
        );
    }

    uint32_t pixel = (static_cast<uint32_t>(r) & 0xFFu) |
                     ((static_cast<uint32_t>(g) & 0xFFu) << 8) |
                     ((static_cast<uint32_t>(b) & 0xFFu) << 16) |
                     (0xFF000000u);

    surf2Dwrite(pixel, params.surface, idx.x * sizeof(uint32_t), idx.y);
}

extern "C" __global__ void __miss__ms() {
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();
    glm::vec3 src(ray_orig.x, ray_orig.y, ray_orig.z);
    glm::vec3 dir(ray_direction.x, ray_direction.y, ray_direction.z);

    glm::vec4 bg = miss_shader(src, dir, params.render_params);
    bg = glm::clamp(bg, 0.0f, 1.0f);
    optixSetPayload_0(static_cast<unsigned int>(bg.r * 255.0f));
    optixSetPayload_1(static_cast<unsigned int>(bg.g * 255.0f));
    optixSetPayload_2(static_cast<unsigned int>(bg.b * 255.0f));
}

extern "C" __global__ void __closesthit__ch() {
    const float2 bary = optixGetTriangleBarycentrics();
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();

    glm::vec3 src(ray_orig.x, ray_orig.y, ray_orig.z);
    glm::vec3 dir(ray_direction.x, ray_direction.y, ray_direction.z);
    glm::vec3 tuv(optixGetRayTmax(), bary.x, bary.y);

    const uint32_t prim_id = params.primitives_start + optixGetPrimitiveIndex();
    bool continue_traversal = false;

    glm::vec4 out_color = fragment_hit_shader(prim_id, src, dir, tuv, params.render_params, continue_traversal);

    unsigned int depth = optixGetPayload_3();
    constexpr unsigned int MAX_DEPTH = 8;

    if (continue_traversal && depth < MAX_DEPTH) {
        unsigned int r = 0, g = 0, b = 0;
        unsigned int next_depth = depth + 1;
        optixTrace(
            params.handle,
            make_float3(src.x, src.y, src.z),
            make_float3(dir.x, dir.y, dir.z),
            1e-4f,                     // tmin
            1e16f,                     // tmax
            0.0f,                      // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0,                         // SBT offset
            1,                         // SBT stride
            0,                         // missSBTIndex
            r, g, b, next_depth        // Payload values
        );

        glm::vec4 secondary_color(
            static_cast<float>(r) / 255.0f,
            static_cast<float>(g) / 255.0f,
            static_cast<float>(b) / 255.0f,
            1.0f
        );
        out_color = blend_colors(out_color, secondary_color);
    }

    out_color = glm::clamp(out_color, 0.0f, 1.0f);

    optixSetPayload_0(static_cast<unsigned int>(out_color.r * 255.0f));
    optixSetPayload_1(static_cast<unsigned int>(out_color.g * 255.0f));
    optixSetPayload_2(static_cast<unsigned int>(out_color.b * 255.0f));
}
