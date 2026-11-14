#pragma once
#include "../../external/soa_vector/src/soa_vector.hpp"
#include <glm/glm.hpp>
#include "../types.hpp"
#include "../types/NagataPatch.cuh"
#include "../geometry_common.cuh"
#include "rt_shaders.cuh"
#include "../acceleration_structures/octree/OctreeGeometry.cuh"
#include "../acceleration_structures/uniform_grid/UniformGridGeometry.cuh"
#include "../acceleration_structures/kdtree/KDTreeGeometry.cuh"



#include "RenderParameters.hpp"

#ifdef __CUDACC__
inline __device__
void raygen(const RenderParameters& p)
{
    auto fragment_global_id=p.initial_rays_generated+threadIdx.x;
    if (fragment_global_id<p.width*p.height)
    {
        constexpr int band_height=6;//6 pixels per horizontal band
        const int pixels_per_band=p.width*band_height;

        auto band=fragment_global_id / pixels_per_band;
        auto local_id=fragment_global_id % pixels_per_band;

        auto row=band*band_height + local_id%band_height;
        auto col=local_id/band_height;

        // auto point=screen_to_space(row,col,p.width,p.height,1);
    }
}

__device__
inline void write_texture(cudaSurfaceObject_t surface,uint32_t x, uint32_t y, glm::vec4 value)
{
    value=glm::clamp(value,0.f,1.f);
    uchar4 pixelColor = make_uchar4(
    (unsigned char)( value.r * 255.0f),
    (unsigned char)( value.g * 255.0f),
    (unsigned char)( value.b * 255.0f),
    (unsigned char)( value.a * 255.0f)
);

    // WRITE TO SURFACE
    // The X coordinate in surf2Dwrite is in BYTES, not elements
    surf2Dwrite(pixelColor, surface, x * sizeof(uchar4), y);
}
#endif


template <ASType AsType, bool HasNagata = true>
__host__ __device__ inline
glm::vec4 trace_pixel_first_hit(uint32_t x, uint32_t y, const RenderParameters& p)
{
    glm::vec3 source, direction;
    p.calculate_camera_ray(x, y, source, direction);
    if constexpr (AsType == ASType::UniformGrid)
        return p.as_uniform_grid.template first_hit<HasNagata>(p, source, direction);
    else if constexpr (AsType == ASType::KDTree)
        return p.as_kdtree.template first_hit<HasNagata>(p, source, direction);
    else
        return p.as_octree.template first_hit<HasNagata>(p, source, direction);
}

__host__ __device__ inline
glm::vec4 trace_pixel_first_hit(uint32_t x, uint32_t y, const RenderParameters& p)
{
    if (p.active_as_type == ASType::UniformGrid)
        return trace_pixel_first_hit<ASType::UniformGrid>(x, y, p);
    if (p.active_as_type == ASType::KDTree)
        return trace_pixel_first_hit<ASType::KDTree>(x, y, p);
    return trace_pixel_first_hit<ASType::Octree>(x, y, p);
}

#ifdef __CUDACC__
__device__ inline
glm::uvec2 threadPixelPos()
{
    //Warp will take 8x4 pixels
    //if blockDim.x is 64 then 2 warps will create a single localized square 8x8 pixels
    auto x=blockIdx.x * blockDim.y*8 + threadIdx.y*8;
    auto y=blockIdx.y * blockDim.z*8 + threadIdx.z*8;
    x+=threadIdx.x%8;
    y+=threadIdx.x/8;
    return {x,y};
}

template <ASType AsType, bool HasNagata = true>
__global__
void draw_first_hit_kernel(cudaSurfaceObject_t surface, const __grid_constant__ RenderParameters p)
{
    auto threadPixel=threadPixelPos();

    if (threadPixel.x<p.width && threadPixel.y<p.height)
    {
        glm::vec4 output;
        if (p.measure_time)
        {
            __syncwarp();
            long long start = clock64();
            asm volatile("": : :"memory");
            output=trace_pixel_first_hit<AsType, HasNagata>(threadPixel.x,threadPixel.y,p);
            __threadfence();
            asm volatile("": : :"memory");
            __syncwarp();
            long long end = clock64();

            output.r=static_cast<float>(end-start)/static_cast<float>(p.time_max);
            write_texture(surface,threadPixel.x,threadPixel.y,output);
        }
        else
        {
            output=trace_pixel_first_hit<AsType, HasNagata>(threadPixel.x,threadPixel.y,p);
            write_texture(surface,threadPixel.x,threadPixel.y,output);
        }
    }
}
#endif
