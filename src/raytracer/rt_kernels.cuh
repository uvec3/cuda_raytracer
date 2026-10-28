#pragma once
#include "../../external/soa_vector/src/soa_vector.hpp"
#include <glm/glm.hpp>
#include "../types.hpp"

struct RenderParameters
{
    uint32_t width;
    uint32_t height;

    glm::vec2 ax_c;
    glm::vec2 az_c;

    glm::vec2 mouse_pos;
    glm::vec3 view_pos{0.5,0,0.5};
    double t;
    float az;
    float ax;
    glm::mat3 rotation;
    float near_ = 1;

    bool show_iterations=true;
    bool measure_time=false;
    bool show_triangles=true;
    bool show_boxes=true;

    int time_max=50000;

    soa::soa_view<Box> as;
    soa::soa_view<BoxContent> containers;
    soa::soa_view<Primitive> primitives;


    soa::soa_view<Triangle> triangles;
    soa::soa_view<Vertex> vertices;

    soa::soa_view<Ray> rays;
    uint32_t number_of_rays=0;
    uint32_t initial_rays_generated=0;

    int root_box;
};


__host__ __device__ inline
glm::vec3 screen_to_space(uint32_t x, uint32_t y, uint32_t width, uint32_t height, float near_plane)
{
    auto point=glm::vec2((float)x/(float)width,(float)y/(float)height);
    point=point*2.f-1.f;
    point.x*=float(width)/float(height);
    auto point3d=glm::vec3(point.x,near_plane,-point.y);
    return  point3d;
}

inline __device__
void raygen(RenderParameters& p)
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

        auto point=screen_to_space(row,col,p.width,p.height,1);
    }
}