#pragma once
#include "../../external/soa_vector/src/soa_vector.hpp"
#include <glm/glm.hpp>
#include "../types.hpp"
#include "../types/NagataPatch.cuh"
#include "../geometry_common.cuh"
#include "../acceleration_structures/octree/OctreeGeometry.cuh"

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

    float ambient_light=0.1;
    float diffuse_light=0.9;

    bool fixed_light_source=false;
    glm::vec3 light_source{0.5, 0.5,1.5};

    bool show_iterations=true;
    bool measure_time=false;
    bool show_triangles=true;
    bool show_boxes=true;
    bool use_fake_normals=true;
    bool renderNagata=false;

    int time_max=50000;

    soa::soa_view<int> as;
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
    // The X coordinate in surf2Dwrite is in BYTES, not elements!
    surf2Dwrite(pixelColor, surface, x * sizeof(uchar4), y);
}

 __host__ __device__ inline
glm::vec4 fragment_shader(uint32_t primitive_id, glm::vec3 ray_src, glm::vec3  ray_dir, glm::vec3 tuv, RenderParameters& p)
{
    using namespace glm;

    auto v0_id=p.triangles[primitive_id].v1;
    auto v1_id=p.triangles[primitive_id].v2;
    auto v2_id=p.triangles[primitive_id].v3;
    glm::vec3 normal;
    if (p.use_fake_normals)
    {
        auto n0=p.vertices[v0_id].normal;
        auto n1=p.vertices[v1_id].normal;
        auto n2=p.vertices[v2_id].normal;
        normal = n0*(1-tuv[1]-tuv[2])+n1*tuv[1]+n2*tuv[2];
        normal=normalize(normal);
    }
    else
    {
        auto v0=p.vertices[v0_id].position;
        auto v1=p.vertices[v1_id].position;
        auto v2=p.vertices[v2_id].position;
        normal=glm::normalize(glm::cross(v1-v0, v2-v0));
    }

    float diffuse;
    if (p.fixed_light_source)
    {
        auto intersection=ray_src+ray_dir*tuv[0];
        auto l=p.light_source-intersection;
        auto dist=length(l);
        diffuse = clamp( dot(l/dist,normal),0.f,1.f);
    }
    else
    {
        diffuse = abs(dot(normalize(ray_dir),normal));
    }

    return vec4(diffuse*p.diffuse_light+p.ambient_light);
}


__host__ __device__ inline
glm::vec4 octree_first_hit(RenderParameters p, glm::vec3 source, glm::vec3 direction)
{
    //current source can propagate through the acceleration structure
    auto current_source=source;
    static constexpr int max_iterations=500;
    //counts how many boxes were hit on the way
    int box_hits=0;
    //records triangles
    float hit_primitive_color=-1;

    //current container and box
    int32_t container_id=INT32_MAX;
    CurrentBox current_box{.id = p.root_box};

    //count the number of iterations to prevent infinite or too long propagations
    //one iteration is a one horizontal step through octree
    int it=0;
    while (container_id!=0 && hit_primitive_color<0 && it<max_iterations)
    {
        //AS TRAVERSAL STEP (returns after hitting non-empty leaf node)
        TraversalResult result=traverse_stackless(&p.as[0], current_box, source, current_source, direction, it);
        container_id=result.content;

        if (container_id>0)
        {
            uint32_t closest_triangle=UINT32_MAX;
            glm::vec3 closest_tuv{INFINITY,0,0};
            ++box_hits;

            if (p.show_triangles)
            {
                auto primitives_count=p.containers[container_id].number_of_primitives;
                auto primitives_start=p.containers[container_id].start;
                glm::vec3 content_box_center=result.stack.get_center();
                for (auto i=0;i<primitives_count;++i)
                {
                    if (p.primitives[primitives_start+i].type == PrimitiveType::Triangle)
                    {
                        auto triangle_id=p.primitives[primitives_start+i].id;
                        auto v0_id=p.triangles[triangle_id].v1;
                        auto v1_id=p.triangles[triangle_id].v2;
                        auto v2_id=p.triangles[triangle_id].v3;

                        glm::vec3 tuv;


                        bool hit;
                        if (p.renderNagata)
                        {
                            hit=intersectNagataOnTheFly(source,direction,current_box.t_min,current_box.t_max,
                                                        p.vertices[v0_id].position,p.vertices[v1_id].position,p.vertices[v2_id].position,
                                                        p.vertices[v0_id].normal,p.vertices[v1_id].normal,p.vertices[v2_id].normal,
                                                        tuv[0],tuv[1],tuv[2]);
                        }
                        else
                        {
                            hit=intersectRayTriangle(source,direction, current_box.t_min, current_box.t_max,
                                                     p.vertices[v0_id].position,p.vertices[v1_id].position,p.vertices[v2_id].position,
                                                     tuv.x,tuv.y,tuv.z);
                        }

                        if (hit)
                        {
                            if (tuv[0]<closest_tuv[0])
                            {
                                closest_tuv = tuv;
                                closest_triangle=triangle_id;
                            }
                        }
                    }
                }
            }

            if (closest_triangle!=UINT32_MAX)
            {
                hit_primitive_color=fragment_shader(closest_triangle,source,direction,closest_tuv,p).r;
            }
        }

        if (current_box.stack.depth()==0)//even though the content was return the stack depth 0 means that that was the last box
            break;
    }

    if (it==max_iterations)//return white to indicate that the loop was interrupted
        return glm::vec4{1.f};
    //composes the final color result: number of boxes - red, number of leafs - green, geometry - blue
    return {float(it)/25.0f*p.show_iterations, (box_hits/10.f)*p.show_boxes,hit_primitive_color,1};
}

__host__ __device__ inline
glm::vec4 trace_pixel_first_hit(uint32_t x, uint32_t y, RenderParameters p)
{
    auto direction=screen_to_space(x,y,p.width,p.height,p.near_);

    //apply rotation
    //complex multiplication with  p.ax_c : rotates the camera up/down
    direction={direction.x, direction.y*p.ax_c.x-direction.z*p.ax_c.y, direction.y*p.ax_c.y+direction.z*p.ax_c.x};
    //complex multiplication with  p.az_c : rotates the camera left/right
    direction={ direction.x*p.az_c.x-direction.y*p.az_c.y, direction.x*p.az_c.y+direction.y*p.az_c.x, direction.z};

    //for some calculation it is better to keep normalized vector(e.g. nagata patch)
    direction=glm::normalize(direction);

    //keep source at one point
    auto source=p.view_pos;

    return octree_first_hit(p, source, direction);
}

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

__global__
inline void  draw_first_hit_kernel(cudaSurfaceObject_t surface, RenderParameters p)
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
            output=trace_pixel_first_hit(threadPixel.x,threadPixel.y,p);
            __threadfence();
            asm volatile("": : :"memory");
            __syncwarp();
            long long end = clock64();

            output.r=static_cast<float>(end-start)/static_cast<float>(p.time_max);
            write_texture(surface,threadPixel.x,threadPixel.y,output);
        }
        else
        {
            output=trace_pixel_first_hit(threadPixel.x,threadPixel.y,p);
            write_texture(surface,threadPixel.x,threadPixel.y,output);
        }
    }
}
