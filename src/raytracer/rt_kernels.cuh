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
    glm::vec3 view_pos{0.5,0,0.3};
    double t;
    float az{0};
    float ax{-0.5};
    glm::mat3 rotation;
    float near_ = 1;

    float ambient_light=0.1;
    float diffuse_light=0.9;

    bool fixed_light_source=false;
    glm::vec3 light_source{0.5, 0.5,1.5};

    bool show_iterations=false;
    bool measure_time=false;
    bool show_triangles=true;
    bool show_boxes=false;
    bool use_fake_normals=true;
    bool renderNagata=false;

    bool enableShadows=false;
    bool enableReflection=false;
    bool enableRefraction=false;

    glm::vec4 background_color=glm::vec4(0,0,0,1);
    cudaTextureObject_t skybox;
    bool use_skybox=true;

    int time_max=50000;

    //as
    soa::soa_view<int> as;
    soa::soa_view<BoxContent> containers;
    soa::soa_view<PrimitiveChunk16> primitives;
    soa::soa_view<Vertex> vertices;

    //scene
    soa::soa_view<Material> materials;

    //temp
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
    // The X coordinate in surf2Dwrite is in BYTES, not elements
    surf2Dwrite(pixelColor, surface, x * sizeof(uchar4), y);
}


//returns whether to continue tracing the ray
 __host__ __device__ inline
bool fragment_hit_shader(glm::vec4& out_color, uint32_t primitive_id, glm::vec3& ray_src, glm::vec3& ray_dir, glm::vec3 tuv,const RenderParameters& p)
{
    using namespace glm;

    float w0 = 1.0f - tuv[1] - tuv[2];
    float w1 = tuv[1];
    float w2 = tuv[2];

    PrimitiveChunk16 chunk0 = p.primitives[primitive_id];
    uint32_t mat_id = chunk0.material_id();
    glm::vec3 normal(0.0f, 0.0f, 1.0f);
    glm::vec2 uv_interp(0.0f);

    if (chunk0.type() == PrimitiveType::Triangle)
    {
        Triangle tri = chunk0.as<Triangle>();

        if (p.use_fake_normals)
        {
            glm::vec3 n0 = glm::vec3(p.vertices[tri.v0].normal);
            glm::vec3 n1 = glm::vec3(p.vertices[tri.v1].normal);
            glm::vec3 n2 = glm::vec3(p.vertices[tri.v2].normal);
            normal = normalize(n0 * w0 + n1 * w1 + n2 * w2);
        }
        else
        {
            glm::vec3 v0 = glm::vec3(p.vertices[tri.v0].position);
            glm::vec3 v1 = glm::vec3(p.vertices[tri.v1].position);
            glm::vec3 v2 = glm::vec3(p.vertices[tri.v2].position);
            normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
        }

        auto uv0 = glm::vec2(p.vertices[tri.v0].uv);
        auto uv1 = glm::vec2(p.vertices[tri.v1].uv);
        auto uv2 = glm::vec2(p.vertices[tri.v2].uv);
        uv_interp = uv0 * w0 + uv1 * w1 + uv2 * w2;
    }
    else if (chunk0.type() == PrimitiveType::FullTriangle)
    {
        InlinedTriangle ftri = load_primitive<InlinedTriangle>(p.primitives, primitive_id, chunk0);

        if (p.use_fake_normals)
        {
            auto n0 = ftri.n0_compressed.unpack();
            auto n1 = ftri.n1_compressed.unpack();
            auto n2 = ftri.n2_compressed.unpack();
            normal = normalize(n0 * w0 + n1 * w1 + n2 * w2);
        }
        else
        {
            normal = glm::normalize(glm::cross(ftri.v1 - ftri.v0, ftri.v2 - ftri.v0));
        }

        glm::vec2 uv0 = glm::unpackHalf2x16(ftri.uv_0);
        glm::vec2 uv1 = glm::unpackHalf2x16(ftri.uv_1);
        glm::vec2 uv2 = glm::unpackHalf2x16(ftri.uv_2);
        uv_interp = uv0 * w0 + uv1 * w1 + uv2 * w2;
    }
    else if (chunk0.type() == PrimitiveType::NagataPatch)
    {
        NagataPatchPrimitive patch = load_primitive<NagataPatchPrimitive>(p.primitives, primitive_id, chunk0);

        if (p.use_fake_normals)
        {
            glm::vec3 n0 = glm::vec3(p.vertices[patch.v0].normal);
            glm::vec3 n1 = glm::vec3(p.vertices[patch.v1].normal);
            glm::vec3 n2 = glm::vec3(p.vertices[patch.v2].normal);
            normal = normalize(n0 * w0 + n1 * w1 + n2 * w2);
        }
        else
        {
            glm::vec3 v0 = glm::vec3(p.vertices[patch.v0].position);
            glm::vec3 v1 = glm::vec3(p.vertices[patch.v1].position);
            glm::vec3 v2 = glm::vec3(p.vertices[patch.v2].position);
            normal = evaluateNagataNormal2(v0,v1,v2, patch.C12,patch.C23,patch.C31,w0,w1);
        }

        glm::vec2 uv0 = p.vertices[patch.v0].uv;
        glm::vec2 uv1 = p.vertices[patch.v1].uv;
        glm::vec2 uv2 = p.vertices[patch.v2].uv;
        uv_interp = uv0 * w0 + uv1 * w1 + uv2 * w2;
    }

    vec3 intersection=ray_src+ray_dir*tuv[0];

    float diffuse;
    if (p.fixed_light_source)
    {
        auto l=p.light_source-intersection;
        auto dist=length(l);
        diffuse = clamp( dot(l/dist,normal),0.f,1.f);
    }
    else
    {
        diffuse = abs(dot(normalize(ray_dir),normal));
    }

    glm::vec4 base_color(1.0f);
    glm::vec3 emissive(0.0f);

    if (mat_id < p.materials.size())
    {
        auto mat = p.materials[mat_id];
        base_color = mat.baseColor;
        emissive = mat.emissiveFactor;

#if defined(__CUDA_ARCH__)//only use texture on gpu
        cudaTextureObject_t tex = mat.baseColorTexture;
        if (tex != 0)
        {
            float4 sampled = tex2D<float4>(tex, uv_interp.x, uv_interp.y);
            base_color *= glm::vec4(sampled.x, sampled.y, sampled.z, sampled.w);
        }
#endif
    }

    float lighting = diffuse * p.diffuse_light + p.ambient_light;
    out_color += glm::vec4(glm::vec3(base_color) * lighting + emissive, base_color.a);

    if (p.enableRefraction)
    {
        float eta = 0.75;

        //out
        if (glm::dot(ray_dir,normal) > 0.0f) {
            eta = 1/eta;
            normal=-normal;
        }
        auto refracted=glm::refract(ray_dir,normal,eta);
        if (refracted!=glm::vec3(0))
        {
            ray_dir=refracted;
            ray_src=intersection;
            return true;
        }
        else
            return false;
    }

    return false;
}

__host__ __device__ inline
glm::vec4 miss_shader(glm::vec3 ray_src, glm::vec3 ray_dir,const RenderParameters& p)
{

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
    glm::vec4 hit_primitive_color{0};

    //current container and box
    int32_t container_id=INT32_MAX;
    CurrentBox current_box{.id = p.root_box};

    //count the number of iterations to prevent infinite or too long propagations
    //one iteration is a one horizontal step through octree
    int it=0;
    bool continue_traversal=true;
    while (container_id!=0 && continue_traversal && it<max_iterations)
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
                for (;;++container_id)
                {
                    auto primitive=p.containers[container_id];
                    uint32_t prim_id=primitive.id();
                    PrimitiveChunk16 chunk0=p.primitives[prim_id];
                    if (chunk0.type() == PrimitiveType::Triangle)
                    {
                        Triangle tri = chunk0.as<Triangle>();

                        glm::vec3 tuv;
                        bool hit = intersectRayTriangle(
                            source, direction, current_box.t_min, current_box.t_max,
                            glm::vec3(p.vertices[tri.v0].position), glm::vec3(p.vertices[tri.v1].position), glm::vec3(p.vertices[tri.v2].position),
                            tuv.x, tuv.y, tuv.z);

                        if (hit)
                        {
                            if (tuv[0] < closest_tuv[0])
                            {
                                closest_tuv = tuv;
                                closest_triangle = prim_id;
                            }
                        }
                    }
                    else if (chunk0.type() == PrimitiveType::FullTriangle)
                    {
                        InlinedTriangle ftri = load_primitive<InlinedTriangle>(p.primitives, prim_id, chunk0);

                        glm::vec3 tuv;
                        bool hit = intersectRayTriangle(
                            source, direction, current_box.t_min, current_box.t_max,
                            ftri.v0, ftri.v1, ftri.v2,
                            tuv.x, tuv.y, tuv.z);

                        if (hit)
                        {
                            if (tuv[0] < closest_tuv[0])
                            {
                                closest_tuv = tuv;
                                closest_triangle = prim_id;
                            }
                        }
                    }
                    else if (chunk0.type() == PrimitiveType::NagataPatch)
                    {
                        NagataPatchPrimitive patch = load_primitive<NagataPatchPrimitive>(p.primitives, prim_id, chunk0);

                        glm::vec3 tuv;
                        bool hit = intersectNagataPrecomputed(
                            source, direction, current_box.t_min, current_box.t_max,
                            glm::vec3(p.vertices[patch.v0].position), glm::vec3(p.vertices[patch.v1].position), glm::vec3(p.vertices[patch.v2].position),
                            patch.C12, patch.C23, patch.C31,
                            tuv.x, tuv.y, tuv.z);

                        if (hit)
                        {
                            if (tuv[0] < closest_tuv[0])
                            {
                                closest_tuv = tuv;
                                closest_triangle = prim_id;
                            }
                        }
                    }

                    if (primitive.is_last())//if it is the last in chain
                        break;
                }
            }

            if (closest_triangle!=UINT32_MAX)
            {

                continue_traversal=fragment_hit_shader(hit_primitive_color,closest_triangle,source,direction,closest_tuv,p);
                if (continue_traversal)
                {
                    current_box.t_min=1e-4f;
                    current_box.t_max=1e-4f;
                }
            }
        }

        if (current_box.stack.depth()==0)//even though the content was return the stack depth 0 means that that was the last box
            break;
    }

    if (it==max_iterations)//return white to indicate that the loop was interrupted
        return glm::vec4{1.f};

    if (!p.show_iterations && !p.show_boxes)
    {
        return glm::vec4(glm::vec3(hit_primitive_color), 1.0f);
    }

    //composes the final color result: number of boxes - red, number of leafs - green, geometry - blue (or additive when geometry has color)
    float iter_overlay = (float(it) / 25.0f) * p.show_iterations;
    float box_overlay = (box_hits / 10.0f) * p.show_boxes;
    return glm::vec4(
        hit_primitive_color.r + iter_overlay,
        hit_primitive_color.g + box_overlay,
        hit_primitive_color.b,
        1.0f);
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
