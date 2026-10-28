#include "Raytracer.cuh"

#include "Scene.hpp"
#include "../cuda_common.hpp"
#include "rt_kernels.cuh"

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


__host__ __device__ __forceinline__
float triangle_simple_lighting(glm::vec3 v0,glm::vec3 v1,glm::vec3 v2, glm::vec3 dir, float u, float v)
{
    using namespace glm;
    auto normal = normalize(cross(v1-v0, v2-v0));
    return abs(dot(normalize(dir),normal));
}


__host__ __device__
glm::vec4 draw_pixel(uint32_t x, uint32_t y, RenderParameters p)
{
    static constexpr int max_iterations=500;

    auto direction=screen_to_space(x,y,p.width,p.height,p.near_);

    //apply rotation
    direction={direction.x, direction.y*p.ax_c.x-direction.z*p.ax_c.y, direction.y*p.ax_c.y+direction.z*p.ax_c.x};
    direction={ direction.x*p.az_c.x-direction.y*p.az_c.y, direction.x*p.az_c.y+direction.y*p.az_c.x, direction.z};

    auto source=p.view_pos;

    // direction=glm::normalize(direction);



    int it=0;

    int32_t container_id=INT32_MAX;
    int box_hits=0;
    float hit_primitive_color=-1;

    CurrentBox current_box{.id = p.root_box};
    while (container_id!=0 && hit_primitive_color<0 && it<max_iterations)
    {
        //AS TRAVERSAL
        DEVICE_ONLY(__syncwarp());
        auto result=traverse_stackless(&p.as[0].child, current_box,source,direction,it);
        container_id=result.content;
        DEVICE_ONLY(__syncwarp());

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
                        auto v1_id=p.triangles[triangle_id].v1;
                        auto v2_id=p.triangles[triangle_id].v2;
                        auto v3_id=p.triangles[triangle_id].v3;

                        glm::vec3 tuv;
                        bool hit=intersectRayTriangle(source,direction,
                            p.vertices[v1_id].position,p.vertices[v2_id].position,p.vertices[v3_id].position,
                            tuv.x,tuv.y,tuv.z);

                        if (hit)
                        {
                            auto hit_point=source+direction*tuv[0];
                            auto diff=glm::abs(content_box_center-hit_point);
                            auto half_box_size=result.stack.half_box_size();
                            auto gr_then=glm::greaterThan(diff,glm::vec3(half_box_size));
                            if ( glm::any(gr_then) )
                                hit=false;
                        }

                        if (hit)
                        {
                            if (tuv[0]>0 && tuv[0]<closest_tuv[0])
                            {
                                closest_tuv=tuv;
                                closest_triangle=triangle_id;
                                hit_primitive_color=triangle_simple_lighting(p.vertices[v1_id].position,p.vertices[v2_id].position,p.vertices[v3_id].position,
                                    direction,tuv[1],tuv[2]);
                            }
                        }
                    }
                }
            }
        }

        if (current_box.stack.depth()==0)//even though the content was return the stack depth 0 means that that was the last box
            break;
    }

    if (it==max_iterations)
        return glm::vec4{0.f};
    return {float(it)/25.0f*p.show_iterations, (box_hits/10.f)*p.show_boxes,hit_primitive_color,1};

}


__device__
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
void  draw_kernel(cudaSurfaceObject_t surface, RenderParameters p)
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
            output=draw_pixel(threadPixel.x,threadPixel.y,p);
            __threadfence();
            asm volatile("": : :"memory");
            __syncwarp();
            long long end = clock64();

            output.r=static_cast<float>(end-start)/static_cast<float>(p.time_max);
            write_texture(surface,threadPixel.x,threadPixel.y,output);
        }
        else
        {
            output=draw_pixel(threadPixel.x,threadPixel.y,p);
            write_texture(surface,threadPixel.x,threadPixel.y,output);
        }
    }
}

Raytracer::Raytracer()
{
    start_time=std::chrono::high_resolution_clock::now();
}

void Raytracer::update_parameters(Scene& scene, uint32_t width, uint32_t height)
{
    parameters.t= (std::chrono::high_resolution_clock::now()-start_time).count() / 1000000000.0;
    parameters.ax_c={cos(parameters.ax),sin(parameters.ax)};
    parameters.az_c={cos(parameters.az),sin(parameters.az)};
    parameters.root_box=scene.get_root_box();
    parameters.width=width;
    parameters.height=height;

    parameters.vertices=scene.as.vertices.view();
    parameters.triangles=scene.as.triangles.view();
    parameters.as=scene.as.as.view();
    parameters.containers=scene.as.box_content.view();
    parameters.primitives=scene.as.primitives.view();
}

void Raytracer::draw(Scene& scene, cudaSurfaceObject_t surface, uint32_t width, uint32_t height)
{
    update_parameters(scene,width, height);
    uint32_t block_dim_x=16;//must be multiple of 8
    uint32_t block_dim_y=16;//must be multiple of 8
    dim3 grid_dim{(width-1)/block_dim_x+1,(height-1)/block_dim_y+1 ,1};
    draw_kernel<<<grid_dim,{64,block_dim_x/8,block_dim_y/8}>>>(surface,parameters);
}