#pragma once
#include <cstdint>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "../types.hpp"
#include "../geometry_common.cuh"
#include "../acceleration_structures/octree/OctreeGeometry.cuh"


enum class ProjectionType
{
    Perspective,
};

struct RenderParameters
{
    //image
    uint32_t width = 0;
    uint32_t height = 0;


    //general
    glm::vec2 mouse_pos = {0.0f, 0.0f};
    double t = 0.0;

    //config
    bool show_iterations = false;
    bool measure_time = false;
    bool show_triangles = true;
    bool show_boxes = false;
    int time_max = 50000;

    //lighting
    float ambient_light = 0.1f;
    float diffuse_light = 0.9f;
    bool fixed_light_source = false;
    glm::vec3 light_source = {0.5f, 0.5f, 1.5f};
    bool use_fake_normals = true;
    bool renderNagata = false;
    bool enableShadows = false;
    bool enableReflection = false;
    bool enableRefraction = false;

    //background
    glm::vec4 background_color = {0.0f, 0.0f, 0.0f, 1.0f};
    cudaTextureObject_t skybox = 0;

    //camera logical
    float az = 0.0f;
    float ax = -0.5f;
    float fov = 45.0f;

    //camera rt
    bool perspective_projection=true;
    glm::mat3 rotation = glm::mat3(1.0f);
    float near_ = 1.0f;
    glm::vec3 view_pos = {0.5f, 0.0f, 0.3f};
    glm::vec2 ax_c = {1.0f, 0.0f};
    glm::vec2 az_c = {1.0f, 0.0f};

    //AS
    OctreeASDeviceView as_octree{};

    //geometry context
    soa::soa_view<PrimitiveChunk16> primitives{};
    soa::soa_view<Vertex> vertices{};
    soa::soa_view<Material> materials{};
    soa::soa_view<Space> spaces{};

    //temporary
    soa::soa_view<Ray> rays{};
    uint32_t number_of_rays = 0;
    uint32_t initial_rays_generated = 0;
    int root_box = -1;


    void update_camera()
    {
        //update near
        auto rad=fov/180.0*glm::pi<double>();
        double tg=glm::tan(rad / 2);
        near_=static_cast<float>(1.0/tg);

        //update rotations
        ax_c={cos(ax),sin(ax)};
        az_c={cos(az),sin(az)};
    }


    __host__ __device__ inline
    glm::vec3 screen_to_space(uint32_t x, uint32_t y) const
    {
        auto point = glm::vec2((float)x / (float)width, (float)y / (float)height);
        point = point * 2.f - 1.f;
        point.x *= float(width) / float(height);
        if (perspective_projection)
        {
            return glm::vec3(point.x, near_, -point.y);
        }
        else
        {
            float max_degree=fov/180.f*glm::pi<float>()/2;
            float az=point.x*max_degree;
            float ay=-point.y*max_degree;

            glm::vec3 dir{0,cos(ay)*near_,sin(ay)*near_};
            dir={dir.y*sin(az),dir.y*cos(az),dir.z};

            return dir;
        }
    }


    __host__ __device__ inline void calculate_camera_ray(uint32_t x, uint32_t y, glm::vec3& source, glm::vec3& direction) const
    {
        direction = screen_to_space(x, y);

        // Apply camera rotation
        // Complex multiplication with ax_c: rotates the camera up/down
        direction = {
            direction.x,
            direction.y * ax_c.x - direction.z * ax_c.y,
            direction.y * ax_c.y + direction.z * ax_c.x
        };
        // Complex multiplication with az_c: rotates the camera left/right
        direction = {
            direction.x * az_c.x - direction.y * az_c.y,
            direction.x * az_c.y + direction.y * az_c.x,
            direction.z
        };

        direction = glm::normalize(direction);
        source = view_pos;
    }
};
