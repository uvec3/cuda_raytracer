#pragma once

#include <span>
#include "../../types.hpp"
#include "../../pool/soa_pool_cuda.hpp"
#include "../../geometry_common.cuh"
#include "OctreeGeometry.cuh"
#include "../as_concept.cuh"
#include "../../concepts.hpp"

struct InsertionContext;
struct Mesh;

struct OctreeASDeviceView
{
    soa::soa_view<Space> spaces;
    soa::soa_view<int> as;
    soa::soa_view<PrimitiveChunk16> primitives;
    soa::soa_view<BoxContent> box_content;
    int traverse();
};
static_assert(ASDeviceView<OctreeASDeviceView>);//OctreeASDeviceView implements ASDeviceView concept

class OctreeAS
{
public:
    soa_pool<Space> spaces;
    soa_pool<int> as;
    soa_pool<BoxContent> box_content;
    soa_pool<Vertex> vertices;
    soa_pool<PrimitiveChunk16> primitives;

    soa::cuda_device_vector<Space> d_spaces;
    soa::cuda_device_vector<int> d_as;
    soa::cuda_device_vector<BoxContent> d_box_content;
    soa::cuda_device_vector<Vertex> d_vertices;
    soa::cuda_device_vector<PrimitiveChunk16> d_primitives;

    void sync_to_device()
    {
        d_spaces = spaces;
        d_as = as;
        d_box_content = box_content;
        d_vertices = vertices;
        d_primitives = primitives;
    }

    OctreeASDeviceView device_view()
    {
        return OctreeASDeviceView{
            .spaces = d_spaces,
            .as = d_as,
            .primitives = d_primitives,
            .box_content = d_box_content
        };
    }

    void config_ui();
    void register_containers(ContainerExplorer& explorer);

    OctreeAS();

    uint32_t create_space();
    void free_space(uint32_t space_id);

    void insert_triangles(uint32_t space_id, uint32_t primitives_start, uint32_t number_of_triangles, uint32_t occupied_slots,
                          int depth_min, int depth_max, float triangle_box_stop_ratio, bool use_gpu = true);
    void insert_triangles_gpu(uint32_t space_id, uint32_t primitives_start, uint32_t number_of_triangles, uint32_t occupied_slots,
                              int depth_min, int depth_max, float triangle_box_stop_ratio, auto find_triangle_cube_relation);
    void insert_triangles_cpu(uint32_t space_id, uint32_t primitives_start, uint32_t number_of_triangles, uint32_t occupied_slots,
                              int depth_min, int depth_max, float triangle_box_stop_ratio, auto find_triangle_cube_relation);

    uint32_t create_space_for_models(range_of<std::pair<glm::mat4,Mesh&>> auto&& meshes)
    {
        size_t triangles_size = 0;
        size_t vertices_size = 0;

        for (auto& mesh : meshes)
        {
            triangles_size += mesh.second.triangles.size();
            vertices_size += mesh.second.vertices.size();
        }

        auto space_id = spaces.alloc_element();
        auto vertices_start = vertices.alloc(vertices_size);

        spaces[space_id].vertices_start = vertices_start;
        spaces[space_id].vertices_size = vertices_size;
        spaces[space_id].primitives_start = 0;
        spaces[space_id].primitives_size = 0;

        soa::soa_vector<Triangle> temp_triangles(triangles_size);
        size_t tri_offset = 0;
        size_t vert_offset = vertices_start;

        for (auto& [transform, mesh] : meshes)
        {
            // copy triangles into temporary indexed array
            for (size_t i = 0; i < mesh.triangles.size(); ++i)
            {
                const auto& src_triangle = mesh.triangles[i];
                auto&& dst_triangle = temp_triangles[tri_offset + i];

                dst_triangle.type = PrimitiveType::Triangle;
                dst_triangle.material_id = src_triangle.material_id;
                dst_triangle.v0 = src_triangle.v0 + vert_offset;
                dst_triangle.v1 = src_triangle.v1 + vert_offset;
                dst_triangle.v2 = src_triangle.v2 + vert_offset;
            }

            // copy vertices
            for (size_t i = 0; i < mesh.vertices.size(); ++i)
            {
                const auto& src_vertex = mesh.vertices[i];
                auto&& dst_vertex = vertices[vert_offset + i];

                dst_vertex = src_vertex;

                dst_vertex.position = transform * glm::vec4(dst_vertex.position, 1);
                dst_vertex.normal = glm::normalize(transform * glm::vec4(dst_vertex.normal, 0));
                if (config.swapYZ)
                {
                    std::swap(dst_vertex.position.y, dst_vertex.position.z);
                    dst_vertex.position.y = -dst_vertex.position.y;
                    std::swap(dst_vertex.normal.y, dst_vertex.normal.z);
                    dst_vertex.normal.y = -dst_vertex.normal.y;
                }
            }

            tri_offset += mesh.triangles.size();
            vert_offset += mesh.vertices.size();
        }

        // load vertices and temporary triangles to device
        d_vertices = vertices;
        soa::cuda_device_vector<Triangle> d_temp_triangles = temp_triangles;

        normalize_space(space_id, config.stretch);
        if (config.sort_space)
        {
            sort_space(space_id, d_temp_triangles);
        }

        uint32_t occupied_slots = Triangle::occupied_slots;
        if (config.triangle_type == PrimitiveType::FullTriangle)
            occupied_slots = InlinedTriangle::occupied_slots;
        else if (config.triangle_type == PrimitiveType::NagataPatch)
            occupied_slots = NagataPatchPrimitive::occupied_slots;

        pack_primitives(space_id, d_temp_triangles, static_cast<uint32_t>(triangles_size));

        insert_triangles(space_id, spaces[space_id].primitives_start, static_cast<uint32_t>(triangles_size), occupied_slots,
                         config.min_depth, config.max_depth, config.triangle_to_box_stop_ratio, config.use_gpu);

        sync_to_device();

        return space_id;
    }

    int divide_box(InsertionContext& insertion_context, BoxInfo box_info);
    void insert_triangles_recursive(int box_id, BoxStack stack, InsertionContext& ic, uint32_t start, uint32_t size, const std::vector<std::pair<uint32_t,
                                    uint32_t>>& parent_triangles, auto find_triangle_cube_relation);
    void add_box_contents(InsertionContext& insertion_context, uint32_t box_id, std::span<const uint32_t> prim_ids);
    void free_box(int box_id);
    void free_box_content(int box_id);
    void normalize_space(uint32_t space_id, bool stretch);
    void sort_space(uint32_t space_id, soa::cuda_device_vector<Triangle>& d_temp_triangles);
    void pack_primitives(uint32_t space_id, const soa::cuda_device_vector<Triangle>& d_temp_triangles, uint32_t triangles_count);


    struct Config
    {
        int min_depth = 1;
        int max_depth = 19;
        float triangle_to_box_stop_ratio = 1.1f;
        bool stretch = false;
        bool swapYZ = true;

        bool use_gpu = true;
        bool sort_space = true;
        PrimitiveType triangle_type = PrimitiveType::Triangle;
    };

    Config config;

};
static_assert(AS<OctreeAS>); //OctreeAS implements AS concept
