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
    soa::soa_view<Box> as;
    soa::soa_view<Primitive> primitives;
    soa::soa_view<BoxContent> box_content;
    int traverse();
};
static_assert(ASDeviceView<OctreeASDeviceView>);//OctreeASDeviceView implements ASDeviceView concept

class OctreeAS
{
public:
    soa_pool<Space> spaces;
    soa::soa_cuda_unified_pool<Box> as;
    soa::soa_cuda_unified_pool<Primitive> primitives;
    soa::soa_cuda_unified_pool<BoxContent> box_content;
    soa::soa_cuda_unified_pool<Vertex> vertices;
    soa::soa_cuda_unified_pool<Triangle> triangles;
    OctreeASDeviceView device_view()
    {
        return OctreeASDeviceView{
            .spaces = spaces,
            .as=as,
            .primitives = primitives,
            .box_content = box_content
        };
    }

    OctreeAS();

    uint32_t create_space();
    void free_space(uint32_t space_id);


    void normalize_space(uint32_t space_id, bool stretch);
    void insert_triangles(uint32_t space_id, uint32_t triangle_first, uint32_t number_of_triangles, int depth_min, int depth_max, float
                          triangle_box_stop_ratio);

    uint32_t create_space_for_models(range_of<std::pair<glm::mat4,Mesh&>> auto&& meshes, bool stretch, bool swapYZ, int depth_min, int depth_max, float triangle_to_box_stop_ratio)
    {
        size_t triangles_size=0;
        size_t vertices_size=0;

        for (auto& mesh:meshes)
        {
            triangles_size+=mesh.second.triangles.size();
            vertices_size+=mesh.second.vertices.size();
        }

        auto space_id = spaces.alloc_element();
        auto triangles_start = triangles.alloc(triangles_size);
        auto vertices_start = vertices.alloc(vertices_size);

        spaces[space_id].first_triangle = triangles_start;
        spaces[space_id].number_of_triangles = triangles_size;
        spaces[space_id].vertices_start = vertices_start;
        spaces[space_id].vertices_size = vertices_size;

        for (auto& [transform, mesh]:meshes)
        {
            //copy triangles
            for (int i = 0; i < mesh.triangles.size(); ++i)
            {
                const auto& src_triangle = mesh.triangles[i];
                auto&& dst_triangle = triangles[triangles_start + i];

                dst_triangle.material_id = src_triangle.material_id;
                dst_triangle.v1 = src_triangle.v1 + vertices_start;
                dst_triangle.v2 = src_triangle.v2 + vertices_start;
                dst_triangle.v3 = src_triangle.v3 + vertices_start;
            }

            //copy vertices
            for (int i = 0; i < mesh.vertices.size(); ++i)
            {
                const auto& src_vertex = mesh.vertices[i];
                auto&& dst_vertex =vertices[vertices_start + i];

                dst_vertex= src_vertex;


                dst_vertex.position=transform*glm::vec4(dst_vertex.position,1);
                if (swapYZ)
                {
                    std::swap(dst_vertex.position.y,dst_vertex.position.z);
                }
            }

            triangles_start+=mesh.triangles.size();
            vertices_start+=mesh.vertices.size();
        }
        normalize_space(space_id, stretch);
        insert_triangles(space_id, spaces[space_id].first_triangle,triangles_size, depth_min, depth_max, triangle_to_box_stop_ratio);
        return space_id;
    }

private:
    int divide_box(InsertionContext& insertion_context, BoxInfo box_info);
    void insert_triangles_recursive(int box_id, BoxStack stack, InsertionContext& ic, uint32_t start, uint32_t size, const std::vector<std::pair<uint32_t,
                                    uint32_t>>& parent_triangles);
    void add_box_contents(InsertionContext& insertion_context, uint32_t box_id, PrimitiveType type, std::span<uint32_t> primitives);
    void free_box(int box_id);
    void free_box_content(int box_id);
};
static_assert(AS<OctreeAS>); //OctreeAS implements AS concept
