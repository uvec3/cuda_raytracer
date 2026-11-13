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

// static_assert(ASDeviceView<OctreeASDeviceView>); // OctreeASDeviceView implements ASDeviceView concept

class OctreeAS
{
public:
    soa_pool<int> as;
    soa_pool<BoxContent> box_content;

    soa::cuda_device_vector<int> d_as;
    soa::cuda_device_vector<BoxContent> d_box_content;

    GeometryContext* m_geometry_context = nullptr;

    OctreeAS(GeometryContext* geometry_context = nullptr);

    void bind_geometry_context(GeometryContext* geometry_context)
    {
        m_geometry_context = geometry_context;
    }

    void sync_to_device()
    {
        d_as = as;
        d_box_content = box_content;
    }

    OctreeASDeviceView device_view()
    {
        return OctreeASDeviceView{
            .as = d_as,
            .box_content = d_box_content
        };
    }

    void config_ui();
    void register_containers(ContainerExplorer& explorer);

    void build(uint32_t space_id, Space space);
    void free_space(uint32_t space_id, Space space);

    void insert_triangles(uint32_t space_id, Space space, uint32_t primitives_start, uint32_t number_of_triangles, uint32_t occupied_slots,
                          int depth_min, int depth_max, float triangle_box_stop_ratio, bool use_gpu = true);
    void insert_triangles_gpu(uint32_t space_id, Space space, uint32_t primitives_start, uint32_t number_of_triangles, uint32_t occupied_slots,
                              int depth_min, int depth_max, float triangle_box_stop_ratio, auto find_triangle_cube_relation);
    void insert_triangles_cpu(uint32_t space_id, Space space, uint32_t primitives_start, uint32_t number_of_triangles, uint32_t occupied_slots,
                              int depth_min, int depth_max, float triangle_box_stop_ratio, auto find_triangle_cube_relation);

    int divide_box(InsertionContext& insertion_context, BoxInfo box_info);
    void insert_triangles_recursive(int box_id, BoxStack stack, InsertionContext& ic, uint32_t start, uint32_t size, const std::vector<std::pair<uint32_t,
                                    uint32_t>>& parent_triangles, auto find_triangle_cube_relation);
    void add_box_contents(InsertionContext& insertion_context, uint32_t box_id, std::span<const uint32_t> prim_ids);
    void free_box(int box_id);
    void free_box_content(int box_id);

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

// static_assert(AS<OctreeAS>); // OctreeAS implements AS concept
