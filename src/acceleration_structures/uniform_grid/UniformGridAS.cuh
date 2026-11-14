#pragma once

#include <unordered_map>
#include "../../types.hpp"
#include "../../pool/soa_pool_cuda.hpp"
#include "../../geometry_common.cuh"
#include "UniformGridGeometry.cuh"
#include "../as_concept.cuh"
#include "../../concepts.hpp"

class UniformGridAS
{
public:
    soa_pool<GridInfo> grids;
    soa_pool<uint32_t> as;
    soa_pool<BoxContent> box_content;

    soa::cuda_device_vector<GridInfo> d_grids;
    soa::cuda_device_vector<uint32_t> d_as;
    soa::cuda_device_vector<BoxContent> d_box_content;

    GeometryContext* m_geometry_context = nullptr;

    UniformGridAS(GeometryContext* geometry_context = nullptr);

    void bind_geometry_context(GeometryContext* geometry_context)
    {
        m_geometry_context = geometry_context;
    }

    void sync_to_device()
    {
        d_grids = grids;
        d_as = as;
        d_box_content = box_content;
    }

    UniformGridASDeviceView device_view()
    {
        return UniformGridASDeviceView{
            .grids = d_grids,
            .as = d_as,
            .box_content = d_box_content
        };
    }

    void config_ui();
    void info_ui(uint32_t space_id, Space space);
    void register_containers(ContainerExplorer& explorer);

    void build(uint32_t space_id, Space space);
    void free_space(uint32_t space_id, Space space);

    void insert_triangles(uint32_t space_id, Space space, glm::uvec3 dims, float cell_size,
                          uint32_t primitives_start, uint32_t number_of_triangles, uint32_t occupied_slots);
    void insert_triangles_gpu(uint32_t space_id, Space space, glm::uvec3 dims, float cell_size,
                              uint32_t primitives_start, uint32_t number_of_triangles, uint32_t occupied_slots,
                              auto find_triangle_cube_relation);

    struct Config
    {
        int resolution = 20;
        bool swapYZ = true;
        bool sort_space = true;
        PrimitiveType triangle_type = PrimitiveType::Triangle;
    };

    Config config;

private:
    std::unordered_map<uint32_t, double> m_build_times_ms;
};
