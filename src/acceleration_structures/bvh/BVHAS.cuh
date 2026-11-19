#pragma once

#include <unordered_map>
#include <utility>
#include <vector>
#include "../../types.hpp"
#include "../../pool/soa_pool_cuda.hpp"
#include "../../geometry_common.cuh"
#include "BVHGeometry.cuh"
#include "../as_concept.cuh"
#include "../../concepts.hpp"

enum class BVHSplitMode : int
{
    SAH = 0,
    SpatialMedian = 1,
    ObjectMedian = 2
};

class BVHAS
{
public:
    soa_pool<BVHTreeInfo> trees;
    soa_pool<BVHNode> as;
    soa_pool<BVHNode16> as16;
    soa_pool<BoxContent> box_content;

    soa::cuda_device_vector<BVHTreeInfo> d_trees;
    soa::cuda_device_vector<BVHNode> d_as;
    soa::cuda_device_vector<BVHNode16> d_as16;
    soa::cuda_device_vector<BoxContent> d_box_content;

    GeometryContext* m_geometry_context = nullptr;

    BVHAS(GeometryContext* geometry_context = nullptr);

    void bind_geometry_context(GeometryContext* geometry_context)
    {
        m_geometry_context = geometry_context;
    }

    void sync_to_device()
    {
        d_trees = trees;
        d_as = as;
        d_as16 = as16;
        d_box_content = box_content;
    }

    BVHASDeviceView device_view()
    {
        return BVHASDeviceView{
            .trees = d_trees,
            .as = d_as,
            .as16 = d_as16,
            .box_content = d_box_content
        };
    }

    void config_ui();
    void info_ui(uint32_t space_id, Space space);
    void register_containers(ContainerExplorer& explorer);

    void build(uint32_t space_id, Space space, PrimitiveType primitive_type = PrimitiveType::Triangle);
    void free_space(uint32_t space_id, Space space);

    void build_gpu(uint32_t space_id, Space space, uint32_t number_of_triangles, uint32_t occupied_slots);
    void build_cpu(uint32_t space_id, Space space, uint32_t number_of_triangles, uint32_t occupied_slots);

    struct Config
    {
        BVHSplitMode split_mode = BVHSplitMode::SAH;
        int max_depth = 30;
        int max_leaf_primitives = 4;
        float traversal_cost = 1.0f;
        float intersection_cost = 1.5f;
        bool use_gpu = true;
        bool use_bvh16 = true;
    };

    Config config;

private:
    void finalize_tree_pools(
        Space space,
        glm::vec3 bounds_min,
        glm::vec3 bounds_max,
        std::vector<BVHNode>& host_nodes,
        std::vector<std::pair<uint32_t, uint32_t>>& leaf_pairs,
        uint32_t total_leaves);

    std::unordered_map<uint32_t, double> m_build_times_ms;
};
