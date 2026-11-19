#pragma once

#include <span>
#include <unordered_map>
#include <utility>
#include <vector>
#include "../../types.hpp"
#include "../../pool/soa_pool_cuda.hpp"
#include "../../geometry_common.cuh"
#include "KDTreeGeometry.cuh"
#include "../as_concept.cuh"
#include "../../concepts.hpp"

enum class KDSplitMode : int
{
    SAH = 0,
    SpatialMedian = 1
};

class KDTreeAS
{
public:
    soa_pool<KDTreeInfo> trees;
    soa_pool<KDNode> as;
    soa_pool<BoxContent> box_content;

    soa::cuda_device_vector<KDTreeInfo> d_trees;
    soa::cuda_device_vector<KDNode> d_as;
    soa::cuda_device_vector<BoxContent> d_box_content;

    GeometryContext* m_geometry_context = nullptr;

    KDTreeAS(GeometryContext* geometry_context = nullptr);

    void bind_geometry_context(GeometryContext* geometry_context)
    {
        m_geometry_context = geometry_context;
    }

    void sync_to_device()
    {
        d_trees = trees;
        d_as = as;
        d_box_content = box_content;
    }

    KDTreeASDeviceView device_view()
    {
        return KDTreeASDeviceView{
            .trees = d_trees,
            .as = d_as,
            .box_content = d_box_content
        };
    }

    void config_ui();
    void info_ui(uint32_t space_id, Space space);
    void register_containers(ContainerExplorer& explorer);

    void build(uint32_t space_id, Space space, PrimitiveType primitive_type = PrimitiveType::Triangle);
    void free_space(uint32_t space_id, Space space);

    void build_gpu(uint32_t space_id, Space space, uint32_t number_of_triangles, uint32_t occupied_slots, bool is_subspace = false);
    void build_cpu(uint32_t space_id, Space space, uint32_t number_of_triangles, uint32_t occupied_slots, bool is_subspace = false);
    void build_subspaces_cpu(uint32_t space_id, Space space, std::span<const SubspacePrimitive> subspaces);

    struct Config
    {
        KDSplitMode split_mode = KDSplitMode::SAH;
        int max_depth = 30;
        int max_leaf_primitives = 8;
        float traversal_cost = 1.0f;
        float intersection_cost = 1.5f;
        bool use_gpu = true;
    };

    Config config;

private:
    void finalize_tree_pools(
        Space space,
        glm::vec3 bounds_min,
        glm::vec3 bounds_max,
        std::vector<KDNode>& host_nodes,
        std::vector<std::pair<uint32_t, uint32_t>>& leaf_pairs,
        uint32_t total_leaves);

    std::unordered_map<uint32_t, double> m_build_times_ms;
};
