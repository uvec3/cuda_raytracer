#pragma once

#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../types.hpp"
#include "../../pool/soa_pool_cuda.hpp"
#include "../../geometry_common.cuh"
#include "../../concepts.hpp"
#include "HeterogeneousGeometry.cuh"
#include "../octree/OctreeAS.cuh"
#include "../uniform_grid/UniformGridAS.cuh"
#include "../kdtree/KDTreeAS.cuh"
#include "../bvh/BVHAS.cuh"
#include "../../raytracer/RenderParameters.hpp"

enum class ProbeRayPattern : uint32_t
{
    CoherentTiles = 0,
    RandomSphere = 1
};

struct ASConfigVariant
{
    std::string name;
    ASType as_type = ASType::BVH;
    bool enabled = true;

    BVHAS::Config bvh_config{};
    KDTreeAS::Config kdtree_config{};
    UniformGridAS::Config grid_config{};
    OctreeAS::Config octree_config{};
};

struct MeshCandidateResult
{
    std::string variant_name;
    ASType as_type = ASType::None;
    double build_ms = 0.0;
    float probe_trace_ms = 0.0f;
    size_t as_nodes_bytes = 0;
    size_t box_content_bytes = 0;
    double bytes_per_triangle = 0.0;
    bool within_memory_limit = false;
};

struct MeshBuildRecord
{
    std::string mesh_name;
    uint32_t mesh_node_id = UINT32_MAX;
    uint32_t space_id = 0;
    uint32_t instance_count = 1;
    uint32_t num_triangles = 0;
    uint32_t num_vertices = 0;
    glm::vec3 local_bounds_min{0.0f};
    glm::vec3 local_bounds_max{0.0f};
    size_t selected_variant_idx = 0;
    std::string selected_variant_name;
    ASType selected_as_type = ASType::BVH;
    double selected_build_ms = 0.0;
    size_t selected_as_nodes_bytes = 0;
    size_t selected_box_content_bytes = 0;
    std::vector<MeshCandidateResult> candidates;
};

struct HeterogeneousInstanceRecord
{
    std::string instance_name;
    uint32_t instance_node_id = UINT32_MAX;
    uint32_t mesh_node_id = UINT32_MAX;
    uint32_t default_space_id = UINT32_MAX;
    uint32_t active_space_id = UINT32_MAX;
    ASType default_as_type = ASType::BVH;
    ASType active_as_type = ASType::BVH;
    uint32_t instance_slot = 0;
    glm::vec4 s2l_row0{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec4 s2l_row1{0.0f, 1.0f, 0.0f, 0.0f};
    glm::vec4 s2l_row2{0.0f, 0.0f, 1.0f, 0.0f};
};

class HeterogeneousAS
{
public:
    soa_pool<HeterogeneousTreeInfo> trees;
    soa_pool<HeterogeneousInstanceInfo> instances;
    soa::cuda_device_vector<HeterogeneousTreeInfo> d_trees;
    soa::cuda_device_vector<HeterogeneousInstanceInfo> d_instances;

    GeometryContext* m_geometry_context = nullptr;
    OctreeAS* m_octree = nullptr;
    UniformGridAS* m_uniform_grid = nullptr;
    KDTreeAS* m_kdtree = nullptr;
    BVHAS* m_bvh = nullptr;

    struct Config
    {
        ASType tlas_type = ASType::BVH; // Supports BVH (BVH16), KDTree, UniformGrid, Octree
        bool rebuild_blas = true;
        float max_memory_bytes_per_triangle = 256.0f;
        int num_probe_rays = 16384*16;
        uint32_t random_seed = 1u;
        ProbeRayPattern probe_ray_pattern = ProbeRayPattern::CoherentTiles;
    };

    Config config;
    std::vector<ASConfigVariant> variants;

    HeterogeneousAS(
        GeometryContext* geometry_context = nullptr,
        OctreeAS* octree = nullptr,
        UniformGridAS* uniform_grid = nullptr,
        KDTreeAS* kdtree = nullptr,
        BVHAS* bvh = nullptr);

    void set_probe_render_params(const RenderParameters& params) { m_probe_render_params = params; }
    const RenderParameters& get_probe_render_params() const { return m_probe_render_params; }

    void bind_context(
        GeometryContext* geometry_context,
        OctreeAS* octree,
        UniformGridAS* uniform_grid,
        KDTreeAS* kdtree,
        BVHAS* bvh)
    {
        m_geometry_context = geometry_context;
        m_octree = octree;
        m_uniform_grid = uniform_grid;
        m_kdtree = kdtree;
        m_bvh = bvh;
    }

    void init_default_variants();

    void sync_to_device()
    {
        d_trees = trees;
        d_instances = instances;
    }

    HeterogeneousASDeviceView device_view()
    {
        return HeterogeneousASDeviceView{
            .trees = d_trees,
            .instances = d_instances
        };
    }

    MeshBuildRecord evaluate_and_build_blas(
        uint32_t blas_space_id,
        Space blas_space,
        uint32_t num_triangles,
        uint32_t num_vertices,
        const std::string& mesh_name,
        uint32_t mesh_node_id,
        uint32_t instance_count,
        glm::vec3 local_bounds_min,
        glm::vec3 local_bounds_max,
        PrimitiveType primitive_type = PrimitiveType::Triangle);

    MeshBuildRecord make_existing_blas_record(
        uint32_t blas_space_id,
        Space blas_space,
        uint32_t num_triangles,
        uint32_t num_vertices,
        const std::string& mesh_name,
        uint32_t mesh_node_id,
        uint32_t instance_count,
        glm::vec3 local_bounds_min,
        glm::vec3 local_bounds_max) const;

    void finalize_tlas(
        uint32_t tlas_space_id,
        Space tlas_space,
        std::vector<MeshBuildRecord>&& mesh_records,
        std::vector<HeterogeneousInstanceRecord>&& instance_records,
        double blas_eval_prep_ms,
        double tlas_prep_ms = 0.0,
        std::span<const SubspacePrimitive> subspaces = {});

    void pack_subspaces(
        uint32_t tlas_space_id,
        Space tlas_space,
        const std::vector<SubspacePrimitive>& subspaces);

    std::vector<uint32_t> get_owned_blas_spaces(uint32_t space_id) const;
    bool is_blas_space_owned(uint32_t blas_space_id) const;
    void on_blas_space_rebuilt(
        uint32_t blas_space_id,
        ASType new_as_type,
        double new_build_ms = 0.0,
        size_t new_as_nodes_bytes = 0,
        size_t new_box_content_bytes = 0);
    void compute_as_memory_bytes(
        ASType as_type,
        Space space,
        size_t& out_nodes_bytes,
        size_t& out_content_bytes) const;
    void bind_instance_node_override(uint32_t instance_node_id, uint32_t override_space_id, ASType override_as_type);
    void unbind_instance_node_override(uint32_t instance_node_id);
    void free_space(uint32_t space_id, Space space);

    bool tlas_type_combo_ui(const char* label = "TLAS Type");
    void config_ui();
    void info_ui(uint32_t space_id, Space space);
    void register_containers(ContainerExplorer& explorer);

    const std::vector<MeshBuildRecord>* get_mesh_records(uint32_t space_id) const;
    const std::vector<HeterogeneousInstanceRecord>* get_instance_records(uint32_t space_id) const;
    const MeshBuildRecord* find_mesh_record_by_space(uint32_t blas_space_id) const;
    double get_build_time_ms(uint32_t space_id) const;
    double get_tlas_build_time_ms(uint32_t space_id) const;
    double get_blas_build_time_ms(uint32_t space_id) const;
    double get_blas_eval_prep_time_ms(uint32_t space_id) const;

private:
    struct SpaceMetadata
    {
        ASType tlas_as_type = ASType::BVH;
        int32_t tlas_root_box = -1;
        uint32_t instances_offset = 0;
        uint32_t instances_count = 0;
        double tlas_build_ms = 0.0;
        double blas_build_ms = 0.0;
        double blas_eval_and_prep_ms = 0.0;
        double total_build_ms = 0.0;
        std::vector<MeshBuildRecord> mesh_records;
        std::vector<HeterogeneousInstanceRecord> instance_records;
    };

    struct SavedASConfigs
    {
        BVHAS::Config bvh;
        KDTreeAS::Config kdtree;
        UniformGridAS::Config grid;
        OctreeAS::Config octree;
    };

    SavedASConfigs save_standalone_configs() const;
    void restore_standalone_configs(const SavedASConfigs& saved);

    void apply_variant_config(const ASConfigVariant& variant);
    ASType configure_tlas_builder(ASType requested_type);
    void build_single_as(ASType as_type, uint32_t space_id, Space space, PrimitiveType primitive_type);
    void free_single_as_only(ASType as_type, uint32_t space_id, Space space);
    void free_tlas_inner_as(uint32_t space_id, Space space, ASType tlas_as_type, int32_t tlas_root_box);
    float benchmark_candidate_gpu(
        ASType as_type,
        int32_t root_box,
        bool has_nagata,
        uint32_t ray_count,
        uint32_t seed);

    struct ProbeResources
    {
        uchar4* d_pixels = nullptr;
        uint32_t pixel_capacity = 0;
        cudaEvent_t ev_start = nullptr;
        cudaEvent_t ev_stop = nullptr;

        ProbeResources() = default;
        ~ProbeResources();
        ProbeResources(const ProbeResources&) = delete;
        ProbeResources& operator=(const ProbeResources&) = delete;
        ProbeResources(ProbeResources&& other) noexcept;
        ProbeResources& operator=(ProbeResources&& other) noexcept;

        void ensure(uint32_t ray_count);
        void release() noexcept;
    };

    RenderParameters m_probe_render_params{};
    ProbeResources m_probe_resources{};
    std::unordered_map<uint32_t, SpaceMetadata> m_space_metadata;
};
