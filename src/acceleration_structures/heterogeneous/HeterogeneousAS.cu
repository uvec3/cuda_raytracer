#include "HeterogeneousAS.cuh"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <limits>

#include "imgui.h"
#include "../../cuda_common.hpp"
#include "../as_concept.cuh"

namespace heterogeneous_host_detail
{
    struct ProbeRenderParams
    {
        soa::soa_view<PrimitiveChunk16> primitives;
        soa::soa_view<Vertex> vertices;
    };

    __device__ __forceinline__ uint32_t pcg_hash(uint32_t input)
    {
        uint32_t state = input * 747796405u + 2891336453u;
        uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }

    __device__ __forceinline__ float next_rand_float(uint32_t& rng_state)
    {
        rng_state = pcg_hash(rng_state);
        return static_cast<float>(rng_state & 0x00FFFFFFu) * (1.0f / 16777216.0f);
    }

    __device__ __forceinline__ void make_probe_ray(
        uint32_t ray_idx,
        uint32_t seed,
        glm::vec3& out_orig,
        glm::vec3& out_dir)
    {
        uint32_t rng = seed ^ (ray_idx * 1664525u + 1013904223u);
        const float u = next_rand_float(rng);
        const float v = next_rand_float(rng);
        const float tx = 0.05f + 0.90f * next_rand_float(rng);
        const float ty = 0.05f + 0.90f * next_rand_float(rng);
        const float tz = 0.05f + 0.90f * next_rand_float(rng);

        const float z = 1.0f - 2.0f * u;
        const float phi = 6.28318530718f * v;
        const float r_xy = sqrtf(fmaxf(0.0f, 1.0f - z * z));
        const glm::vec3 unit_sphere(r_xy * cosf(phi), r_xy * sinf(phi), z);

        out_orig = glm::vec3(0.5f) + 0.9f * unit_sphere;
        const glm::vec3 target(tx, ty, tz);
        out_dir = glm::normalize(target - out_orig);
    }

    template <ASType CandidateType, bool HasNagata>
    __global__ void probe_subspace_candidate_kernel(
        BVHASDeviceView bvh_view,
        KDTreeASDeviceView kdtree_view,
        UniformGridASDeviceView grid_view,
        OctreeASDeviceView octree_view,
        ProbeRenderParams params,
        int32_t root_box,
        uint32_t ray_count,
        uint32_t seed,
        uint32_t* out_checksum)
    {
        constexpr uint32_t block_threads = HeterogeneousASDeviceView::PROBE_BLOCK_THREADS;
        constexpr uint32_t blas_stack_size = HeterogeneousASDeviceView::BLAS_STACK_SIZE;
        constexpr bool needs_stack = (CandidateType == ASType::BVH || CandidateType == ASType::KDTree);
        constexpr uint32_t active_stack_size = needs_stack ? blas_stack_size : 1u;

        __shared__ BVHStackEntry shared_stack[active_stack_size][block_threads];
        void* stack_base = needs_stack ? static_cast<void*>(&shared_stack[0][threadIdx.x]) : nullptr;

        const uint32_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (ray_idx >= ray_count || root_box < 0)
            return;

        glm::vec3 orig{}, dir{};
        make_probe_ray(ray_idx, seed, orig, dir);

        uint32_t closest_tri = UINT32_MAX;
        glm::vec3 closest_tuv{INFINITY, 0.0f, 0.0f};
        int it = 0;

        const auto& as_view = [&]() -> const auto& {
            if constexpr (CandidateType == ASType::BVH)
                return bvh_view;
            else if constexpr (CandidateType == ASType::KDTree)
                return kdtree_view;
            else if constexpr (CandidateType == ASType::UniformGrid)
                return grid_view;
            else
                return octree_view;
        }();

        heterogeneous_detail::traverse_as_primitives<CandidateType, HasNagata, block_threads, blas_stack_size, true, false>(
            as_view,
            root_box,
            stack_base,
            params,
            orig,
            dir,
            0.0f,
            closest_tuv,
            closest_tri,
            it,
            nullptr);

        if (closest_tri != UINT32_MAX && out_checksum != nullptr)
        {
            atomicAdd(out_checksum, (closest_tri & 0xFFFFu) ^ static_cast<uint32_t>(it));
        }
    }

    inline const char* as_type_short_name(ASType t)
    {
        switch (t)
        {
        case ASType::BVH: return "BVH16";
        case ASType::KDTree: return "KDTree";
        case ASType::UniformGrid: return "UniformGrid";
        case ASType::Octree: return "Octree";
        default: return "None";
        }
    }
} // namespace heterogeneous_host_detail

HeterogeneousAS::HeterogeneousAS(
    GeometryContext* geometry_context,
    OctreeAS* octree,
    UniformGridAS* uniform_grid,
    KDTreeAS* kdtree,
    BVHAS* bvh)
    : m_geometry_context(geometry_context)
    , m_octree(octree)
    , m_uniform_grid(uniform_grid)
    , m_kdtree(kdtree)
    , m_bvh(bvh)
{
    init_default_variants();
    sync_to_device();
}

void HeterogeneousAS::init_default_variants()
{
    variants.clear();

    {
        ASConfigVariant v{};
        v.name = "BVH16 SAH (Leaf <= 4)";
        v.as_type = ASType::BVH;
        v.enabled = true;
        v.bvh_config.split_mode = BVHSplitMode::SAH;
        v.bvh_config.max_depth = 30;
        v.bvh_config.max_leaf_primitives = 4;
        v.bvh_config.use_gpu = true;
        v.bvh_config.use_bvh16 = true;
        variants.push_back(std::move(v));
    }

    {
        ASConfigVariant v{};
        v.name = "BVH16 Compact (Leaf <= 8)";
        v.as_type = ASType::BVH;
        v.enabled = true;
        v.bvh_config.split_mode = BVHSplitMode::SAH;
        v.bvh_config.max_depth = 30;
        v.bvh_config.max_leaf_primitives = 8;
        v.bvh_config.use_gpu = true;
        v.bvh_config.use_bvh16 = true;
        variants.push_back(std::move(v));
    }

    {
        ASConfigVariant v{};
        v.name = "KDTree SAH";
        v.as_type = ASType::KDTree;
        v.enabled = true;
        v.kdtree_config.split_mode = KDSplitMode::SAH;
        v.kdtree_config.max_depth = 30;
        v.kdtree_config.max_leaf_primitives = 8;
        v.kdtree_config.use_gpu = true;
        variants.push_back(std::move(v));
    }

    {
        ASConfigVariant v{};
        v.name = "UniformGrid (0.5 vox/tri)";
        v.as_type = ASType::UniformGrid;
        v.enabled = true;
        v.grid_config.voxels_per_triangle = 0.5f;
        variants.push_back(std::move(v));
    }

    {
        ASConfigVariant v{};
        v.name = "UniformGrid (2.0 vox/tri)";
        v.as_type = ASType::UniformGrid;
        v.enabled = true;
        v.grid_config.voxels_per_triangle = 2.0f;
        variants.push_back(std::move(v));
    }

    {
        ASConfigVariant v{};
        v.name = "UniformGrid (4.0 vox/tri)";
        v.as_type = ASType::UniformGrid;
        v.enabled = true;
        v.grid_config.voxels_per_triangle = 4.0f;
        variants.push_back(std::move(v));
    }

    {
        ASConfigVariant v{};
        v.name = "Octree";
        v.as_type = ASType::Octree;
        v.enabled = true;
        v.octree_config.min_depth = 1;
        v.octree_config.max_depth = 19;
        v.octree_config.triangle_to_box_stop_ratio = 1.1f;
        v.octree_config.stretch = false;
        v.octree_config.use_gpu = true;
        variants.push_back(std::move(v));
    }
}

HeterogeneousAS::SavedASConfigs HeterogeneousAS::save_standalone_configs() const
{
    return SavedASConfigs{
        .bvh = m_bvh ? m_bvh->config : BVHAS::Config{},
        .kdtree = m_kdtree ? m_kdtree->config : KDTreeAS::Config{},
        .grid = m_uniform_grid ? m_uniform_grid->config : UniformGridAS::Config{},
        .octree = m_octree ? m_octree->config : OctreeAS::Config{}
    };
}

void HeterogeneousAS::restore_standalone_configs(const SavedASConfigs& saved)
{
    if (m_bvh) m_bvh->config = saved.bvh;
    if (m_kdtree) m_kdtree->config = saved.kdtree;
    if (m_uniform_grid) m_uniform_grid->config = saved.grid;
    if (m_octree) m_octree->config = saved.octree;
}

void HeterogeneousAS::apply_variant_config(const ASConfigVariant& variant)
{
    if (variant.as_type == ASType::BVH && m_bvh)
    {
        m_bvh->config = variant.bvh_config;
        m_bvh->config.use_bvh16 = true; // Strictly enforce BVH16 only
    }
    else if (variant.as_type == ASType::KDTree && m_kdtree)
    {
        m_kdtree->config = variant.kdtree_config;
    }
    else if (variant.as_type == ASType::UniformGrid && m_uniform_grid)
    {
        m_uniform_grid->config = variant.grid_config;
    }
    else if (variant.as_type == ASType::Octree && m_octree)
    {
        m_octree->config = variant.octree_config;
    }
}

void HeterogeneousAS::build_single_as(ASType as_type, uint32_t space_id, Space space, PrimitiveType primitive_type)
{
    space.as_type = as_type;
    space.root_box = -1;
    space.as_start = 0;
    space.as_size = 0;
    space.box_content_start = 0;
    space.box_content_size = 0;

    if (as_type == ASType::BVH && m_bvh)
        m_bvh->build(space_id, space, primitive_type);
    else if (as_type == ASType::KDTree && m_kdtree)
        m_kdtree->build(space_id, space, primitive_type);
    else if (as_type == ASType::UniformGrid && m_uniform_grid)
        m_uniform_grid->build(space_id, space, primitive_type);
    else if (as_type == ASType::Octree && m_octree)
        m_octree->build(space_id, space, primitive_type);
}

void HeterogeneousAS::free_single_as_only(ASType as_type, uint32_t space_id, Space space)
{
    space.as_type = as_type;
    if (as_type == ASType::BVH && m_bvh)
        m_bvh->free_space(space_id, space);
    else if (as_type == ASType::KDTree && m_kdtree)
        m_kdtree->free_space(space_id, space);
    else if (as_type == ASType::UniformGrid && m_uniform_grid)
        m_uniform_grid->free_space(space_id, space);
    else if (as_type == ASType::Octree && m_octree)
        m_octree->free_space(space_id, space);

    space.root_box = -1;
    space.as_start = 0;
    space.as_size = 0;
    space.box_content_start = 0;
    space.box_content_size = 0;
}

void HeterogeneousAS::free_tlas_inner_as(
    uint32_t space_id,
    Space space,
    ASType tlas_as_type,
    int32_t tlas_root_box)
{
    int t_root = tlas_root_box;
    ASType t_type = tlas_as_type;
    uint32_t t_vstart = space.vertices_start, t_vsize = space.vertices_size;
    uint32_t t_astart = space.as_start, t_asize = space.as_size;
    uint32_t t_bstart = space.box_content_start, t_bsize = space.box_content_size;
    uint32_t t_pstart = space.primitives_start, t_psize = space.primitives_size;
    glm::vec3 t_tr = space.normalization_translate;
    glm::vec3 t_sc = space.normalization_scale;
    bool t_nagata = space.has_nagata;

    Space tlas_proxy{t_root, t_type, t_vstart, t_vsize, t_astart, t_asize,
                     t_bstart, t_bsize, t_pstart, t_psize, t_tr, t_sc, t_nagata};
    free_single_as_only(t_type, space_id, tlas_proxy);
}

void HeterogeneousAS::compute_as_memory_bytes(
    ASType as_type,
    Space space,
    size_t& out_nodes_bytes,
    size_t& out_content_bytes) const
{
    out_content_bytes = static_cast<size_t>(space.box_content_size) * sizeof(BoxContent);
    if (as_type == ASType::BVH)
    {
        const bool is_bvh16 = (m_bvh != nullptr &&
            space.root_box >= 0 &&
            m_bvh->trees.is_allocated(static_cast<size_t>(space.root_box)))
            ? (m_bvh->trees[static_cast<size_t>(space.root_box)].use_bvh16 != 0u)
            : true;
        out_nodes_bytes = static_cast<size_t>(space.as_size) * (is_bvh16 ? sizeof(BVHNode16) : sizeof(BVHNode));
    }
    else if (as_type == ASType::KDTree)
        out_nodes_bytes = static_cast<size_t>(space.as_size) * sizeof(KDNode);
    else if (as_type == ASType::UniformGrid)
        out_nodes_bytes = static_cast<size_t>(space.as_size) * sizeof(uint32_t);
    else if (as_type == ASType::Octree)
        out_nodes_bytes = static_cast<size_t>(space.as_size) * sizeof(int);
    else
        out_nodes_bytes = 0;
}

float HeterogeneousAS::benchmark_candidate_gpu(
    ASType as_type,
    int32_t root_box,
    bool has_nagata,
    uint32_t ray_count,
    uint32_t seed) const
{
    if (root_box < 0 || ray_count == 0 || m_geometry_context == nullptr)
        return FLT_MAX;

    uint32_t* d_checksum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_checksum, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_checksum, 0, sizeof(uint32_t)));

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    const heterogeneous_host_detail::ProbeRenderParams params{
        .primitives = m_geometry_context->primitives.view(),
        .vertices = m_geometry_context->vertices.view()
    };

    const BVHASDeviceView bvh_view = m_bvh ? m_bvh->device_view() : BVHASDeviceView{};
    const KDTreeASDeviceView kd_view = m_kdtree ? m_kdtree->device_view() : KDTreeASDeviceView{};
    const UniformGridASDeviceView grid_view = m_uniform_grid ? m_uniform_grid->device_view() : UniformGridASDeviceView{};
    const OctreeASDeviceView oct_view = m_octree ? m_octree->device_view() : OctreeASDeviceView{};

    constexpr uint32_t block_threads = HeterogeneousASDeviceView::PROBE_BLOCK_THREADS;
    const uint32_t grid_blocks = (ray_count - 1u) / block_threads + 1u;

    auto launch_probe = [&]<ASType Type>()
    {
        if (has_nagata)
        {
            heterogeneous_host_detail::probe_subspace_candidate_kernel<Type, true>
                <<<grid_blocks, block_threads>>>(
                    bvh_view, kd_view, grid_view, oct_view, params,
                    root_box, ray_count, seed, d_checksum);
        }
        else
        {
            heterogeneous_host_detail::probe_subspace_candidate_kernel<Type, false>
                <<<grid_blocks, block_threads>>>(
                    bvh_view, kd_view, grid_view, oct_view, params,
                    root_box, ray_count, seed, d_checksum);
        }
    };

    auto dispatch = [&]()
    {
        if (as_type == ASType::BVH)
            launch_probe.template operator()<ASType::BVH>();
        else if (as_type == ASType::KDTree)
            launch_probe.template operator()<ASType::KDTree>();
        else if (as_type == ASType::UniformGrid)
            launch_probe.template operator()<ASType::UniformGrid>();
        else if (as_type == ASType::Octree)
            launch_probe.template operator()<ASType::Octree>();
    };

    // Warmup launch avoids cold JIT/L1I penalty on the first tested candidate
    dispatch();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev_start, 0));
    dispatch();
    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaGetLastError());

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_checksum);
    return elapsed_ms;
}

MeshBuildRecord HeterogeneousAS::evaluate_and_build_blas(
    uint32_t blas_space_id,
    Space blas_space,
    uint32_t num_triangles,
    uint32_t num_vertices,
    const std::string& mesh_name,
    uint32_t mesh_node_id,
    uint32_t instance_count,
    glm::vec3 local_bounds_min,
    glm::vec3 local_bounds_max,
    PrimitiveType primitive_type)
{
    assert(m_geometry_context != nullptr);

    MeshBuildRecord record{};
    record.mesh_name = mesh_name;
    record.mesh_node_id = mesh_node_id;
    record.space_id = blas_space_id;
    record.instance_count = instance_count;
    record.num_triangles = num_triangles;
    record.num_vertices = num_vertices;
    record.local_bounds_min = local_bounds_min;
    record.local_bounds_max = local_bounds_max;

    if (variants.empty())
        init_default_variants();

    const SavedASConfigs saved_configs = save_standalone_configs();
    const uint32_t ray_count = static_cast<uint32_t>(std::max(config.num_probe_rays, 256));
    const double max_mem_per_tri = static_cast<double>(config.max_memory_bytes_per_triangle);

    std::vector<size_t> candidate_variant_indices;
    candidate_variant_indices.reserve(variants.size());

    for (size_t i = 0; i < variants.size(); ++i)
    {
        if (!variants[i].enabled)
            continue;

        apply_variant_config(variants[i]);

        const auto t0 = std::chrono::high_resolution_clock::now();
        build_single_as(variants[i].as_type, blas_space_id, blas_space, primitive_type);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        size_t nodes_bytes = 0;
        size_t content_bytes = 0;
        compute_as_memory_bytes(variants[i].as_type, blas_space, nodes_bytes, content_bytes);

        const double denom = static_cast<double>(std::max(1u, num_triangles));
        const double bytes_per_tri = static_cast<double>(nodes_bytes + content_bytes) / denom;
        const bool within_limit = bytes_per_tri <= max_mem_per_tri;

        const float probe_ms = benchmark_candidate_gpu(
            variants[i].as_type,
            blas_space.root_box,
            blas_space.has_nagata,
            ray_count,
            config.random_seed);

        free_single_as_only(variants[i].as_type, blas_space_id, blas_space);

        MeshCandidateResult cand{};
        cand.variant_name = variants[i].name;
        cand.as_type = variants[i].as_type;
        cand.build_ms = build_ms;
        cand.probe_trace_ms = probe_ms;
        cand.as_nodes_bytes = nodes_bytes;
        cand.box_content_bytes = content_bytes;
        cand.bytes_per_triangle = bytes_per_tri;
        cand.within_memory_limit = within_limit;

        record.candidates.push_back(std::move(cand));
        candidate_variant_indices.push_back(i);
    }

    // Fallback to a default BVH16 preset if all variants were disabled in UI
    if (record.candidates.empty())
    {
        ASConfigVariant fallback{};
        fallback.name = "BVH16 Fallback";
        fallback.as_type = ASType::BVH;
        fallback.bvh_config.use_bvh16 = true;

        apply_variant_config(fallback);
        const auto final_t0 = std::chrono::high_resolution_clock::now();
        build_single_as(ASType::BVH, blas_space_id, blas_space, primitive_type);
        const auto final_t1 = std::chrono::high_resolution_clock::now();
        const double final_ms = std::chrono::duration<double, std::milli>(final_t1 - final_t0).count();

        size_t nodes_bytes = 0;
        size_t content_bytes = 0;
        compute_as_memory_bytes(ASType::BVH, blas_space, nodes_bytes, content_bytes);
        const double bytes_per_tri = static_cast<double>(nodes_bytes + content_bytes) /
            static_cast<double>(std::max(1u, num_triangles));

        MeshCandidateResult cand{};
        cand.variant_name = fallback.name;
        cand.as_type = ASType::BVH;
        cand.build_ms = final_ms;
        cand.as_nodes_bytes = nodes_bytes;
        cand.box_content_bytes = content_bytes;
        cand.bytes_per_triangle = bytes_per_tri;
        cand.within_memory_limit = (bytes_per_tri <= max_mem_per_tri);
        record.candidates.push_back(std::move(cand));

        record.selected_variant_idx = 0;
        record.selected_variant_name = fallback.name;
        record.selected_as_type = ASType::BVH;
        record.selected_build_ms = final_ms;
        record.selected_as_nodes_bytes = nodes_bytes;
        record.selected_box_content_bytes = content_bytes;
        restore_standalone_configs(saved_configs);
        return record;
    }

    size_t best_pos = std::numeric_limits<size_t>::max();
    float best_time = FLT_MAX;
    for (size_t c = 0; c < record.candidates.size(); ++c)
    {
        if (record.candidates[c].within_memory_limit && record.candidates[c].probe_trace_ms < best_time)
        {
            best_time = record.candidates[c].probe_trace_ms;
            best_pos = c;
        }
    }

    if (best_pos == std::numeric_limits<size_t>::max())
    {
        double min_bpt = DBL_MAX;
        for (size_t c = 0; c < record.candidates.size(); ++c)
        {
            if (record.candidates[c].bytes_per_triangle < min_bpt)
            {
                min_bpt = record.candidates[c].bytes_per_triangle;
                best_pos = c;
            }
        }
    }

    const size_t winning_variant_idx = candidate_variant_indices[best_pos];
    record.selected_variant_idx = winning_variant_idx;
    record.selected_variant_name = variants[winning_variant_idx].name;
    record.selected_as_type = variants[winning_variant_idx].as_type;

    apply_variant_config(variants[winning_variant_idx]);
    const auto final_t0 = std::chrono::high_resolution_clock::now();
    build_single_as(variants[winning_variant_idx].as_type, blas_space_id, blas_space, primitive_type);
    const auto final_t1 = std::chrono::high_resolution_clock::now();
    record.selected_build_ms = std::chrono::duration<double, std::milli>(final_t1 - final_t0).count();
    compute_as_memory_bytes(
        record.selected_as_type,
        blas_space,
        record.selected_as_nodes_bytes,
        record.selected_box_content_bytes);

    restore_standalone_configs(saved_configs);
    return record;
}

MeshBuildRecord HeterogeneousAS::make_existing_blas_record(
    uint32_t blas_space_id,
    Space blas_space,
    uint32_t num_triangles,
    uint32_t num_vertices,
    const std::string& mesh_name,
    uint32_t mesh_node_id,
    uint32_t instance_count,
    glm::vec3 local_bounds_min,
    glm::vec3 local_bounds_max) const
{
    MeshBuildRecord record{};
    record.mesh_name = mesh_name;
    record.mesh_node_id = mesh_node_id;
    record.space_id = blas_space_id;
    record.instance_count = instance_count;
    record.num_triangles = num_triangles;
    record.num_vertices = num_vertices;
    record.local_bounds_min = local_bounds_min;
    record.local_bounds_max = local_bounds_max;
    record.selected_as_type = blas_space.as_type;
    record.selected_variant_name = std::string("Existing (") +
        heterogeneous_host_detail::as_type_short_name(blas_space.as_type) + ")";
    record.selected_build_ms = 0.0;
    compute_as_memory_bytes(
        blas_space.as_type,
        blas_space,
        record.selected_as_nodes_bytes,
        record.selected_box_content_bytes);
    return record;
}

void HeterogeneousAS::pack_subspaces(
    uint32_t /*tlas_space_id*/,
    Space tlas_space,
    const std::vector<SubspacePrimitive>& subspaces)
{
    if (subspaces.empty() || m_geometry_context == nullptr)
        return;

    const uint32_t count = static_cast<uint32_t>(subspaces.size());
    const uint32_t slots = count * SubspacePrimitive::occupied_slots;
    const uint32_t prim_start = static_cast<uint32_t>(m_geometry_context->primitives.alloc(slots));
    tlas_space.primitives_start = prim_start;
    tlas_space.primitives_size = slots;

    static_assert(sizeof(SubspacePrimitive) == SubspacePrimitive::occupied_slots * sizeof(PrimitiveChunk16));
    CUDA_CHECK(cudaMemcpyAsync(
        &m_geometry_context->primitives[prim_start],
        subspaces.data(),
        static_cast<size_t>(count) * sizeof(SubspacePrimitive),
        cudaMemcpyHostToDevice));
}

ASType HeterogeneousAS::configure_tlas_builder(ASType requested_type)
{
    ASType tlas_as_type = requested_type;
    if (tlas_as_type != ASType::BVH &&
        tlas_as_type != ASType::KDTree &&
        tlas_as_type != ASType::UniformGrid &&
        tlas_as_type != ASType::Octree)
    {
        tlas_as_type = ASType::BVH;
    }

    if (tlas_as_type == ASType::BVH && m_bvh)
    {
        m_bvh->config.use_bvh16 = true;
        m_bvh->config.split_mode = BVHSplitMode::SAH;
        m_bvh->config.max_leaf_primitives = 1;
        m_bvh->config.use_gpu = true;
    }
    else if (tlas_as_type == ASType::KDTree && m_kdtree)
    {
        m_kdtree->config.split_mode = KDSplitMode::SAH;
        m_kdtree->config.max_leaf_primitives = 2;
        m_kdtree->config.use_gpu = true;
    }
    else if (tlas_as_type == ASType::UniformGrid && m_uniform_grid)
    {
        m_uniform_grid->config.voxels_per_triangle = 12.0f;
    }
    else if (tlas_as_type == ASType::Octree && m_octree)
    {
        m_octree->config.min_depth = 1;
        m_octree->config.max_depth = 19;
        m_octree->config.triangle_to_box_stop_ratio = 2.f;
        m_octree->config.stretch = false;
        m_octree->config.use_gpu = true;
    }

    return tlas_as_type;
}

void HeterogeneousAS::finalize_tlas(
    uint32_t tlas_space_id,
    Space tlas_space,
    std::vector<MeshBuildRecord>&& mesh_records,
    std::vector<HeterogeneousInstanceRecord>&& instance_records,
    double blas_eval_prep_ms,
    double tlas_prep_ms,
    std::span<const SubspacePrimitive> subspaces)
{
    assert(m_geometry_context != nullptr);

    const auto tlas_t0 = std::chrono::high_resolution_clock::now();
    const SavedASConfigs saved_configs = save_standalone_configs();
    const ASType tlas_as_type = configure_tlas_builder(config.tlas_type);

    constexpr size_t kCpuSubspaceTlasThreshold = 512u;
    if (!subspaces.empty() && subspaces.size() <= kCpuSubspaceTlasThreshold)
    {
        tlas_space.as_type = tlas_as_type;
        tlas_space.root_box = -1;
        tlas_space.as_start = 0;
        tlas_space.as_size = 0;
        tlas_space.box_content_start = 0;
        tlas_space.box_content_size = 0;

        if (tlas_as_type == ASType::BVH && m_bvh)
            m_bvh->build_subspaces_cpu(tlas_space_id, tlas_space, subspaces);
        else if (tlas_as_type == ASType::KDTree && m_kdtree)
            m_kdtree->build_subspaces_cpu(tlas_space_id, tlas_space, subspaces);
        else if (tlas_as_type == ASType::UniformGrid && m_uniform_grid)
            m_uniform_grid->build_subspaces_cpu(tlas_space_id, tlas_space, subspaces);
        else if (tlas_as_type == ASType::Octree && m_octree)
            m_octree->build_subspaces_cpu(tlas_space_id, tlas_space, subspaces);
    }
    else
    {
        build_single_as(tlas_as_type, tlas_space_id, tlas_space, PrimitiveType::Subspace);
    }
    restore_standalone_configs(saved_configs);

    const auto tlas_t1 = std::chrono::high_resolution_clock::now();
    const double tlas_ms = std::max(0.0, tlas_prep_ms) +
        std::chrono::duration<double, std::milli>(tlas_t1 - tlas_t0).count();

    const uint32_t inst_count = static_cast<uint32_t>(instance_records.size());
    uint32_t inst_offset = 0;
    if (inst_count > 0)
    {
        inst_offset = static_cast<uint32_t>(instances.alloc(inst_count));
        for (uint32_t i = 0; i < inst_count; ++i)
        {
            const uint32_t slot = inst_offset + i;
            instance_records[i].instance_slot = slot;
            instances[slot].s2l_row0 = instance_records[i].s2l_row0;
            instances[slot].s2l_row1 = instance_records[i].s2l_row1;
            instances[slot].s2l_row2 = instance_records[i].s2l_row2;
            instances[slot].space_id = static_cast<int32_t>(instance_records[i].active_space_id);
        }
    }

    const int32_t tlas_root_box = tlas_space.root_box;
    const uint32_t tree_id = static_cast<uint32_t>(trees.alloc_element());
    trees[tree_id].tlas_space_id = static_cast<int32_t>(tlas_space_id);
    trees[tree_id].tlas_root_box = tlas_root_box;
    trees[tree_id].tlas_as_type = static_cast<uint32_t>(tlas_as_type);
    trees[tree_id].blas_offset = inst_offset;
    trees[tree_id].blas_count = inst_count;

    tlas_space.root_box = static_cast<int>(tree_id);
    tlas_space.as_type = ASType::Heterogeneous;

    double sum_selected_blas_ms = 0.0;
    for (const auto& rec : mesh_records)
    {
        sum_selected_blas_ms += rec.selected_build_ms;
    }
    const double clamped_eval_prep_ms = std::max(0.0, blas_eval_prep_ms);

    m_space_metadata[tlas_space_id] = SpaceMetadata{
        .tlas_as_type = tlas_as_type,
        .tlas_root_box = tlas_root_box,
        .instances_offset = inst_offset,
        .instances_count = inst_count,
        .tlas_build_ms = tlas_ms,
        .blas_build_ms = sum_selected_blas_ms,
        .blas_eval_and_prep_ms = clamped_eval_prep_ms,
        .total_build_ms = sum_selected_blas_ms + clamped_eval_prep_ms + tlas_ms,
        .mesh_records = std::move(mesh_records),
        .instance_records = std::move(instance_records)
    };

    soa::upload_soa_range_async(d_trees, trees, tree_id, 1);
    if (inst_count > 0)
    {
        soa::upload_soa_range_async(d_instances, instances, inst_offset, inst_count);
    }
}

std::vector<uint32_t> HeterogeneousAS::get_owned_blas_spaces(uint32_t space_id) const
{
    std::vector<uint32_t> result;
    auto it = m_space_metadata.find(space_id);
    if (it == m_space_metadata.end())
        return result;

    result.reserve(it->second.mesh_records.size());
    for (const auto& rec : it->second.mesh_records)
    {
        result.push_back(rec.space_id);
    }
    return result;
}

bool HeterogeneousAS::is_blas_space_owned(uint32_t blas_space_id) const
{
    for (const auto& [tlas_id, meta] : m_space_metadata)
    {
        for (const auto& rec : meta.mesh_records)
        {
            if (rec.space_id == blas_space_id)
                return true;
        }
    }
    return false;
}

void HeterogeneousAS::on_blas_space_rebuilt(
    uint32_t blas_space_id,
    ASType new_as_type,
    double new_build_ms,
    size_t new_as_nodes_bytes,
    size_t new_box_content_bytes)
{
    const char* short_name = heterogeneous_host_detail::as_type_short_name(new_as_type);
    for (auto& [tlas_id, meta] : m_space_metadata)
    {
        bool touched = false;
        for (auto& rec : meta.mesh_records)
        {
            if (rec.space_id == blas_space_id)
            {
                rec.selected_as_type = new_as_type;
                rec.selected_variant_name = std::string("Manual (") + short_name + ")";
                rec.selected_build_ms = (new_as_type == ASType::None) ? 0.0 : new_build_ms;
                rec.selected_as_nodes_bytes = (new_as_type == ASType::None) ? 0 : new_as_nodes_bytes;
                rec.selected_box_content_bytes = (new_as_type == ASType::None) ? 0 : new_box_content_bytes;
                touched = true;
            }
        }
        for (auto& inst_rec : meta.instance_records)
        {
            if (inst_rec.default_space_id == blas_space_id)
            {
                inst_rec.default_as_type = new_as_type;
                if (inst_rec.active_space_id == blas_space_id)
                {
                    inst_rec.active_as_type = new_as_type;
                }
            }
        }
        if (touched)
        {
            double sum_blas = 0.0;
            for (const auto& rec : meta.mesh_records)
            {
                sum_blas += rec.selected_build_ms;
            }
            meta.blas_build_ms = sum_blas;
            meta.total_build_ms = meta.blas_build_ms + meta.blas_eval_and_prep_ms + meta.tlas_build_ms;
        }
    }
}

void HeterogeneousAS::bind_instance_node_override(uint32_t instance_node_id, uint32_t override_space_id, ASType override_as_type)
{
    if (instance_node_id == UINT32_MAX)
        return;

    bool updated = false;
    for (auto& [tlas_id, meta] : m_space_metadata)
    {
        for (auto& inst_rec : meta.instance_records)
        {
            if (inst_rec.instance_node_id == instance_node_id)
            {
                inst_rec.active_space_id = override_space_id;
                inst_rec.active_as_type = override_as_type;
                instances[inst_rec.instance_slot].space_id = static_cast<int32_t>(override_space_id);
                updated = true;
            }
        }
    }
    if (updated)
    {
        d_instances = instances;
    }
}

void HeterogeneousAS::unbind_instance_node_override(uint32_t instance_node_id)
{
    if (instance_node_id == UINT32_MAX)
        return;

    bool updated = false;
    for (auto& [tlas_id, meta] : m_space_metadata)
    {
        for (auto& inst_rec : meta.instance_records)
        {
            if (inst_rec.instance_node_id == instance_node_id)
            {
                inst_rec.active_space_id = inst_rec.default_space_id;
                inst_rec.active_as_type = inst_rec.default_as_type;
                instances[inst_rec.instance_slot].space_id = static_cast<int32_t>(inst_rec.default_space_id);
                updated = true;
            }
        }
    }
    if (updated)
    {
        d_instances = instances;
    }
}

void HeterogeneousAS::free_space(uint32_t space_id, Space space)
{
    if (space.as_type != ASType::Heterogeneous)
    {
        m_space_metadata.erase(space_id);
        return;
    }

    const int tree_id = space.root_box;
    auto it = m_space_metadata.find(space_id);
    if (it != m_space_metadata.end())
    {
        if (it->second.instances_count > 0)
        {
            instances.free(it->second.instances_offset);
        }

        free_tlas_inner_as(space_id, space, it->second.tlas_as_type, it->second.tlas_root_box);
        m_space_metadata.erase(it);
    }

    if (tree_id >= 0)
    {
        trees.free(static_cast<size_t>(tree_id));
    }
}

const std::vector<MeshBuildRecord>* HeterogeneousAS::get_mesh_records(uint32_t space_id) const
{
    auto it = m_space_metadata.find(space_id);
    if (it == m_space_metadata.end())
        return nullptr;
    return &it->second.mesh_records;
}

const std::vector<HeterogeneousInstanceRecord>* HeterogeneousAS::get_instance_records(uint32_t space_id) const
{
    auto it = m_space_metadata.find(space_id);
    if (it == m_space_metadata.end())
        return nullptr;
    return &it->second.instance_records;
}

const MeshBuildRecord* HeterogeneousAS::find_mesh_record_by_space(uint32_t blas_space_id) const
{
    for (const auto& [tlas_id, meta] : m_space_metadata)
    {
        for (const auto& rec : meta.mesh_records)
        {
            if (rec.space_id == blas_space_id)
                return &rec;
        }
    }
    return nullptr;
}

double HeterogeneousAS::get_build_time_ms(uint32_t space_id) const
{
    auto it = m_space_metadata.find(space_id);
    return (it != m_space_metadata.end()) ? it->second.total_build_ms : 0.0;
}

double HeterogeneousAS::get_tlas_build_time_ms(uint32_t space_id) const
{
    auto it = m_space_metadata.find(space_id);
    return (it != m_space_metadata.end()) ? it->second.tlas_build_ms : 0.0;
}

double HeterogeneousAS::get_blas_build_time_ms(uint32_t space_id) const
{
    auto it = m_space_metadata.find(space_id);
    return (it != m_space_metadata.end()) ? it->second.blas_build_ms : 0.0;
}

double HeterogeneousAS::get_blas_eval_prep_time_ms(uint32_t space_id) const
{
    auto it = m_space_metadata.find(space_id);
    return (it != m_space_metadata.end()) ? it->second.blas_eval_and_prep_ms : 0.0;
}

bool HeterogeneousAS::tlas_type_combo_ui(const char* label)
{
    const char* tlas_names[] = {"BVH16 (SAH)", "KD-Tree (SAH)", "Uniform Grid", "Octree"};
    int tlas_idx = 0;
    if (config.tlas_type == ASType::KDTree) tlas_idx = 1;
    else if (config.tlas_type == ASType::UniformGrid) tlas_idx = 2;
    else if (config.tlas_type == ASType::Octree) tlas_idx = 3;

    if (ImGui::Combo(label, &tlas_idx, tlas_names, 4))
    {
        if (tlas_idx == 0) config.tlas_type = ASType::BVH;
        else if (tlas_idx == 1) config.tlas_type = ASType::KDTree;
        else if (tlas_idx == 2) config.tlas_type = ASType::UniformGrid;
        else if (tlas_idx == 3) config.tlas_type = ASType::Octree;
        return true;
    }
    return false;
}

void HeterogeneousAS::config_ui()
{
    tlas_type_combo_ui("TLAS Type");
    ImGui::Checkbox("Rebuild BLAS", &config.rebuild_blas);

    ImGui::DragFloat("Max AS Bytes / Tri", &config.max_memory_bytes_per_triangle, 1.0f, 4.0f, 512.0f, "%.1f B");
    ImGui::DragInt("Probe Rays / Candidate", &config.num_probe_rays, 256.0f, 1024, 262144);

    if (ImGui::TreeNode("Candidate Presets (BVH uses BVH16 only)"))
    {
        for (size_t i = 0; i < variants.size(); ++i)
        {
            ImGui::PushID(static_cast<int>(i));
            ImGui::Checkbox(variants[i].name.c_str(), &variants[i].enabled);
            ImGui::PopID();
        }
        ImGui::TreePop();
    }
}

void HeterogeneousAS::info_ui(uint32_t space_id, Space space)
{
    auto it = m_space_metadata.find(space_id);
    if (it == m_space_metadata.end())
        return;

    const auto& meta = it->second;
    ImGui::Text("  Total Build Time: %.2f ms", meta.total_build_ms);
    ImGui::Text("    TLAS Build:     %.2f ms", meta.tlas_build_ms);
    ImGui::Text("    BLAS Build:     %.2f ms", meta.blas_build_ms);
    ImGui::Text("    BLAS Eval+Prep: %.2f ms", meta.blas_eval_and_prep_ms);
    ImGui::Text("  TLAS Type:        %s (%u instances across %u unique meshes)",
                heterogeneous_host_detail::as_type_short_name(meta.tlas_as_type),
                static_cast<uint32_t>(meta.instance_records.size()),
                static_cast<uint32_t>(meta.mesh_records.size()));

    size_t tlas_nodes_bytes = 0;
    size_t tlas_content_bytes = 0;
    compute_as_memory_bytes(meta.tlas_as_type, space, tlas_nodes_bytes, tlas_content_bytes);

    uint32_t unique_tris = 0;
    uint32_t unique_verts = 0;
    size_t total_blas_as_bytes = 0;
    uint32_t count_bvh = 0, count_kd = 0, count_grid = 0, count_oct = 0;

    for (const auto& rec : meta.mesh_records)
    {
        unique_tris += rec.num_triangles;
        unique_verts += rec.num_vertices;
        if (rec.selected_as_type == ASType::BVH) ++count_bvh;
        else if (rec.selected_as_type == ASType::KDTree) ++count_kd;
        else if (rec.selected_as_type == ASType::UniformGrid) ++count_grid;
        else if (rec.selected_as_type == ASType::Octree) ++count_oct;

        total_blas_as_bytes += rec.selected_as_nodes_bytes + rec.selected_box_content_bytes;
    }

    ImGui::Text("  Selections: BVH16=%u, KD=%u, Grid=%u, Oct=%u", count_bvh, count_kd, count_grid, count_oct);
    ImGui::Text("  Unique Tris: %u, Unique Verts: %u", unique_tris, unique_verts);
    ImGui::Text("  TLAS AS:    %s", format_bytes(tlas_nodes_bytes + tlas_content_bytes).c_str());
    ImGui::Text("  BLAS AS:    %s", format_bytes(total_blas_as_bytes).c_str());

    if (ImGui::TreeNode("Per-Mesh AS Breakdown"))
    {
        for (size_t i = 0; i < meta.mesh_records.size(); ++i)
        {
            const auto& rec = meta.mesh_records[i];
            if (ImGui::TreeNode(reinterpret_cast<void*>(static_cast<uintptr_t>(i)),
                                "%s [%s, %u tris, %u inst]",
                                rec.mesh_name.empty() ? "Mesh" : rec.mesh_name.c_str(),
                                heterogeneous_host_detail::as_type_short_name(rec.selected_as_type),
                                rec.num_triangles,
                                rec.instance_count))
            {
                if (rec.mesh_node_id != UINT32_MAX)
                    ImGui::Text("  Selected: %s (Node #%u, Space #%u)", rec.selected_variant_name.c_str(), rec.mesh_node_id, rec.space_id);
                else
                    ImGui::Text("  Selected: %s (Space #%u)", rec.selected_variant_name.c_str(), rec.space_id);

                ImGui::Text("  BLAS Build Time: %.2f ms", rec.selected_build_ms);
                for (const auto& cand : rec.candidates)
                {
                    ImGui::Text("    %-24s | %6.3f ms probe | %6.2f ms build | %6.1f B/tri %s",
                                cand.variant_name.c_str(),
                                cand.probe_trace_ms,
                                cand.build_ms,
                                cand.bytes_per_triangle,
                                cand.within_memory_limit ? "" : "[over]");
                }
                ImGui::TreePop();
            }
        }
        ImGui::TreePop();
    }

    if (ImGui::TreeNode("Per-Instance Node Bindings"))
    {
        for (size_t i = 0; i < meta.instance_records.size(); ++i)
        {
            const auto& inst = meta.instance_records[i];
            const bool is_override = (inst.active_space_id != inst.default_space_id);
            ImGui::Text("  %s (Node #%u) -> [%s, Space #%u%s]",
                        inst.instance_name.empty() ? "Instance" : inst.instance_name.c_str(),
                        inst.instance_node_id,
                        heterogeneous_host_detail::as_type_short_name(inst.active_as_type),
                        inst.active_space_id,
                        is_override ? " override" : " shared");
        }
        ImGui::TreePop();
    }
}
