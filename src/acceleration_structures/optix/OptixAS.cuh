#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <unordered_map>

#include "../../types.hpp"
#include "../../pool/soa_pool_cuda.hpp"
#include "../../geometry_common.cuh"
#include "../as_concept.cuh"
#include "../../concepts.hpp"

struct GeometryContext;
struct Space;
struct RenderParameters;
class ContainerExplorer;

struct OptixSpaceData
{
    OptixTraversableHandle gas_handle = 0;
    CUdeviceptr gas_output_buffer = 0;
    size_t gas_output_size = 0;
    uint32_t primitives_start = 0;
};

class OptixAS
{
    GeometryContext* m_geometry_context = nullptr;

    // Persistent OptiX Pipeline resources
    OptixDeviceContext m_context = nullptr;
    OptixModule m_module = nullptr;
    OptixPipeline m_pipeline = nullptr;
    OptixProgramGroup m_raygen_pg = nullptr;
    OptixProgramGroup m_miss_pg = nullptr;
    OptixProgramGroup m_hitgroup_pg = nullptr;

    CUdeviceptr d_rg_sbt = 0;
    CUdeviceptr d_ms_sbt = 0;
    CUdeviceptr d_hg_sbt = 0;
    OptixShaderBindingTable m_sbt = {};

    bool m_pipeline_initialized = false;

    // GAS per space
    std::unordered_map<uint32_t, OptixSpaceData> m_spaces;

    CUdeviceptr d_params = 0;

    void init_pipeline();
    void cleanup_gas();
    void cleanup_pipeline();

public:
    OptixAS(GeometryContext* geometry_context = nullptr);
    ~OptixAS();

    OptixAS(const OptixAS&) = delete;
    OptixAS& operator=(const OptixAS&) = delete;
    OptixAS(OptixAS&& other) noexcept;
    OptixAS& operator=(OptixAS&& other) noexcept;

    void bind_geometry_context(GeometryContext* geometry_context)
    {
        m_geometry_context = geometry_context;
    }

    void build(uint32_t space_id, Space space);
    void free_space(uint32_t space_id, Space space);

    void render(uint32_t space_id, cudaSurfaceObject_t surface, const RenderParameters& p, cudaStream_t stream = 0);
    void render(cudaSurfaceObject_t surface, const RenderParameters& p, cudaStream_t stream = 0);

    bool is_valid(uint32_t space_id = UINT32_MAX) const
    {
        if (space_id == UINT32_MAX)
            return !m_spaces.empty();
        return m_spaces.find(space_id) != m_spaces.end();
    }

    OptixTraversableHandle get_handle(uint32_t space_id = UINT32_MAX) const
    {
        if (m_spaces.empty())
            return 0;
        if (space_id == UINT32_MAX)
            return m_spaces.begin()->second.gas_handle;
        auto it = m_spaces.find(space_id);
        return (it != m_spaces.end()) ? it->second.gas_handle : 0;
    }

    void register_containers(ContainerExplorer& explorer) {}
};
