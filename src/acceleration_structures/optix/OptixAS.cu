#include "OptixAS.cuh"

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <cuda_runtime.h>
#include <iostream>

#include "params.hpp"
#include "../../cuda_common.hpp"
#include "../../raytracer/rt_kernels.cuh"
#include "shaders_optix_ir.h"

#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        OptixResult res = call;                                                \
        if (res != OPTIX_SUCCESS) {                                            \
            std::cerr << "OptiX Error: " << optixGetErrorName(res) << " ("     \
                      << res << ") at line " << __LINE__ << std::endl;         \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static void optix_log_callback(unsigned int level, const char* tag, const char* message, void*)
{
    std::cout << "[OptiX][" << level << "][" << tag << "]: " << message << std::endl;
}

OptixAS::OptixAS(GeometryContext* geometry_context)
    : m_geometry_context(geometry_context)
{
}

OptixAS::OptixAS(OptixAS&& other) noexcept
    : m_geometry_context(other.m_geometry_context)
    , m_context(other.m_context)
    , m_module(other.m_module)
    , m_pipeline(other.m_pipeline)
    , m_raygen_pg(other.m_raygen_pg)
    , m_miss_pg(other.m_miss_pg)
    , m_hitgroup_pg(other.m_hitgroup_pg)
    , d_rg_sbt(other.d_rg_sbt)
    , d_ms_sbt(other.d_ms_sbt)
    , d_hg_sbt(other.d_hg_sbt)
    , m_sbt(other.m_sbt)
    , m_pipeline_initialized(other.m_pipeline_initialized)
    , m_spaces(std::move(other.m_spaces))
    , d_params(other.d_params)
{
    other.m_context = nullptr;
    other.m_module = nullptr;
    other.m_pipeline = nullptr;
    other.m_raygen_pg = nullptr;
    other.m_miss_pg = nullptr;
    other.m_hitgroup_pg = nullptr;
    other.d_rg_sbt = 0;
    other.d_ms_sbt = 0;
    other.d_hg_sbt = 0;
    other.m_sbt = {};
    other.m_pipeline_initialized = false;
    other.m_spaces.clear();
    other.d_params = 0;
}

OptixAS& OptixAS::operator=(OptixAS&& other) noexcept
{
    if (this != &other)
    {
        cleanup_pipeline();
        m_geometry_context = other.m_geometry_context;
        m_context = other.m_context;
        m_module = other.m_module;
        m_pipeline = other.m_pipeline;
        m_raygen_pg = other.m_raygen_pg;
        m_miss_pg = other.m_miss_pg;
        m_hitgroup_pg = other.m_hitgroup_pg;
        d_rg_sbt = other.d_rg_sbt;
        d_ms_sbt = other.d_ms_sbt;
        d_hg_sbt = other.d_hg_sbt;
        m_sbt = other.m_sbt;
        m_pipeline_initialized = other.m_pipeline_initialized;
        m_spaces = std::move(other.m_spaces);
        d_params = other.d_params;

        other.m_context = nullptr;
        other.m_module = nullptr;
        other.m_pipeline = nullptr;
        other.m_raygen_pg = nullptr;
        other.m_miss_pg = nullptr;
        other.m_hitgroup_pg = nullptr;
        other.d_rg_sbt = 0;
        other.d_ms_sbt = 0;
        other.d_hg_sbt = 0;
        other.m_sbt = {};
        other.m_pipeline_initialized = false;
        other.m_spaces.clear();
        other.d_params = 0;
    }
    return *this;
}

OptixAS::~OptixAS()
{
    cleanup_pipeline();
}

void OptixAS::cleanup_gas()
{
    for (auto& [id, data] : m_spaces)
    {
        if (data.gas_output_buffer)
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(data.gas_output_buffer)));
        }
    }
    m_spaces.clear();
}

void OptixAS::cleanup_pipeline()
{
    cleanup_gas();

    if (d_params)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));
        d_params = 0;
    }
    if (d_rg_sbt)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_rg_sbt)));
        d_rg_sbt = 0;
    }
    if (d_ms_sbt)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ms_sbt)));
        d_ms_sbt = 0;
    }
    if (d_hg_sbt)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_hg_sbt)));
        d_hg_sbt = 0;
    }
    if (m_pipeline)
    {
        optixPipelineDestroy(m_pipeline);
        m_pipeline = nullptr;
    }
    if (m_raygen_pg)
    {
        optixProgramGroupDestroy(m_raygen_pg);
        m_raygen_pg = nullptr;
    }
    if (m_miss_pg)
    {
        optixProgramGroupDestroy(m_miss_pg);
        m_miss_pg = nullptr;
    }
    if (m_hitgroup_pg)
    {
        optixProgramGroupDestroy(m_hitgroup_pg);
        m_hitgroup_pg = nullptr;
    }
    if (m_module)
    {
        optixModuleDestroy(m_module);
        m_module = nullptr;
    }
    if (m_context)
    {
        optixDeviceContextDestroy(m_context);
        m_context = nullptr;
    }
    m_pipeline_initialized = false;
}

void OptixAS::init_pipeline()
{
    if (m_pipeline_initialized)
        return;

    // 1. Initialize CUDA and OptiX
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());

    // 2. Create OptixDeviceContext
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &optix_log_callback;
    options.logCallbackLevel = 4;
#ifdef OPTIX_ENABLE_VALIDATION
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#else
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
#endif
    CUcontext cuCtx = 0;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &m_context));

    // 3. Create OptiX Module from embedded bytecode
    OptixModuleCompileOptions module_compile_options = {};
    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 4; // r, g, b, depth
    pipeline_compile_options.numAttributeValues = 2; // u, v barycentrics
#ifdef OPTIX_ENABLE_VALIDATION
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
#else
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    pipeline_compile_options.usesPrimitiveTypeFlags = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    OPTIX_CHECK(optixModuleCreate(
        m_context,
        &module_compile_options,
        &pipeline_compile_options,
        reinterpret_cast<const char*>(embedded_optix_ir),
        sizeof(embedded_optix_ir),
        nullptr, nullptr,
        &m_module
    ));

    // 4. Create Program Groups
    OptixProgramGroupOptions pg_options = {};

    OptixProgramGroupDesc rg_desc = {};
    rg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rg_desc.raygen.module = m_module;
    rg_desc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK(optixProgramGroupCreate(m_context, &rg_desc, 1, &pg_options, nullptr, nullptr, &m_raygen_pg));

    OptixProgramGroupDesc ms_desc = {};
    ms_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    ms_desc.miss.module = m_module;
    ms_desc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK(optixProgramGroupCreate(m_context, &ms_desc, 1, &pg_options, nullptr, nullptr, &m_miss_pg));

    OptixProgramGroupDesc hg_desc = {};
    hg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hg_desc.hitgroup.moduleCH = m_module;
    hg_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OPTIX_CHECK(optixProgramGroupCreate(m_context, &hg_desc, 1, &pg_options, nullptr, nullptr, &m_hitgroup_pg));

    // 5. Link Pipeline
    OptixProgramGroup program_groups[] = { m_raygen_pg, m_miss_pg, m_hitgroup_pg };
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 8;

    OPTIX_CHECK(optixPipelineCreate(
        m_context,
        &pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        3,
        nullptr, nullptr,
        &m_pipeline
    ));

    // 6. Configure Stack Sizes
    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK(optixUtilAccumulateStackSizes(m_raygen_pg, &stack_sizes, m_pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(m_miss_pg, &stack_sizes, m_pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(m_hitgroup_pg, &stack_sizes, m_pipeline));

    uint32_t direct_callable_stack_size_from_traversal = 0;
    uint32_t direct_callable_stack_size_from_state = 0;
    uint32_t continuation_stack_size = 0;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        8, // maxTraceDepth
        0, // maxCCDepth
        0, // maxDCDepth
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size
    ));
    OPTIX_CHECK(optixPipelineSetStackSize(
        m_pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        1 // maxTraversableGraphDepth
    ));

    // 7. Populate SBT
    EmptySbtRecord rg_sbt, ms_sbt, hg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(m_raygen_pg, &rg_sbt));
    OPTIX_CHECK(optixSbtRecordPackHeader(m_miss_pg, &ms_sbt));
    OPTIX_CHECK(optixSbtRecordPackHeader(m_hitgroup_pg, &hg_sbt));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rg_sbt), sizeof(EmptySbtRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ms_sbt), sizeof(EmptySbtRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hg_sbt), sizeof(EmptySbtRecord)));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_rg_sbt), &rg_sbt, sizeof(EmptySbtRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_ms_sbt), &ms_sbt, sizeof(EmptySbtRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hg_sbt), &hg_sbt, sizeof(EmptySbtRecord), cudaMemcpyHostToDevice));

    m_sbt.raygenRecord = d_rg_sbt;
    m_sbt.missRecordBase = d_ms_sbt;
    m_sbt.missRecordStrideInBytes = sizeof(EmptySbtRecord);
    m_sbt.missRecordCount = 1;
    m_sbt.hitgroupRecordBase = d_hg_sbt;
    m_sbt.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
    m_sbt.hitgroupRecordCount = 1;

    m_pipeline_initialized = true;
}

void OptixAS::build(uint32_t space_id, Space space)
{
    if (!m_pipeline_initialized)
    {
        init_pipeline();
    }

    if (space.primitives_size == 0 || space.vertices_size == 0)
    {
        return;
    }

    const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    auto vert_view = m_geometry_context->vertices.view();
    glm::vec3* d_positions = static_cast<glm::vec3*>(vert_view.m_data[0]);

    auto prim_view = m_geometry_context->primitives.view();
    uint8_t* d_prims = static_cast<uint8_t*>(prim_view.m_data[0]);

    CUdeviceptr d_vertex_buffer = reinterpret_cast<CUdeviceptr>(d_positions);
    triangle_input.triangleArray.vertexBuffers = &d_vertex_buffer;
    triangle_input.triangleArray.numVertices = space.vertices_start + space.vertices_size;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(glm::vec3);

    uint32_t num_triangles = space.primitives_size / Triangle::occupied_slots;
    CUdeviceptr d_index_buffer = reinterpret_cast<CUdeviceptr>(
        d_prims + space.primitives_start * sizeof(PrimitiveChunk16) + offsetof(Triangle, v0)
    );
    triangle_input.triangleArray.indexBuffer = d_index_buffer;
    triangle_input.triangleArray.numIndexTriplets = num_triangles;
    triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes = sizeof(PrimitiveChunk16);
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    // If space_id already has a GAS built, free it first
    auto existing_it = m_spaces.find(space_id);
    if (existing_it != m_spaces.end())
    {
        if (existing_it->second.gas_output_buffer)
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(existing_it->second.gas_output_buffer)));
        }
        m_spaces.erase(existing_it);
    }

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &accel_options, &triangle_input, 1, &gas_buffer_sizes));

    CUdeviceptr d_temp_buffer = 0;
    CUdeviceptr d_gas_output_buffer = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));

    OptixTraversableHandle gas_handle = 0;
    OPTIX_CHECK(optixAccelBuild(
        m_context,
        0,
        &accel_options,
        &triangle_input,
        1,
        d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_gas_output_buffer,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle,
        nullptr,
        0
    ));

    CUDA_CHECK(cudaStreamSynchronize(0));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));

    OptixSpaceData data{};
    data.gas_handle = gas_handle;
    data.gas_output_buffer = d_gas_output_buffer;
    data.gas_output_size = gas_buffer_sizes.outputSizeInBytes;
    data.primitives_start = space.primitives_start;
    m_spaces[space_id] = data;
}

void OptixAS::free_space(uint32_t space_id, Space space)
{
    auto it = m_spaces.find(space_id);
    if (it != m_spaces.end())
    {
        if (it->second.gas_output_buffer)
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(it->second.gas_output_buffer)));
        }
        m_spaces.erase(it);
    }
}

void OptixAS::render(uint32_t space_id, cudaSurfaceObject_t surface, const RenderParameters& p, cudaStream_t stream)
{
    if (!m_pipeline_initialized || m_spaces.empty() || p.width == 0 || p.height == 0)
        return;

    auto it = m_spaces.find(space_id);
    if (it == m_spaces.end())
    {
        if (space_id == UINT32_MAX)
            it = m_spaces.begin();
        else
            return;
    }

    Params params = {};
    params.render_params = p;
    params.handle = it->second.gas_handle;
    params.surface = surface;
    params.primitives_start = it->second.primitives_start;

    if (d_params == 0)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params)));
    }
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_params), &params, sizeof(Params), cudaMemcpyHostToDevice, stream));

    OPTIX_CHECK(optixLaunch(
        m_pipeline,
        stream,
        d_params,
        sizeof(Params),
        &m_sbt,
        p.width,
        p.height,
        1
    ));
}

void OptixAS::render(cudaSurfaceObject_t surface, const RenderParameters& p, cudaStream_t stream)
{
    render(UINT32_MAX, surface, p, stream);
}
