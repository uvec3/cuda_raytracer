#include "UniformGridAS.cuh"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#include "imgui.h"
#include "../../cuda_common.hpp"

namespace uniform_grid_detail
{
    __host__ __device__ inline PrimitiveType get_primitive_triangle_vertices(
        soa::soa_view<PrimitiveChunk16> primitives,
        soa::soa_view<Vertex> vertices,
        uint32_t prim_slot_id,
        glm::vec3& v0, glm::vec3& v1, glm::vec3& v2,
        glm::vec3& n0, glm::vec3& n1, glm::vec3& n2)
    {
        PrimitiveChunk16 chunk0 = primitives[prim_slot_id];
        PrimitiveType ptype = chunk0.type();
        if (ptype == PrimitiveType::Triangle)
        {
            Triangle tri = chunk0.as<Triangle>();
            v0 = glm::vec3(vertices[tri.v0].position);
            v1 = glm::vec3(vertices[tri.v1].position);
            v2 = glm::vec3(vertices[tri.v2].position);
            n0 = glm::vec3(vertices[tri.v0].normal);
            n1 = glm::vec3(vertices[tri.v1].normal);
            n2 = glm::vec3(vertices[tri.v2].normal);
        }
        else if (ptype == PrimitiveType::FullTriangle)
        {
            InlinedTriangle ftri = load_primitive<InlinedTriangle>(primitives, prim_slot_id, chunk0);
            v0 = ftri.v0;
            v1 = ftri.v1;
            v2 = ftri.v2;
            n0 = ftri.n0_compressed.unpack();
            n1 = ftri.n1_compressed.unpack();
            n2 = ftri.n2_compressed.unpack();
        }
        else if (ptype == PrimitiveType::NagataPatch)
        {
            NagataPatchPrimitive patch = load_primitive<NagataPatchPrimitive>(primitives, prim_slot_id, chunk0);
            v0 = glm::vec3(vertices[patch.v0].position);
            v1 = glm::vec3(vertices[patch.v1].position);
            v2 = glm::vec3(vertices[patch.v2].position);
            n0 = glm::vec3(vertices[patch.v0].normal);
            n1 = glm::vec3(vertices[patch.v1].normal);
            n2 = glm::vec3(vertices[patch.v2].normal);
        }
        return ptype;
    }

    __device__ __forceinline__ void get_primitive_cell_range(
        PrimitiveType type,
        const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
        const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2,
        glm::uvec3 dims,
        float cell_size,
        glm::ivec3& out_min,
        glm::ivec3& out_max)
    {
        glm::vec3 tri_min;
        glm::vec3 tri_max;
        if (type == PrimitiveType::NagataPatch)
        {
            NagataPatch::fromTriangle(v0, v1, v2, n0, n1, n2).computeBoundingBox(tri_min, tri_max);
        }
        else
        {
            tri_min = glm::min(glm::min(v0, v1), v2);
            tri_max = glm::max(glm::max(v0, v1), v2);
        }

        const float eps = 0.5f * cell_size * 1e-4f + 1e-6f;
        const glm::ivec3 max_cell = glm::ivec3(dims) - 1;
        out_min = glm::clamp(glm::ivec3(glm::floor((tri_min - eps) / cell_size)), glm::ivec3(0), max_cell);
        out_max = glm::clamp(glm::ivec3(glm::floor((tri_max + eps) / cell_size)), glm::ivec3(0), max_cell);
    }

    struct MaxVec3Functor
    {
        __host__ __device__ glm::vec3 operator()(const glm::vec3& a, const glm::vec3& b) const
        {
            return glm::max(a, b);
        }
    };

    inline uint32_t axis_cell_count(float extent, float cell_size, uint32_t max_res)
    {
        if (extent <= 0.0f)
            return 1u;
        const uint32_t count = static_cast<uint32_t>(std::ceil((extent / cell_size) - 1e-5f));
        return std::min(std::max(count, 1u), max_res);
    }
} // namespace uniform_grid_detail

__global__ void count_triangle_grid_cells_kernel(
    soa::soa_view<PrimitiveChunk16> primitives,
    soa::soa_view<Vertex> vertices,
    uint32_t primitives_start,
    uint32_t number_of_triangles,
    uint32_t occupied_slots,
    glm::uvec3 dims,
    float cell_size,
    uint32_t* tri_counts,
    auto find_triangle_cube_relation)
{
    uint32_t local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_idx >= number_of_triangles)
        return;

    uint32_t t_id = primitives_start + local_idx * occupied_slots;
    glm::vec3 v0, v1, v2, n0, n1, n2;
    PrimitiveType ptype = uniform_grid_detail::get_primitive_triangle_vertices(
        primitives, vertices, t_id, v0, v1, v2, n0, n1, n2);

    glm::ivec3 c_min, c_max;
    uniform_grid_detail::get_primitive_cell_range(ptype, v0, v1, v2, n0, n1, n2, dims, cell_size, c_min, c_max);

    const float half_size = 0.5f * cell_size;
    uint32_t count = 0;

    for (int iz = c_min.z; iz <= c_max.z; ++iz)
    {
        for (int iy = c_min.y; iy <= c_max.y; ++iy)
        {
            for (int ix = c_min.x; ix <= c_max.x; ++ix)
            {
                glm::vec3 cell_center = (glm::vec3(ix, iy, iz) + 0.5f) * cell_size;
                if (find_triangle_cube_relation(cell_center, half_size, v0, v1, v2, n0, n1, n2) != TriangleCubeRelation::Outside)
                {
                    ++count;
                }
            }
        }
    }

    tri_counts[local_idx] = count;
}

__global__ void emit_triangle_grid_cells_kernel(
    soa::soa_view<PrimitiveChunk16> primitives,
    soa::soa_view<Vertex> vertices,
    uint32_t primitives_start,
    uint32_t number_of_triangles,
    uint32_t occupied_slots,
    glm::uvec3 dims,
    float cell_size,
    const uint32_t* tri_counts,
    const uint32_t* tri_offsets,
    soa::soa_view<CellPrimPair> pairs,
    auto find_triangle_cube_relation)
{
    uint32_t local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_idx >= number_of_triangles || tri_counts[local_idx] == 0)
        return;

    uint32_t t_id = primitives_start + local_idx * occupied_slots;
    glm::vec3 v0, v1, v2, n0, n1, n2;
    PrimitiveType ptype = uniform_grid_detail::get_primitive_triangle_vertices(
        primitives, vertices, t_id, v0, v1, v2, n0, n1, n2);

    glm::ivec3 c_min, c_max;
    uniform_grid_detail::get_primitive_cell_range(ptype, v0, v1, v2, n0, n1, n2, dims, cell_size, c_min, c_max);

    const float half_size = 0.5f * cell_size;
    uint32_t write_pos = tri_offsets[local_idx];

    for (int iz = c_min.z; iz <= c_max.z; ++iz)
    {
        for (int iy = c_min.y; iy <= c_max.y; ++iy)
        {
            for (int ix = c_min.x; ix <= c_max.x; ++ix)
            {
                glm::vec3 cell_center = (glm::vec3(ix, iy, iz) + 0.5f) * cell_size;
                if (find_triangle_cube_relation(cell_center, half_size, v0, v1, v2, n0, n1, n2) != TriangleCubeRelation::Outside)
                {
                    uint32_t cell_id = uniform_grid_cell_index(glm::uvec3(ix, iy, iz), dims);
                    pairs[write_pos].cell_id = cell_id;
                    pairs[write_pos].prim_id = t_id;
                    ++write_pos;
                }
            }
        }
    }
}

__global__ void populate_uniform_grid_kernel(
    const uint32_t* sorted_cells,
    const uint32_t* sorted_prims,
    uint32_t total_pairs,
    uint32_t bc_start,
    uint32_t* local_as,
    soa::soa_view<BoxContent> local_bc)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_pairs)
        return;

    uint32_t cell = sorted_cells[i];
    uint32_t prim = sorted_prims[i];
    bool is_first = (i == 0) || (sorted_cells[i - 1] != cell);
    bool is_last = (i + 1 == total_pairs) || (sorted_cells[i + 1] != cell);

    local_bc[1 + i] = BoxContent(prim, is_last);
    if (is_first)
    {
        local_as[cell] = bc_start + 1 + i;
    }
}

UniformGridAS::UniformGridAS(GeometryContext* geometry_context)
    : m_geometry_context(geometry_context)
{
    // Reserve element 0 for empty cell sentinel (0 = empty)
    as.alloc(1);
    as[0] = 0;
    box_content.alloc(1);
    sync_to_device();
}

void UniformGridAS::config_ui()
{
    ImGui::DragInt("Resolution (longest axis)", &config.resolution, 1.0f, 1, 2048);
    if (config.resolution < 1)
        config.resolution = 1;

    ImGui::Checkbox("Swap y and z", &config.swapYZ);
    ImGui::Checkbox("Sort space", &config.sort_space);

    const char* tri_type_names[] = {"Triangle (16B Indexed)", "FullTriangle (64B Inline)", "NagataPatch (96B Inline Curved)"};
    int tri_type_idx = 0;
    if (config.triangle_type == PrimitiveType::FullTriangle)
        tri_type_idx = 1;
    else if (config.triangle_type == PrimitiveType::NagataPatch)
        tri_type_idx = 2;

    if (ImGui::Combo("Triangle storage type", &tri_type_idx, tri_type_names, 3))
    {
        if (tri_type_idx == 0) config.triangle_type = PrimitiveType::Triangle;
        else if (tri_type_idx == 1) config.triangle_type = PrimitiveType::FullTriangle;
        else if (tri_type_idx == 2) config.triangle_type = PrimitiveType::NagataPatch;
    }
}

void UniformGridAS::info_ui(uint32_t space_id, Space space)
{
    auto it = m_build_times_ms.find(space_id);
    if (it != m_build_times_ms.end())
    {
        ImGui::Text("  Build Time: %.2f ms", it->second);
    }

    if (space.root_box >= 0)
    {
        auto grid = grids[static_cast<size_t>(space.root_box)];
        ImGui::Text("  Grid Dims:  %u x %u x %u (cell %.4f)", grid.dims.x, grid.dims.y, grid.dims.z, grid.cell_size);
    }

    const size_t cells_bytes = static_cast<size_t>(space.as_size) * sizeof(uint32_t);
    const size_t content_bytes = static_cast<size_t>(space.box_content_size) * sizeof(BoxContent);
    const size_t grid_meta_bytes = (space.root_box >= 0) ? (sizeof(glm::uvec3) + sizeof(float) + sizeof(uint32_t)) : 0;
    const size_t as_bytes = cells_bytes + content_bytes + grid_meta_bytes;
    const size_t geom_bytes = space_geometry_bytes(space);
    const size_t total_bytes = as_bytes + geom_bytes;

    ImGui::Text("  Cells:      %u", (uint32_t)space.as_size);
    ImGui::Text("  Contents:   %u", (uint32_t)space.box_content_size);
    ImGui::Text("  Primitives: %u slots", (uint32_t)space.primitives_size);
    ImGui::Text("  Vertices:   %u", (uint32_t)space.vertices_size);
    ImGui::Text("  AS Memory:  %s", format_bytes(as_bytes).c_str());
    ImGui::Text("    Cells:    %s", format_bytes(cells_bytes).c_str());
    ImGui::Text("    Contents: %s", format_bytes(content_bytes).c_str());
    ImGui::Text("  Geo Memory: %s", format_bytes(geom_bytes).c_str());
    ImGui::Text("  Total Size: %s", format_bytes(total_bytes).c_str());
}

void UniformGridAS::free_space(uint32_t space_id, Space space)
{
    m_build_times_ms.erase(space_id);
    if (space.as_type != ASType::UniformGrid)
        return;

    if (space.root_box >= 0)
    {
        grids.free(static_cast<size_t>(space.root_box));
    }
    if (space.as_size > 0)
    {
        as.free(space.as_start);
    }
    if (space.box_content_size > 0)
    {
        box_content.free(space.box_content_start);
    }
    sync_to_device();
}

void UniformGridAS::insert_triangles_gpu(
    [[maybe_unused]] uint32_t space_id,
    Space space,
    glm::uvec3 dims,
    float cell_size,
    uint32_t primitives_start,
    uint32_t number_of_triangles,
    uint32_t occupied_slots,
    auto find_triangle_cube_relation)
{
    if (number_of_triangles == 0)
        return;

    constexpr uint32_t block_size = 256;
    const uint32_t tri_blocks = (number_of_triangles - 1) / block_size + 1;

    thrust::device_vector<uint32_t> d_tri_counts(number_of_triangles, 0);
    thrust::device_vector<uint32_t> d_tri_offsets(number_of_triangles, 0);

    count_triangle_grid_cells_kernel<<<tri_blocks, block_size>>>(
        m_geometry_context->primitives.view(),
        m_geometry_context->vertices.view(),
        primitives_start,
        number_of_triangles,
        occupied_slots,
        dims,
        cell_size,
        d_tri_counts.data().get(),
        find_triangle_cube_relation);
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::exclusive_scan(d_tri_counts.begin(), d_tri_counts.end(), d_tri_offsets.begin());

    uint32_t last_offset = 0;
    uint32_t last_count = 0;
    CUDA_CHECK(cudaMemcpy(&last_offset, d_tri_offsets.data().get() + number_of_triangles - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_count, d_tri_counts.data().get() + number_of_triangles - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    const uint32_t total_pairs = last_offset + last_count;

    const uint32_t total_cells = dims.x * dims.y * dims.z;

    const auto grid_id = static_cast<uint32_t>(grids.alloc_element());
    const auto as_start = static_cast<uint32_t>(as.alloc(total_cells));
    const auto bc_start = static_cast<uint32_t>(box_content.alloc(total_pairs + 1));

    grids[grid_id].dims = dims;
    grids[grid_id].cell_size = cell_size;
    grids[grid_id].as_start = as_start;

    space.root_box = static_cast<int>(grid_id);
    space.as_start = as_start;
    space.as_size = total_cells;
    space.box_content_start = bc_start;
    space.box_content_size = total_pairs + 1;

    soa::cuda_device_vector<uint32_t> d_local_as(total_cells);
    thrust::fill_n(thrust::device, &d_local_as[0], total_cells, 0u);

    soa::cuda_device_vector<BoxContent> d_local_bc(total_pairs + 1);
    {
        auto bc_begin = thrust::make_zip_iterator(d_local_bc.get_tuple());
        auto bc_end = thrust::make_zip_iterator(d_local_bc.get_tuple(total_pairs + 1));
        using BcZipValue = typename decltype(bc_begin)::value_type;
        thrust::fill(thrust::device, bc_begin, bc_end, BcZipValue{});
    }

    if (total_pairs > 0)
    {
        soa::cuda_device_vector<CellPrimPair> d_pairs(total_pairs);

        emit_triangle_grid_cells_kernel<<<tri_blocks, block_size>>>(
            m_geometry_context->primitives.view(),
            m_geometry_context->vertices.view(),
            primitives_start,
            number_of_triangles,
            occupied_slots,
            dims,
            cell_size,
            d_tri_counts.data().get(),
            d_tri_offsets.data().get(),
            d_pairs.view(),
            find_triangle_cube_relation);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto pair_tuple = d_pairs.get_tuple();
        uint32_t* d_cell_ids = cuda::std::get<0>(pair_tuple);
        uint32_t* d_prim_ids = cuda::std::get<1>(pair_tuple);
        thrust::sort_by_key(thrust::device, d_cell_ids, d_cell_ids + total_pairs, d_prim_ids);
        CUDA_CHECK(cudaDeviceSynchronize());

        const uint32_t pair_blocks = (total_pairs - 1) / block_size + 1;
        populate_uniform_grid_kernel<<<pair_blocks, block_size>>>(
            d_cell_ids,
            d_prim_ids,
            total_pairs,
            bc_start,
            &d_local_as[0],
            d_local_bc.view());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (d_as.size() < as.size()) d_as.resize(as.size());
    if (d_box_content.size() < box_content.size()) d_box_content.resize(box_content.size());

    d_grids = grids;
    d_as.range(as_start) = d_local_as;
    d_box_content.range(bc_start) = d_local_bc;

    as.range(as_start) = d_as.range(as_start, total_cells);
    box_content.range(bc_start) = d_box_content.range(bc_start, total_pairs + 1);
}

void UniformGridAS::insert_triangles(
    uint32_t space_id,
    Space space,
    glm::uvec3 dims,
    float cell_size,
    uint32_t primitives_start,
    uint32_t number_of_triangles,
    uint32_t occupied_slots)
{
    if (config.triangle_type == PrimitiveType::NagataPatch)
    {
        insert_triangles_gpu(
            space_id, space, dims, cell_size, primitives_start, number_of_triangles, occupied_slots,
            FindCubeIntersection<PrimitiveApproximation::Nagata>{});
    }
    else
    {
        insert_triangles_gpu(
            space_id, space, dims, cell_size, primitives_start, number_of_triangles, occupied_slots,
            FindCubeIntersection<PrimitiveApproximation::Precise>{});
    }
}

void UniformGridAS::build(uint32_t space_id, Space space)
{
    assert(m_geometry_context != nullptr);
    auto t0 = std::chrono::high_resolution_clock::now();

    uint32_t occupied_slots = Triangle::occupied_slots;
    if (config.triangle_type == PrimitiveType::FullTriangle)
        occupied_slots = InlinedTriangle::occupied_slots;
    else if (config.triangle_type == PrimitiveType::NagataPatch)
        occupied_slots = NagataPatchPrimitive::occupied_slots;

    const uint32_t number_of_triangles = space.primitives_size / occupied_slots;
    if (number_of_triangles == 0)
        return;

    const auto prim_view = m_geometry_context->primitives.view();
    const auto vert_view = m_geometry_context->vertices.view();
    const uint32_t prim_start = space.primitives_start;

    const glm::vec3 aabb_max = thrust::transform_reduce(
        thrust::device,
        thrust::make_counting_iterator<uint32_t>(0),
        thrust::make_counting_iterator<uint32_t>(number_of_triangles),
        [=] __device__(uint32_t idx) -> glm::vec3
        {
            const uint32_t t_id = prim_start + idx * occupied_slots;
            glm::vec3 v0, v1, v2, n0, n1, n2;
            uniform_grid_detail::get_primitive_triangle_vertices(prim_view, vert_view, t_id, v0, v1, v2, n0, n1, n2);
            return glm::max(glm::max(v0, v1), v2);
        },
        glm::vec3(0.0f),
        uniform_grid_detail::MaxVec3Functor{});

    const uint32_t res = static_cast<uint32_t>(std::max(config.resolution, 1));
    const float max_extent = std::max(std::max(aabb_max.x, aabb_max.y), std::max(aabb_max.z, 1e-4f));
    const float cell_size = max_extent / static_cast<float>(res);
    const glm::uvec3 dims{
        uniform_grid_detail::axis_cell_count(aabb_max.x, cell_size, res),
        uniform_grid_detail::axis_cell_count(aabb_max.y, cell_size, res),
        uniform_grid_detail::axis_cell_count(aabb_max.z, cell_size, res)};

    insert_triangles(space_id, space, dims, cell_size, space.primitives_start, number_of_triangles, occupied_slots);

    auto t1 = std::chrono::high_resolution_clock::now();
    m_build_times_ms[space_id] = std::chrono::duration<double, std::milli>(t1 - t0).count();
}
