#include "Scene.hpp"
#include "glm/glm.hpp"
#include "../cuda_common.hpp"
#include "../../external/tiny_gltf_v3.h"
#include "../../external/stb_image.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include "imgui.h"
#include "../device_algorithms.cuh"
#include "../types/NagataPatch.cuh"

#include <unordered_map>
#include <unordered_set>

glm::mat4 Node::getTransform() const
{
    if (m_dirty)
    {
        // recalculate the local transform matrix
        glm::mat4 matTranslate = glm::translate(glm::mat4(1), m_translation);
        glm::mat4 matRotate = glm::mat4_cast(m_rotation);
        glm::mat4 matScale = glm::scale(glm::mat4(1), m_scale);
        m_transform = matTranslate * matRotate * matScale;
        m_dirty = false;
    }
    return m_transform;
}

void Node::setTransform(glm::mat4 &transform)
{
    glm::vec3 skew;
    glm::vec4 perspective;
    glm::decompose(transform, m_scale, m_rotation, m_translation, skew, perspective);
    m_transform = transform;
    m_dirty = false;
}

void Node::get_mesh_transform_relative(glm::mat4 current_transform,
                                       std::vector<std::pair<glm::mat4, Mesh&>>& result,
                                       const std::vector<Node>& nodes,
                                       std::vector<Mesh>& meshes) const
{
    if (meshId > 0 && meshId - 1 < meshes.size())
    {
        result.push_back({current_transform, meshes[meshId - 1]});
    }

    for (uint32_t child_id : children)
    {
        if (child_id < nodes.size())
        {
            const auto& child = nodes[child_id];
            child.get_mesh_transform_relative(current_transform * child.getTransform(), result, nodes, meshes);
        }
    }
}

size_t Node::count_subtree_meshes(const std::vector<Node>& nodes) const
{
    size_t count = (meshId > 0) ? 1 : 0;
    for (uint32_t child_id : children)
    {
        if (child_id < nodes.size())
        {
            count += nodes[child_id].count_subtree_meshes(nodes);
        }
    }
    return count;
}

size_t Node::count_subtree_triangles(const std::vector<Node>& nodes, const std::vector<Mesh>& meshes) const
{
    size_t count = (meshId > 0 && meshId - 1 < meshes.size()) ? meshes[meshId - 1].d_triangles.size() : 0;
    for (uint32_t child_id : children)
    {
        if (child_id < nodes.size())
        {
            count += nodes[child_id].count_subtree_triangles(nodes, meshes);
        }
    }
    return count;
}

size_t Node::count_subtree_vertices(const std::vector<Node>& nodes, const std::vector<Mesh>& meshes) const
{
    size_t count = (meshId > 0 && meshId - 1 < meshes.size()) ? meshes[meshId - 1].d_vertices.size() : 0;
    for (uint32_t child_id : children)
    {
        if (child_id < nodes.size())
        {
            count += nodes[child_id].count_subtree_vertices(nodes, meshes);
        }
    }
    return count;
}

Scene::Scene(std::string_view name)
    : m_name(name)
    , as_octree(&geometry_context)
    , as_uniform_grid(&geometry_context)
    , as_kdtree(&geometry_context)
    , as_bvh(&geometry_context)
    , as_heterogeneous(&geometry_context, &as_octree, &as_uniform_grid, &as_kdtree, &as_bvh)
    , as_optix(&geometry_context)
{
    m_active_space = root_space;
    Node root{};
    root.name = m_name.empty() ? "Model Root" : m_name;
    m_nodes.push_back(std::move(root));
    root_node_id = 0;
}

Scene::~Scene()
{
    if (m_d_aabb_scratch != nullptr)
    {
        cudaFree(m_d_aabb_scratch);
        m_d_aabb_scratch = nullptr;
    }
    destroy_textures();
}

Scene::Scene(Scene&& other) noexcept
    : config(other.config)
    , m_meshes(std::move(other.m_meshes))
    , d_source_vertices(std::move(other.d_source_vertices))
    , d_source_triangles(std::move(other.d_source_triangles))
    , m_textures(std::move(other.m_textures))
    , m_materials(std::move(other.m_materials))
    , d_materials(std::move(other.d_materials))
    , m_nodes(std::move(other.m_nodes))
    , m_animations(std::move(other.m_animations))
    , root_node_id(other.root_node_id)
    , m_mesh_node_ids(std::move(other.m_mesh_node_ids))
    , default_scene(other.default_scene)
    , root_space(other.root_space)
    , spaces(std::move(other.spaces))
    , geometry_context(std::move(other.geometry_context))
    , as_octree(std::move(other.as_octree))
    , as_uniform_grid(std::move(other.as_uniform_grid))
    , as_kdtree(std::move(other.as_kdtree))
    , as_bvh(std::move(other.as_bvh))
    , as_heterogeneous(std::move(other.as_heterogeneous))
    , as_optix(std::move(other.as_optix))
    , m_is_loaded(other.m_is_loaded)
    , m_name(std::move(other.m_name))
    , m_active_space(other.m_active_space)
    , m_root_box(other.m_root_box)
    , m_d_aabb_scratch(other.m_d_aabb_scratch)
{
    as_octree.bind_geometry_context(&geometry_context);
    as_uniform_grid.bind_geometry_context(&geometry_context);
    as_kdtree.bind_geometry_context(&geometry_context);
    as_bvh.bind_geometry_context(&geometry_context);
    as_heterogeneous.bind_context(&geometry_context, &as_octree, &as_uniform_grid, &as_kdtree, &as_bvh);
    as_optix.bind_geometry_context(&geometry_context);
    other.m_textures.clear();
    other.m_is_loaded = false;
    other.m_d_aabb_scratch = nullptr;
}

Scene& Scene::operator=(Scene&& other) noexcept
{
    if (this != &other)
    {
        if (m_d_aabb_scratch != nullptr)
        {
            cudaFree(m_d_aabb_scratch);
            m_d_aabb_scratch = nullptr;
        }
        destroy_textures();
        config = other.config;
        m_meshes = std::move(other.m_meshes);
        d_source_vertices = std::move(other.d_source_vertices);
        d_source_triangles = std::move(other.d_source_triangles);
        m_textures = std::move(other.m_textures);
        m_materials = std::move(other.m_materials);
        d_materials = std::move(other.d_materials);
        m_nodes = std::move(other.m_nodes);
        m_animations = std::move(other.m_animations);
        root_node_id = other.root_node_id;
        m_mesh_node_ids = std::move(other.m_mesh_node_ids);
        default_scene = other.default_scene;
        root_space = other.root_space;
        spaces = std::move(other.spaces);
        geometry_context = std::move(other.geometry_context);
        as_octree = std::move(other.as_octree);
        as_uniform_grid = std::move(other.as_uniform_grid);
        as_kdtree = std::move(other.as_kdtree);
        as_bvh = std::move(other.as_bvh);
        as_heterogeneous = std::move(other.as_heterogeneous);
        as_optix = std::move(other.as_optix);
        m_is_loaded = other.m_is_loaded;
        m_name = std::move(other.m_name);
        m_active_space = other.m_active_space;
        m_root_box = other.m_root_box;
        m_d_aabb_scratch = other.m_d_aabb_scratch;

        as_octree.bind_geometry_context(&geometry_context);
        as_uniform_grid.bind_geometry_context(&geometry_context);
        as_kdtree.bind_geometry_context(&geometry_context);
        as_bvh.bind_geometry_context(&geometry_context);
        as_heterogeneous.bind_context(&geometry_context, &as_octree, &as_uniform_grid, &as_kdtree, &as_bvh);
        as_optix.bind_geometry_context(&geometry_context);
        other.m_textures.clear();
        other.m_is_loaded = false;
        other.m_d_aabb_scratch = nullptr;
    }
    return *this;
}

void Scene::config_ui()
{
    ImGui::Checkbox("Swap y and z", &config.swapYZ);
    ImGui::Checkbox("Sort space", &config.sort_space);
    const char* items[] = { "Triangle", "FullTriangle", "NagataPatch" };
    int current_item = 0;
    if (config.triangle_type == PrimitiveType::FullTriangle) current_item = 1;
    else if (config.triangle_type == PrimitiveType::NagataPatch) current_item = 2;
    if (ImGui::Combo("Triangle storage type", &current_item, items, IM_ARRAYSIZE(items)))
    {
        if (current_item == 0) config.triangle_type = PrimitiveType::Triangle;
        else if (current_item == 1) config.triangle_type = PrimitiveType::FullTriangle;
        else if (current_item == 2) config.triangle_type = PrimitiveType::NagataPatch;
    }
}

void Scene::destroy_textures()
{
    for (auto& tex : m_textures)
    {
        if (tex.texObj != 0)
        {
            cudaDestroyTextureObject(tex.texObj);
            tex.texObj = 0;
        }
        if (tex.cudaArray != nullptr)
        {
            cudaFreeArray(tex.cudaArray);
            tex.cudaArray = nullptr;
        }
    }
    m_textures.clear();
}

void Scene::sync_materials_to_device()
{
    geometry_context.materials = m_materials;
    d_materials = geometry_context.materials;
}

void Scene::load(std::string_view data, bool load_textures)
{
    tg3_parse_options opts;
    tg3_error_stack errors;
    tg3_model model;

    // 1. Initialize tracking structures
    tg3_parse_options_init(&opts);
    opts.borrow_input_buffers = 1;
    opts.skip_extras_values = 0;
    opts.images_as_is = 1;
    tg3_error_stack_init(&errors);

    // 2. Parse directly from the memory buffer
    // Automatically handles parsing the GLB header, JSON chunk, and binary chunk.
    tg3_error_code err = tg3_parse_glb(&model,&errors,reinterpret_cast<const uint8_t*>(data.data()),data.size(),nullptr,10,&opts);

    if (err != TG3_OK) {
        for (uint32_t i = 0; i < errors.count; i++) {
            fprintf(stderr, "Err [%d]: %s\n",
                    (int)errors.entries[i].severity,
                    errors.entries[i].message ? errors.entries[i].message : "Unknown");
        }
        tg3_error_stack_free(&errors);
        return;
    }

    std::vector<uint32_t> textureIds = load_textures ? loadTextures(model) : std::vector<uint32_t>{};
    std::vector<uint32_t> materialIds = loadMaterials(model, textureIds);
    std::vector<uint32_t> meshIds = loadMeshes(model, materialIds);

    std::vector<uint32_t> importedGltfNodes = load_nodes(model, meshIds);
    load_animations(model, importedGltfNodes);

    sync_materials_to_device();

    // 4. Clean up allocated arena memory
    tg3_model_free(&model);
    tg3_error_stack_free(&errors);

    m_is_loaded=true;
}

bool Scene::is_loaded() const
{
    return m_is_loaded;
}

const std::string& Scene::get_name() const
{
    return m_name;
}

void Scene::set_active_space(uint32_t space_id)
{
    TEST_ASSERT(spaces.is_allocated(space_id));
    m_active_space = space_id;
    m_root_box = spaces[m_active_space].root_box;
}

static void reset_space_fields(Space space)
{
    space.root_box = -1;
    space.as_type = ASType::None;
    space.vertices_start = 0;
    space.vertices_size = 0;
    space.as_start = 0;
    space.as_size = 0;
    space.box_content_start = 0;
    space.box_content_size = 0;
    space.primitives_start = 0;
    space.primitives_size = 0;
    space.normalization_translate = glm::vec3(0.0f);
    space.normalization_scale = glm::vec3(1.0f);
    space.has_nagata = false;
}

uint32_t Scene::create_space()
{
    auto space_id = static_cast<uint32_t>(spaces.alloc_element());
    reset_space_fields(spaces[space_id]);
    geometry_context.d_spaces = spaces;
    return space_id;
}

void Scene::release_space_resources(uint32_t space_id)
{
    if (!spaces.is_allocated(space_id))
        return;
    auto space = spaces[space_id];
    if (space.as_type == ASType::Octree)
    {
        as_octree.free_space(space_id, space);
    }
    else if (space.as_type == ASType::UniformGrid)
    {
        as_uniform_grid.free_space(space_id, space);
    }
    else if (space.as_type == ASType::KDTree)
    {
        as_kdtree.free_space(space_id, space);
    }
    else if (space.as_type == ASType::BVH)
    {
        as_bvh.free_space(space_id, space);
    }
    else if (space.as_type == ASType::Heterogeneous)
    {
        const auto owned_blas = as_heterogeneous.get_owned_blas_spaces(space_id);
        as_heterogeneous.free_space(space_id, space);
        for (uint32_t blas_id : owned_blas)
        {
            if (!as_heterogeneous.is_blas_space_owned(blas_id))
            {
                for (auto& node : m_nodes)
                {
                    if (node.as_ref.is_valid() && node.as_ref.space_id == blas_id)
                        node.as_ref = {};
                }
                free_space(blas_id);
            }
        }
    }
    else if (space.as_type == ASType::Optix)
    {
        as_optix.free_space(space_id, space);
    }
    if (space.vertices_size > 0)
    {
        geometry_context.vertices.free(space.vertices_start);
    }
    if (space.primitives_size > 0)
    {
        geometry_context.primitives.free(space.primitives_start);
    }
    reset_space_fields(spaces[space_id]);
    geometry_context.d_spaces = spaces;
}

void Scene::free_tlas_space_only(uint32_t space_id)
{
    if (!spaces.is_allocated(space_id) || spaces[space_id].as_type != ASType::Heterogeneous)
        return;

    auto space = spaces[space_id];
    as_heterogeneous.free_space(space_id, space);
    if (space.vertices_size > 0)
    {
        geometry_context.vertices.free(space.vertices_start);
    }
    if (space.primitives_size > 0)
    {
        geometry_context.primitives.free(space.primitives_start);
    }
    reset_space_fields(spaces[space_id]);
    spaces.free(space_id);
    soa::upload_soa_range_async(geometry_context.d_spaces, spaces, space_id, 1);

    if (m_active_space == space_id)
    {
        m_active_space = UINT32_MAX;
        m_root_box = -1;
    }
}

void Scene::free_space(uint32_t space_id)
{
    if (!spaces.is_allocated(space_id))
        return;
    release_space_resources(space_id);
    spaces.free(space_id);
    geometry_context.d_spaces = spaces;

    if (m_active_space == space_id)
    {
        m_active_space = UINT32_MAX;
        m_root_box = -1;
    }
}

__global__ void scene_normalize_positions_kernel(
    soa::soa_view<Vertex> vertices,
    uint32_t vertices_start,
    uint32_t vertices_size,
    glm::vec3 translate,
    glm::vec3 scale,
    float offset)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vertices_size) return;
    glm::vec3 p = vertices[vertices_start + i].position;
    vertices[vertices_start + i].position = (p + translate) * scale + offset;
}

void Scene::normalize_space(uint32_t space_id, glm::vec3 aabb_min, glm::vec3 aabb_max, bool stretch, bool margin)
{
    auto vertices_size = spaces[space_id].vertices_size;
    if (vertices_size == 0)
        return;

    auto aabb_size = aabb_max - aabb_min;

    const float scale_factor = margin ? 0.98f : 1.0f;
    const float offset = margin ? 0.01f : 0.0f;

    glm::vec3 normalization_translate = -aabb_min;
    glm::vec3 normalization_scale;
    if (stretch)
    {
        glm::vec3 safe_size = glm::max(aabb_size, glm::vec3(1e-7f));
        normalization_scale = glm::vec3(scale_factor) / safe_size;
    }
    else
    {
        float max_axis = glm::max(glm::max(aabb_size.x, aabb_size.y), aabb_size.z);
        normalization_scale = (max_axis > 0.0f) ? glm::vec3(scale_factor / max_axis) : glm::vec3(1.0f);
    }

    uint32_t blocks = (vertices_size + 255u) / 256u;
    scene_normalize_positions_kernel<<<blocks, 256>>>(
        geometry_context.vertices.view(),
        spaces[space_id].vertices_start,
        vertices_size,
        normalization_translate,
        normalization_scale,
        offset);
    CUDA_CHECK(cudaGetLastError());

    spaces[space_id].normalization_translate = normalization_translate;
    spaces[space_id].normalization_scale = normalization_scale;
}

void Scene::sort_space(uint32_t space_id, soa::cuda_device_vector<Triangle>& d_temp_triangles)
{
    auto triangles_size = static_cast<uint32_t>(d_temp_triangles.size());
    if (triangles_size == 0 || spaces[space_id].vertices_size == 0)
        return;

    auto triangles_range = d_temp_triangles.view();

    thrust::device_vector<uint64_t> morton_codes(triangles_size, thrust::no_init);
    calculate_morton_codes<<<(triangles_size - 1) / 128 + 1, 128>>>(geometry_context.vertices.view(), triangles_range, morton_codes.data().get(), triangles_size);
    auto t = d_temp_triangles.get_tuple();
    thrust::sort_by_key(thrust::device, morton_codes.begin(), morton_codes.end(), thrust::zip_iterator{t});

    morton_codes.resize(0);
    morton_codes.shrink_to_fit();

    auto vertices_start = spaces[space_id].vertices_start;
    auto vertices_size = spaces[space_id].vertices_size;

    auto map = sort_by_first_appearance_in_triangles(triangles_range,
        triangles_size, vertices_start, vertices_size,
        thrust::make_zip_iterator(geometry_context.vertices.get_tuple(vertices_start)));

    remap_triangle_vertices<<<(triangles_size - 1) / 256 + 1, 256>>>(triangles_range, map.data().get(), vertices_start, triangles_size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void scene_pack_triangles_kernel(
    soa::soa_view<Triangle> src_triangles,
    uint32_t triangles_count,
    soa::soa_view<PrimitiveChunk16> dst_primitives,
    uint32_t primitives_start)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= triangles_count) return;

    Triangle tri = src_triangles[i];
    tri.type = PrimitiveType::Triangle;
    store_primitive(dst_primitives, primitives_start + i, tri);
}

__global__ void scene_pack_full_triangles_kernel(
    soa::soa_view<Triangle> src_triangles,
    soa::soa_view<Vertex> vertices,
    uint32_t triangles_count,
    soa::soa_view<PrimitiveChunk16> dst_primitives,
    uint32_t primitives_start)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= triangles_count) return;

    Triangle tri = src_triangles[i];
    InlinedTriangle ftri{};
    ftri.type = PrimitiveType::FullTriangle;
    ftri.material_id = tri.material_id;

    glm::vec2 uv0 = vertices[tri.v0].uv;
    glm::vec2 uv1 = vertices[tri.v1].uv;
    glm::vec2 uv2 = vertices[tri.v2].uv;
    ftri.uv_0 = glm::packHalf2x16(glm::vec2(uv0.x, uv0.y));
    ftri.uv_1 = glm::packHalf2x16(glm::vec2(uv1.x, uv1.y));
    ftri.uv_2 = glm::packHalf2x16(glm::vec2(uv2.x, uv2.y));

    ftri.v0 = glm::vec3(vertices[tri.v0].position);
    ftri.n0_compressed.pack(glm::vec3(vertices[tri.v0].normal));

    ftri.v1 = glm::vec3(vertices[tri.v1].position);
    ftri.n1_compressed.pack(glm::vec3(vertices[tri.v1].normal));

    ftri.v2 = glm::vec3(vertices[tri.v2].position);
    ftri.n2_compressed.pack(glm::vec3(vertices[tri.v2].normal));

    uint32_t dst_slot = primitives_start + i * InlinedTriangle::occupied_slots;
    store_primitive(dst_primitives, dst_slot, ftri);
}

__global__ void scene_pack_nagata_patches_kernel(
    soa::soa_view<Triangle> src_triangles,
    soa::soa_view<Vertex> vertices,
    uint32_t triangles_count,
    soa::soa_view<PrimitiveChunk16> dst_primitives,
    uint32_t primitives_start)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= triangles_count) return;

    Triangle tri = src_triangles[i];
    glm::vec3 v0 = glm::vec3(vertices[tri.v0].position);
    glm::vec3 v1 = glm::vec3(vertices[tri.v1].position);
    glm::vec3 v2 = glm::vec3(vertices[tri.v2].position);
    glm::vec3 n0 = glm::vec3(vertices[tri.v0].normal);
    glm::vec3 n1 = glm::vec3(vertices[tri.v1].normal);
    glm::vec3 n2 = glm::vec3(vertices[tri.v2].normal);

    NagataPatch patch = NagataPatch::fromTriangle(v0, v1, v2, n0, n1, n2);

    NagataPatchPrimitive prim{};
    prim.type = PrimitiveType::NagataPatch;
    prim.material_id = tri.material_id;

    prim.v0 = tri.v0;
    prim.v1 = tri.v1;
    prim.v2 = tri.v2;

    prim.C12 = patch.C12;
    prim.C23 = patch.C23;
    prim.C31 = patch.C31;

    uint32_t dst_slot = primitives_start + i * NagataPatchPrimitive::occupied_slots;
    store_primitive(dst_primitives, dst_slot, prim);
}

void Scene::pack_primitives(uint32_t space_id, const soa::cuda_device_vector<Triangle>& d_temp_triangles, uint32_t triangles_count, PrimitiveType triangle_type)
{
    if (triangles_count == 0)
    {
        spaces[space_id].has_nagata = false;
        return;
    }

    spaces[space_id].has_nagata = (triangle_type == PrimitiveType::NagataPatch);

    if (triangle_type == PrimitiveType::FullTriangle)
    {
        uint32_t primitives_size = triangles_count * InlinedTriangle::occupied_slots;
        auto primitives_start = static_cast<uint32_t>(geometry_context.primitives.alloc(primitives_size));
        spaces[space_id].primitives_start = primitives_start;
        spaces[space_id].primitives_size = primitives_size;

        uint32_t blocks = (triangles_count - 1) / 256 + 1;
        scene_pack_full_triangles_kernel<<<blocks, 256>>>(
            const_cast<soa::cuda_device_vector<Triangle>&>(d_temp_triangles).view(),
            geometry_context.vertices.view(),
            triangles_count,
            geometry_context.primitives.view(),
            primitives_start);
        CUDA_CHECK(cudaDeviceSynchronize());

        if (spaces[space_id].vertices_size > 0)
        {
            geometry_context.vertices.free(spaces[space_id].vertices_start);
            spaces[space_id].vertices_start = 0;
            spaces[space_id].vertices_size = 0;
        }
    }
    else if (triangle_type == PrimitiveType::NagataPatch)
    {
        uint32_t primitives_size = triangles_count * NagataPatchPrimitive::occupied_slots;
        auto primitives_start = static_cast<uint32_t>(geometry_context.primitives.alloc(primitives_size));
        spaces[space_id].primitives_start = primitives_start;
        spaces[space_id].primitives_size = primitives_size;

        uint32_t blocks = (triangles_count - 1) / 256 + 1;
        scene_pack_nagata_patches_kernel<<<blocks, 256>>>(
            const_cast<soa::cuda_device_vector<Triangle>&>(d_temp_triangles).view(),
            geometry_context.vertices.view(),
            triangles_count,
            geometry_context.primitives.view(),
            primitives_start);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    else
    {
        uint32_t primitives_size = triangles_count * Triangle::occupied_slots;
        auto primitives_start = static_cast<uint32_t>(geometry_context.primitives.alloc(primitives_size));
        spaces[space_id].primitives_start = primitives_start;
        spaces[space_id].primitives_size = primitives_size;

        uint32_t blocks = (triangles_count - 1) / 256 + 1;
        scene_pack_triangles_kernel<<<blocks, 256>>>(
            const_cast<soa::cuda_device_vector<Triangle>&>(d_temp_triangles).view(),
            triangles_count,
            geometry_context.primitives.view(),
            primitives_start);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

int Scene::get_root_box() const
{
    return m_root_box;
}

uint32_t Scene::get_active_space() const
{
    return m_active_space;
}

struct SubMesh
{
    size_t vertexStart = 0;
    size_t vertexCount = 0;
    size_t indexStart = 0;
    size_t indexCount = 0;

    const tg3_str_int_pair* attribute_position=nullptr;
    const tg3_str_int_pair* attribute_normal=nullptr;
    const tg3_str_int_pair* attribute_color=nullptr;
    const tg3_str_int_pair* attribute_texcoord=nullptr;

    uint32_t materialId = 0;
};

namespace
{
    float readComponentAsFloat(const uint8_t* ptr, int32_t componentType, bool normalized)
    {
        switch (componentType)
        {
        case TG3_COMPONENT_TYPE_FLOAT:
            return *reinterpret_cast<const float*>(ptr);
        case TG3_COMPONENT_TYPE_DOUBLE:
            return static_cast<float>(*reinterpret_cast<const double*>(ptr));
        case TG3_COMPONENT_TYPE_BYTE:
        {
            const int8_t v = *reinterpret_cast<const int8_t*>(ptr);
            return normalized ? std::max(static_cast<float>(v) / 127.0f, -1.0f) : static_cast<float>(v);
        }
        case TG3_COMPONENT_TYPE_UNSIGNED_BYTE:
        {
            const uint8_t v = *reinterpret_cast<const uint8_t*>(ptr);
            return normalized ? static_cast<float>(v) / 255.0f : static_cast<float>(v);
        }
        case TG3_COMPONENT_TYPE_SHORT:
        {
            const int16_t v = *reinterpret_cast<const int16_t*>(ptr);
            return normalized ? std::max(static_cast<float>(v) / 32767.0f, -1.0f) : static_cast<float>(v);
        }
        case TG3_COMPONENT_TYPE_UNSIGNED_SHORT:
        {
            const uint16_t v = *reinterpret_cast<const uint16_t*>(ptr);
            return normalized ? static_cast<float>(v) / 65535.0f : static_cast<float>(v);
        }
        case TG3_COMPONENT_TYPE_INT:
        {
            const int32_t v = *reinterpret_cast<const int32_t*>(ptr);
            return normalized ? std::max(static_cast<float>(static_cast<double>(v) / 2147483647.0), -1.0f) : static_cast<float>(v);
        }
        case TG3_COMPONENT_TYPE_UNSIGNED_INT:
        {
            const uint32_t v = *reinterpret_cast<const uint32_t*>(ptr);
            return normalized ? static_cast<float>(static_cast<double>(v) / 4294967295.0) : static_cast<float>(v);
        }
        default:
            assert(false && "Unsupported TG3_COMPONENT_TYPE");
            return 0.0f;
        }
    }

    uint32_t readComponentAsUint32(const uint8_t* ptr, int32_t componentType)
    {
        switch (componentType)
        {
        case TG3_COMPONENT_TYPE_BYTE:
            return static_cast<uint32_t>(*reinterpret_cast<const int8_t*>(ptr));
        case TG3_COMPONENT_TYPE_UNSIGNED_BYTE:
            return static_cast<uint32_t>(*reinterpret_cast<const uint8_t*>(ptr));
        case TG3_COMPONENT_TYPE_SHORT:
            return static_cast<uint32_t>(*reinterpret_cast<const int16_t*>(ptr));
        case TG3_COMPONENT_TYPE_UNSIGNED_SHORT:
            return static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(ptr));
        case TG3_COMPONENT_TYPE_INT:
            return static_cast<uint32_t>(*reinterpret_cast<const int32_t*>(ptr));
        case TG3_COMPONENT_TYPE_UNSIGNED_INT:
            return *reinterpret_cast<const uint32_t*>(ptr);
        case TG3_COMPONENT_TYPE_FLOAT:
            return static_cast<uint32_t>(*reinterpret_cast<const float*>(ptr));
        case TG3_COMPONENT_TYPE_DOUBLE:
            return static_cast<uint32_t>(*reinterpret_cast<const double*>(ptr));
        default:
            assert(false && "Unsupported TG3_COMPONENT_TYPE");
            return 0;
        }
    }

    cudaTextureAddressMode mapWrapMode(int32_t wrap)
    {
        switch (wrap)
        {
        case TG3_TEXTURE_WRAP_CLAMP_TO_EDGE:
            return cudaAddressModeClamp;
        case TG3_TEXTURE_WRAP_MIRRORED_REPEAT:
            return cudaAddressModeMirror;
        case TG3_TEXTURE_WRAP_REPEAT:
        default:
            return cudaAddressModeWrap;
        }
    }

    cudaTextureFilterMode mapFilterMode(int32_t magFilter, int32_t minFilter)
    {
        if (magFilter == TG3_TEXTURE_FILTER_NEAREST ||
            minFilter == TG3_TEXTURE_FILTER_NEAREST ||
            minFilter == TG3_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST ||
            minFilter == TG3_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR)
        {
            return cudaFilterModePoint;
        }
        return cudaFilterModeLinear;
    }

    int b64DecodeChar(unsigned char c)
    {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    }

    std::vector<uint8_t> decodeDataUri(const char* uri, uint32_t uri_len)
    {
        if (!uri || uri_len < 5 || std::memcmp(uri, "data:", 5) != 0)
            return {};
        const char* p = uri + 5;
        const char* end = uri + uri_len;
        while (p < end && *p != ';') ++p;
        if (p >= end) return {};
        ++p;
        if (static_cast<size_t>(end - p) < 7 || std::memcmp(p, "base64,", 7) != 0)
            return {};
        p += 7;
        size_t input_len = static_cast<size_t>(end - p);
        while (input_len > 0 && p[input_len - 1] == '=') --input_len;

        std::vector<uint8_t> out;
        out.reserve((input_len * 3) / 4 + 1);
        uint32_t accum = 0;
        int bits = 0;
        for (size_t i = 0; i < input_len; ++i)
        {
            int val = b64DecodeChar(static_cast<unsigned char>(p[i]));
            if (val < 0) continue;
            accum = (accum << 6) | static_cast<uint32_t>(val);
            bits += 6;
            if (bits >= 8)
            {
                bits -= 8;
                out.push_back(static_cast<uint8_t>((accum >> bits) & 0xFFu));
            }
        }
        return out;
    }

    struct ResolvedTexture
    {
        uint32_t index = 0;
        cudaTextureObject_t obj = 0;
    };

    struct MaterialTransmissionData
    {
        float transmissionFactor = 0.0f;
        int32_t transmissionTextureIdx = -1;
    };

    struct MaterialVolumeData
    {
        float thicknessFactor = 0.0f;
        int32_t thicknessTextureIdx = -1;
        glm::vec3 attenuationColor = glm::vec3(1.0f);
        float attenuationDistance = std::numeric_limits<float>::infinity();
    };

    const tg3_value* find_extension_in_container(const tg3_extras_ext& ext_container, const char* ext_name)
    {
        for (uint32_t i = 0; i < ext_container.extensions_count; ++i)
        {
            const auto& ext = ext_container.extensions[i];
            if (tg3_str_equals_cstr(ext.name, ext_name))
                return &ext.value;
        }
        return nullptr;
    }

    const tg3_value* find_material_extension(const tg3_material* tg3mat, const char* ext_name)
    {
        if (!tg3mat)
            return nullptr;
        if (const tg3_value* ext_val = find_extension_in_container(tg3mat->ext, ext_name))
            return ext_val;
        return find_extension_in_container(tg3mat->pbr_metallic_roughness.ext, ext_name);
    }

    const tg3_value* find_object_member(const tg3_value& obj, const char* key)
    {
        if (obj.type != TG3_VALUE_OBJECT || !obj.object_data)
            return nullptr;
        for (uint32_t i = 0; i < obj.object_count; ++i)
        {
            const auto& kv = obj.object_data[i];
            if (tg3_str_equals_cstr(kv.key, key))
                return &kv.value;
        }
        return nullptr;
    }

    std::optional<float> read_json_float(const tg3_value& val)
    {
        if (val.type == TG3_VALUE_REAL)
        {
            const float f = static_cast<float>(val.real_val);
            return std::isfinite(f) ? std::optional<float>(f) : std::nullopt;
        }
        if (val.type == TG3_VALUE_INT)
        {
            return static_cast<float>(val.int_val);
        }
        return std::nullopt;
    }

    std::optional<float> read_ext_float(const tg3_value& obj, const char* key)
    {
        const tg3_value* member = find_object_member(obj, key);
        return member ? read_json_float(*member) : std::nullopt;
    }

    std::optional<glm::vec3> read_ext_vec3(const tg3_value& obj, const char* key)
    {
        const tg3_value* member = find_object_member(obj, key);
        if (!member || member->type != TG3_VALUE_ARRAY || !member->array_data || member->array_count != 3)
            return std::nullopt;

        glm::vec3 result(1.0f);
        for (uint32_t i = 0; i < 3; ++i)
        {
            const auto comp = read_json_float(member->array_data[i]);
            if (!comp)
                return std::nullopt;
            result[i] = *comp;
        }
        return result;
    }

    int32_t read_ext_texture_index(const tg3_value& obj, const char* key)
    {
        const tg3_value* tex_obj = find_object_member(obj, key);
        if (!tex_obj)
            return -1;
        const tg3_value* idx_val = find_object_member(*tex_obj, "index");
        if (!idx_val || idx_val->type != TG3_VALUE_INT)
            return -1;
        if (idx_val->int_val < 0 || idx_val->int_val > std::numeric_limits<int32_t>::max())
            return -1;
        return static_cast<int32_t>(idx_val->int_val);
    }

    float get_material_ior(const tg3_material* tg3mat, float metallicFactor)
    {
        // Default IOR fallback: 1.5 for dielectric (metallicFactor < 1.0), 1.0 for conductor/metal (metallicFactor == 1.0)
        const float fallback_ior = (metallicFactor < 1.0f) ? 1.5f : 1.0f;

        if (const tg3_value* ext_val = find_material_extension(tg3mat, "KHR_materials_ior"))
        {
            if (auto ior = read_ext_float(*ext_val, "ior"))
            {
                return (*ior > 0.0f) ? *ior : fallback_ior;
            }
            // Extension is present but "ior" property omitted -> glTF spec default is 1.5
            return 1.5f;
        }

        return fallback_ior;
    }

    float clamp_unit(float value)
    {
        return (value < 0.0f) ? 0.0f : ((value > 1.0f) ? 1.0f : value);
    }

    MaterialTransmissionData get_material_transmission(const tg3_material* tg3mat)
    {
        MaterialTransmissionData data{};
        const tg3_value* ext_val = find_material_extension(tg3mat, "KHR_materials_transmission");
        if (!ext_val)
            return data;

        if (auto factor = read_ext_float(*ext_val, "transmissionFactor"))
        {
            data.transmissionFactor = clamp_unit(*factor);
        }
        data.transmissionTextureIdx = read_ext_texture_index(*ext_val, "transmissionTexture");
        return data;
    }

    MaterialVolumeData get_material_volume(const tg3_material* tg3mat)
    {
        MaterialVolumeData data{};
        const tg3_value* ext_val = find_material_extension(tg3mat, "KHR_materials_volume");
        if (!ext_val)
            return data;

        if (auto thickness = read_ext_float(*ext_val, "thicknessFactor"))
        {
            data.thicknessFactor = (*thickness > 0.0f) ? *thickness : 0.0f;
        }
        data.thicknessTextureIdx = read_ext_texture_index(*ext_val, "thicknessTexture");
        if (auto color = read_ext_vec3(*ext_val, "attenuationColor"))
        {
            data.attenuationColor = glm::vec3(
                clamp_unit(color->x),
                clamp_unit(color->y),
                clamp_unit(color->z));
        }
        if (auto distance = read_ext_float(*ext_val, "attenuationDistance"))
        {
            if (*distance > 0.0f)
            {
                data.attenuationDistance = *distance;
            }
        }
        return data;
    }

    ResolvedTexture resolve_material_texture(
        int32_t gltfTexIdx,
        const std::vector<uint32_t>& textureIds,
        const std::vector<Texture>& textures)
    {
        ResolvedTexture resolved{};
        if (gltfTexIdx >= 0 && static_cast<uint32_t>(gltfTexIdx) < textureIds.size())
        {
            resolved.index = textureIds[gltfTexIdx];
            if (resolved.index > 0 && resolved.index - 1 < textures.size())
            {
                resolved.obj = textures[resolved.index - 1].texObj;
            }
        }
        return resolved;
    }
} // namespace

std::vector<uint32_t> Scene::loadTextures(const tg3_model& model)
{
    std::vector<uint32_t> textureIds(model.textures_count, 0);
    if (model.textures_count == 0)
        return textureIds;

    m_textures.reserve(m_textures.size() + model.textures_count);

    for (uint32_t i = 0; i < model.textures_count; ++i)
    {
        const tg3_texture* tg3tex = &model.textures[i];
        Texture tex;
        tex.name = (tg3tex->name.data != nullptr && tg3tex->name.len > 0)
            ? std::string(tg3tex->name.data, tg3tex->name.len)
            : ("Texture " + std::to_string(i));

        if (tg3tex->sampler >= 0 && static_cast<uint32_t>(tg3tex->sampler) < model.samplers_count)
        {
            const tg3_sampler* samp = &model.samplers[tg3tex->sampler];
            tex.minFilter = samp->min_filter;
            tex.magFilter = samp->mag_filter;
            tex.wrapS = samp->wrap_s;
            tex.wrapT = samp->wrap_t;
        }

        if (tg3tex->source >= 0 && static_cast<uint32_t>(tg3tex->source) < model.images_count)
        {
            const tg3_image* img = &model.images[tg3tex->source];
            if ((tex.name.empty() || tex.name.starts_with("Texture ")) && img->name.data != nullptr && img->name.len > 0)
            {
                tex.name = std::string(img->name.data, img->name.len);
            }

            const uint8_t* rawData = nullptr;
            size_t rawSize = 0;
            std::vector<uint8_t> decodedUriBuffer;

            if (img->image.data != nullptr && img->image.count > 0)
            {
                rawData = img->image.data;
                rawSize = static_cast<size_t>(img->image.count);
            }
            else if (img->buffer_view >= 0 && static_cast<uint32_t>(img->buffer_view) < model.buffer_views_count)
            {
                const tg3_buffer_view* bv = &model.buffer_views[img->buffer_view];
                if (bv->buffer >= 0 && static_cast<uint32_t>(bv->buffer) < model.buffers_count)
                {
                    const tg3_buffer* buf = &model.buffers[bv->buffer];
                    if (buf->data.data != nullptr && bv->byte_offset + bv->byte_length <= buf->data.count)
                    {
                        rawData = buf->data.data + bv->byte_offset;
                        rawSize = static_cast<size_t>(bv->byte_length);
                    }
                }
            }
            else if (img->uri.data != nullptr && img->uri.len > 0 && tg3_is_data_uri(img->uri.data, img->uri.len))
            {
                decodedUriBuffer = decodeDataUri(img->uri.data, img->uri.len);
                rawData = decodedUriBuffer.data();
                rawSize = decodedUriBuffer.size();
            }

            if (rawData != nullptr && rawSize > 0)
            {
                int w = 0, h = 0, comp = 0;
                stbi_uc* pixels = nullptr;
                bool freeStbi = false;

                if (img->as_is == 0 && img->width > 0 && img->height > 0 && img->component == 4 &&
                    rawSize == static_cast<size_t>(img->width) * static_cast<size_t>(img->height) * 4)
                {
                    w = img->width;
                    h = img->height;
                    comp = img->component;
                    pixels = const_cast<stbi_uc*>(rawData);
                }
                else
                {
                    pixels = stbi_load_from_memory(rawData, static_cast<int>(rawSize), &w, &h, &comp, STBI_rgb_alpha);
                    freeStbi = true;
                }

                if (pixels != nullptr && w > 0 && h > 0)
                {
                    tex.width = w;
                    tex.height = h;
                    tex.channels = comp;

                    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
                    CUDA_CHECK(cudaMallocArray(&tex.cudaArray, &channelDesc, static_cast<size_t>(w), static_cast<size_t>(h)));
                    CUDA_CHECK(cudaMemcpy2DToArray(
                        tex.cudaArray, 0, 0,
                        pixels, static_cast<size_t>(w) * sizeof(uchar4),
                        static_cast<size_t>(w) * sizeof(uchar4), static_cast<size_t>(h),
                        cudaMemcpyHostToDevice));

                    cudaResourceDesc resDesc{};
                    resDesc.resType = cudaResourceTypeArray;
                    resDesc.res.array.array = tex.cudaArray;

                    cudaTextureDesc texDesc{};
                    texDesc.addressMode[0] = mapWrapMode(tex.wrapS);
                    texDesc.addressMode[1] = mapWrapMode(tex.wrapT);
                    texDesc.filterMode = mapFilterMode(tex.magFilter, tex.minFilter);
                    texDesc.readMode = cudaReadModeNormalizedFloat;
                    texDesc.normalizedCoords = 1;

                    CUDA_CHECK(cudaCreateTextureObject(&tex.texObj, &resDesc, &texDesc, nullptr));
                }

                if (freeStbi && pixels != nullptr)
                {
                    stbi_image_free(pixels);
                }
            }
        }

        m_textures.push_back(std::move(tex));
        textureIds[i] = static_cast<uint32_t>(m_textures.size());
    }

    return textureIds;
}

std::vector<uint32_t> Scene::loadMeshes(const tg3_model& model, const std::vector<uint32_t>& materialIds)
{
    std::vector<uint32_t> meshIds(model.meshes_count);
    const size_t prev_mesh_count = m_meshes.size();
    const size_t prev_vert_count = d_source_vertices.size();
    const size_t prev_tri_count = d_source_triangles.size();
    m_meshes.reserve(prev_mesh_count + model.meshes_count);

    struct MeshSpan
    {
        size_t vert_offset = 0;
        size_t vert_count = 0;
        size_t tri_offset = 0;
        size_t tri_count = 0;
    };
    std::vector<MeshSpan> mesh_spans(model.meshes_count);

    soa::soa_vector<Vertex> host_vertices;
    soa::soa_vector<Triangle> host_triangles;

    // load all gltf mesh and primitive data into temporary host staging buffers
    for (uint32_t i = 0; i < model.meshes_count; ++i)
    {
        Mesh mesh;
        const tg3_mesh* tg3mesh = &model.meshes[i];
        mesh.name = tg3mesh->name.data != nullptr
            ? std::string(tg3mesh->name.data, tg3mesh->name.len)
            : "No Name";

        // lambda to perform attribute data copy
        auto writeAttribute = [&model]<typename T>(T* dst, const tg3_str_int_pair* attr, size_t start)
        {
            const tg3_accessor* accessor = &model.accessors[attr->value];
            const tg3_buffer_view* bufferView = &model.buffer_views[accessor->buffer_view];
            const tg3_buffer* buffer = &model.buffers[bufferView->buffer];
            const size_t bufferOffset = bufferView->byte_offset + accessor->byte_offset;
            const int32_t compSize = tg3_component_size(accessor->component_type);
            const int32_t numComps = tg3_num_components(accessor->type);
            assert(compSize > 0 && numComps > 0);
            const size_t stride = bufferView->byte_stride != 0
                ? bufferView->byte_stride
                : static_cast<size_t>(compSize * numComps);
            const bool normalized = accessor->normalized != 0;
            const uint8_t* basePtr = buffer->data.data + bufferOffset;

            if constexpr (std::is_same_v<T, glm::vec3>)
            {
                if (accessor->component_type == TG3_COMPONENT_TYPE_FLOAT &&
                    accessor->type == TG3_TYPE_VEC3 &&
                    stride == sizeof(glm::vec3))
                {
                    memcpy(&dst[start], basePtr, accessor->count * sizeof(glm::vec3));
                    return;
                }
            }

            for (uint64_t idx = 0; idx < accessor->count; ++idx)
            {
                const uint8_t* elemPtr = basePtr + idx * stride;
                float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                const int32_t count = std::min<int32_t>(numComps, 4);
                for (int32_t c = 0; c < count; ++c)
                {
                    values[c] = readComponentAsFloat(elemPtr + c * compSize, accessor->component_type, normalized);
                }

                if constexpr (std::is_same_v<T, glm::vec4>)
                {
                    dst[start + idx] = glm::vec4(values[0], values[1], values[2], values[3]);
                }
                else if constexpr (std::is_same_v<T, glm::vec3>)
                {
                    dst[start + idx] = glm::vec3(values[0], values[1], values[2]);
                }
                else if constexpr (std::is_same_v<T, glm::vec2>)
                {
                    dst[start + idx] = glm::vec2(values[0], values[1]);
                }
            }
        };


        //extract submeshes
        std::vector<SubMesh> subMeshes(tg3mesh->primitives_count);
        std::vector<uint32_t> indices;
        for (uint32_t s = 0; s < tg3mesh->primitives_count; ++s)
        {
            const tg3_primitive* primitive = &tg3mesh->primitives[s];
            if (primitive->mode != -1 && primitive->mode != TG3_MODE_TRIANGLES)
            {
                continue;
            }

            if (primitive->material >= 0 && static_cast<uint32_t>(primitive->material) < materialIds.size())
                subMeshes[s].materialId = materialIds[primitive->material];


            for (uint32_t a = 0; a < primitive->attributes_count; ++a)
            {
                const tg3_str_int_pair* attr = &primitive->attributes[a];
                if (tg3_str_equals_cstr(attr->key, "POSITION"))
                {
                    const tg3_accessor* accessor = &model.accessors[attr->value];
                    assert(accessor->type == TG3_TYPE_VEC3 && tg3_component_size(accessor->component_type) > 0);
                    subMeshes[s].vertexCount = accessor->count;
                    subMeshes[s].attribute_position = attr;
                }
                else if (tg3_str_equals_cstr(attr->key, "NORMAL"))
                {
                    const tg3_accessor* accessor = &model.accessors[attr->value];
                    assert(accessor->type == TG3_TYPE_VEC3 && tg3_component_size(accessor->component_type) > 0);
                    subMeshes[s].attribute_normal = attr;
                }
                else if (tg3_str_equals_cstr(attr->key, "COLOR_0"))
                {
                    const tg3_accessor* accessor = &model.accessors[attr->value];
                    assert(accessor->type == TG3_TYPE_VEC3 || accessor->type == TG3_TYPE_VEC4);
                    assert(tg3_component_size(accessor->component_type) > 0);
                    subMeshes[s].attribute_color = attr;
                }
                else if (tg3_str_equals_cstr(attr->key, "TEXCOORD_0"))
                {
                    const tg3_accessor* accessor = &model.accessors[attr->value];
                    assert(accessor->type == TG3_TYPE_VEC2 && tg3_component_size(accessor->component_type) > 0);
                    subMeshes[s].attribute_texcoord = attr;
                }
            }


            // copy index data
            if (primitive->indices != -1)
            {
                const tg3_accessor* accessor = &model.accessors[primitive->indices];
                const tg3_buffer_view* bufferView = &model.buffer_views[accessor->buffer_view];
                const tg3_buffer* buffer = &model.buffers[bufferView->buffer];
                auto start = indices.size();
                indices.resize(start + accessor->count);

                subMeshes[s].indexStart = start;
                subMeshes[s].indexCount = accessor->count;

                const int32_t compSize = tg3_component_size(accessor->component_type);
                assert(compSize > 0);
                const size_t stride = bufferView->byte_stride != 0
                    ? bufferView->byte_stride
                    : static_cast<size_t>(compSize);
                const uint8_t* basePtr = buffer->data.data + bufferView->byte_offset + accessor->byte_offset;

                if (accessor->component_type == TG3_COMPONENT_TYPE_UNSIGNED_INT && stride == sizeof(uint32_t))
                {
                    memcpy(&indices[start], basePtr, accessor->count * sizeof(uint32_t));
                }
                else
                {
                    for (uint64_t idx = 0; idx < accessor->count; ++idx)
                    {
                        indices[start + idx] = readComponentAsUint32(basePtr + idx * stride, accessor->component_type);
                    }
                }
            }
            else if (subMeshes[s].vertexCount > 0)
            {
                auto start = indices.size();
                indices.resize(start + subMeshes[s].vertexCount);

                subMeshes[s].indexStart = start;
                subMeshes[s].indexCount = subMeshes[s].vertexCount;

                for (size_t idx = 0; idx < subMeshes[s].vertexCount; ++idx)
                {
                    indices[start + idx] = static_cast<uint32_t>(idx);
                }
            }
        }

        size_t total_number_of_vertices = 0;
        for (const auto& m : subMeshes)
        {
            total_number_of_vertices += m.vertexCount;
        }

        const size_t mesh_vert_offset = host_vertices.size();
        const size_t mesh_tri_offset = host_triangles.size();
        if (total_number_of_vertices > 0)
        {
            host_vertices.reserve(mesh_vert_offset + total_number_of_vertices);
        }

        for (uint32_t s = 0; s < tg3mesh->primitives_count; ++s)
        {
            if (subMeshes[s].vertexCount == 0)
                continue;

            const size_t global_vert_start = host_vertices.allocate(subMeshes[s].vertexCount);
            subMeshes[s].vertexStart = global_vert_start - mesh_vert_offset;

            if (subMeshes[s].attribute_position)
            {
                writeAttribute(&host_vertices[0].position, subMeshes[s].attribute_position, global_vert_start);
            }
            else
            {
                std::fill_n(&host_vertices[0].position + global_vert_start, subMeshes[s].vertexCount, glm::vec3(0.0f));
            }

            if (subMeshes[s].attribute_normal)
            {
                writeAttribute(&host_vertices[0].normal, subMeshes[s].attribute_normal, global_vert_start);
            }
            else
            {
                std::fill_n(&host_vertices[0].normal + global_vert_start, subMeshes[s].vertexCount, glm::vec3(0.0f));
            }

            if (subMeshes[s].attribute_color)
            {
                writeAttribute(&host_vertices[0].color, subMeshes[s].attribute_color, global_vert_start);
            }
            else
            {
                std::fill_n(&host_vertices[0].color + global_vert_start, subMeshes[s].vertexCount, glm::vec4(0.0f));
            }

            if (subMeshes[s].attribute_texcoord)
            {
                writeAttribute(&host_vertices[0].uv, subMeshes[s].attribute_texcoord, global_vert_start);
            }
            else
            {
                std::fill_n(&host_vertices[0].uv + global_vert_start, subMeshes[s].vertexCount, glm::vec2(0.0f));
            }
        }

        const size_t mesh_vert_count = host_vertices.size() - mesh_vert_offset;
        if (mesh_vert_count > 0)
        {
            glm::vec3 bmin(std::numeric_limits<float>::infinity());
            glm::vec3 bmax(-std::numeric_limits<float>::infinity());
            const glm::vec3* pos_ptr = &host_vertices[0].position + mesh_vert_offset;
            for (size_t v = 0; v < mesh_vert_count; ++v)
            {
                bmin = glm::min(bmin, pos_ptr[v]);
                bmax = glm::max(bmax, pos_ptr[v]);
            }
            mesh.local_bounds_min = bmin;
            mesh.local_bounds_max = bmax;
        }
        else
        {
            mesh.local_bounds_min = glm::vec3(0.0f);
            mesh.local_bounds_max = glm::vec3(0.0f);
        }

        if (indices.size() >= 3)
        {
            host_triangles.reserve(mesh_tri_offset + indices.size() / 3);
        }
        for (const auto& submesh : subMeshes)
        {
            auto number_of_triangles = submesh.indexCount / 3;
            if (number_of_triangles == 0)
                continue;

            auto start = host_triangles.allocate(number_of_triangles);
            for (size_t t = 0; t < number_of_triangles; ++t)
            {
                host_triangles[start + t].type = PrimitiveType::Triangle;
                host_triangles[start + t].v0 = static_cast<uint32_t>(submesh.vertexStart + indices[submesh.indexStart + t * 3]);
                host_triangles[start + t].v1 = static_cast<uint32_t>(submesh.vertexStart + indices[submesh.indexStart + t * 3 + 1]);
                host_triangles[start + t].v2 = static_cast<uint32_t>(submesh.vertexStart + indices[submesh.indexStart + t * 3 + 2]);
                host_triangles[start + t].material_id = static_cast<uint16_t>(submesh.materialId);
            }
        }

        mesh_spans[i] = MeshSpan{
            mesh_vert_offset,
            mesh_vert_count,
            mesh_tri_offset,
            host_triangles.size() - mesh_tri_offset
        };

        m_meshes.push_back(std::move(mesh));
        meshIds[i] = static_cast<uint32_t>(m_meshes.size());
    }

    // Upload all source vertices and triangles once to device storage and assign per-mesh device views
    if (prev_vert_count == 0)
    {
        if (!host_vertices.empty())
        {
            d_source_vertices = host_vertices;
        }
    }
    else if (!host_vertices.empty())
    {
        d_source_vertices.resize(prev_vert_count + host_vertices.size());
        d_source_vertices.range(prev_vert_count, host_vertices.size()) = host_vertices;
    }

    if (prev_tri_count == 0)
    {
        if (!host_triangles.empty())
        {
            d_source_triangles = host_triangles;
        }
    }
    else if (!host_triangles.empty())
    {
        d_source_triangles.resize(prev_tri_count + host_triangles.size());
        d_source_triangles.range(prev_tri_count, host_triangles.size()) = host_triangles;
    }

    // Refresh existing mesh views if appending reallocated d_source_*
    if (prev_mesh_count > 0 && (!host_vertices.empty() || !host_triangles.empty()))
    {
        size_t v_off = 0;
        size_t t_off = 0;
        for (size_t k = 0; k < prev_mesh_count; ++k)
        {
            const size_t v_cnt = m_meshes[k].d_vertices.size();
            const size_t t_cnt = m_meshes[k].d_triangles.size();
            m_meshes[k].d_vertices = (v_cnt > 0) ? d_source_vertices.view(v_off, v_cnt) : soa::soa_view<Vertex>{};
            m_meshes[k].d_triangles = (t_cnt > 0) ? d_source_triangles.view(t_off, t_cnt) : soa::soa_view<Triangle>{};
            v_off += v_cnt;
            t_off += t_cnt;
        }
    }

    for (uint32_t i = 0; i < model.meshes_count; ++i)
    {
        const auto& span = mesh_spans[i];
        Mesh& dst_mesh = m_meshes[prev_mesh_count + i];
        dst_mesh.d_vertices = (span.vert_count > 0)
            ? d_source_vertices.view(prev_vert_count + span.vert_offset, span.vert_count)
            : soa::soa_view<Vertex>{};
        dst_mesh.d_triangles = (span.tri_count > 0)
            ? d_source_triangles.view(prev_tri_count + span.tri_offset, span.tri_count)
            : soa::soa_view<Triangle>{};
    }

    return meshIds;
}

std::vector<uint32_t> Scene::loadMaterials(const tg3_model& model, const std::vector<uint32_t>& textureIds)
{
    std::vector<uint32_t> materialIds(model.materials_count);
    m_materials.reserve(m_materials.size() + model.materials_count + 1);

    // Ensure slot 0 is a default fallback material (when primitive->material == -1)
    if (m_materials.empty())
    {
        auto idx0 = m_materials.allocate(1);
        m_materials[idx0].baseColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
        m_materials[idx0].metallicFactor = 1.0f;
        m_materials[idx0].roughnessFactor = 1.0f;
        m_materials[idx0].emissiveFactor = glm::vec3(0.0f, 0.0f, 0.0f);
        m_materials[idx0].baseColorTextureIndex = 0;
        m_materials[idx0].normalTextureIndex = 0;
        m_materials[idx0].transmissionTextureIndex = 0;
        m_materials[idx0].thicknessTextureIndex = 0;
        m_materials[idx0].baseColorTexture = 0;
        m_materials[idx0].normalTexture = 0;
        m_materials[idx0].transmissionTexture = 0;
        m_materials[idx0].thicknessTexture = 0;
        m_materials[idx0].ior = 1;
        m_materials[idx0].transmissionFactor = 0.0f;
        m_materials[idx0].thicknessFactor = 0.0f;
        m_materials[idx0].attenuationColor = glm::vec3(1.0f);
        m_materials[idx0].attenuationDistance = std::numeric_limits<float>::infinity();
    }

    for (uint32_t i = 0; i < model.materials_count; ++i)
    {
        const tg3_material *tg3mat = &model.materials[i];
        auto matIdx = m_materials.allocate(1);
        auto&& mat = m_materials[matIdx];

        mat.baseColor = glm::vec4(
            static_cast<float>(tg3mat->pbr_metallic_roughness.base_color_factor[0]),
            static_cast<float>(tg3mat->pbr_metallic_roughness.base_color_factor[1]),
            static_cast<float>(tg3mat->pbr_metallic_roughness.base_color_factor[2]),
            static_cast<float>(tg3mat->pbr_metallic_roughness.base_color_factor[3]));
        mat.metallicFactor = static_cast<float>(tg3mat->pbr_metallic_roughness.metallic_factor);
        mat.roughnessFactor = static_cast<float>(tg3mat->pbr_metallic_roughness.roughness_factor);
        mat.emissiveFactor = glm::vec3(
            static_cast<float>(tg3mat->emissive_factor[0]),
            static_cast<float>(tg3mat->emissive_factor[1]),
            static_cast<float>(tg3mat->emissive_factor[2]));

        const auto baseTex = resolve_material_texture(
            tg3mat->pbr_metallic_roughness.base_color_texture.index, textureIds, m_textures);
        const auto normTex = resolve_material_texture(
            tg3mat->normal_texture.index, textureIds, m_textures);
        const auto transmission = get_material_transmission(tg3mat);
        const auto volume = get_material_volume(tg3mat);
        const auto transTex = resolve_material_texture(
            transmission.transmissionTextureIdx, textureIds, m_textures);
        const auto thickTex = resolve_material_texture(
            volume.thicknessTextureIdx, textureIds, m_textures);

        mat.baseColorTextureIndex = baseTex.index;
        mat.normalTextureIndex = normTex.index;
        mat.transmissionTextureIndex = transTex.index;
        mat.thicknessTextureIndex = thickTex.index;
        mat.baseColorTexture = baseTex.obj;
        mat.normalTexture = normTex.obj;
        mat.transmissionTexture = transTex.obj;
        mat.thicknessTexture = thickTex.obj;
        mat.ior = get_material_ior(tg3mat, mat.metallicFactor);
        mat.transmissionFactor = transmission.transmissionFactor;
        mat.thicknessFactor = volume.thicknessFactor;
        mat.attenuationColor = volume.attenuationColor;
        mat.attenuationDistance = volume.attenuationDistance;

        materialIds[i] = static_cast<uint32_t>(matIdx);
    }
    return materialIds;
}


uint32_t Scene::importNode(const tg3_model& model, uint32_t nodeIndex, const std::vector<uint32_t>& meshIds, std::vector<uint32_t>& importedGltfNodes)
{
    if (nodeIndex < importedGltfNodes.size() && importedGltfNodes[nodeIndex] != UINT32_MAX)
    {
        return importedGltfNodes[nodeIndex];
    }

    const tg3_node& tg3Node = model.nodes[nodeIndex];
    const uint32_t my_id = static_cast<uint32_t>(m_nodes.size());
    m_nodes.emplace_back();
    if (nodeIndex < importedGltfNodes.size())
    {
        importedGltfNodes[nodeIndex] = my_id;
    }

    {
        Node& node = m_nodes[my_id];
        node.name = (tg3Node.name.data != nullptr && tg3Node.name.len > 0)
            ? std::string(tg3Node.name.data, tg3Node.name.len)
            : ("Node " + std::to_string(nodeIndex));

        if (tg3Node.has_matrix)
        {
            glm::mat4 transform(1);
            float* transformPtr = &transform[0][0];
            for (int i = 0; i < 16; ++i)
            {
                transformPtr[i] = static_cast<float>(tg3Node.matrix[i]);
            }
            node.setTransform(transform);
        }
        else
        {
            glm::vec3 translation(tg3Node.translation[0], tg3Node.translation[1], tg3Node.translation[2]);
            glm::quat rotation(tg3Node.rotation[3], tg3Node.rotation[0], tg3Node.rotation[1], tg3Node.rotation[2]);
            glm::vec3 scale(tg3Node.scale[0], tg3Node.scale[1], tg3Node.scale[2]);

            node.setTranslation(translation);
            node.setRotation(rotation);
            node.setScale(scale);
        }
    }

    std::vector<uint32_t> child_ids;
    child_ids.reserve(tg3Node.children_count + (tg3Node.mesh != -1 ? 1u : 0u));

    if (tg3Node.mesh != -1)
    {
        const uint32_t mesh_id_1based = meshIds[tg3Node.mesh];
        if (mesh_id_1based > 0 && (mesh_id_1based - 1) < m_mesh_node_ids.size())
        {
            child_ids.push_back(m_mesh_node_ids[mesh_id_1based - 1]);
        }
    }

    for (uint32_t i = 0; i < tg3Node.children_count; ++i)
    {
        uint32_t childIndex = static_cast<uint32_t>(tg3Node.children[i]);
        child_ids.push_back(importNode(model, childIndex, meshIds, importedGltfNodes));
    }

    m_nodes[my_id].children = std::move(child_ids);
    return my_id;
}

std::vector<uint32_t> Scene::load_nodes(const tg3_model& model, const std::vector<uint32_t>& meshIds)
{
    default_scene = (model.default_scene >= 0 && static_cast<uint32_t>(model.default_scene) < model.scenes_count)
        ? static_cast<uint32_t>(model.default_scene)
        : 0;

    m_nodes.clear();
    m_mesh_node_ids.clear();

    root_node_id = static_cast<uint32_t>(m_nodes.size());
    {
        Node root{};
        root.name = m_name.empty() ? "Model Root" : m_name;
        m_nodes.push_back(std::move(root));
    }

    m_mesh_node_ids.reserve(m_meshes.size());
    for (size_t i = 0; i < m_meshes.size(); ++i)
    {
        const uint32_t mesh_node_id = static_cast<uint32_t>(m_nodes.size());
        Node mesh_node{};
        mesh_node.name = m_meshes[i].name.empty()
            ? ("Mesh " + std::to_string(i))
            : m_meshes[i].name;
        mesh_node.meshId = static_cast<uint32_t>(i + 1);
        m_nodes.push_back(std::move(mesh_node));
        m_mesh_node_ids.push_back(mesh_node_id);
    }

    std::vector<uint32_t> importedGltfNodes(model.nodes_count, UINT32_MAX);
    std::vector<uint32_t> subscene_ids;
    subscene_ids.reserve(model.scenes_count);

    for (uint32_t scene_id = 0; scene_id < model.scenes_count; ++scene_id)
    {
        const tg3_scene* scene = &model.scenes[scene_id];
        const uint32_t subscene_id = static_cast<uint32_t>(m_nodes.size());
        {
            Node subscene_node{};
            subscene_node.name = (scene->name.data != nullptr && scene->name.len > 0)
                ? std::string(scene->name.data, scene->name.len)
                : ("Subscene " + std::to_string(scene_id));
            m_nodes.push_back(std::move(subscene_node));
        }

        std::vector<uint32_t> scene_child_ids;
        scene_child_ids.reserve(scene->nodes_count);
        for (uint32_t i = 0; i < scene->nodes_count; ++i)
        {
            scene_child_ids.push_back(importNode(model, static_cast<uint32_t>(scene->nodes[i]), meshIds, importedGltfNodes));
        }
        m_nodes[subscene_id].children = std::move(scene_child_ids);
        subscene_ids.push_back(subscene_id);
    }

    m_nodes[root_node_id].children = std::move(subscene_ids);
    return importedGltfNodes;
}

namespace
{
    struct AccessorView
    {
        const uint8_t* base_ptr = nullptr;
        uint64_t count = 0;
        size_t stride = 0;
        int32_t comp_size = 0;
        int32_t num_comps = 0;
        int32_t component_type = 0;
        int32_t type = 0;
        bool normalized = false;
    };

    bool resolve_accessor_view(const tg3_model& model, int32_t accessor_idx, AccessorView& out)
    {
        if (accessor_idx < 0 || static_cast<uint32_t>(accessor_idx) >= model.accessors_count || !model.accessors)
            return false;

        const tg3_accessor* accessor = &model.accessors[accessor_idx];
        if (accessor->buffer_view < 0 ||
            static_cast<uint32_t>(accessor->buffer_view) >= model.buffer_views_count ||
            !model.buffer_views)
            return false;

        const tg3_buffer_view* bv = &model.buffer_views[accessor->buffer_view];
        if (bv->buffer < 0 ||
            static_cast<uint32_t>(bv->buffer) >= model.buffers_count ||
            !model.buffers)
            return false;

        const tg3_buffer* buf = &model.buffers[bv->buffer];
        const int32_t comp_size = tg3_component_size(accessor->component_type);
        const int32_t num_comps = tg3_num_components(accessor->type);
        if (comp_size <= 0 || num_comps <= 0)
            return false;

        const size_t elem_size = static_cast<size_t>(comp_size) * static_cast<size_t>(num_comps);
        const size_t stride = (bv->byte_stride != 0) ? static_cast<size_t>(bv->byte_stride) : elem_size;
        if (stride < elem_size)
            return false;

        if (accessor->count > 0)
        {
            if (buf->data.data == nullptr)
                return false;

            const uint64_t last_elem_offset = (accessor->count - 1u) * static_cast<uint64_t>(stride);
            if (stride != 0 && last_elem_offset / static_cast<uint64_t>(stride) != (accessor->count - 1u))
                return false;

            const uint64_t span = last_elem_offset + static_cast<uint64_t>(elem_size);
            if (span < last_elem_offset)
                return false;

            const uint64_t end_in_bv = accessor->byte_offset + span;
            if (end_in_bv < accessor->byte_offset || end_in_bv > bv->byte_length)
                return false;

            const uint64_t buffer_offset = bv->byte_offset + accessor->byte_offset;
            if (buffer_offset < bv->byte_offset)
                return false;

            const uint64_t end_in_buf = buffer_offset + span;
            if (end_in_buf < buffer_offset || end_in_buf > buf->data.count)
                return false;

            out.base_ptr = buf->data.data + buffer_offset;
        }

        out.count = accessor->count;
        out.stride = stride;
        out.comp_size = comp_size;
        out.num_comps = num_comps;
        out.component_type = accessor->component_type;
        out.type = accessor->type;
        out.normalized = (accessor->normalized != 0);
        return true;
    }

    bool readAccessorScalars(const tg3_model& model, int32_t accessor_idx, std::vector<float>& out)
    {
        AccessorView view{};
        if (!resolve_accessor_view(model, accessor_idx, view) || view.type != TG3_TYPE_SCALAR)
            return false;

        out.resize(static_cast<size_t>(view.count));
        if (view.count == 0)
            return true;

        if (view.component_type == TG3_COMPONENT_TYPE_FLOAT &&
            view.stride == sizeof(float) &&
            !view.normalized)
        {
            std::memcpy(out.data(), view.base_ptr, static_cast<size_t>(view.count) * sizeof(float));
            return true;
        }

        for (uint64_t i = 0; i < view.count; ++i)
        {
            out[static_cast<size_t>(i)] = readComponentAsFloat(
                view.base_ptr + i * view.stride,
                view.component_type,
                view.normalized);
        }
        return true;
    }

    bool readAccessorVec4s(const tg3_model& model, int32_t accessor_idx, std::vector<glm::vec4>& out, int32_t& out_type)
    {
        AccessorView view{};
        if (!resolve_accessor_view(model, accessor_idx, view))
            return false;
        if (view.type != TG3_TYPE_VEC3 && view.type != TG3_TYPE_VEC4)
            return false;

        out_type = view.type;
        out.resize(static_cast<size_t>(view.count));
        if (view.count == 0)
            return true;

        if (sizeof(glm::vec4) == 4 * sizeof(float) &&
            view.component_type == TG3_COMPONENT_TYPE_FLOAT &&
            view.type == TG3_TYPE_VEC4 &&
            view.stride == sizeof(glm::vec4) &&
            !view.normalized)
        {
            std::memcpy(out.data(), view.base_ptr, static_cast<size_t>(view.count) * sizeof(glm::vec4));
            return true;
        }

        for (uint64_t i = 0; i < view.count; ++i)
        {
            const uint8_t* elem_ptr = view.base_ptr + i * view.stride;
            float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int32_t c = 0; c < view.num_comps; ++c)
            {
                values[c] = readComponentAsFloat(
                    elem_ptr + static_cast<size_t>(c) * static_cast<size_t>(view.comp_size),
                    view.component_type,
                    view.normalized);
            }
            out[static_cast<size_t>(i)] = glm::vec4(values[0], values[1], values[2], values[3]);
        }
        return true;
    }
} // namespace

void Scene::load_animations(const tg3_model& model, const std::vector<uint32_t>& importedGltfNodes)
{
    m_animations.clear();
    if (model.animations_count == 0 || model.animations == nullptr)
        return;

    m_animations.reserve(model.animations_count);

    for (uint32_t i = 0; i < model.animations_count; ++i)
    {
        const tg3_animation& tg3Anim = model.animations[i];
        Animation anim;
        anim.name = (tg3Anim.name.data != nullptr && tg3Anim.name.len > 0)
            ? std::string(tg3Anim.name.data, tg3Anim.name.len)
            : ("Animation " + std::to_string(i));

        anim.samplers.resize(tg3Anim.samplers_count);
        std::vector<bool> sampler_valid(tg3Anim.samplers_count, false);
        std::vector<int32_t> sampler_output_type(tg3Anim.samplers_count, 0);

        for (uint32_t s = 0; s < tg3Anim.samplers_count; ++s)
        {
            const tg3_animation_sampler& tg3Sampler = tg3Anim.samplers[s];
            AnimationSampler& sampler = anim.samplers[s];

            if (tg3_str_equals_cstr(tg3Sampler.interpolation, "STEP"))
            {
                sampler.interpolation = AnimationInterpolation::Step;
            }
            else if (tg3_str_equals_cstr(tg3Sampler.interpolation, "CUBICSPLINE"))
            {
                sampler.interpolation = AnimationInterpolation::CubicSpline;
            }
            else
            {
                sampler.interpolation = AnimationInterpolation::Linear;
            }

            std::vector<float> inputs;
            std::vector<glm::vec4> outputs;
            int32_t output_type = 0;
            if (!readAccessorScalars(model, tg3Sampler.input, inputs) ||
                !readAccessorVec4s(model, tg3Sampler.output, outputs, output_type))
            {
                continue;
            }

            if (inputs.empty())
                continue;

            const size_t expected_outputs = (sampler.interpolation == AnimationInterpolation::CubicSpline)
                ? (inputs.size() * 3u)
                : inputs.size();
            if (outputs.size() != expected_outputs)
                continue;

            sampler.inputs = std::move(inputs);
            sampler.outputs = std::move(outputs);
            sampler_valid[s] = true;
            sampler_output_type[s] = output_type;
        }

        anim.channels.reserve(tg3Anim.channels_count);
        for (uint32_t c = 0; c < tg3Anim.channels_count; ++c)
        {
            const tg3_animation_channel& tg3Channel = tg3Anim.channels[c];
            if (tg3Channel.sampler < 0 ||
                static_cast<uint32_t>(tg3Channel.sampler) >= tg3Anim.samplers_count ||
                !sampler_valid[tg3Channel.sampler])
            {
                continue;
            }

            if (tg3Channel.target.node < 0 ||
                static_cast<uint32_t>(tg3Channel.target.node) >= importedGltfNodes.size())
            {
                continue;
            }

            const uint32_t mapped_node_id = importedGltfNodes[tg3Channel.target.node];
            if (mapped_node_id == UINT32_MAX || mapped_node_id >= m_nodes.size())
            {
                continue;
            }

            AnimationPath path{};
            int32_t required_output_type = 0;
            if (tg3_str_equals_cstr(tg3Channel.target.path, "translation"))
            {
                path = AnimationPath::Translation;
                required_output_type = TG3_TYPE_VEC3;
            }
            else if (tg3_str_equals_cstr(tg3Channel.target.path, "rotation"))
            {
                path = AnimationPath::Rotation;
                required_output_type = TG3_TYPE_VEC4;
            }
            else if (tg3_str_equals_cstr(tg3Channel.target.path, "scale"))
            {
                path = AnimationPath::Scale;
                required_output_type = TG3_TYPE_VEC3;
            }
            else
            {
                continue;
            }

            if (sampler_output_type[tg3Channel.sampler] != required_output_type)
            {
                continue;
            }

            AnimationChannel channel{};
            channel.sampler_index = static_cast<uint32_t>(tg3Channel.sampler);
            channel.node_id = mapped_node_id;
            channel.path = path;
            anim.channels.push_back(channel);
        }

        if (anim.channels.empty())
            continue;

        float min_time = std::numeric_limits<float>::infinity();
        float max_time = -std::numeric_limits<float>::infinity();
        bool has_timestamps = false;
        for (const auto& ch : anim.channels)
        {
            const auto& inputs = anim.samplers[ch.sampler_index].inputs;
            for (float t : inputs)
            {
                if (std::isfinite(t))
                {
                    min_time = std::min(min_time, t);
                    max_time = std::max(max_time, t);
                    has_timestamps = true;
                }
            }
        }
        if (has_timestamps)
        {
            anim.start_time = min_time;
            anim.end_time = max_time;
        }

        m_animations.push_back(std::move(anim));
    }
}

std::vector<std::pair<glm::mat4, Mesh&>> Scene::get_mesh_transform(const Node* node)
{
    std::vector<std::pair<glm::mat4, Mesh&>> result;
    if (m_nodes.empty() || root_node_id >= m_nodes.size())
        return result;

    if (node != nullptr)
    {
        node->get_mesh_transform_relative(glm::mat4(1.0f), result, m_nodes, m_meshes);
        return result;
    }

    const Node& root = m_nodes[root_node_id];
    if (root.children.empty())
    {
        root.get_mesh_transform_relative(glm::mat4(1.0f), result, m_nodes, m_meshes);
        return result;
    }

    uint32_t target_scene = (default_scene < root.children.size()) ? default_scene : 0;
    const uint32_t target_id = root.children[target_scene];
    if (target_id < m_nodes.size())
    {
        m_nodes[target_id].get_mesh_transform_relative(glm::mat4(1.0f), result, m_nodes, m_meshes);
    }
    return result;
}

bool Scene::is_mesh_node_shared(uint32_t node_id) const
{
    if (node_id >= m_nodes.size() || m_nodes[node_id].meshId == 0)
        return false;

    uint32_t ref_count = 0;
    for (const auto& n : m_nodes)
    {
        for (uint32_t cid : n.children)
        {
            if (cid == node_id)
            {
                ++ref_count;
                if (ref_count > 1)
                    return true;
            }
        }
    }
    return false;
}

namespace
{
    void collect_subtree_instances_impl(
        const std::vector<Node>& nodes,
        const std::vector<Mesh>& meshes,
        uint32_t node_id,
        uint32_t parent_node_id,
        glm::mat4 current_transform,
        std::vector<MeshInstanceDesc>& out)
    {
        if (node_id >= nodes.size())
            return;

        const Node& node = nodes[node_id];
        if (node.meshId > 0 && node.meshId - 1 < meshes.size())
        {
            MeshInstanceDesc desc{};
            desc.transform = current_transform;
            desc.mesh = const_cast<Mesh*>(&meshes[node.meshId - 1]);
            desc.mesh_node_id = node_id;
            desc.instance_node_id = (parent_node_id != UINT32_MAX) ? parent_node_id : node_id;
            desc.instance_name = (parent_node_id != UINT32_MAX && parent_node_id < nodes.size() && !nodes[parent_node_id].name.empty())
                ? nodes[parent_node_id].name
                : node.name;
            out.push_back(std::move(desc));
        }

        for (uint32_t child_id : node.children)
        {
            if (child_id < nodes.size())
            {
                collect_subtree_instances_impl(
                    nodes,
                    meshes,
                    child_id,
                    node_id,
                    current_transform * nodes[child_id].getTransform(),
                    out);
            }
        }
    }

    void collect_subtree_node_ids(
        const std::vector<Node>& nodes,
        uint32_t node_id,
        std::unordered_set<uint32_t>& visited,
        std::vector<uint32_t>& out)
    {
        if (node_id >= nodes.size() || !visited.insert(node_id).second)
            return;

        out.push_back(node_id);
        for (uint32_t child_id : nodes[node_id].children)
        {
            collect_subtree_node_ids(nodes, child_id, visited, out);
        }
    }

    uint32_t find_mesh_node_id(
        const std::vector<Mesh>& meshes,
        const std::vector<uint32_t>& mesh_node_ids,
        const Mesh* mesh)
    {
        if (mesh == nullptr || meshes.empty())
            return UINT32_MAX;

        if (mesh >= meshes.data() && mesh < meshes.data() + meshes.size())
        {
            const size_t idx = static_cast<size_t>(mesh - meshes.data());
            if (idx < mesh_node_ids.size())
                return mesh_node_ids[idx];
        }
        return UINT32_MAX;
    }

    uint32_t find_reusable_blas_space_for_mesh_node(
        soa_pool<Space>& spaces,
        const HeterogeneousAS& as_heterogeneous,
        uint32_t mesh_node_id)
    {
        uint32_t candidate = UINT32_MAX;
        for (uint32_t sid = 0; sid < static_cast<uint32_t>(spaces.size()); ++sid)
        {
            if (!spaces.is_allocated(sid) || spaces[sid].as_type != ASType::Heterogeneous)
                continue;

            const auto* recs = as_heterogeneous.get_mesh_records(sid);
            if (!recs)
                continue;

            for (const auto& rec : *recs)
            {
                if (rec.mesh_node_id == mesh_node_id && spaces.is_allocated(rec.space_id))
                {
                    candidate = rec.space_id;
                }
            }
        }
        return candidate;
    }

    bool is_reusable_heterogeneous_blas_type(ASType type)
    {
        return type == ASType::BVH ||
               type == ASType::KDTree ||
               type == ASType::UniformGrid ||
               type == ASType::Octree;
    }

    bool is_valid_heterogeneous_blas_space(soa_pool<Space>& spaces, uint32_t space_id)
    {
        return space_id != UINT32_MAX &&
               spaces.is_allocated(space_id) &&
               is_reusable_heterogeneous_blas_type(spaces[space_id].as_type) &&
               spaces[space_id].root_box >= 0;
    }

    void refresh_heterogeneous_tlas_nagata_flags(
        soa_pool<Space>& spaces,
        const HeterogeneousAS& as_heterogeneous)
    {
        for (uint32_t tlas_id = 0; tlas_id < static_cast<uint32_t>(spaces.size()); ++tlas_id)
        {
            if (!spaces.is_allocated(tlas_id) || spaces[tlas_id].as_type != ASType::Heterogeneous)
                continue;

            const auto* inst_recs = as_heterogeneous.get_instance_records(tlas_id);
            if (!inst_recs)
                continue;

            bool any_nagata = false;
            for (const auto& irec : *inst_recs)
            {
                if (irec.active_space_id < spaces.size() && spaces.is_allocated(irec.active_space_id))
                {
                    if (spaces[irec.active_space_id].has_nagata)
                    {
                        any_nagata = true;
                        break;
                    }
                }
            }

            if (spaces[tlas_id].has_nagata != any_nagata)
            {
                spaces[tlas_id].has_nagata = any_nagata;
            }
        }
    }
} // namespace

void Scene::collect_subtree_instances(uint32_t node_id, glm::mat4 current_transform, std::vector<MeshInstanceDesc>& out) const
{
    collect_subtree_instances_impl(m_nodes, m_meshes, node_id, UINT32_MAX, current_transform, out);
}

namespace
{
    struct MeshTransformTask
    {
        soa::soa_view<Vertex> src_vertices;
        soa::soa_view<Triangle> src_triangles;
        glm::mat4 transform;
        glm::mat3 normal_matrix;
        uint32_t dst_vert_offset;
        uint32_t dst_tri_offset;
        uint32_t vert_count;
        uint32_t tri_count;
    };

    __device__ __forceinline__ int32_t scene_float_to_ordered_int(float f)
    {
        int32_t bits = __float_as_int(f);
        return (bits < 0) ? (bits ^ 0x7FFFFFFF) : bits;
    }

    inline float scene_ordered_int_to_float(int32_t ordered)
    {
        int32_t bits = (ordered < 0) ? (ordered ^ 0x7FFFFFFF) : ordered;
        return std::bit_cast<float>(bits);
    }

    __device__ __forceinline__ uint32_t scene_find_vert_task(
        const MeshTransformTask* tasks,
        uint32_t num_tasks,
        uint32_t global_idx)
    {
        uint32_t lo = 0;
        uint32_t hi = num_tasks;
        while (lo < hi)
        {
            uint32_t mid = lo + ((hi - lo) >> 1);
            if (tasks[mid].dst_vert_offset <= global_idx)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo - 1;
    }

    __device__ __forceinline__ uint32_t scene_find_tri_task(
        const MeshTransformTask* tasks,
        uint32_t num_tasks,
        uint32_t global_idx)
    {
        uint32_t lo = 0;
        uint32_t hi = num_tasks;
        while (lo < hi)
        {
            uint32_t mid = lo + ((hi - lo) >> 1);
            if (tasks[mid].dst_tri_offset <= global_idx)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo - 1;
    }

    __device__ __forceinline__ glm::vec3 scene_transform_one_vertex(
        const MeshTransformTask& task,
        uint32_t local_idx,
        soa::soa_view<Vertex> dst_vertices,
        uint32_t dst_idx)
    {
        glm::vec3 pos = glm::vec3(task.transform * glm::vec4(glm::vec3(task.src_vertices[local_idx].position), 1.0f));
        glm::vec3 n = task.normal_matrix * glm::vec3(task.src_vertices[local_idx].normal);
        float nlen2 = glm::dot(n, n);
        if (nlen2 > 1e-20f)
        {
            n *= rsqrtf(nlen2);
        }

        dst_vertices[dst_idx].position = pos;
        dst_vertices[dst_idx].normal = n;
        dst_vertices[dst_idx].color = glm::vec4(task.src_vertices[local_idx].color);
        dst_vertices[dst_idx].uv = glm::vec2(task.src_vertices[local_idx].uv);
        return pos;
    }

    __device__ __forceinline__ void scene_reduce_block_aabb(
        glm::vec3 local_min,
        glm::vec3 local_max,
        int32_t* d_aabb_i)
    {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            local_min.x = fminf(local_min.x, __shfl_down_sync(0xFFFFFFFFu, local_min.x, offset));
            local_min.y = fminf(local_min.y, __shfl_down_sync(0xFFFFFFFFu, local_min.y, offset));
            local_min.z = fminf(local_min.z, __shfl_down_sync(0xFFFFFFFFu, local_min.z, offset));
            local_max.x = fmaxf(local_max.x, __shfl_down_sync(0xFFFFFFFFu, local_max.x, offset));
            local_max.y = fmaxf(local_max.y, __shfl_down_sync(0xFFFFFFFFu, local_max.y, offset));
            local_max.z = fmaxf(local_max.z, __shfl_down_sync(0xFFFFFFFFu, local_max.z, offset));
        }

        __shared__ float s_min_x[8], s_min_y[8], s_min_z[8];
        __shared__ float s_max_x[8], s_max_y[8], s_max_z[8];

        const uint32_t lane = threadIdx.x & 31u;
        const uint32_t warp = threadIdx.x >> 5;
        if (lane == 0)
        {
            s_min_x[warp] = local_min.x;
            s_min_y[warp] = local_min.y;
            s_min_z[warp] = local_min.z;
            s_max_x[warp] = local_max.x;
            s_max_y[warp] = local_max.y;
            s_max_z[warp] = local_max.z;
        }
        __syncthreads();

        if (warp == 0)
        {
            const uint32_t num_warps = (blockDim.x + 31u) >> 5;
            glm::vec3 bmin = (lane < num_warps)
                ? glm::vec3(s_min_x[lane], s_min_y[lane], s_min_z[lane])
                : glm::vec3(FLT_MAX);
            glm::vec3 bmax = (lane < num_warps)
                ? glm::vec3(s_max_x[lane], s_max_y[lane], s_max_z[lane])
                : glm::vec3(-FLT_MAX);

            #pragma unroll
            for (int offset = 4; offset > 0; offset >>= 1)
            {
                bmin.x = fminf(bmin.x, __shfl_down_sync(0xFFFFFFFFu, bmin.x, offset));
                bmin.y = fminf(bmin.y, __shfl_down_sync(0xFFFFFFFFu, bmin.y, offset));
                bmin.z = fminf(bmin.z, __shfl_down_sync(0xFFFFFFFFu, bmin.z, offset));
                bmax.x = fmaxf(bmax.x, __shfl_down_sync(0xFFFFFFFFu, bmax.x, offset));
                bmax.y = fmaxf(bmax.y, __shfl_down_sync(0xFFFFFFFFu, bmax.y, offset));
                bmax.z = fmaxf(bmax.z, __shfl_down_sync(0xFFFFFFFFu, bmax.z, offset));
            }

            if (lane == 0 && bmin.x <= bmax.x)
            {
                atomicMin(&d_aabb_i[0], scene_float_to_ordered_int(bmin.x));
                atomicMin(&d_aabb_i[1], scene_float_to_ordered_int(bmin.y));
                atomicMin(&d_aabb_i[2], scene_float_to_ordered_int(bmin.z));
                atomicMax(&d_aabb_i[3], scene_float_to_ordered_int(bmax.x));
                atomicMax(&d_aabb_i[4], scene_float_to_ordered_int(bmax.y));
                atomicMax(&d_aabb_i[5], scene_float_to_ordered_int(bmax.z));
            }
        }
    }

    template <PrimitiveType PType>
    __device__ __forceinline__ void scene_store_packed_primitive(
        const Triangle& src_tri,
        uint32_t vert_base,
        uint32_t dst_tri_idx,
        soa::soa_view<Vertex> vertices,
        soa::soa_view<PrimitiveChunk16> dst_primitives,
        uint32_t primitives_start)
    {
        const uint32_t v0_idx = src_tri.v0 + vert_base;
        const uint32_t v1_idx = src_tri.v1 + vert_base;
        const uint32_t v2_idx = src_tri.v2 + vert_base;

        if constexpr (PType == PrimitiveType::FullTriangle)
        {
            InlinedTriangle ftri{};
            ftri.type = PrimitiveType::FullTriangle;
            ftri.material_id = src_tri.material_id;

            glm::vec2 uv0 = vertices[v0_idx].uv;
            glm::vec2 uv1 = vertices[v1_idx].uv;
            glm::vec2 uv2 = vertices[v2_idx].uv;
            ftri.uv_0 = glm::packHalf2x16(uv0);
            ftri.uv_1 = glm::packHalf2x16(uv1);
            ftri.uv_2 = glm::packHalf2x16(uv2);

            ftri.v0 = glm::vec3(vertices[v0_idx].position);
            ftri.n0_compressed.pack(glm::vec3(vertices[v0_idx].normal));
            ftri.v1 = glm::vec3(vertices[v1_idx].position);
            ftri.n1_compressed.pack(glm::vec3(vertices[v1_idx].normal));
            ftri.v2 = glm::vec3(vertices[v2_idx].position);
            ftri.n2_compressed.pack(glm::vec3(vertices[v2_idx].normal));

            uint32_t dst_slot = primitives_start + dst_tri_idx * InlinedTriangle::occupied_slots;
            store_primitive(dst_primitives, dst_slot, ftri);
        }
        else if constexpr (PType == PrimitiveType::NagataPatch)
        {
            glm::vec3 p0 = glm::vec3(vertices[v0_idx].position);
            glm::vec3 p1 = glm::vec3(vertices[v1_idx].position);
            glm::vec3 p2 = glm::vec3(vertices[v2_idx].position);
            glm::vec3 n0 = glm::vec3(vertices[v0_idx].normal);
            glm::vec3 n1 = glm::vec3(vertices[v1_idx].normal);
            glm::vec3 n2 = glm::vec3(vertices[v2_idx].normal);

            NagataPatch patch = NagataPatch::fromTriangle(p0, p1, p2, n0, n1, n2);

            NagataPatchPrimitive prim{};
            prim.type = PrimitiveType::NagataPatch;
            prim.material_id = src_tri.material_id;
            prim.v0 = v0_idx;
            prim.v1 = v1_idx;
            prim.v2 = v2_idx;
            prim.C12 = patch.C12;
            prim.C23 = patch.C23;
            prim.C31 = patch.C31;

            uint32_t dst_slot = primitives_start + dst_tri_idx * NagataPatchPrimitive::occupied_slots;
            store_primitive(dst_primitives, dst_slot, prim);
        }
        else
        {
            Triangle tri{};
            tri.type = PrimitiveType::Triangle;
            tri.material_id = src_tri.material_id;
            tri.v0 = v0_idx;
            tri.v1 = v1_idx;
            tri.v2 = v2_idx;

            uint32_t dst_slot = primitives_start + dst_tri_idx * Triangle::occupied_slots;
            store_primitive(dst_primitives, dst_slot, tri);
        }
    }

    inline void transform_aabb(
        const glm::vec3& bmin,
        const glm::vec3& bmax,
        const glm::mat4& M,
        glm::vec3& out_min,
        glm::vec3& out_max)
    {
        out_min = glm::vec3(std::numeric_limits<float>::infinity());
        out_max = glm::vec3(-std::numeric_limits<float>::infinity());
        for (int c = 0; c < 8; ++c)
        {
            glm::vec3 corner(
                (c & 1) ? bmax.x : bmin.x,
                (c & 2) ? bmax.y : bmin.y,
                (c & 4) ? bmax.z : bmin.z);
            glm::vec3 tp = glm::vec3(M * glm::vec4(corner, 1.0f));
            out_min = glm::min(out_min, tp);
            out_max = glm::max(out_max, tp);
        }
    }
} // namespace

__global__ void scene_transform_vertices_single_kernel(
    MeshTransformTask task,
    soa::soa_view<Vertex> dst_vertices,
    uint32_t vertices_start,
    int32_t* d_aabb_i)
{
    glm::vec3 local_min(FLT_MAX);
    glm::vec3 local_max(-FLT_MAX);
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < task.vert_count; i += stride)
    {
        glm::vec3 pos = scene_transform_one_vertex(task, i, dst_vertices, vertices_start + i);
        local_min = glm::min(local_min, pos);
        local_max = glm::max(local_max, pos);
    }
    scene_reduce_block_aabb(local_min, local_max, d_aabb_i);
}

__global__ void scene_transform_vertices_batched_kernel(
    const MeshTransformTask* tasks,
    uint32_t num_tasks,
    uint32_t total_vertices,
    soa::soa_view<Vertex> dst_vertices,
    uint32_t vertices_start,
    int32_t* d_aabb_i)
{
    glm::vec3 local_min(FLT_MAX);
    glm::vec3 local_max(-FLT_MAX);
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_vertices; i += stride)
    {
        uint32_t t = scene_find_vert_task(tasks, num_tasks, i);
        const MeshTransformTask& task = tasks[t];
        uint32_t local_idx = i - task.dst_vert_offset;
        glm::vec3 pos = scene_transform_one_vertex(task, local_idx, dst_vertices, vertices_start + i);
        local_min = glm::min(local_min, pos);
        local_max = glm::max(local_max, pos);
    }
    scene_reduce_block_aabb(local_min, local_max, d_aabb_i);
}

__global__ void scene_offset_triangles_single_kernel(
    MeshTransformTask task,
    uint32_t vertices_start,
    soa::soa_view<Triangle> dst_triangles)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= task.tri_count) return;
    Triangle src = task.src_triangles[i];
    Triangle dst{};
    dst.type = PrimitiveType::Triangle;
    dst.material_id = src.material_id;
    dst.v0 = src.v0 + vertices_start;
    dst.v1 = src.v1 + vertices_start;
    dst.v2 = src.v2 + vertices_start;
    dst_triangles[i] = dst;
}

__global__ void scene_offset_triangles_batched_kernel(
    const MeshTransformTask* tasks,
    uint32_t num_tasks,
    uint32_t total_triangles,
    uint32_t vertices_start,
    soa::soa_view<Triangle> dst_triangles)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_triangles) return;
    uint32_t t = scene_find_tri_task(tasks, num_tasks, i);
    const MeshTransformTask& task = tasks[t];
    uint32_t local_idx = i - task.dst_tri_offset;
    uint32_t vert_base = vertices_start + task.dst_vert_offset;
    Triangle src = task.src_triangles[local_idx];
    Triangle dst{};
    dst.type = PrimitiveType::Triangle;
    dst.material_id = src.material_id;
    dst.v0 = src.v0 + vert_base;
    dst.v1 = src.v1 + vert_base;
    dst.v2 = src.v2 + vert_base;
    dst_triangles[i] = dst;
}

template <PrimitiveType PType>
__global__ void scene_direct_pack_single_kernel(
    MeshTransformTask task,
    uint32_t vertices_start,
    soa::soa_view<Vertex> vertices,
    soa::soa_view<PrimitiveChunk16> dst_primitives,
    uint32_t primitives_start)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= task.tri_count) return;
    Triangle src = task.src_triangles[i];
    scene_store_packed_primitive<PType>(src, vertices_start, i, vertices, dst_primitives, primitives_start);
}

template <PrimitiveType PType>
__global__ void scene_direct_pack_batched_kernel(
    const MeshTransformTask* tasks,
    uint32_t num_tasks,
    uint32_t total_triangles,
    uint32_t vertices_start,
    soa::soa_view<Vertex> vertices,
    soa::soa_view<PrimitiveChunk16> dst_primitives,
    uint32_t primitives_start)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_triangles) return;
    uint32_t t = scene_find_tri_task(tasks, num_tasks, i);
    const MeshTransformTask& task = tasks[t];
    uint32_t local_idx = i - task.dst_tri_offset;
    Triangle src = task.src_triangles[local_idx];
    scene_store_packed_primitive<PType>(
        src,
        vertices_start + task.dst_vert_offset,
        i,
        vertices,
        dst_primitives,
        primitives_start);
}

template<range_of<std::pair<glm::mat4, Mesh&>> Range>
void Scene::prepare_and_pack_space(
    uint32_t space_id,
    Range&& meshes,
    ASType type,
    bool swap_yz,
    bool stretch,
    bool do_sort,
    PrimitiveType triangle_type)
{
    glm::mat4 S_yz(1.0f);
    S_yz[1][1] = 0.0f;  S_yz[1][2] = 1.0f;
    S_yz[2][1] = -1.0f; S_yz[2][2] = 0.0f;

    std::vector<MeshTransformTask> tasks;
    uint32_t total_vertices = 0;
    uint32_t total_triangles = 0;

    for (auto& [transform, mesh] : meshes)
    {
        const uint32_t v_cnt = static_cast<uint32_t>(mesh.d_vertices.size());
        const uint32_t t_cnt = static_cast<uint32_t>(mesh.d_triangles.size());
        if (v_cnt == 0)
            continue;

        const glm::mat4 M = swap_yz ? (S_yz * transform) : transform;
        const glm::mat3 upper(M);
        const float det = glm::determinant(upper);
        const glm::mat3 normal_matrix = (std::fabs(det) > 1e-18f)
            ? glm::transpose(glm::inverse(upper))
            : upper;

        MeshTransformTask task{};
        task.src_vertices = mesh.d_vertices;
        task.src_triangles = mesh.d_triangles;
        task.transform = M;
        task.normal_matrix = normal_matrix;
        task.dst_vert_offset = total_vertices;
        task.dst_tri_offset = total_triangles;
        task.vert_count = v_cnt;
        task.tri_count = t_cnt;
        tasks.push_back(task);

        total_vertices += v_cnt;
        total_triangles += t_cnt;
    }

    spaces[space_id].as_type = type;
    if (total_vertices == 0)
    {
        spaces[space_id].vertices_start = 0;
        spaces[space_id].vertices_size = 0;
        spaces[space_id].primitives_start = 0;
        spaces[space_id].primitives_size = 0;
        spaces[space_id].has_nagata = false;
        return;
    }

    const uint32_t vertices_start = static_cast<uint32_t>(geometry_context.vertices.alloc(total_vertices));
    spaces[space_id].vertices_start = vertices_start;
    spaces[space_id].vertices_size = total_vertices;

    if (m_d_aabb_scratch == nullptr)
    {
        CUDA_CHECK(cudaMalloc(&m_d_aabb_scratch, 6 * sizeof(int32_t)));
    }
    const int32_t init_aabb_i[6] = { INT32_MAX, INT32_MAX, INT32_MAX, INT32_MIN, INT32_MIN, INT32_MIN };
    CUDA_CHECK(cudaMemcpy(m_d_aabb_scratch, init_aabb_i, sizeof(init_aabb_i), cudaMemcpyHostToDevice));

    thrust::device_vector<MeshTransformTask> d_tasks;
    const uint32_t vert_blocks = std::min((total_vertices + 255u) / 256u, 2048u);
    if (tasks.size() == 1)
    {
        scene_transform_vertices_single_kernel<<<vert_blocks, 256>>>(
            tasks[0],
            geometry_context.vertices.view(),
            vertices_start,
            m_d_aabb_scratch);
    }
    else
    {
        d_tasks = tasks;
        scene_transform_vertices_batched_kernel<<<vert_blocks, 256>>>(
            thrust::raw_pointer_cast(d_tasks.data()),
            static_cast<uint32_t>(tasks.size()),
            total_vertices,
            geometry_context.vertices.view(),
            vertices_start,
            m_d_aabb_scratch);
    }
    CUDA_CHECK(cudaGetLastError());

    int32_t h_aabb_i[6];
    CUDA_CHECK(cudaMemcpy(h_aabb_i, m_d_aabb_scratch, sizeof(h_aabb_i), cudaMemcpyDeviceToHost));
    const glm::vec3 aabb_min(
        scene_ordered_int_to_float(h_aabb_i[0]),
        scene_ordered_int_to_float(h_aabb_i[1]),
        scene_ordered_int_to_float(h_aabb_i[2]));
    const glm::vec3 aabb_max(
        scene_ordered_int_to_float(h_aabb_i[3]),
        scene_ordered_int_to_float(h_aabb_i[4]),
        scene_ordered_int_to_float(h_aabb_i[5]));

    normalize_space(space_id, aabb_min, aabb_max, stretch);

    if (total_triangles == 0)
    {
        spaces[space_id].primitives_start = 0;
        spaces[space_id].primitives_size = 0;
        spaces[space_id].has_nagata = false;
        return;
    }

    const uint32_t tri_blocks = (total_triangles + 255u) / 256u;
    if (!do_sort)
    {
        spaces[space_id].has_nagata = (triangle_type == PrimitiveType::NagataPatch);

        uint32_t slots_per_prim = Triangle::occupied_slots;
        if (triangle_type == PrimitiveType::FullTriangle)
            slots_per_prim = InlinedTriangle::occupied_slots;
        else if (triangle_type == PrimitiveType::NagataPatch)
            slots_per_prim = NagataPatchPrimitive::occupied_slots;

        const uint32_t primitives_size = total_triangles * slots_per_prim;
        const uint32_t primitives_start = static_cast<uint32_t>(geometry_context.primitives.alloc(primitives_size));
        spaces[space_id].primitives_start = primitives_start;
        spaces[space_id].primitives_size = primitives_size;

        auto launch_direct_pack = [&]<PrimitiveType PType>()
        {
            if (tasks.size() == 1)
            {
                scene_direct_pack_single_kernel<PType><<<tri_blocks, 256>>>(
                    tasks[0],
                    vertices_start,
                    geometry_context.vertices.view(),
                    geometry_context.primitives.view(),
                    primitives_start);
            }
            else
            {
                scene_direct_pack_batched_kernel<PType><<<tri_blocks, 256>>>(
                    thrust::raw_pointer_cast(d_tasks.data()),
                    static_cast<uint32_t>(tasks.size()),
                    total_triangles,
                    vertices_start,
                    geometry_context.vertices.view(),
                    geometry_context.primitives.view(),
                    primitives_start);
            }
            CUDA_CHECK(cudaGetLastError());
        };

        if (triangle_type == PrimitiveType::FullTriangle)
        {
            launch_direct_pack.template operator()<PrimitiveType::FullTriangle>();
            CUDA_CHECK(cudaDeviceSynchronize());
            if (spaces[space_id].vertices_size > 0)
            {
                geometry_context.vertices.free(spaces[space_id].vertices_start);
                spaces[space_id].vertices_start = 0;
                spaces[space_id].vertices_size = 0;
            }
        }
        else if (triangle_type == PrimitiveType::NagataPatch)
        {
            launch_direct_pack.template operator()<PrimitiveType::NagataPatch>();
            if (tasks.size() > 1)
                CUDA_CHECK(cudaDeviceSynchronize());
        }
        else
        {
            launch_direct_pack.template operator()<PrimitiveType::Triangle>();
            if (tasks.size() > 1)
                CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    else
    {
        soa::cuda_device_vector<Triangle> d_temp_triangles(total_triangles);
        if (tasks.size() == 1)
        {
            scene_offset_triangles_single_kernel<<<tri_blocks, 256>>>(
                tasks[0],
                vertices_start,
                d_temp_triangles.view());
        }
        else
        {
            scene_offset_triangles_batched_kernel<<<tri_blocks, 256>>>(
                thrust::raw_pointer_cast(d_tasks.data()),
                static_cast<uint32_t>(tasks.size()),
                total_triangles,
                vertices_start,
                d_temp_triangles.view());
        }
        CUDA_CHECK(cudaGetLastError());

        sort_space(space_id, d_temp_triangles);
        pack_primitives(space_id, d_temp_triangles, total_triangles, triangle_type);
    }
}

template<range_of<std::pair<glm::mat4, Mesh&>> Range>
uint32_t Scene::build_as_into_space(uint32_t space_id, Range&& meshes, ASType type)
{
    if (type != ASType::Octree &&
        type != ASType::Optix &&
        type != ASType::UniformGrid &&
        type != ASType::KDTree &&
        type != ASType::BVH)
    {
        return space_id;
    }

    const bool stretch = (type == ASType::Octree) ? as_octree.config.stretch : false;
    const bool do_sort = (type == ASType::Optix) ? false : config.sort_space;
    const PrimitiveType prim_type = (type == ASType::Optix) ? PrimitiveType::Triangle : config.triangle_type;

    prepare_and_pack_space(space_id, meshes, type, config.swapYZ, stretch, do_sort, prim_type);

    switch (type)
    {
    case ASType::Octree:
        as_octree.build(space_id, spaces[space_id], prim_type);
        break;
    case ASType::Optix:
        as_optix.build(space_id, spaces[space_id]);
        break;
    case ASType::UniformGrid:
        as_uniform_grid.build(space_id, spaces[space_id], prim_type);
        break;
    case ASType::KDTree:
        as_kdtree.build(space_id, spaces[space_id], prim_type);
        break;
    case ASType::BVH:
        as_bvh.build(space_id, spaces[space_id], prim_type);
        break;
    default:
        break;
    }

    geometry_context.d_spaces = spaces;
    return space_id;
}

uint32_t Scene::build_heterogeneous_instances(const std::vector<MeshInstanceDesc>& instances)
{
    std::vector<MeshInstanceDesc> valid_instances;
    valid_instances.reserve(instances.size());
    for (const auto& desc : instances)
    {
        if (desc.mesh != nullptr && !desc.mesh->d_triangles.empty() && !desc.mesh->d_vertices.empty())
        {
            valid_instances.push_back(desc);
        }
    }

    if (valid_instances.empty())
        return UINT32_MAX;

    const bool rebuild_blas = as_heterogeneous.config.rebuild_blas;
    const bool swap_yz = config.swapYZ;

    glm::mat4 S_yz(1.0f);
    S_yz[1][1] = 0.0f;  S_yz[1][2] = 1.0f;
    S_yz[2][1] = -1.0f; S_yz[2][2] = 0.0f;
    const glm::mat4 S_yz_inv = glm::transpose(S_yz);

    struct UniqueMeshEntry
    {
        Mesh* mesh = nullptr;
        uint32_t mesh_node_id = UINT32_MAX;
        uint32_t instance_count = 0;
        uint32_t blas_space_id = UINT32_MAX;
        glm::vec3 local_bounds_min{std::numeric_limits<float>::infinity()};
        glm::vec3 local_bounds_max{-std::numeric_limits<float>::infinity()};
    };

    std::vector<UniqueMeshEntry> unique_meshes;
    std::unordered_map<const Mesh*, uint32_t> mesh_to_unique_idx;
    std::vector<uint32_t> inst_unique_idx(valid_instances.size(), 0);
    std::vector<glm::vec3> inst_swapped_mins(valid_instances.size(), glm::vec3(std::numeric_limits<float>::infinity()));
    std::vector<glm::vec3> inst_swapped_maxs(valid_instances.size(), glm::vec3(-std::numeric_limits<float>::infinity()));

    glm::vec3 scene_aabb_min(std::numeric_limits<float>::infinity());
    glm::vec3 scene_aabb_max(-std::numeric_limits<float>::infinity());

    for (size_t i = 0; i < valid_instances.size(); ++i)
    {
        const auto& desc = valid_instances[i];
        auto [it, inserted] = mesh_to_unique_idx.emplace(desc.mesh, static_cast<uint32_t>(unique_meshes.size()));
        if (inserted)
        {
            UniqueMeshEntry entry{};
            entry.mesh = desc.mesh;
            entry.mesh_node_id = (desc.mesh_node_id != UINT32_MAX)
                ? desc.mesh_node_id
                : find_mesh_node_id(m_meshes, m_mesh_node_ids, desc.mesh);

            if (swap_yz)
            {
                transform_aabb(
                    desc.mesh->local_bounds_min,
                    desc.mesh->local_bounds_max,
                    S_yz,
                    entry.local_bounds_min,
                    entry.local_bounds_max);
            }
            else
            {
                entry.local_bounds_min = desc.mesh->local_bounds_min;
                entry.local_bounds_max = desc.mesh->local_bounds_max;
            }
            unique_meshes.push_back(std::move(entry));
        }
        else if (unique_meshes[it->second].mesh_node_id == UINT32_MAX && desc.mesh_node_id != UINT32_MAX)
        {
            unique_meshes[it->second].mesh_node_id = desc.mesh_node_id;
        }

        const uint32_t u_idx = it->second;
        inst_unique_idx[i] = u_idx;
        unique_meshes[u_idx].instance_count += 1;

        const glm::mat4 inst_M = swap_yz ? (S_yz * desc.transform) : desc.transform;
        glm::vec3 inst_min;
        glm::vec3 inst_max;
        transform_aabb(desc.mesh->local_bounds_min, desc.mesh->local_bounds_max, inst_M, inst_min, inst_max);
        inst_swapped_mins[i] = inst_min;
        inst_swapped_maxs[i] = inst_max;
        scene_aabb_min = glm::min(scene_aabb_min, inst_min);
        scene_aabb_max = glm::max(scene_aabb_max, inst_max);
    }

    const glm::vec3 scene_aabb_size = scene_aabb_max - scene_aabb_min;
    const float scene_max_axis = glm::max(glm::max(scene_aabb_size.x, scene_aabb_size.y), scene_aabb_size.z);
    const glm::vec3 scene_trans = -scene_aabb_min;
    const glm::vec3 scene_scale = (scene_max_axis > 0.0f) ? glm::vec3(1.0f / scene_max_axis) : glm::vec3(1.0f);
    const glm::mat4 M_swapped_to_scene =
        glm::scale(glm::mat4(1.0f), scene_scale) *
        glm::translate(glm::mat4(1.0f), scene_trans);

    std::vector<MeshBuildRecord> mesh_records;
    mesh_records.reserve(unique_meshes.size());
    bool any_has_nagata = false;
    double blas_eval_prep_ms = 0.0;

    for (auto& u : unique_meshes)
    {
        uint32_t reused_blas_space_id = UINT32_MAX;
        if (!rebuild_blas && u.mesh_node_id < m_nodes.size())
        {
            const ASRef& node_ref = m_nodes[u.mesh_node_id].as_ref;
            if (node_ref.is_valid() &&
                is_valid_heterogeneous_blas_space(spaces, node_ref.space_id) &&
                spaces[node_ref.space_id].as_type == node_ref.type)
            {
                reused_blas_space_id = node_ref.space_id;
            }
        }

        if (reused_blas_space_id != UINT32_MAX)
        {
            u.blas_space_id = reused_blas_space_id;
            MeshBuildRecord rec = as_heterogeneous.make_existing_blas_record(
                reused_blas_space_id,
                spaces[reused_blas_space_id],
                static_cast<uint32_t>(u.mesh->d_triangles.size()),
                static_cast<uint32_t>(u.mesh->d_vertices.size()),
                u.mesh->name,
                u.mesh_node_id,
                u.instance_count,
                u.local_bounds_min,
                u.local_bounds_max);
            any_has_nagata = any_has_nagata || static_cast<bool>(spaces[reused_blas_space_id].has_nagata);
            mesh_records.push_back(std::move(rec));
            continue;
        }

        const auto mesh_t0 = std::chrono::high_resolution_clock::now();
        const uint32_t blas_space_id = create_space();
        u.blas_space_id = blas_space_id;

        std::pair<glm::mat4, Mesh&> single_mesh[1] = {{glm::mat4(1.0f), *u.mesh}};
        prepare_and_pack_space(
            blas_space_id,
            single_mesh,
            ASType::None,
            swap_yz,
            false,
            config.sort_space,
            config.triangle_type);
        geometry_context.d_spaces = spaces;

        MeshBuildRecord rec = as_heterogeneous.evaluate_and_build_blas(
            blas_space_id,
            spaces[blas_space_id],
            static_cast<uint32_t>(u.mesh->d_triangles.size()),
            static_cast<uint32_t>(u.mesh->d_vertices.size()),
            u.mesh->name,
            u.mesh_node_id,
            u.instance_count,
            u.local_bounds_min,
            u.local_bounds_max,
            config.triangle_type);
        geometry_context.d_spaces = spaces;
        const auto mesh_t1 = std::chrono::high_resolution_clock::now();
        const double mesh_total_ms = std::chrono::duration<double, std::milli>(mesh_t1 - mesh_t0).count();
        blas_eval_prep_ms += std::max(0.0, mesh_total_ms - rec.selected_build_ms);

        any_has_nagata = any_has_nagata || static_cast<bool>(spaces[blas_space_id].has_nagata);

        if (u.mesh_node_id < m_nodes.size())
        {
            ASRef& node_ref = m_nodes[u.mesh_node_id].as_ref;
            if (node_ref.is_valid() && node_ref.space_id != blas_space_id && !as_heterogeneous.is_blas_space_owned(node_ref.space_id))
            {
                free_space(node_ref.space_id);
            }
            node_ref = ASRef{blas_space_id, rec.selected_as_type};
        }

        mesh_records.push_back(std::move(rec));
    }

    std::vector<SubspacePrimitive> subspaces;
    std::vector<HeterogeneousInstanceRecord> instance_records;
    subspaces.reserve(valid_instances.size());
    instance_records.reserve(valid_instances.size());

    for (size_t i = 0; i < valid_instances.size(); ++i)
    {
        const auto& desc = valid_instances[i];
        const uint32_t u_idx = inst_unique_idx[i];
        const auto& u = unique_meshes[u_idx];
        const auto& mrec = mesh_records[u_idx];

        const glm::mat4 m_local_swapped_to_swapped = swap_yz
            ? (S_yz * desc.transform * S_yz_inv)
            : desc.transform;
        const glm::mat4 m_local_to_scene = M_swapped_to_scene * m_local_swapped_to_swapped;

        glm::mat4 m_scene_to_local(1.0f);
        const float det = glm::determinant(m_local_to_scene);
        if (std::fabs(det) > 1e-18f)
        {
            m_scene_to_local = glm::inverse(m_local_to_scene);
        }

        uint32_t active_space_id = u.blas_space_id;
        ASType active_as_type = mrec.selected_as_type;
        if (!rebuild_blas &&
            desc.instance_node_id < m_nodes.size() &&
            desc.instance_node_id != u.mesh_node_id)
        {
            const ASRef& inst_ref = m_nodes[desc.instance_node_id].as_ref;
            if (inst_ref.is_valid() &&
                is_valid_heterogeneous_blas_space(spaces, inst_ref.space_id) &&
                spaces[inst_ref.space_id].as_type == inst_ref.type)
            {
                active_space_id = inst_ref.space_id;
                active_as_type = inst_ref.type;
                any_has_nagata = any_has_nagata || static_cast<bool>(spaces[active_space_id].has_nagata);
            }
        }

        HeterogeneousInstanceRecord irec{};
        irec.instance_name = desc.instance_name.empty() ? u.mesh->name : desc.instance_name;
        irec.instance_node_id = desc.instance_node_id;
        irec.mesh_node_id = u.mesh_node_id;
        irec.default_space_id = u.blas_space_id;
        irec.active_space_id = active_space_id;
        irec.default_as_type = mrec.selected_as_type;
        irec.active_as_type = active_as_type;
        irec.s2l_row0 = glm::vec4(m_scene_to_local[0][0], m_scene_to_local[1][0], m_scene_to_local[2][0], m_scene_to_local[3][0]);
        irec.s2l_row1 = glm::vec4(m_scene_to_local[0][1], m_scene_to_local[1][1], m_scene_to_local[2][1], m_scene_to_local[3][1]);
        irec.s2l_row2 = glm::vec4(m_scene_to_local[0][2], m_scene_to_local[1][2], m_scene_to_local[2][2], m_scene_to_local[3][2]);
        instance_records.push_back(std::move(irec));

        SubspacePrimitive sub{};
        sub.type = PrimitiveType::Subspace;
        sub.material_id = 0;
        sub.space_id = static_cast<int>(u.blas_space_id);
        sub.blas_info_id = static_cast<uint32_t>(i);
        sub.blas_as_type = static_cast<uint32_t>(mrec.selected_as_type);
        sub.bounds_min = (inst_swapped_mins[i] + scene_trans) * scene_scale - glm::vec3(1e-5f);
        sub.bounds_max = (inst_swapped_maxs[i] + scene_trans) * scene_scale + glm::vec3(1e-5f);
        subspaces.push_back(sub);
    }

    const auto tlas_prep_t0 = std::chrono::high_resolution_clock::now();
    const uint32_t tlas_space_id = create_space();
    spaces[tlas_space_id].normalization_translate = scene_trans;
    spaces[tlas_space_id].normalization_scale = scene_scale;
    spaces[tlas_space_id].has_nagata = any_has_nagata;

    as_heterogeneous.pack_subspaces(tlas_space_id, spaces[tlas_space_id], subspaces);
    const auto tlas_prep_t1 = std::chrono::high_resolution_clock::now();
    const double tlas_prep_ms = std::chrono::duration<double, std::milli>(tlas_prep_t1 - tlas_prep_t0).count();

    as_heterogeneous.finalize_tlas(
        tlas_space_id,
        spaces[tlas_space_id],
        std::move(mesh_records),
        std::move(instance_records),
        blas_eval_prep_ms,
        tlas_prep_ms,
        subspaces);

    soa::upload_soa_range_async(geometry_context.d_spaces, spaces, tlas_space_id, 1);
    return tlas_space_id;
}

uint32_t Scene::build_node_as(uint32_t node_id, ASType type)
{
    if (node_id >= m_nodes.size() || type == ASType::None)
        return UINT32_MAX;

    if (type == ASType::Heterogeneous)
    {
        std::vector<MeshInstanceDesc> instances;
        collect_subtree_instances(node_id, glm::mat4(1.0f), instances);
        if (instances.empty())
            return UINT32_MAX;

        const bool rebuild_blas = as_heterogeneous.config.rebuild_blas;
        const bool was_active_space =
            m_nodes[node_id].as_ref.is_valid() &&
            m_nodes[node_id].as_ref.space_id == m_active_space;

        std::unordered_set<uint32_t> visited;
        std::vector<uint32_t> subtree_nodes;
        collect_subtree_node_ids(m_nodes, node_id, visited, subtree_nodes);

        uint32_t recovered_target_leaf_blas = UINT32_MAX;
        if (!rebuild_blas &&
            m_nodes[node_id].meshId > 0 &&
            m_nodes[node_id].as_ref.is_valid() &&
            m_nodes[node_id].as_ref.type == ASType::Heterogeneous)
        {
            if (const auto* recs = as_heterogeneous.get_mesh_records(m_nodes[node_id].as_ref.space_id))
            {
                for (const auto& rec : *recs)
                {
                    if (rec.mesh_node_id == node_id && is_valid_heterogeneous_blas_space(spaces, rec.space_id))
                    {
                        recovered_target_leaf_blas = rec.space_id;
                        break;
                    }
                }
            }
        }

        for (uint32_t sub_id : subtree_nodes)
        {
            ASRef& sub_ref = m_nodes[sub_id].as_ref;
            if (!sub_ref.is_valid())
                continue;

            if (rebuild_blas)
            {
                const uint32_t old_space = sub_ref.space_id;
                if (!as_heterogeneous.is_blas_space_owned(old_space))
                {
                    as_heterogeneous.unbind_instance_node_override(sub_id);
                    free_space(old_space);
                    sub_ref = {};
                }
                continue;
            }

            const bool is_target = (sub_id == node_id);
            const bool is_mesh_leaf = (m_nodes[sub_id].meshId > 0);
            const bool is_single_mesh_instance = (!is_mesh_leaf && m_nodes[sub_id].count_subtree_meshes(m_nodes) == 1);

            if (sub_ref.type == ASType::Heterogeneous)
            {
                free_tlas_space_only(sub_ref.space_id);
                sub_ref = {};
            }
            else if (is_mesh_leaf || (!is_target && is_single_mesh_instance))
            {
                if (!is_valid_heterogeneous_blas_space(spaces, sub_ref.space_id) ||
                    spaces[sub_ref.space_id].as_type != sub_ref.type)
                {
                    as_heterogeneous.unbind_instance_node_override(sub_id);
                    const uint32_t old_space = sub_ref.space_id;
                    if (!as_heterogeneous.is_blas_space_owned(old_space))
                    {
                        free_space(old_space);
                    }
                    sub_ref = {};
                }
            }
            else
            {
                const uint32_t old_space = sub_ref.space_id;
                if (!as_heterogeneous.is_blas_space_owned(old_space))
                {
                    as_heterogeneous.unbind_instance_node_override(sub_id);
                    free_space(old_space);
                }
                sub_ref = {};
            }
        }

        if (recovered_target_leaf_blas != UINT32_MAX &&
            is_valid_heterogeneous_blas_space(spaces, recovered_target_leaf_blas))
        {
            m_nodes[node_id].as_ref = ASRef{recovered_target_leaf_blas, spaces[recovered_target_leaf_blas].as_type};
        }

        const uint32_t new_space_id = build_heterogeneous_instances(instances);
        if (new_space_id == UINT32_MAX)
            return UINT32_MAX;

        m_nodes[node_id].as_ref = ASRef{new_space_id, ASType::Heterogeneous};
        if (was_active_space || m_active_space == new_space_id)
        {
            set_active_space(new_space_id);
        }
        return new_space_id;
    }

    std::vector<std::pair<glm::mat4, Mesh&>> relative_meshes;
    m_nodes[node_id].get_mesh_transform_relative(glm::mat4(1.0f), relative_meshes, m_nodes, m_meshes);
    if (relative_meshes.empty())
        return UINT32_MAX;

    uint32_t target_space_id = UINT32_MAX;
    const ASRef existing = m_nodes[node_id].as_ref;
    if (existing.is_valid() && existing.type != ASType::Heterogeneous && spaces.is_allocated(existing.space_id))
    {
        target_space_id = existing.space_id;
    }
    else if (m_nodes[node_id].meshId > 0)
    {
        target_space_id = find_reusable_blas_space_for_mesh_node(spaces, as_heterogeneous, node_id);
    }

    if (target_space_id != UINT32_MAX)
    {
        release_space_resources(target_space_id);
    }
    else
    {
        if (existing.is_valid() && existing.type == ASType::Heterogeneous)
        {
            free_space(existing.space_id);
            m_nodes[node_id].as_ref = {};
        }
        target_space_id = create_space();
    }

    const auto rebuild_t0 = std::chrono::high_resolution_clock::now();
    build_as_into_space(target_space_id, relative_meshes, type);
    const auto rebuild_t1 = std::chrono::high_resolution_clock::now();
    const double rebuild_ms = std::chrono::duration<double, std::milli>(rebuild_t1 - rebuild_t0).count();

    m_nodes[node_id].as_ref = ASRef{target_space_id, type};

    if (as_heterogeneous.is_blas_space_owned(target_space_id))
    {
        size_t nodes_bytes = 0;
        size_t content_bytes = 0;
        as_heterogeneous.compute_as_memory_bytes(type, spaces[target_space_id], nodes_bytes, content_bytes);
        as_heterogeneous.on_blas_space_rebuilt(target_space_id, type, rebuild_ms, nodes_bytes, content_bytes);
    }
    else if (m_nodes[node_id].meshId == 0 && m_nodes[node_id].count_subtree_meshes(m_nodes) == 1)
    {
        as_heterogeneous.bind_instance_node_override(node_id, target_space_id, type);
    }

    refresh_heterogeneous_tlas_nagata_flags(spaces, as_heterogeneous);
    geometry_context.d_spaces = spaces;

    if (m_active_space == target_space_id)
    {
        set_active_space(target_space_id);
    }
    return target_space_id;
}

void Scene::free_node_as(uint32_t node_id)
{
    if (node_id >= m_nodes.size())
        return;

    as_heterogeneous.unbind_instance_node_override(node_id);

    Node& node = m_nodes[node_id];
    if (node.as_ref.is_valid())
    {
        const uint32_t space_id = node.as_ref.space_id;
        if (as_heterogeneous.is_blas_space_owned(space_id))
        {
            release_space_resources(space_id);
            as_heterogeneous.on_blas_space_rebuilt(space_id, ASType::None, 0.0);
            if (m_active_space == space_id)
            {
                m_root_box = -1;
            }
        }
        else
        {
            free_space(space_id);
        }
        node.as_ref = {};
    }

    refresh_heterogeneous_tlas_nagata_flags(spaces, as_heterogeneous);
    geometry_context.d_spaces = spaces;
}

namespace
{
    inline glm::quat anim_quat_from_vec4(const glm::vec4& v)
    {
        return glm::quat(v.w, v.x, v.y, v.z);
    }

    inline glm::quat anim_safe_normalize_quat(const glm::quat& q)
    {
        const float len2 = glm::dot(q, q);
        if (len2 > 1e-20f)
            return q * (1.0f / std::sqrt(len2));
        return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    }

    inline void anim_eval_cubic_spline_weights(float u, float& h00, float& h10, float& h01, float& h11)
    {
        const float u2 = u * u;
        const float u3 = u2 * u;
        h00 = 2.0f * u3 - 3.0f * u2 + 1.0f;
        h10 = u3 - 2.0f * u2 + u;
        h01 = -2.0f * u3 + 3.0f * u2;
        h11 = u3 - u2;
    }

    inline bool anim_find_interval(
        const std::vector<float>& inputs,
        float time_sec,
        size_t& out_prev_idx,
        size_t& out_next_idx,
        float& out_dt,
        float& out_u)
    {
        if (inputs.empty())
            return false;

        if (inputs.size() == 1 || time_sec <= inputs.front())
        {
            out_prev_idx = 0;
            out_next_idx = 0;
            out_dt = 0.0f;
            out_u = 0.0f;
            return false;
        }

        if (time_sec >= inputs.back())
        {
            out_prev_idx = inputs.size() - 1u;
            out_next_idx = out_prev_idx;
            out_dt = 0.0f;
            out_u = 0.0f;
            return false;
        }

        const auto it = std::upper_bound(inputs.begin(), inputs.end(), time_sec);
        out_next_idx = static_cast<size_t>(std::distance(inputs.begin(), it));
        out_prev_idx = (out_next_idx > 0u) ? (out_next_idx - 1u) : 0u;
        const float t0 = inputs[out_prev_idx];
        const float t1 = inputs[out_next_idx];
        out_dt = t1 - t0;
        out_u = (out_dt > 1e-8f) ? ((time_sec - t0) / out_dt) : 0.0f;
        return true;
    }

    inline glm::vec3 sample_animation_vec3(const AnimationSampler& sampler, float time_sec)
    {
        if (sampler.inputs.empty() || sampler.outputs.empty())
            return glm::vec3(0.0f);

        size_t prev_idx = 0;
        size_t next_idx = 0;
        float dt = 0.0f;
        float u = 0.0f;
        if (!anim_find_interval(sampler.inputs, time_sec, prev_idx, next_idx, dt, u))
        {
            const size_t val_idx = (sampler.interpolation == AnimationInterpolation::CubicSpline)
                ? (prev_idx * 3u + 1u)
                : prev_idx;
            return (val_idx < sampler.outputs.size()) ? glm::vec3(sampler.outputs[val_idx]) : glm::vec3(0.0f);
        }

        if (sampler.interpolation == AnimationInterpolation::Step)
        {
            return (prev_idx < sampler.outputs.size()) ? glm::vec3(sampler.outputs[prev_idx]) : glm::vec3(0.0f);
        }

        if (sampler.interpolation == AnimationInterpolation::CubicSpline)
        {
            const size_t v0_idx = prev_idx * 3u + 1u;
            const size_t b0_idx = prev_idx * 3u + 2u;
            const size_t a1_idx = next_idx * 3u;
            const size_t v1_idx = next_idx * 3u + 1u;
            if (v1_idx >= sampler.outputs.size() || b0_idx >= sampler.outputs.size())
                return glm::vec3(0.0f);

            const glm::vec3 v0 = glm::vec3(sampler.outputs[v0_idx]);
            const glm::vec3 b0 = dt * glm::vec3(sampler.outputs[b0_idx]);
            const glm::vec3 a1 = dt * glm::vec3(sampler.outputs[a1_idx]);
            const glm::vec3 v1 = glm::vec3(sampler.outputs[v1_idx]);

            float h00 = 0.0f, h10 = 0.0f, h01 = 0.0f, h11 = 0.0f;
            anim_eval_cubic_spline_weights(u, h00, h10, h01, h11);
            return h00 * v0 + h10 * b0 + h01 * v1 + h11 * a1;
        }

        if (next_idx >= sampler.outputs.size())
            return glm::vec3(0.0f);
        return glm::mix(glm::vec3(sampler.outputs[prev_idx]), glm::vec3(sampler.outputs[next_idx]), u);
    }

    inline glm::quat sample_animation_quat(const AnimationSampler& sampler, float time_sec)
    {
        if (sampler.inputs.empty() || sampler.outputs.empty())
            return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

        size_t prev_idx = 0;
        size_t next_idx = 0;
        float dt = 0.0f;
        float u = 0.0f;
        if (!anim_find_interval(sampler.inputs, time_sec, prev_idx, next_idx, dt, u))
        {
            const size_t val_idx = (sampler.interpolation == AnimationInterpolation::CubicSpline)
                ? (prev_idx * 3u + 1u)
                : prev_idx;
            return (val_idx < sampler.outputs.size())
                ? anim_safe_normalize_quat(anim_quat_from_vec4(sampler.outputs[val_idx]))
                : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        }

        if (sampler.interpolation == AnimationInterpolation::Step)
        {
            return (prev_idx < sampler.outputs.size())
                ? anim_safe_normalize_quat(anim_quat_from_vec4(sampler.outputs[prev_idx]))
                : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        }

        if (sampler.interpolation == AnimationInterpolation::CubicSpline)
        {
            const size_t v0_idx = prev_idx * 3u + 1u;
            const size_t b0_idx = prev_idx * 3u + 2u;
            const size_t a1_idx = next_idx * 3u;
            const size_t v1_idx = next_idx * 3u + 1u;
            if (v1_idx >= sampler.outputs.size() || b0_idx >= sampler.outputs.size())
                return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

            const glm::vec4 v0 = sampler.outputs[v0_idx];
            const glm::vec4 b0 = dt * sampler.outputs[b0_idx];
            const glm::vec4 a1 = dt * sampler.outputs[a1_idx];
            const glm::vec4 v1 = sampler.outputs[v1_idx];

            float h00 = 0.0f, h10 = 0.0f, h01 = 0.0f, h11 = 0.0f;
            anim_eval_cubic_spline_weights(u, h00, h10, h01, h11);
            const glm::vec4 p = h00 * v0 + h10 * b0 + h01 * v1 + h11 * a1;
            return anim_safe_normalize_quat(anim_quat_from_vec4(p));
        }

        if (next_idx >= sampler.outputs.size())
            return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        const glm::quat q0 = anim_safe_normalize_quat(anim_quat_from_vec4(sampler.outputs[prev_idx]));
        const glm::quat q1 = anim_safe_normalize_quat(anim_quat_from_vec4(sampler.outputs[next_idx]));
        return anim_safe_normalize_quat(glm::slerp(q0, q1, u));
    }

    inline float resolve_animation_eval_time(const Animation& anim, float time_sec, bool loop)
    {
        if (!loop)
            return time_sec;

        const float duration = anim.duration();
        if (duration <= 1e-6f)
            return anim.start_time;

        if (time_sec >= anim.start_time && time_sec <= anim.end_time)
            return time_sec;

        float offset = std::fmod(time_sec - anim.start_time, duration);
        if (offset < 0.0f)
            offset += duration;
        return anim.start_time + offset;
    }
} // namespace

bool Scene::apply_animation(uint32_t animation_index, float time_sec, bool loop)
{
    if (animation_index >= m_animations.size())
        return false;

    const Animation& anim = m_animations[animation_index];
    const float eval_time = resolve_animation_eval_time(anim, time_sec, loop);
    bool any_updated = false;
    for (const AnimationChannel& ch : anim.channels)
    {
        if (ch.sampler_index >= anim.samplers.size() || ch.node_id >= m_nodes.size())
            continue;

        const AnimationSampler& sampler = anim.samplers[ch.sampler_index];
        if (sampler.inputs.empty())
            continue;

        Node& target_node = m_nodes[ch.node_id];
        switch (ch.path)
        {
        case AnimationPath::Translation:
            target_node.setTranslation(sample_animation_vec3(sampler, eval_time));
            any_updated = true;
            break;
        case AnimationPath::Rotation:
            target_node.setRotation(sample_animation_quat(sampler, eval_time));
            any_updated = true;
            break;
        case AnimationPath::Scale:
            target_node.setScale(sample_animation_vec3(sampler, eval_time));
            any_updated = true;
            break;
        }
    }

    return any_updated;
}

bool Scene::apply_animations(const std::vector<uint32_t>& animation_indices, float time_sec, bool loop)
{
    bool any_updated = false;
    for (uint32_t anim_idx : animation_indices)
    {
        if (apply_animation(anim_idx, time_sec, loop))
        {
            any_updated = true;
        }
    }
    return any_updated;
}

bool Scene::rebuild_active_as()
{
    if (m_active_space == UINT32_MAX || !spaces.is_allocated(m_active_space))
        return false;

    uint32_t target_node_id = UINT32_MAX;
    for (uint32_t node_id = 0; node_id < static_cast<uint32_t>(m_nodes.size()); ++node_id)
    {
        const ASRef& ref = m_nodes[node_id].as_ref;
        if (ref.is_valid() && ref.space_id == m_active_space)
        {
            target_node_id = node_id;
            break;
        }
    }

    if (target_node_id == UINT32_MAX)
        return false;

    const uint32_t new_space = build_node_as(target_node_id, m_nodes[target_node_id].as_ref.type);
    if (new_space == UINT32_MAX)
        return false;

    set_active_space(new_space);
    return true;
}
