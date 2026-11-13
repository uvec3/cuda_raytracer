#include "Scene.hpp"
#include "glm/glm.hpp"
#include "../cuda_common.hpp"
#include "../../external/tiny_gltf_v3.h"
#include "../../external/stb_image.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cstring>
#include <optional>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include "../device_algorithms.cuh"
#include "../types/NagataPatch.cuh"

glm::mat4 Node::getTransform()
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
                                       std::vector<Mesh>& meshes)
{
    if (meshId > 0 && meshId - 1 < meshes.size())
    {
        result.push_back({current_transform, meshes[meshId - 1]});
    }

    for (auto& child : children)
    {
        child.get_mesh_transform_relative(current_transform * child.getTransform(), result, meshes);
    }
}

size_t Node::count_subtree_meshes() const
{
    size_t count = (meshId > 0) ? 1 : 0;
    for (const auto& child : children)
    {
        count += child.count_subtree_meshes();
    }
    return count;
}

size_t Node::count_subtree_triangles(const std::vector<Mesh>& meshes) const
{
    size_t count = (meshId > 0 && meshId - 1 < meshes.size()) ? meshes[meshId - 1].triangles.size() : 0;
    for (const auto& child : children)
    {
        count += child.count_subtree_triangles(meshes);
    }
    return count;
}

size_t Node::count_subtree_vertices(const std::vector<Mesh>& meshes) const
{
    size_t count = (meshId > 0 && meshId - 1 < meshes.size()) ? meshes[meshId - 1].vertices.size() : 0;
    for (const auto& child : children)
    {
        count += child.count_subtree_vertices(meshes);
    }
    return count;
}

Scene::Scene(std::string_view name)
    : m_name(name)
    , as_octree(&geometry_context)
    , as_uniform_grid(&geometry_context)
    , as_optix(&geometry_context)
{
    m_active_space = root_space;
}

Scene::~Scene()
{
    destroy_textures();
}

Scene::Scene(Scene&& other) noexcept
    : m_meshes(std::move(other.m_meshes))
    , m_textures(std::move(other.m_textures))
    , m_materials(std::move(other.m_materials))
    , d_materials(std::move(other.d_materials))
    , root_node(std::move(other.root_node))
    , default_scene(other.default_scene)
    , root_space(other.root_space)
    , spaces(std::move(other.spaces))
    , geometry_context(std::move(other.geometry_context))
    , as_octree(std::move(other.as_octree))
    , as_uniform_grid(std::move(other.as_uniform_grid))
    , as_optix(std::move(other.as_optix))
    , m_is_loaded(other.m_is_loaded)
    , m_name(std::move(other.m_name))
    , m_active_space(other.m_active_space)
    , m_root_box(other.m_root_box)
{
    as_octree.bind_geometry_context(&geometry_context);
    as_optix.bind_geometry_context(&geometry_context);
    other.m_textures.clear();
    other.m_is_loaded = false;
}

Scene& Scene::operator=(Scene&& other) noexcept
{
    if (this != &other)
    {
        destroy_textures();
        m_meshes = std::move(other.m_meshes);
        m_textures = std::move(other.m_textures);
        m_materials = std::move(other.m_materials);
        d_materials = std::move(other.d_materials);
        root_node = std::move(other.root_node);
        default_scene = other.default_scene;
        root_space = other.root_space;
        spaces = std::move(other.spaces);
        geometry_context = std::move(other.geometry_context);
        as_octree = std::move(other.as_octree);
        as_uniform_grid = std::move(other.as_uniform_grid);
        as_optix = std::move(other.as_optix);
        m_is_loaded = other.m_is_loaded;
        m_name = std::move(other.m_name);
        m_active_space = other.m_active_space;
        m_root_box = other.m_root_box;

        as_octree.bind_geometry_context(&geometry_context);
    as_optix.bind_geometry_context(&geometry_context);
        other.m_textures.clear();
        other.m_is_loaded = false;
    }
    return *this;
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

void Scene::load(std::string_view data)
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

    std::vector<uint32_t> textureIds = loadTextures(model);
    std::vector<uint32_t> materialIds = loadMaterials(model, textureIds);
    std::vector<uint32_t> meshIds = loadMeshes(model, materialIds);

    load_nodes(model, meshIds);

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

uint32_t Scene::create_space()
{
    auto space_id = static_cast<uint32_t>(spaces.alloc_element());
    spaces[space_id].root_box = -1;
    spaces[space_id].as_type = ASType::None;
    spaces[space_id].vertices_start = 0;
    spaces[space_id].vertices_size = 0;
    spaces[space_id].as_start = 0;
    spaces[space_id].as_size = 0;
    spaces[space_id].box_content_start = 0;
    spaces[space_id].box_content_size = 0;
    spaces[space_id].primitives_start = 0;
    spaces[space_id].primitives_size = 0;
    spaces[space_id].normalization_translate = glm::vec3(0.0f);
    spaces[space_id].normalization_scale = glm::vec3(1.0f);
    geometry_context.d_spaces = spaces;
    return space_id;
}

void Scene::free_space(uint32_t space_id)
{
    if (!spaces.is_allocated(space_id))
        return;
    auto space = spaces[space_id];
    as_octree.free_space(space_id, space);
    as_optix.free_space(space_id, space);
    if (space.vertices_size > 0)
    {
        geometry_context.vertices.free(space.vertices_start);
    }
    if (space.primitives_size > 0)
    {
        geometry_context.primitives.free(space.primitives_start);
    }
    spaces.free(space_id);
    geometry_context.d_spaces = spaces;

    if (m_active_space == space_id)
    {
        m_active_space = UINT32_MAX;
        m_root_box = -1;
    }
}

struct MinVec3 {
    __device__ glm::vec3 operator()(const glm::vec3& a, const glm::vec3& b) const {
        return glm::min(a, b);
    }
};

struct MaxVec3 {
    __device__ glm::vec3 operator()(const glm::vec3& a, const glm::vec3& b) const {
        return glm::max(a, b);
    }
};

void Scene::normalize_space(uint32_t space_id, bool stretch)
{
    auto vertices_start = &geometry_context.vertices[spaces[space_id].vertices_start].position;
    auto vertices_size = spaces[space_id].vertices_size;

    constexpr float inf = std::numeric_limits<float>::infinity();
    auto aabb_min = thrust::reduce(thrust::device, vertices_start, vertices_start + vertices_size, glm::vec3(inf), MinVec3{});
    auto aabb_max = thrust::reduce(thrust::device, vertices_start, vertices_start + vertices_size, glm::vec3(-inf), MaxVec3{});
    auto aabb_size = aabb_max - aabb_min;

    glm::vec3 normalization_translate = -aabb_min;
    glm::vec3 normalization_scale;
    if (stretch)
    {
        normalization_scale = glm::vec3(1) / (aabb_size) * 0.98f;
    }
    else
    {
        normalization_scale = glm::vec3(1) / glm::vec3(glm::max(glm::max(aabb_size.x, aabb_size.y), aabb_size.z)) * 0.98f;
    }

    thrust::transform(thrust::device, vertices_start, vertices_start + vertices_size, vertices_start, [=] __device__(const glm::vec3& v)
    {
        return (v + normalization_translate) * normalization_scale + 0.01f;
    });

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
        return;

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

    float get_material_ior(const tg3_material* tg3mat, float metallicFactor)
    {
        // Default IOR fallback: 1.5 for dielectric (metallicFactor < 1.0), 1.0 for conductor/metal (metallicFactor == 1.0)
        const float fallback_ior = (metallicFactor < 1.0f) ? 1.5f : 1.0f;

        if (!tg3mat)
            return fallback_ior;

        auto extract_ior_from_ext = [](const tg3_extras_ext& ext_container) -> std::optional<float>
        {
            for (uint32_t i = 0; i < ext_container.extensions_count; ++i)
            {
                const auto& ext = ext_container.extensions[i];
                if (tg3_str_equals_cstr(ext.name, "KHR_materials_ior"))
                {
                    if (ext.value.type == TG3_VALUE_OBJECT)
                    {
                        for (uint32_t j = 0; j < ext.value.object_count; ++j)
                        {
                            const auto& kv = ext.value.object_data[j];
                            if (tg3_str_equals_cstr(kv.key, "ior"))
                            {
                                if (kv.value.type == TG3_VALUE_REAL)
                                    return static_cast<float>(kv.value.real_val);
                                if (kv.value.type == TG3_VALUE_INT)
                                    return static_cast<float>(kv.value.int_val);
                            }
                        }
                    }
                    // Extension is present but "ior" property omitted -> glTF spec default is 1.5
                    return 1.5f;
                }
            }
            return std::nullopt;
        };

        if (auto ior = extract_ior_from_ext(tg3mat->ext))
        {
            return (*ior > 0.0f) ? *ior : fallback_ior;
        }

        if (auto ior = extract_ior_from_ext(tg3mat->pbr_metallic_roughness.ext))
        {
            return (*ior > 0.0f) ? *ior : fallback_ior;
        }

        return fallback_ior;
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
    m_meshes.reserve(m_meshes.size() + model.meshes_count);

    // load all gltf mesh and primitive data
    for (uint32_t i = 0; i < model.meshes_count; ++i)
    {
        Mesh mesh;
        const tg3_mesh* tg3mesh = &model.meshes[i];
        mesh.name = tg3mesh->name.data != nullptr
            ? std::string(tg3mesh->name.data, tg3mesh->name.len)
            : "No Name";

        // lambda to perform attribute data copy
        auto writeAttribute = [&model]<typename T>(T* dst, const tg3_str_int_pair* attr, uint32_t start)
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

        uint32_t total_number_of_vertices = 0;
        for (const auto& m : subMeshes)
        {
            total_number_of_vertices += m.vertexCount;
        }
        mesh.vertices.reserve(total_number_of_vertices);


        for (uint32_t s = 0; s < tg3mesh->primitives_count; ++s)
        {
            if (subMeshes[s].vertexCount == 0)
                continue;

            subMeshes[s].vertexStart = mesh.vertices.allocate(subMeshes[s].vertexCount);

            if (subMeshes[s].attribute_position)
            {
                writeAttribute(&mesh.vertices[0].position, subMeshes[s].attribute_position, subMeshes[s].vertexStart);
            }

            if (subMeshes[s].attribute_normal)
            {
                writeAttribute(&mesh.vertices[0].normal, subMeshes[s].attribute_normal, subMeshes[s].vertexStart);
            }

            if (subMeshes[s].attribute_color)
            {
                writeAttribute(&mesh.vertices[0].color, subMeshes[s].attribute_color, subMeshes[s].vertexStart);
            }

            if (subMeshes[s].attribute_texcoord)
            {
                writeAttribute(&mesh.vertices[0].uv, subMeshes[s].attribute_texcoord, subMeshes[s].vertexStart);
            }
        }

        mesh.triangles.reserve(indices.size() / 3);
        for (const auto& submesh : subMeshes)
        {
            auto number_of_triangles = submesh.indexCount / 3;
            if (number_of_triangles == 0)
                continue;

            auto start = mesh.triangles.allocate(number_of_triangles);
            for (size_t t = 0; t < number_of_triangles; ++t)
            {
                mesh.triangles[start + t].v0 = submesh.vertexStart + indices[submesh.indexStart + t * 3];
                mesh.triangles[start + t].v1 = submesh.vertexStart + indices[submesh.indexStart + t * 3 + 1];
                mesh.triangles[start + t].v2 = submesh.vertexStart + indices[submesh.indexStart + t * 3 + 2];
                mesh.triangles[start + t].material_id = submesh.materialId;
            }
        }

        m_meshes.push_back(std::move(mesh));
        meshIds[i] = m_meshes.size();
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
        m_materials[idx0].baseColorTexture = 0;
        m_materials[idx0].normalTexture = 0;
        m_materials[idx0].ior = 1;
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

        uint32_t baseTexId = 0;
        cudaTextureObject_t baseTexObj = 0;
        int32_t gltfBaseTexIdx = tg3mat->pbr_metallic_roughness.base_color_texture.index;
        if (gltfBaseTexIdx >= 0 && static_cast<uint32_t>(gltfBaseTexIdx) < textureIds.size())
        {
            baseTexId = textureIds[gltfBaseTexIdx];
            if (baseTexId > 0 && baseTexId - 1 < m_textures.size())
            {
                baseTexObj = m_textures[baseTexId - 1].texObj;
            }
        }

        uint32_t normTexId = 0;
        cudaTextureObject_t normTexObj = 0;
        int32_t gltfNormTexIdx = tg3mat->normal_texture.index;
        if (gltfNormTexIdx >= 0 && static_cast<uint32_t>(gltfNormTexIdx) < textureIds.size())
        {
            normTexId = textureIds[gltfNormTexIdx];
            if (normTexId > 0 && normTexId - 1 < m_textures.size())
            {
                normTexObj = m_textures[normTexId - 1].texObj;
            }
        }

        mat.baseColorTextureIndex = baseTexId;
        mat.normalTextureIndex = normTexId;
        mat.baseColorTexture = baseTexObj;
        mat.normalTexture = normTexObj;
        mat.ior = get_material_ior(tg3mat, mat.metallicFactor);

        materialIds[i] = static_cast<uint32_t>(matIdx);
    }
    return materialIds;
}


Node Scene::importNode(const tg3_model& model, uint32_t nodeIndex, const std::vector<uint32_t>& meshIds)
{
    const tg3_node& tg3Node = model.nodes[nodeIndex];
    Node node;
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

    if (tg3Node.mesh != -1)
    {
        node.meshId = meshIds[tg3Node.mesh];
    }

    node.children.reserve(tg3Node.children_count);
    for (uint32_t i = 0; i < tg3Node.children_count; ++i)
    {
        int32_t childIndex = tg3Node.children[i];
        node.children.push_back(importNode(model, childIndex, meshIds));
    }

    return node;
}

void Scene::load_nodes(const tg3_model& model, const std::vector<uint32_t>& meshIds)
{
    default_scene = (model.default_scene >= 0 && static_cast<uint32_t>(model.default_scene) < model.scenes_count)
        ? static_cast<uint32_t>(model.default_scene)
        : 0;

    root_node = Node{};
    root_node.name = m_name.empty() ? "Model Root" : m_name;
    root_node.children.reserve(model.scenes_count);

    for (uint32_t scene_id = 0; scene_id < model.scenes_count; ++scene_id)
    {
        const tg3_scene* scene = &model.scenes[scene_id];

        Node subscene_node;
        subscene_node.name = (scene->name.data != nullptr && scene->name.len > 0)
            ? std::string(scene->name.data, scene->name.len)
            : ("Subscene " + std::to_string(scene_id));
        subscene_node.children.reserve(scene->nodes_count);

        for (uint32_t i = 0; i < scene->nodes_count; ++i)
        {
            subscene_node.children.push_back(importNode(model, scene->nodes[i], meshIds));
        }

        root_node.children.push_back(std::move(subscene_node));
    }
}

std::vector<std::pair<glm::mat4, Mesh&>> Scene::get_mesh_transform(const Node* node)
{
    std::vector<std::pair<glm::mat4, Mesh&>> result;
    if (node != nullptr)
    {
        const_cast<Node*>(node)->get_mesh_transform_relative(glm::mat4(1.0f), result, m_meshes);
        return result;
    }

    if (root_node.children.empty())
    {
        root_node.get_mesh_transform_relative(glm::mat4(1.0f), result, m_meshes);
        return result;
    }

    uint32_t target_scene = (default_scene < root_node.children.size()) ? default_scene : 0;
    root_node.children[target_scene].get_mesh_transform_relative(glm::mat4(1.0f), result, m_meshes);
    return result;
}
