#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#define GLM_FORCE_PURE
#include <glm/glm.hpp>
#undef min
#undef max

enum class ASType : uint32_t
{
    None = 0,
    Octree = 1,
    BVH = 2
};

struct ASRef
{
    uint32_t space_id = UINT32_MAX;
    ASType type = ASType::None;

    bool is_valid() const { return space_id != UINT32_MAX && type != ASType::None; }
};

enum class PrimitiveType: uint32_t
{
    Count,
    Sphere,
    Triangle,
    Subspace,
    NagataPatch,
    Dynamic//??
};

struct Transform
{
    glm::vec2& rx;
    glm::vec2& ry;
    glm::vec2& rz;
    float& scale;
};

struct Vertex
{
    glm::vec3& position;
    glm::vec3& normal;
    glm::vec3& color;
    glm::vec3& uv;
};

struct alignas(16) Triangle
{
    static constexpr int occupied_slots=1;//how many 16 byte slots it occupies
    PrimitiveType type: 8;
    uint32_t material_id: 24;
    uint32_t v0;
    uint32_t v1;
    uint32_t v2;
};

struct CompressedNormal
{
    uint32_t  octNormal;

    // Pack unit normal into 32-bit octahedral representation (Cigolle et al. 2014)
    void pack(glm::vec3 n)
    {
        n /= (std::abs(n.x) + std::abs(n.y) + std::abs(n.z));
        if (n.z < 0.0f) {
            float ox = (1.0f - std::abs(n.y)) * (n.x >= 0.0f ? 1.0f : -1.0f);
            float oy = (1.0f - std::abs(n.x)) * (n.y >= 0.0f ? 1.0f : -1.0f);
            n.x = ox; n.y = oy;
        }
        octNormal= glm::packSnorm2x16(glm::vec2(n.x, n.y));
    }

    [[nodiscard]] glm::vec3 unpack() const
    {
        glm::vec2 f = glm::unpackSnorm2x16(octNormal);
        glm::vec3 n(f.x, f.y, 1.0f - std::abs(f.x) - std::abs(f.y));
        float t = glm::clamp(-n.z, 0.0f, 1.0f);
        n.x += (n.x >= 0.0f) ? -t : t;
        n.y += (n.y >= 0.0f) ? -t : t;
        return glm::normalize(n);
    }
};

struct alignas(16) FullTriangle
{
    static constexpr int occupied_slots=4;//how many 16-byte slots it occupies

    //16 bytes
    PrimitiveType type: 8;
    uint32_t material_id: 24;
    uint32_t uv_0;//keep uv separately as they are not needed for intersecion tests
    uint32_t uv_1;
    uint32_t uv_2;


    //16 bytes
    glm::vec3 v0;
    CompressedNormal n0_compressed;

    //16 bytes
    glm::vec3 v1;
    CompressedNormal n1_compressed;

    //16 bytes
    glm::vec3 v2;
    CompressedNormal n2_compressed;
};


struct alignas(16) Sphere
{
    static constexpr int occupied_slots=1;//how many 16 byte slots it occupies
    PrimitiveType type : 8 = PrimitiveType::Sphere;
    uint32_t material_id: 24;

    glm::vec3 center;
    float radius;
};

struct alignas(8) Primitive
{
    uint32_t id; //for first primitive holds number of primitives that follow
    PrimitiveType type;//first primitive is PrimitiveType::Count
};

struct Space
{
    int& root_box;
    uint32_t& first_triangle;
    uint32_t& number_of_triangles;
    uint32_t& vertices_start;
    uint32_t& vertices_size;

    uint32_t& as_start;
    uint32_t& as_size;

    uint32_t& box_content_start;
    uint32_t& box_content_size;

    uint32_t& primitives_start;
    uint32_t& primitives_size;

    glm::vec3& normalization_translate;
    glm::vec3& normalization_scale;
};

struct alignas(8) BoxContent
{
    uint32_t start;
    uint32_t number_of_primitives;
};

struct SpaceView
{
    uint32_t& space_id;
    uint32_t& parent_space_id;
    Transform transform;
};

struct Texture
{
    std::string name;
    int32_t width = 0;
    int32_t height = 0;
    int32_t channels = 0;
    int32_t minFilter = -1;
    int32_t magFilter = -1;
    int32_t wrapS = 10497;
    int32_t wrapT = 10497;
    cudaArray_t cudaArray = nullptr;
    cudaTextureObject_t texObj = 0;
};

struct MaterialValue
{
    glm::vec4 baseColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    glm::vec3 emissiveFactor = glm::vec3(0.0f, 0.0f, 0.0f);
    uint32_t baseColorTextureIndex = 0;
    uint32_t normalTextureIndex = 0;
    cudaTextureObject_t baseColorTexture = 0;
    cudaTextureObject_t normalTexture = 0;
};

struct Material
{
    using soa_value_type = MaterialValue;

    glm::vec4& baseColor;
    float& metallicFactor;
    float& roughnessFactor;
    glm::vec3& emissiveFactor;
    uint32_t& baseColorTextureIndex;
    uint32_t& normalTextureIndex;
    cudaTextureObject_t& baseColorTexture;
    cudaTextureObject_t& normalTexture;
};

struct Ray
{
    glm::vec3& src;
    glm::vec3& dir;
    uint32_t& beam;
};

struct Beam
{
    uint32_t& start;
    uint32_t& size;
    uint32_t& traced;
};
