#pragma once



#include <bit>
#include <cstdint>
#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_FORCE_ALIGNED_GENTYPES
#define GLM_FORCE_CUDA
#define GLM_FORCE_PURE
#include <glm/glm.hpp>
#include <glm/gtc/type_aligned.hpp>
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

enum class PrimitiveType: uint16_t
{
    Count,
    Sphere,
    Triangle,
    FullTriangle,
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
    glm::vec4& color;
    glm::vec2& uv;
};

struct alignas(16) PrimitiveChunk16 {
    uint4 data;

    [[nodiscard]] __host__ __device__ constexpr PrimitiveType type() const {
        return static_cast<PrimitiveType>(data.x & 0xFFFFu);
    }

    [[nodiscard]] __host__ __device__ constexpr uint16_t material_id() const {
        return static_cast<uint16_t>(data.x >> 16);
    }

    template <typename T>
    [[nodiscard]] __host__ __device__ inline T as() const {
        static_assert(sizeof(T) == sizeof(PrimitiveChunk16));
        return std::bit_cast<T>(*this);
    }

    template <typename T>
    [[nodiscard]] __host__ __device__ static inline PrimitiveChunk16 from(const T& prim) {
        static_assert(sizeof(T) == sizeof(PrimitiveChunk16));
        return std::bit_cast<PrimitiveChunk16>(prim);
    }
};
static_assert(sizeof(PrimitiveChunk16) == 16);

template <typename T, typename PrimView>
[[nodiscard]] __host__ __device__ inline T load_primitive(
    const PrimView& primitives,
    uint32_t prim_id,
    PrimitiveChunk16 chunk0)
{
    static_assert(sizeof(T) == T::occupied_slots * sizeof(PrimitiveChunk16));
    if constexpr (T::occupied_slots == 1) {
        return chunk0.as<T>();
    } else {
        struct alignas(16) RawSlots {
            PrimitiveChunk16 slots[T::occupied_slots];
        } raw;
        raw.slots[0] = chunk0;
        #pragma unroll
        for (int i = 1; i < T::occupied_slots; ++i) {
            raw.slots[i] = primitives[prim_id + i];
        }
        return std::bit_cast<T>(raw);
    }
}

template <typename T, typename PrimView>
[[nodiscard]] __host__ __device__ inline T load_primitive(
    const PrimView& primitives,
    uint32_t prim_id)
{
    return load_primitive<T>(primitives, prim_id, primitives[prim_id]);
}

template <typename T, typename PrimView>
__host__ __device__ inline void store_primitive(
    PrimView& primitives,
    uint32_t prim_id,
    const T& prim)
{
    static_assert(sizeof(T) == T::occupied_slots * sizeof(PrimitiveChunk16));
    struct alignas(16) RawSlots {
        PrimitiveChunk16 slots[T::occupied_slots];
    };
    RawSlots raw = std::bit_cast<RawSlots>(prim);
    #pragma unroll
    for (int i = 0; i < T::occupied_slots; ++i) {
        primitives[prim_id + i] = raw.slots[i];
    }
}

struct alignas(16) Triangle
{
    static constexpr int occupied_slots=1;//how many 16 byte slots it occupies
    PrimitiveType type = PrimitiveType::Triangle;
    uint16_t material_id = 0;
    uint32_t v0;
    uint32_t v1;
    uint32_t v2;
};
static_assert(sizeof(Triangle) == 16);

struct CompressedNormal
{
    uint32_t  octNormal;

    // Pack unit normal into 32-bit octahedral representation (Cigolle et al. 2014)
    __host__ __device__ void pack(glm::vec3 n)
    {
        float l1 = glm::abs(n.x) + glm::abs(n.y) + glm::abs(n.z);
        if (l1 > 0.0f) n /= l1;
        if (n.z < 0.0f) {
            float ox = (1.0f - glm::abs(n.y)) * (n.x >= 0.0f ? 1.0f : -1.0f);
            float oy = (1.0f - glm::abs(n.x)) * (n.y >= 0.0f ? 1.0f : -1.0f);
            n.x = ox; n.y = oy;
        }
        octNormal= glm::packSnorm2x16(glm::vec2(n.x, n.y));
    }

    [[nodiscard]] __host__ __device__ glm::vec3 unpack() const
    {
        glm::vec2 f = glm::unpackSnorm2x16(octNormal);
        glm::vec3 n(f.x, f.y, 1.0f - glm::abs(f.x) - glm::abs(f.y));
        float t = glm::clamp(-n.z, 0.0f, 1.0f);
        n.x += (n.x >= 0.0f) ? -t : t;
        n.y += (n.y >= 0.0f) ? -t : t;
        return glm::normalize(n);
    }
};

struct alignas(16) InlinedTriangle
{
    static constexpr int occupied_slots=4;//how many 16-byte slots it occupies

    // Slot 0 (16 bytes): header (4B) + v0 (12B)
    PrimitiveType type = PrimitiveType::FullTriangle;
    uint16_t material_id = 0;
    glm::vec3 v0;

    // Slot 1 (16 bytes): v1 (12B) + n0 (4B)
    glm::vec3 v1;
    CompressedNormal n0_compressed;

    // Slot 2 (16 bytes): v2 (12B) + n1 (4B)
    glm::vec3 v2;
    CompressedNormal n1_compressed;

    // Slot 3 (16 bytes): n2 (4B) + uv_0..2 (12B)
    CompressedNormal n2_compressed;
    uint32_t uv_0;
    uint32_t uv_1;
    uint32_t uv_2;
};
static_assert(sizeof(InlinedTriangle) == 64);

struct alignas(16) NagataPatchPrimitive
{
    static constexpr int occupied_slots = 4;

    PrimitiveType type = PrimitiveType::NagataPatch;
    uint16_t material_id = 0;
    uint32_t v0;
    uint32_t v1;
    uint32_t v2;

    glm::vec3 C12;
    uint32_t padding_1;
    glm::vec3 C23;
    uint32_t padding_2;
    glm::vec3 C31;
    uint32_t pad1 = 0;

};


struct alignas(16) Sphere
{
    static constexpr int occupied_slots=2;//how many 16 byte slots it occupies (header 4B + center 12B + radius 4B = 20B -> 32B)
    PrimitiveType type = PrimitiveType::Sphere;
    uint16_t material_id = 0;

    glm::vec3 center;
    float radius;
};
static_assert(sizeof(Sphere) == 32);

struct Space
{
    int& root_box;
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

struct alignas(4) BoxContent
{
    uint32_t data = 0;

    __host__ __device__ constexpr BoxContent() = default;
    __host__ __device__ constexpr BoxContent(uint32_t prim_id, bool is_last_flag)
        : data((prim_id << 1) | (is_last_flag ? 1u : 0u)) {}

    [[nodiscard]] __host__ __device__ constexpr bool is_last() const { return (data & 1u) != 0u; }
    [[nodiscard]] __host__ __device__ constexpr uint32_t id() const { return data >> 1; }
    __host__ __device__ constexpr void set(uint32_t prim_id, bool is_last_flag) {
        data = (prim_id << 1) | (is_last_flag ? 1u : 0u);
    }
};
static_assert(sizeof(BoxContent) == 4);

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

struct Material
{
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