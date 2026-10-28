#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <cuda.h>
#define GLM_FORCE_CUDA
#define GLM_FORCE_PURE
#include <glm/glm.hpp>
#undef min
#undef max

enum class PrimitiveType: int
{
    Count,
    Sphere,
    Triangle,
    Subspace,
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
    glm::vec3& color;
    glm::vec3& normal;
    glm::vec3& uv;

};

struct Triangle
{
    uint32_t& v1;
    uint32_t& v2;
    uint32_t& v3;
    uint32_t& material_id;
};

struct Sphere
{
    glm::vec4& center_and_radius;
    uint32_t& material_id;
};

struct Primitive
{
    PrimitiveType& type;//first primitive is PrimitiveType::Count
    uint32_t& id; //for first primitive holds number of primitives that follow
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

struct Box
{
    int32_t &child;
    uint32_t& parent;
    uint32_t& space_id;
};

struct BoxContent
{
    uint32_t& start;
    uint32_t& number_of_primitives;
};


struct SpaceView
{
    uint32_t& space_id;
    uint32_t& parent_space_id;
    Transform transform;
};

struct Material
{
    glm::vec4 baseColor = glm::vec4(1, 1, 1, 1);
    uint32_t textureIndex = 0;
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

