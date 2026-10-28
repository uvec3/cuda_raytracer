#pragma once
#include "soa/src/soa_vector.hpp"

struct Mesh
{
    std::string name;
    soa::soa_vector<Vertex> vertices;
    soa::soa_vector<Triangle> triangles;
    glm::mat3 transform=1;
    uint32_t space_id=UINT32_MAX;
};
