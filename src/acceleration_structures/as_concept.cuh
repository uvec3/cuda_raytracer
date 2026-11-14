#pragma once
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <string>

#include "../types.hpp"

class ContainerExplorer;

inline size_t space_geometry_bytes(const Space& space)
{
    const size_t prim_bytes = static_cast<size_t>(space.primitives_size) * sizeof(PrimitiveChunk16);
    const size_t vert_bytes = static_cast<size_t>(space.vertices_size) *
        (sizeof(glm::vec3) * 2 + sizeof(glm::vec4) + sizeof(glm::vec2));
    return prim_bytes + vert_bytes;
}

inline std::string format_bytes(size_t bytes)
{
    char buf[64];
    if (bytes < 1024)
    {
        std::snprintf(buf, sizeof(buf), "%zu B", bytes);
    }
    else if (bytes < 1024 * 1024)
    {
        std::snprintf(buf, sizeof(buf), "%.2f KB (%zu B)", static_cast<double>(bytes) / 1024.0, bytes);
    }
    else
    {
        std::snprintf(buf, sizeof(buf), "%.2f MB (%zu B)", static_cast<double>(bytes) / (1024.0 * 1024.0), bytes);
    }
    return std::string(buf);
}

template <typename T>
concept ASDeviceView = requires (T as_view)//Acceleration structure device view concept for static type checking
{
    { as_view.traverse() } -> std::same_as<int>;
};


template <typename T>
concept AS = requires (T as, ContainerExplorer& explorer)//Acceleration structure concept for static type checking
{
    { as.device_view() } -> ASDeviceView;
    as.register_containers(explorer);
};
