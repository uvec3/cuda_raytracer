#pragma once

#include "../../types.hpp"
#include "../../pool/soa_pool_cuda.hpp"
#include "../../geometry_common.cuh"
#include "../as_concept.cuh"
#include "../../concepts.hpp"

class UniformGridAS
{

    GeometryContext* m_geometry_context = nullptr;
public:
    UniformGridAS(GeometryContext* geometry_context)
    {
        m_geometry_context=geometry_context;
    }

    void build(uint32_t space_id, Space space)
    {

    }
};
