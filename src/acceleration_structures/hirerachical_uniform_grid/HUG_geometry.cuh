#pragma once
#include "glm/glm.hpp"
#include <cuda/std/bit>


struct alignas(64) Grid
{
    static constexpr int dim_x=6;
    static constexpr int dim_y=6;
    static constexpr int dim_z=6;

    uint32_t bits[7];//6x6x6 grid of bits indicating the empty boxes
    uint32_t s;

    void switch_value(glm::u8vec3 pos)
    {
        uint32_t linear_num=(pos.x*dim_y+pos.y)*dim_z+pos.z;
        uint32_t unit=linear_num/sizeof(bits[0]);
        uint32_t bit=linear_num%sizeof(bits[0]);
        uint32_t mask=1<<bit;
        bits[unit]^=mask;
    }

    void set_true(glm::u8vec3 pos)
    {
        uint32_t linear_num=(pos.x*dim_y+pos.y)*dim_z+pos.z;
        uint32_t unit=linear_num/sizeof(bits[0]);
        uint32_t bit=linear_num%sizeof(bits[0]);
        uint32_t mask=1<<bit;
        bits[unit]|=mask;
    }

    void set_false(glm::u8vec3 pos)
    {
        uint32_t linear_num=(pos.x*dim_y+pos.y)*dim_z+pos.z;
        uint32_t unit=linear_num/sizeof(bits[0]);
        uint32_t bit=linear_num%sizeof(bits[0]);
        uint32_t mask=1<<bit;
        bits[unit]&=(~mask);
    }

    bool read(glm::u8vec3 pos)
    {
        uint32_t linear_num=(pos.x*dim_y+pos.y)*dim_z+pos.z;
        uint32_t unit=linear_num/sizeof(bits[0]);
        uint32_t bit=linear_num%sizeof(bits[0]);
        uint32_t mask=1<<bit;
        return bits[unit] & mask;
    }

    uint32_t scan(uint32_t element)//return the number of non empty elements before and including this
    {
        uint32_t unit=element/sizeof(bits[0]);
        uint32_t bit=element%sizeof(bits[0]);
        uint32_t count=0;

        for (int i=0;i<unit;++i)
        {
            count+=cuda::std::popcount(bits[i]);
        }

        uint32_t mask=(1<<(bit+1))-1;
        count+=cuda::std::popcount(bits[unit] & mask);//last unit
    }
};



struct HUG
{
    Grid& grid;//bit mask of the grid

    uint32_t chunks_start;// [32 bytes aligned] begin/end  | byte ids|  indices to primitives
    uint32_t byte_ids_start;
    uint32_t indices_start;

};