#pragma once

#include <algorithm>
#include "soa_vector.hpp"
#include "../LogicalAllocatorOffset.hpp"

template<typename TReference, typename Allocator=soa::DefaultAllocator,int MULT=2>
class soa_pool: public soa::soa_vector<TReference,Allocator>
{
    using Super=soa::soa_vector<TReference,Allocator>;
    LogicalAllocatorOffset l_allocator;

public:

    soa_pool(size_t size = 0,Allocator&& allocator=Allocator{}):Super(size,std::move(allocator)),
    l_allocator(size)
    {
        l_allocator.set_resize_callback([this](auto new_size){resize(new_size*MULT);});
    }

    soa_pool(const soa_pool&) = delete;
    soa_pool& operator=(const soa_pool&) = delete;
    bool is_allocated(uint32_t uint32)
    {
        return  true;
    }

    soa_pool(soa_pool&&) = default;
    soa_pool& operator=(soa_pool&&) = default;



    void resize(size_t new_size)
    {
        Super::resize(new_size);
        l_allocator.resize(new_size);
    }

    size_t alloc_element()
    {
        return l_allocator.alloc(1);
    }

    size_t alloc(size_t size)
    {
        return l_allocator.alloc(size);
    }

    void free(size_t offset)
    {
        l_allocator.free(offset);
    }

    size_t realloc(size_t offset, size_t old_size, size_t new_size)
    {
        if (l_allocator.try_realloc(offset,new_size))
        {
            return offset;
        }

        auto new_offset=l_allocator.alloc(new_size);
        size_t copy_size = std::min(old_size, new_size);
        Super::range(new_offset,new_size).copy_from(Super::range(offset,old_size));
        l_allocator.free(offset,old_size);
        return new_offset;
    }

    void clear(size_t size)
    {
        if (size!=Super::m_size)
        {
            Super::set_capacity(size);
            Super::resize(size);
        }
        l_allocator.clear(size);
    }

    void clear()
    {
        clear(Super::m_size);
    }

    std::string free_info()
    {
        return l_allocator.free_regions_string();
    }

    bool is_allocated(size_t index)
    {
        return l_allocator.is_allocated(index);
    }

    size_t used_size()
    {
        return l_allocator.used_size();
    }


private:
    using Super::allocate;
};
