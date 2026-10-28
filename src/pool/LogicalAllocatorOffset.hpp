#pragma once
#include <cassert>

#include "../../external/OffsetAllocator/offsetAllocator.hpp"
#include <optional>
#include <functional>
#include <sstream>
#include <unordered_map>

class LogicalAllocatorOffset
{
    public:

    LogicalAllocatorOffset(size_t size=0, const std::optional<std::function<void(size_t new_size)>>& resize_callback=std::nullopt):
    m_allocator(size),m_resize_callback(resize_callback),m_size(size)
    {
    }

    LogicalAllocatorOffset(const LogicalAllocatorOffset&)=delete;
    LogicalAllocatorOffset& operator=(const LogicalAllocatorOffset&)=delete;


    LogicalAllocatorOffset(LogicalAllocatorOffset&&)=default;
    LogicalAllocatorOffset& operator=(LogicalAllocatorOffset&&)=default;
    ~LogicalAllocatorOffset()
    {

    }


    void clear(size_t size)
    {
        m_size = size;
        offset_allocation.clear();
        m_allocator = OffsetAllocator::Allocator(size,size);
    }

    size_t alloc(size_t size)
    {
        auto allocation=m_allocator.allocate(size);
        if (allocation.offset==OffsetAllocator::Allocation::NO_SPACE)
        {
            if (m_resize_callback.has_value())
            {
                m_resize_callback.value()(m_size+size);
                allocation=m_allocator.allocate(size);
            }
        }

        assert(allocation.offset<OffsetAllocator::Allocation::NO_ALLOCS);

        offset_allocation[allocation.offset]=allocation;
        return allocation.offset;
    }

    void free(size_t start, [[maybe_unused]] size_t size = 0)
    {
        if (auto it = offset_allocation.find(start); it != offset_allocation.end())
        {
            m_allocator.free(it->second);
            offset_allocation.erase(it);
        }
    }


    bool try_realloc(size_t start, size_t new_size)
    {
        auto it = offset_allocation.find(start);
        if (it == offset_allocation.end()) return false;
        return m_allocator.tryReallocate(it->second, static_cast<uint32_t>(new_size));
    }


    void resize(size_t new_size)
    {
        assert(new_size>m_size);
        m_size=new_size;
        m_allocator.resize(new_size);
    }


    std::string free_regions_string() const
    {
        auto storage_report=m_allocator.storageReportFull();

        std::stringstream ss;
        ss<<"size="<<m_size<<"  ";
        //
        // while (fs!=nullptr)
        // {
        //     ss<<std::format("[{}-({})-{}]  ",fs->start,fs->size,fs->start+fs->size);
        //     fs=fs->next;
        // }

        return ss.str();
    }


    size_t size()
    {
        return m_size;
    }

    size_t used_size() const
    {
        return m_size-m_allocator.storageReport().totalFreeSpace;
    }

    bool is_allocated(uint32_t index)
    {
        //m_allocator.storageReport();
    }


    void set_resize_callback(const std::optional<std::function<void(size_t new_size)>>& resize_callback)
    {
        m_resize_callback=resize_callback;
    }

private:
    std::unordered_map<size_t,OffsetAllocator::Allocation> offset_allocation;
    OffsetAllocator::Allocator m_allocator;
    std::optional<std::function<void(size_t new_size)>> m_resize_callback;
    size_t m_size;
};
