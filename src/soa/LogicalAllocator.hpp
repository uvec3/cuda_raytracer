#pragma once
#include <cassert>
#include <functional>
#include <string>
#include <sstream>
#include <format>
#include <optional>


class LogicalAllocatorInterface
{
public:
    virtual size_t alloc(size_t size)=0;
    virtual void free(size_t start, size_t size) =0;
    virtual bool try_realloc(size_t start, size_t old_size, size_t new_size)=0;
    virtual void resize(size_t new_size)=0;
};

class LogicalAllocator:LogicalAllocatorInterface
{
public:

    LogicalAllocator(size_t size=0, const std::optional<std::function<void(size_t new_size)>>& resize_callback=std::nullopt):
    m_size(size),m_resize_callback(resize_callback)
    {
        m_begin=new FreeSpace{
            .previous = nullptr,
            .next = nullptr,
            .size=size,
            .start=0
        };
        m_end=m_begin;
    }

    LogicalAllocator(const LogicalAllocator&)=delete;
    LogicalAllocator& operator=(const LogicalAllocator&)=delete;


    LogicalAllocator(LogicalAllocator&&)=default;
    LogicalAllocator& operator=(LogicalAllocator&&)=default;
    ~LogicalAllocator()
    {
        clear(0);
        delete m_begin;
    }


    void clear(size_t size)
    {
        FreeSpace *fs=m_begin;
        while (fs!=nullptr&&fs->size < size)
        {
            auto next=fs->next;
            delete fs;
            fs=next;
        }

        m_size=size;
        m_begin=new FreeSpace{
            .previous = nullptr,
            .next = nullptr,
            .size=m_size,
            .start=0
        };
        m_end=m_begin;
    }

    size_t alloc(size_t size) override
    {
        FreeSpace* fs=find_free_space(size);
        if (fs==nullptr)//if no free space found
        {
            if (m_resize_callback.has_value())
            {
                m_resize_callback.value()(m_size+size);
                if (m_end->size>=size)//check if resize happened (last node must be expanded)
                    fs=m_end;
            }
            assert(fs!=nullptr);
        }

        fs->size-=size;
        auto result=fs->start;

        if (fs->size==0)//if whole chuck is used and it is not the last one delete it
        {
            delete_node(fs);
        }
        else//otherwise move start
        {
            fs->start+=size;
        }

        return result;
    }

    void free(size_t start, size_t size) override
    {
        FreeSpace *previous=m_begin;
        FreeSpace* next=nullptr;
        bool merged_with_previous=false;

        if (start<m_begin->start)
        {
            previous=nullptr;
            next=m_begin;
        }
        else
        {
            while (previous->next!=nullptr && start > previous->next->start)
            {
                previous=previous->next;
            }

            next=previous->next;
        }

        if (previous&&start < previous->start+previous->size)
            throw std::runtime_error("Attempt to free unallocated space!");

        if (previous && previous->start+previous->size==start)//merge with previous free region
        {
            previous->size+=size;
            merged_with_previous=true;
        }

        if (next!=nullptr && next->start==start+size)
        {
            if (merged_with_previous)//merge all 3 regions into one
            {
                previous->size+=next->size;
                previous->next=next->next;

            }
            else//merge with next
            {
                next->size+=size;
                next->start-=size;
            }
        }
        else if (!merged_with_previous)//can't merge with previous or next (insert between)
        {
            auto new_node=new FreeSpace
            {
                .previous = previous,
                .next = next,
                .size = size,
                .start = start,
            };

            if (previous)
                previous->next=new_node;
            else
                m_begin=new_node;

            if (next)
                next->previous=new_node;
            else
                m_end=new_node;
        }
    }


    bool try_realloc(size_t start, size_t old_size, size_t new_size) override
    {
        auto end=start+old_size;

        FreeSpace *next=m_begin;
        while (next!=nullptr && end > next->start)
        {
            next=next->next;
        }

        if (new_size>old_size)//extend
        {
            if (next==nullptr)
                return false;

            if (next->start==end && next->size>=(new_size-old_size))//if the next region is free and has enough space
            {
                auto additional_size=new_size-old_size;
                next->start+=additional_size;
                next->size-=additional_size;
                if (next->size==0)
                    delete_node(next);
            }
            else
            {
                return false;
            }
        }
        else if (new_size<old_size)//shrink
        {
            auto diff=old_size-new_size;
            free(start+new_size,diff);
        }

        return true;
    }


    void resize(size_t new_size) override
    {
        if (new_size>m_size)//expand
        {
            if (m_end->start+m_end->size==m_size)
            {
                m_end->size+=(new_size-m_size);
            }
            else
            {
                auto new_end=new FreeSpace{
                    .previous = m_end,
                    .next = nullptr,
                    .size = new_size-m_size,
                    .start = m_size
                };
                m_end->next=new_end;
                m_end=new_end;
            }
        }
        else if (m_end->start+m_end->size==m_size && m_end->size >= m_size-new_size)//shrink if the end is free
        {
            m_end->size-=m_size-new_size;
            if (m_end->size==0)
                delete_node(m_end);
        }
        else
        {
            throw std::runtime_error("Can't resize, not enough free space");
        }

        m_size=new_size;
    }


    std::string free_regions_string() const
    {
        std::stringstream ss;
        ss<<"size="<<m_size<<"  ";


        FreeSpace* fs=m_begin;

        while (fs!=nullptr)
        {
            ss<<std::format("[{}-({})-{}]  ",fs->start,fs->size,fs->start+fs->size);
            fs=fs->next;
        }

        return ss.str();
    }


    size_t size()
    {
        return m_size;
    }


    void set_resize_callback(const std::optional<std::function<void(size_t new_size)>>& resize_callback)
    {
        m_resize_callback=resize_callback;
    }

private:
    struct FreeSpace
    {
        FreeSpace* previous;
        FreeSpace* next;
        size_t size;
        size_t start;
    };

    FreeSpace* m_begin=nullptr;
    FreeSpace* m_end=nullptr;
    std::optional<std::function<void(size_t new_size)>> m_resize_callback;
    size_t m_size;
    std::vector<FreeSpace*> bins[32];



    FreeSpace* find_free_space(size_t size)
    {
        FreeSpace *fs=m_begin;
        while (fs!=nullptr&&fs->size < size)
            fs=fs->next;
        return fs;
    }

    void delete_node(FreeSpace* node)
    {
        //never delete last node
        if (m_begin==m_end)
            return;

        if (node->previous!=nullptr)
            node->previous->next=node->next;

        if (node->next!=nullptr)
        {
            node->next->previous=node->previous;
        }

        if (m_begin==node)
            m_begin=node->next;
        if (m_end==node)
            m_end=node->previous;

        delete node;
    }
};