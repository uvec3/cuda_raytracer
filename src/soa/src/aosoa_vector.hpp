#pragma once
#include "soa_vector.hpp"
#include <array>

namespace soa
{
    template<typename TReference>
    constexpr size_t calculate_block_size(uint32_t internal_array_size)
    {
        size_t size=0;
        auto callback = [&]< std::size_t I, typename T >() -> T&
        {
            size+=sizeof(T);

            T* ptr = nullptr;return *ptr;//result is ignored, doesn't matter
        };
        init_struct_recursive<decltype(callback),TReference>(callback);

        return size*internal_array_size;
    }

    template<typename TReference,size_t number_of_fields>
    constexpr std::array<size_t,number_of_fields> calculate_local_offsets(uint32_t internal_array_size)
    {
        size_t current_offset=0;
        std::array<size_t,number_of_fields> offsets;

        auto callback = [&]< std::size_t I, typename T >() -> T&
        {
            offsets[I]=current_offset;
            current_offset+=sizeof(T)*internal_array_size;

            T* ptr = nullptr;return *ptr;//result is ignored, doesn't matter
        };
        init_struct_recursive<decltype(callback),TReference>(callback);

        return offsets;
    }


    template <typename TReference,uint32_t internal_array_size=32>
    struct aosoa_iterator
    {
        uintptr_t m_data;
        size_t m_index;

        using iterator_category = std::random_access_iterator_tag;
        using value_type = TReference;
        using difference_type = std::ptrdiff_t;
        using pointer = TReference*;
        using reference = TReference;

        CUDA_HOST_DEVICE TReference operator*() const
        {
            static constexpr size_t block_size=calculate_block_size<TReference>(internal_array_size);
            static constexpr size_t number_of_fields=detail::find_arity< TReference >();
            static constexpr std::array<size_t,number_of_fields> offsets=calculate_local_offsets<TReference,number_of_fields>(internal_array_size);

            size_t block_offset = m_index/internal_array_size*block_size;
            size_t local_index  = m_index%internal_array_size;

            uintptr_t block_ptr = m_data+block_offset;
            auto callback = [&]< std::size_t I, typename T >() -> T&
            {
                T* ptr = reinterpret_cast<T*>(block_ptr+offsets[I]+local_index*sizeof(T));
                return *ptr;
            };

            TReference view { init_struct_recursive<decltype(callback),TReference>(callback)};
            return view;
        }


        CUDA_HOST_DEVICE bool operator==(const aosoa_iterator& other) const { return m_index == other.m_index; }
        CUDA_HOST_DEVICE bool operator!=(const aosoa_iterator& other) const {return m_index != other.m_index;  }
        CUDA_HOST_DEVICE bool operator<(const aosoa_iterator& other) const { return m_index < other.m_index; }
        CUDA_HOST_DEVICE bool operator>(const aosoa_iterator& other) const { return m_index > other.m_index; }
        CUDA_HOST_DEVICE bool operator<=(const aosoa_iterator& other) const { return m_index <= other.m_index; }
        CUDA_HOST_DEVICE bool operator>=(const aosoa_iterator& other) const { return m_index >= other.m_index; }

        CUDA_HOST_DEVICE aosoa_iterator& operator++() { ++m_index; return *this; }
        CUDA_HOST_DEVICE aosoa_iterator operator++(int) { aosoa_iterator temp = *this; ++m_index; return temp; }
        CUDA_HOST_DEVICE aosoa_iterator& operator--() { --m_index; return *this; }
        CUDA_HOST_DEVICE aosoa_iterator operator--(int) { aosoa_iterator temp = *this; --m_index; return temp; }

        CUDA_HOST_DEVICE aosoa_iterator& operator+=(difference_type n) { m_index += n; return *this; }
        CUDA_HOST_DEVICE aosoa_iterator& operator-=(difference_type n) { m_index -= n; return *this; }

        CUDA_HOST_DEVICE aosoa_iterator operator+(difference_type n) const { return aosoa_iterator(m_data, m_index + n); }
        CUDA_HOST_DEVICE aosoa_iterator operator-(difference_type n) const { return aosoa_iterator(m_data, m_index - n); }
        CUDA_HOST_DEVICE friend aosoa_iterator operator+(difference_type n, const aosoa_iterator& it) { return it + n; }
        CUDA_HOST_DEVICE difference_type operator-(const aosoa_iterator& other) const { return static_cast<difference_type>(m_index) - static_cast<difference_type>(other.m_index); }

        CUDA_HOST_DEVICE  reference operator[](difference_type n) const { return *(*this + n); }
    };

    template<typename TReference, uint32_t internal_array_size=32, typename Allocator=DefaultAllocator>
    class aosoa_vector
    {
        static constexpr size_t block_size=calculate_block_size<TReference>(internal_array_size);
        static constexpr size_t number_of_fields=detail::find_arity< TReference >();
        static constexpr std::array<size_t,number_of_fields> offsets=calculate_local_offsets<TReference,number_of_fields>(internal_array_size);
        uintptr_t m_data=0;
        size_t m_size;
        size_t m_number_of_blocks;
        size_t m_capacity;
        Allocator m_allocator;

    public:
        template<typename R, uint32_t i, typename A>
        friend class aosoa_vector;

        aosoa_vector(size_t size,Allocator&& allocator=Allocator{}):m_size{size}, m_capacity(size),
        m_allocator(std::forward<Allocator>(allocator))
        {
            m_number_of_blocks=(size-1)/internal_array_size+1;
            m_data=reinterpret_cast<uintptr_t>(m_allocator.alloc(m_number_of_blocks*block_size));
        }

        template <typename OtherAllocator>
        aosoa_vector(const aosoa_vector<TReference,internal_array_size,OtherAllocator>& other):aosoa_vector{other.m_size,Allocator{}}//copy constructor
        {
            if constexpr (std::is_same_v<OtherAllocator,Allocator>)
            {
                m_allocator.memcpy((void*)m_data,(void*)other.m_data,m_number_of_blocks*block_size);
            }
            else
            {
                transfer_data((void*)m_data,(void*)other.m_data,m_number_of_blocks*block_size,m_allocator,other.m_allocator);
            }
        }

        ~aosoa_vector()
        {
            if (m_number_of_blocks>0)
                m_allocator.free((void*)m_data);
        }

        __forceinline__ TReference operator[](uint32_t index)
        {
            size_t block_offset = index/internal_array_size*block_size;
            size_t local_index  = index%internal_array_size;

            uintptr_t block_ptr = m_data+block_offset;
            auto callback = [&]< std::size_t I, typename T >() -> T&
            {
                T* ptr = reinterpret_cast<T*>(block_ptr+offsets[I]+local_index*sizeof(T));
                return *ptr;
            };


            return init_struct_recursive<decltype(callback),TReference>(callback);
        }

        size_t size()
        {
            return m_size;
        }

        __forceinline__ aosoa_iterator<TReference,internal_array_size> begin()
        {
            return aosoa_iterator<TReference,internal_array_size>(m_data,0);
        }

        __forceinline__ aosoa_iterator<TReference,internal_array_size> end()
        {
            return aosoa_iterator<TReference,internal_array_size>(m_data,m_size);
        }
    };
}
