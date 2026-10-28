#pragma once

#include <iterator>
#include "utility"
#include "type_traits"
#include "tuple"
#include "cuda_common.hpp"
#include <cassert>

#define SOA_DEBUG_ASSERT(arg) assert(arg)

namespace detail
{
    // Trait to identify leaf nodes (scalar references)
    template <typename T>
    concept IsLeaf = !std::is_aggregate_v<std::remove_cvref_t<T>> || std::is_default_constructible_v<std::remove_cvref_t
        <T>>;

    // Magic type to detect top-level aggregate arity without triggering brace elision
    struct AnyTop
    {
        template <typename T>
        CUDA_HOST_DEVICE operator T&() const;
    };

    template <typename T, std::size_t... Is>
    CUDA_HOST_DEVICE constexpr auto test_top(std::index_sequence<Is...>) -> decltype(T{((void)Is, AnyTop{})...}, true) { return true; }

    template <typename T, std::size_t... Is>
    CUDA_HOST_DEVICE constexpr bool test_top(...) { return false; }

    template <typename T, std::size_t N = 0, std::size_t Max = 64>
    CUDA_HOST_DEVICE constexpr std::size_t get_top_arity()
    {
        if constexpr (test_top<T>(std::make_index_sequence<N>{}))
        {
            return N;
        }
        else if constexpr (N < Max)
        {
            return get_top_arity<T, N + 1, Max>();
        }
        else
        {
            return Max + 1; // Error
        }
    }

    // Forward declaration required for the lambda to recursively call it
    template <typename T>
    CUDA_HOST_DEVICE auto tie_flat(T& obj);

    template <typename T>
    CUDA_HOST_DEVICE auto tie_flat(T& obj)
    {
        if constexpr (IsLeaf<T>)
        {
            return std::forward_as_tuple(obj);
        }
        else
        {
            constexpr std::size_t N = get_top_arity<std::remove_cvref_t<T>>();
            static_assert(N <= 32, "tie_flat currently supports up to 32 members per sub-struct");

            // Helper lambda to cleanly recursively flatten the unpacked members
            auto make_flat = [](auto&... args)
            {
                return std::tuple_cat(tie_flat(args)...);
            };

            // The necessary evil of C++20 reflection: Explicit destructuring cases
            if constexpr (N == 1)
            {
                auto& [a1] = obj;
                return make_flat(a1);
            }
            else if constexpr (N == 2)
            {
                auto& [a1,a2] = obj;
                return make_flat(a1, a2);
            }
            else if constexpr (N == 3)
            {
                auto& [a1,a2,a3] = obj;
                return make_flat(a1, a2, a3);
            }
            else if constexpr (N == 4)
            {
                auto& [a1,a2,a3,a4] = obj;
                return make_flat(a1, a2, a3, a4);
            }
            else if constexpr (N == 5)
            {
                auto& [a1,a2,a3,a4,a5] = obj;
                return make_flat(a1, a2, a3, a4, a5);
            }
            else if constexpr (N == 6)
            {
                auto& [a1,a2,a3,a4,a5,a6] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6);
            }
            else if constexpr (N == 7)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7);
            }
            else if constexpr (N == 8)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8);
            }
            else if constexpr (N == 9)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9);
            }
            else if constexpr (N == 10)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
            }
            else if constexpr (N == 11)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
            }
            else if constexpr (N == 12)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12);
            }
            else if constexpr (N == 13)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13);
            }
            else if constexpr (N == 14)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14);
            }
            else if constexpr (N == 15)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15);
            }
            else if constexpr (N == 16)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16);
            }
            else if constexpr (N == 17)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17);
            }
            else if constexpr (N == 18)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18);
            }
            else if constexpr (N == 19)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19);
            }
            else if constexpr (N == 20)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
                                 a20);
            }
            else if constexpr (N == 21)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
                                 a20, a21);
            }
            else if constexpr (N == 22)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
                                 a20, a21, a22);
            }
            else if constexpr (N == 23)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
                                 a20, a21, a22, a23);
            }
            else if constexpr (N == 24)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
                                 a20, a21, a22, a23, a24);
            }
            else if constexpr (N == 25)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25] =
                    obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
                                 a20, a21, a22, a23, a24, a25);
            }
            else if constexpr (N == 26)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26] =
                    obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
                                 a20, a21, a22, a23, a24, a25, a26);
            }
            else if constexpr (N == 27)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,
                    a27] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
                                 a20, a21, a22, a23, a24, a25, a26, a27);
            }
            else if constexpr (N == 28)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,
                    a27
                    ,a28] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
                                 a20, a21, a22, a23, a24, a25, a26, a27, a28);
            }
            else if constexpr (N == 29)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,
                    a27
                    ,a28,a29] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
                                 a20, a21, a22, a23, a24, a25, a26, a27, a28, a29);
            }
            else if constexpr (N == 30)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,
                    a27
                    ,a28,a29,a30] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
                                 a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30);
            }
            else if constexpr (N == 31)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,
                    a27
                    ,a28,a29,a30,a31] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
                                 a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31);
            }
            else if constexpr (N == 32)
            {
                auto& [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,
                    a27
                    ,a28,a29,a30,a31,a32] = obj;
                return make_flat(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
                                 a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32);
            }
        }
    }


    template <typename Callback, std::size_t I>
    struct IndexedLeafProvider
    {
        Callback& cb;

        template <typename T>
            requires IsLeaf<T>
        CUDA_HOST_DEVICE constexpr  operator T&() const
        {
            // Explicitly passing the index I and the deduced leaf type T to the callback
            return cb.template operator()<I, T>();
        }
    };

    template <typename Callback, typename Tuple, std::size_t I>
    struct BoundIndexedLeafProvider
    {
        Callback& cb;
        Tuple& flat_src;

        template <typename T>
            requires IsLeaf<T>
        CUDA_HOST_DEVICE constexpr  operator T&() const
        {
            return cb.template operator()<I, T>(std::get<I>(flat_src));
        }
    };



     struct AnyLeaf {
         template < typename T >
         requires IsLeaf< T >
         CUDA_HOST_DEVICE operator T&() const;
     };

    template < typename T, std::size_t... Is >
    CUDA_HOST_DEVICE constexpr auto test_construct(std::index_sequence< Is... >) -> decltype(T{ ((void)Is, AnyLeaf{})... }, true) { return true; }

    template < typename T, std::size_t... Is >
    CUDA_HOST_DEVICE constexpr bool test_construct(...) { return false; }

    template < typename T, std::size_t N = 0, std::size_t Max = 64 >
    CUDA_HOST_DEVICE constexpr std::size_t find_arity() {
        if constexpr (test_construct< T >(std::make_index_sequence< N >{})) {
            return N;
        } else if constexpr (N < Max) {
            return find_arity< T, N + 1, Max >();
        } else {
            return Max + 1;
        }
    }

    template < typename Callback >
    struct LeafProvider {
        Callback& cb;

        template < typename T >
        requires IsLeaf< T >
        CUDA_HOST_DEVICE constexpr  operator T&() const {
            return cb.template operator()< T >();
        }
    };
}

// Extracts the I-th leaf reference from the tuple and passes it to the callback

// Overload 1: Create from scratch (No reference)
template <typename Callback, typename View>
CUDA_HOST_DEVICE constexpr View init_struct_recursive(Callback callback)
{
    constexpr std::size_t N = detail::find_arity<View>();
    return [&]< std::size_t... Is >(std::index_sequence<Is...>)
    {
        // Notice how clean this is now without the comma operator
        return View{detail::IndexedLeafProvider<Callback, Is>{callback}...};
    }(std::make_index_sequence<N>{});
}

// Overload 2: Create from source (1 reference)
template <typename Callback, typename View>
CUDA_HOST_DEVICE constexpr View init_struct_recursive_from(Callback callback, View& src)
{
    auto flat_src = detail::tie_flat(src);
    constexpr std::size_t N = std::tuple_size_v<decltype(flat_src)>;
    return [&]< std::size_t... Is >(std::index_sequence<Is...>)
    {
        return View{detail::BoundIndexedLeafProvider<Callback, decltype(flat_src), Is>{callback, flat_src}...};
    }(std::make_index_sequence<N>{});
}

// Overload 3: Iterate over a single view
template <typename Callback, typename View>
CUDA_HOST_DEVICE constexpr void iterate_struct_recursive(Callback callback, View& src)
{
    auto flat_src = detail::tie_flat(src);
    constexpr std::size_t N = std::tuple_size_v<decltype(flat_src)>;

    [&]< std::size_t... Is >(std::index_sequence<Is...>)
    {
        ( (void)callback.template operator()<Is, std::remove_reference_t<std::tuple_element_t<Is, decltype(flat_src)>>>(
            std::get<Is>(flat_src)), ... );
    }(std::make_index_sequence<N>{});
}

// Overload 4: Iterate over two views simultaneously
template <typename Callback, typename View>
CUDA_HOST_DEVICE constexpr void iterate_struct_pair_recursive(Callback callback, View& src1, const View& src2)
{
    auto flat_src1 = detail::tie_flat(src1);
    auto flat_src2 = detail::tie_flat(src2);
    constexpr std::size_t N = std::tuple_size_v<decltype(flat_src1)>;

    [&]< std::size_t... Is >(std::index_sequence<Is...>)
    {
        ( (void)callback.template operator()<Is, std::remove_reference_t<std::tuple_element_t<
                                                 Is, decltype(flat_src1)>>>(
            std::get<Is>(flat_src1), std::get<Is>(flat_src2)), ... );
    }(std::make_index_sequence<N>{});
}

// Overload 5: Iterate over types of a view without an instance
template <typename Callback, typename View>
CUDA_HOST_DEVICE constexpr void iterate_struct_recursive(Callback callback)
{
    using FlatTuple = decltype(detail::tie_flat(std::declval<View&>()));
    constexpr std::size_t N = std::tuple_size_v<FlatTuple>;

    [&]< std::size_t... Is >(std::index_sequence<Is...>)
    {
        ( (void)callback.template operator()<Is, std::remove_reference_t<std::tuple_element_t<Is, FlatTuple>>>(), ... );
    }(std::make_index_sequence<N>{});
}

namespace soa
{
    struct DefaultAllocator
    {
        static  void* alloc(size_t bytes)
        {
            return ::malloc(bytes);
        }

        static void free(void* ptr)
        {
            return ::free(ptr);
        }

        static void copy(void* dst, const void* src, size_t bytes)
        {
            ::memcpy(dst,src,bytes);
        }

        static void realloc(void** ptr, size_t new_size, size_t old_size)
        {
            *ptr = ::realloc(*ptr, new_size);
        }
    };

    //concept to check whether a specific allocator implements realloc
    template <typename Allocator>
    concept CanRealloc = requires(Allocator a, void** ptr, size_t new_size, size_t old_size) {
        a.realloc(ptr, new_size, old_size);
    };


    template <typename TReference>
    struct soa_element_snapshot
    {
        using flattened_type = decltype(detail::tie_flat(std::declval<TReference&>()));

        template <std::size_t... Is>
        static auto make_storage_type(std::index_sequence<Is...>)
            -> std::tuple<std::remove_cvref_t<std::tuple_element_t<Is, flattened_type>>...>;

        using storage_type = decltype(make_storage_type(
            std::make_index_sequence<std::tuple_size_v<flattened_type>>{}));

        mutable storage_type m_data;

        soa_element_snapshot() = default;

        soa_element_snapshot(const TReference& source)
        {
            *this = source;
        }

        soa_element_snapshot& operator=(const TReference& source)
        {
            auto flat_source = detail::tie_flat(source);
            [&]<std::size_t... Is>(std::index_sequence<Is...>)
            {
                ((std::get<Is>(m_data) = std::get<Is>(flat_source)), ...);
            }(std::make_index_sequence<std::tuple_size_v<flattened_type>>{});
            return *this;
        }

        operator TReference() const
        {
            auto callback = [this]<std::size_t I, typename T>() -> T&
            {
                return std::get<I>(m_data);
            };
            return init_struct_recursive<decltype(callback), TReference>(callback);
        }
    };

    template <typename TReference>
    struct RefWrapper : TReference
    {
        RefWrapper& operator=(const RefWrapper& other)
        {
            auto callback = []< std::size_t I, typename T >(T& dst, const T& src)
            {
                dst = src;
            };

            iterate_struct_pair_recursive<decltype(callback), TReference>(callback, *this, other);
            return *this;
        }

        RefWrapper& operator=(RefWrapper&& other) noexcept
        {
            *this = other;
            return *this;
        };

        RefWrapper& operator=(const soa_element_snapshot<TReference>& other)
        {
            const TReference source = other;
            auto callback = []<std::size_t I, typename T>(T& dst, const T& src)
            {
                dst = src;
            };
            iterate_struct_pair_recursive<decltype(callback), TReference>(callback, *this, source);
            return *this;
        }

        RefWrapper(const RefWrapper& other) = default;
        RefWrapper(RefWrapper&& other) = default;

        RefWrapper(const TReference& other) : TReference(other)
        {
        };

        RefWrapper(TReference&& other) : TReference(std::move(other))
        {
        };


        friend void swap(RefWrapper& a,RefWrapper& b) noexcept
        {
            auto callback = []< std::size_t I, typename T >(T& a, T& b)
            {
                auto tmp=a;
                a=b;
                b=tmp;
            };

            iterate_struct_pair_recursive<decltype(callback), TReference>(callback, a, b);
        }

        friend void swap(RefWrapper&& a,RefWrapper&& b) noexcept
        {
            auto callback = []< std::size_t I, typename T >(T& a, T& b)
            {
                auto tmp=a;
                a=b;
                b=tmp;
            };

            iterate_struct_pair_recursive<decltype(callback), TReference>(callback, a, b);
        }

    };



    template <typename TReference>
    struct soa_iterator
    {
        void* m_data[detail::find_arity<TReference>()];
        size_t m_index;

        using iterator_category = std::random_access_iterator_tag;
        using value_type = soa_element_snapshot<TReference>;
        using difference_type = std::ptrdiff_t;
        // using pointer = RefWrapper<TReference>*;
        using reference = RefWrapper<TReference>;

        CUDA_HOST_DEVICE soa_iterator() = default;

        CUDA_HOST_DEVICE soa_iterator(void* const* data, size_t index) : m_index(index)
        {
            for (size_t i = 0; i < detail::find_arity<TReference>(); ++i) {
                m_data[i] = data[i];
            }
        }

        CUDA_HOST_DEVICE reference operator*() const
        {
            auto callback = [&]< std::size_t I, typename T >() -> T&
            {
                T* ptr = (T*)(m_data[I]) + m_index;
                return *ptr;
            };

            return init_struct_recursive<decltype(callback), TReference>(callback);
        }

        CUDA_HOST_DEVICE [[nodiscard]] size_t index() const
        {
            return m_index;
        }

        CUDA_HOST_DEVICE bool operator==(const soa_iterator& other) const { return m_index == other.m_index; }
        CUDA_HOST_DEVICE bool operator!=(const soa_iterator& other) const {return m_index != other.m_index;  }
        CUDA_HOST_DEVICE bool operator<(const soa_iterator& other) const { return m_index < other.m_index; }
        CUDA_HOST_DEVICE bool operator>(const soa_iterator& other) const { return m_index > other.m_index; }
        CUDA_HOST_DEVICE bool operator<=(const soa_iterator& other) const { return m_index <= other.m_index; }
        CUDA_HOST_DEVICE bool operator>=(const soa_iterator& other) const { return m_index >= other.m_index; }

        CUDA_HOST_DEVICE soa_iterator& operator++() { ++m_index; return *this; }
        CUDA_HOST_DEVICE soa_iterator operator++(int) { soa_iterator temp = *this; ++m_index; return temp; }
        CUDA_HOST_DEVICE soa_iterator& operator--() { --m_index; return *this; }
        CUDA_HOST_DEVICE soa_iterator operator--(int) { soa_iterator temp = *this; --m_index; return temp; }

        CUDA_HOST_DEVICE soa_iterator& operator+=(difference_type n) { m_index += n; return *this; }
        CUDA_HOST_DEVICE soa_iterator& operator-=(difference_type n) { m_index -= n; return *this; }

        CUDA_HOST_DEVICE soa_iterator operator+(difference_type n) const { 
            soa_iterator temp = *this;
            temp.m_index += n;
            return temp; 
        }
        CUDA_HOST_DEVICE soa_iterator operator-(difference_type n) const { 
            soa_iterator temp = *this;
            temp.m_index -= n;
            return temp;
        }
        CUDA_HOST_DEVICE friend soa_iterator operator+(difference_type n, const soa_iterator& it) { return it + n; }
        CUDA_HOST_DEVICE difference_type operator-(const soa_iterator& other) const { return static_cast<difference_type>(m_index) - static_cast<difference_type>(other.m_index); }

        CUDA_HOST_DEVICE  reference operator[](difference_type n) const { return *(*this + n); }
    };

    template <typename R, typename A>
    class soa_vector;

    //lightweight soa view
    template<typename TReference>
    struct soa_view
    {
        void* m_data[detail::find_arity< TReference >()];
        size_t m_size{};


        CUDA_HOST_DEVICE size_t size()
        {
            return  m_size;
        }

        CUDA_HOST_DEVICE constexpr TReference operator[](size_t index) const
        {
            auto callback = [&]< std::size_t I, typename T >() -> T&
            {
                T* ptr = (T*)(m_data[I]) + index;
                return *ptr;
            };

            return init_struct_recursive<decltype(callback), TReference>(callback);
        }

        soa_iterator<TReference> begin()
        {
            return soa_iterator<TReference>((void**)m_data, 0);
        }

        soa_iterator<TReference> end()
        {
            return soa_iterator<TReference>((void**)m_data, m_size);
        }

        void copy_from( const soa_view& other)
        {
            SOA_DEBUG_ASSERT(m_size>=other.m_size);

            auto callback = [&]<std::size_t I, typename T>(){
                T* ptr_dst = (T*)(m_data[I]);
                const T* ptr_src= (T*)(other.m_data[I]);

                for (size_t i=0; i<other.m_size;++i)
                    ptr_dst[i]=ptr_src[i];
            };

            iterate_struct_recursive<decltype(callback), TReference>(callback);
        }

        template <typename R, typename A>
        friend class soa_vector;
    };



    template<typename TReference, typename Allocator=DefaultAllocator>
    class soa_vector
    {
    private:
        static constexpr int number_of_fields=detail::find_arity<TReference>();
        void* m_data[number_of_fields];
        size_t m_capacity;
        size_t m_size;
        Allocator m_allocator;

    public:
        template<typename R,typename A>
        friend class soa_vector;



        soa_vector(size_t size=0,Allocator&& allocator=Allocator{}): m_capacity(size), m_size(size),
        m_allocator(std::forward<Allocator>(allocator))
        {
            for (size_t i = 0; i < number_of_fields; ++i) m_data[i] = nullptr;
            if (m_capacity > 0) {
                auto callback = [&]<std::size_t I, typename T>() {
                    m_data[I] = m_allocator.alloc(sizeof(T) * size);
                };
                iterate_struct_recursive<decltype(callback), TReference>(callback);
            }
        }

        //copy
        soa_vector(const soa_vector& other):soa_vector{other.m_size,Allocator{}}
        {
            if (other.m_size > 0) {
                auto callback = [&]<std::size_t I, typename T>() {
                    m_allocator.copy(m_data[I], other.m_data[I], other.m_size * sizeof(T));
                };
                iterate_struct_recursive<decltype(callback), TReference>(callback);
            }
        }

        template <typename OtherAllocator>
        soa_vector(const soa_vector<TReference,OtherAllocator>& other):soa_vector{other.m_size,Allocator{}}
        {
            if (other.m_size > 0) {
                if constexpr (std::is_same_v<OtherAllocator,Allocator>)
                {
                    auto callback = [&]<std::size_t I, typename T>() {
                        m_allocator.copy(m_data[I], other.m_data[I], other.m_size * sizeof(T));
                    };
                    iterate_struct_recursive<decltype(callback), TReference>(callback);
                }
                else
                {
                    auto callback = [&]<std::size_t I, typename T>() {
                        transfer_data(m_data[I], other.m_data[I], other.m_size * sizeof(T), m_allocator, other.m_allocator);
                    };
                    iterate_struct_recursive<decltype(callback), TReference>(callback);
                }
            }
        }

        soa_vector& operator=(const soa_vector& other)//copy assignment
        {
            printf("operator=\n");
            resize(other.m_size);
            if (other.m_size > 0) {
                auto callback = [&]<std::size_t I, typename T>() {
                    m_allocator.copy(m_data[I], other.m_data[I], other.m_size * sizeof(T));
                };
                iterate_struct_recursive<decltype(callback), TReference>(callback);
            }
            return *this;
        }

        //move
        soa_vector(soa_vector&& other) noexcept
            : m_capacity(other.m_capacity),
              m_size(other.m_size),
              m_allocator(std::move(other.m_allocator))
        {
            for (size_t i = 0; i < detail::find_arity<TReference>(); ++i) {
                m_data[i] = other.m_data[i];
                other.m_data[i] = nullptr;
            }
            other.m_capacity = 0;
            other.m_size = 0;
        }

        soa_vector& operator=(soa_vector&& other) noexcept
        {
            if (this == &other)
                return *this;
            clear();
            for (size_t i = 0; i < detail::find_arity<TReference>(); ++i) {
                m_data[i] = other.m_data[i];
                other.m_data[i] = nullptr;
            }
            m_capacity = other.m_capacity;
            m_size = other.m_size;
            m_allocator = std::move(other.m_allocator);
            other.m_capacity = 0;
            other.m_size = 0;
            return *this;
        }

        ~soa_vector()
        {
            if (m_capacity>0) {
                auto callback = [&]<std::size_t I, typename T>() {
                    m_allocator.free(m_data[I]);
            };
                iterate_struct_recursive<decltype(callback), TReference>(callback);
            }
        }

        //reference wrapper to allow assignment and swap operations
        constexpr RefWrapper<TReference> operator[](size_t index)
        {
            auto callback = [&]< std::size_t I, typename T >() -> T& {
                T* ptr = (T*)(m_data[I]) + index;
                return *ptr;
            };

            return init_struct_recursive<decltype(callback), TReference>(callback);
        }

        size_t size()
        {
            return m_size;
        }

        size_t capacity()
        {
            return m_capacity;
        }

        soa_iterator<TReference> begin()
        {
            return soa_iterator<TReference>((void**)m_data, 0);
        }

        soa_iterator<TReference> end()
        {
            return soa_iterator<TReference>((void**)m_data, m_size);
        }

        soa_view<TReference> view()
        {
            soa_view<TReference> result;
            result.m_size=m_size;
            for (size_t i = 0; i < detail::find_arity<TReference>(); ++i) {
                result.m_data[i] = m_data[i];
            }
            return result;
        }

        operator soa_view<TReference>()
        {
            return view();
        }

        soa_view<TReference> range(size_t offset, size_t size)
        {
            soa_view<TReference> result;
            result.m_size=size;
            auto callback = [&]<std::size_t I, typename T>() {
                result.m_data[I] = (T*)m_data[I] + offset;
            };
            iterate_struct_recursive<decltype(callback), TReference>(callback);
            return result;
        }

        void reserve(size_t capacity)
        {
            if (m_capacity==capacity)
                return;

            if (capacity==0)
            {
                clear();
            }
            else if (m_capacity==0)
            {
                auto callback = [&]<std::size_t I, typename T>()
                {
                    m_data[I] = m_allocator.alloc(sizeof(T) * capacity);
                };
                iterate_struct_recursive<decltype(callback), TReference>(callback);
            }
            else
            {
                auto callback = [&]<std::size_t I, typename T>()
                {
                    void* ptr = m_data[I];
                    if constexpr (CanRealloc<Allocator>)
                    {
                        m_allocator.realloc(&ptr, capacity*sizeof(T), m_capacity*sizeof(T));
                    }
                    else
                    {
                        T* new_data = reinterpret_cast<T*>(m_allocator.alloc(sizeof(T) * capacity));
                        m_allocator.copy(new_data, ptr, std::min(m_capacity, capacity) * sizeof(T));
                        m_allocator.free(ptr);
                        ptr = new_data;
                    }
                    m_data[I] = ptr;
                };
                iterate_struct_recursive<decltype(callback), TReference>(callback);
            }

            m_capacity=capacity;
            if (m_size>m_capacity)
                m_size=m_capacity;
        }

        void resize(size_t size)
        {
            if (m_size!=size)
            {
                if (size>m_capacity)
                {
                    reserve(size);
                }
                m_size=size;
            }
        }

        template<float extension_rate=2.f>
        size_t allocate(size_t number_of_elements=1)
        {
            static_assert(extension_rate>=1);
            auto start=m_size;
            m_size=m_size+number_of_elements;
            if (m_size>m_capacity)
            {
                reserve( static_cast<size_t>(m_size*extension_rate));
            }
            return start;
        }

        void clear()
        {
            if (m_capacity>0) {
                auto callback = [&]<std::size_t I, typename T>() {
                    m_allocator.free(m_data[I]);
                    m_data[I] = nullptr;
                };
                iterate_struct_recursive<decltype(callback), TReference>(callback);
            }
            m_size=0;
            m_capacity=0;
        }

        void shrink_to_fit()
        {
            reserve(m_size);
        }
    };
}
