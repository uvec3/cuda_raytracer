#pragma once
#include "soa_pool.hpp"
#include "../../external/soa_vector/src/soa_cuda.hpp"

namespace soa
{
    template<typename View>
    using soa_cuda_device_pool=soa_pool_generic<cuda_soa_vector<View,CudaDeviceAllocator>>;

    template<typename View>
    using soa_cuda_host_pool=soa_pool_generic<cuda_soa_vector<View,CudaHostAllocator>>;

    template<typename View>
    using soa_cuda_unified_pool=soa_pool_generic<cuda_soa_vector<View,CudaUnifiedAllocator>>;

    template <typename TReference, typename HostSoa>
    inline void upload_soa_range_async(
        cuda_device_vector<TReference>& dst,
        const HostSoa& src,
        size_t offset,
        size_t count,
        cudaStream_t stream = 0)
    {
        if (count == 0)
            return;
        if (dst.size() < src.size())
            dst.resize(src.size());
        const auto dst_view = dst.view(offset, count);
        const auto src_view = src.view(offset, count);
        using FlatTuple = decltype(detail::tie_flat(std::declval<TReference&>()));
        constexpr std::size_t N = std::tuple_size_v<FlatTuple>;
        auto copy_field = [&](void* dst_ptr, const void* src_ptr, size_t bytes)
        {
            CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyHostToDevice, stream));
        };
        [&]<std::size_t... Is>(std::index_sequence<Is...>)
        {
            (copy_field(
                dst_view.m_data[Is],
                src_view.m_data[Is],
                count * sizeof(std::remove_reference_t<std::tuple_element_t<Is, FlatTuple>>)), ...);
        }(std::make_index_sequence<N>{});
    }
}
