#pragma once
#include "soa_cuda.hpp"
#include "soa_pool.hpp"

namespace soa
{
    template<typename View>
    using soa_cuda_device_pool=soa_pool<View,CudaDeviceAllocator>;

    template<typename View>
    using soa_cuda_host_pool=soa_pool<View,CudaHostAllocator>;

    template<typename View>
    using soa_cuda_unified_pool=soa_pool<View,CudaUnifiedAllocator>;
}
