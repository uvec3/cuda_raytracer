#pragma once
#include "soa_pool.hpp"
#include "../../external/soa_vector/src/soa_cuda.hpp"

namespace soa
{
    template<typename View>
    using soa_cuda_device_pool=soa_pool<View,CudaDeviceAllocator>;

    template<typename View>
    using soa_cuda_host_pool=soa_pool<View,CudaHostAllocator>;

    template<typename View>
    using soa_cuda_unified_pool=soa_pool<View,CudaUnifiedAllocator>;
}
