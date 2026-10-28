#pragma once

#include <cuda_runtime_api.h>
#include <iostream>

#include "aosoa_vector.hpp"
#include "soa_vector.hpp"

namespace soa
{
    struct CudaDeviceAllocator
    {
        inline static  void* alloc(size_t bytes)
        {
            void *ptr;
            CUDA_CHECK(cudaMalloc(&ptr,bytes));
            return ptr;
        }

        inline static void free(void* ptr)
        {
            CUDA_CHECK(cudaFree(ptr));
        }

        inline static void copy(void* dst, const void* src, size_t bytes)
        {
            CUDA_CHECK(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToDevice));
        }
    };


    struct CudaHostAllocator
    {
        inline static  void* alloc(size_t bytes)
        {
            void *ptr;
            CUDA_CHECK(cudaMallocHost(&ptr,bytes));
            return ptr;
        }

        inline static void free(void* ptr)
        {
            CUDA_CHECK(cudaFreeHost(ptr));
        }

        inline static void copy(void* dst, const void* src, size_t bytes)
        {
            CUDA_CHECK(cudaMemcpy(dst,src,bytes,cudaMemcpyHostToHost));
        }
    };


    struct CudaUnifiedAllocator
    {
        inline static  void* alloc(size_t bytes)
        {
            void *ptr;
            CUDA_CHECK(cudaMallocManaged(&ptr,bytes));

            int current_dev;
            CUDA_CHECK(cudaGetDevice(&current_dev));

            // Check if the device supports Concurrent Managed Access (Page Migration / Advise)
            int concurrentManagedAccess = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&concurrentManagedAccess, cudaDevAttrConcurrentManagedAccess, current_dev));

            if (concurrentManagedAccess)
            {
                // 1. Hint to keep the physical memory on the GPU
                cudaMemLocation deviceLoc = {.type = cudaMemLocationTypeDevice, .id = current_dev};
                CUDA_CHECK(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, deviceLoc));

                // 2. Hint to set up direct CPU mapping to avoid page migrations upon access
                cudaMemLocation hostLoc = {.type = cudaMemLocationTypeHost, .id = 0};
                CUDA_CHECK(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, hostLoc));

                // // (Optional) Proactively move the memory to the GPU before the kernel starts
                // cudaMemPrefetchAsync(ptr, bytes, deviceLoc, 0, stream);
            }

            return ptr;
        }

        inline static void free(void* ptr)
        {
            CUDA_CHECK(cudaFree(ptr));
        }

        inline static void copy(void* dst, const void* src, size_t bytes)
        {
            CUDA_CHECK(cudaMemcpy(dst,src,bytes,cudaMemcpyDefault));
        }
    };


    //transfer form device to host
    inline void transfer_data(void* ptr_dst, const void* ptr_src, size_t bytes, DefaultAllocator allocatorDst, CudaDeviceAllocator allocatorSrc)
    {
        CUDA_CHECK(cudaMemcpy(ptr_dst,ptr_src,bytes,cudaMemcpyDeviceToHost));
    }

    //transfer form host to device
    inline void transfer_data(void* ptr_dst,const void* ptr_src, size_t bytes,CudaDeviceAllocator  allocatorDst, DefaultAllocator allocatorSrc)
    {
        CUDA_CHECK(cudaMemcpy(ptr_dst,ptr_src,bytes,cudaMemcpyHostToDevice));
    }


    template<typename View>
    using cuda_device_vector=soa_vector<View,CudaDeviceAllocator>;

    template<typename View>
    using cuda_host_vector=soa_vector<View,CudaHostAllocator>;

    template<typename View>
    using cuda_unified_vector=soa_vector<View,CudaUnifiedAllocator>;


    template<typename View,size_t internal_array_size=32>
    using cuda_device_aosoa = aosoa_vector<View,internal_array_size, CudaDeviceAllocator>;

    template<typename View,size_t internal_array_size=32>
    using cuda_host_aosoa = aosoa_vector<View,internal_array_size, CudaHostAllocator>;

    template<typename View,size_t internal_array_size=32>
    using cuda_unified_aosoa = aosoa_vector<View,internal_array_size, CudaUnifiedAllocator>;

}