#pragma once
#include <cstdint>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <cuda/atomic>

struct  DeviceQueueView
{
    using T= uint64_t;

    T* data;
    uint32_t capacity;
    uint32_t* const m_size;
    uint32_t* const consumed;

    __device__ uint32_t push_back(T value)
    {
        cuda::atomic_ref<uint32_t,cuda::thread_scope_device> size_atomic(*m_size);
        auto start=size_atomic.fetch_add(1,cuda::memory_order_relaxed);
        data[start]=value;
        return start;
    }

    __device__ uint32_t allocate_thread(uint32_t size)
    {
        cuda::atomic_ref<uint32_t,cuda::thread_scope_device> size_atomic(*m_size);
        auto start=size_atomic.fetch_add(1,cuda::memory_order_relaxed);

        return start;
    }

    __device__ uint32_t allocate_warp(uint32_t size)
    {
        cuda::atomic_ref<uint32_t,cuda::thread_scope_device> size_atomic(*m_size);
        auto start=size_atomic.fetch_add(1,cuda::memory_order_relaxed);

        return start;
    }


    __device__ uint32_t allocate_block(uint32_t size)
    {
        cuda::atomic_ref<uint32_t,cuda::thread_scope_device> size_atomic(*m_size);
        auto start=size_atomic.fetch_add(1,cuda::memory_order_relaxed);

        return start;
    }


    __device__ uint32_t allocate_device(uint32_t size)
    {
        cuda::atomic_ref<uint32_t,cuda::thread_scope_device> size_atomic(*m_size);
        auto start=size_atomic.fetch_add(1,cuda::memory_order_relaxed);

        return start;
    }

    __device__ uint32_t consume(uint32_t size)
    {
        cuda::atomic_ref<uint32_t,cuda::thread_scope_device> size_atomic(*consumed);
        auto start=size_atomic.fetch_add(size,cuda::memory_order_acquire);
        return start;
    }

    __device__ void reset()
    {
        *consumed=0;
        *m_size=0;
    }

    T& operator[](uint32_t i)
    {
        return data[i];
    }
};


class DeviceQueue
{
    using T= uint64_t;
    T* data;
    uint32_t* size_d;
    uint32_t* consumed_d;
    uint32_t m_capacity;

    DeviceQueue(size_t capacity):m_capacity(capacity)
    {
        cudaMalloc(reinterpret_cast<void**>(&data),m_capacity*sizeof(T));

        cudaMalloc((void**)&size_d,sizeof(*size_d));
        cudaMemset(size_d,0,sizeof(*size_d));

        cudaMalloc((void**)&consumed_d,sizeof(*consumed_d));
        cudaMemset(consumed_d,0,sizeof(*consumed_d));
    }

    DeviceQueueView view()
    {
        return DeviceQueueView{
            .data = data,
            .capacity = m_capacity,
            .m_size = size_d,
            .consumed=consumed_d
        };
    }

    ~DeviceQueue()
    {
        cudaFree(data);
        cudaFree(size_d);
        cudaFree(consumed_d);
    }
};
