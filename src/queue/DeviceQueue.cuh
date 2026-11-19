#pragma once
#include <cassert>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <cuda_runtime_api.h>
#include <cuda/atomic>
#include "../cuda_common.hpp"

template <typename T = uint32_t>
struct DeviceQueueView
{
    static_assert(std::is_unsigned_v<T> && (sizeof(T) == 4 || sizeof(T) == 8),
                  "DeviceQueueView requires a 32-bit or 64-bit unsigned integer type.");

    static constexpr T EMPTY_SENTINEL = static_cast<T>(~T(0));

    T* data;
    uint32_t capacity;
    uint32_t* m_size;
    uint32_t* consumed;
    uint32_t* in_flight;

    __device__ uint32_t reserve_push(uint32_t count) const
    {
        // Increment in_flight BEFORE advancing m_size so no consumer observing
        // the new tail can complete and drop in_flight to zero prematurely.
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(*in_flight)
            .fetch_add(count, cuda::memory_order_relaxed);
        const uint32_t start = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(*m_size)
                                   .fetch_add(count, cuda::memory_order_release);
        assert(start + count <= capacity);
        return start;
    }

    __device__ void publish(uint32_t index, T value) const
    {
        assert(index < capacity && value != EMPTY_SENTINEL);
        cuda::atomic_ref<T, cuda::thread_scope_device>(data[index])
            .store(value, cuda::memory_order_release);
    }

    __device__ uint32_t push_back(T value) const
    {
        const uint32_t idx = reserve_push(1u);
        publish(idx, value);
        return idx;
    }

    __device__ T load_wait(uint32_t index) const
    {
        assert(index < capacity);
        cuda::atomic_ref<T, cuda::thread_scope_device> slot(data[index]);
        T val = slot.load(cuda::memory_order_acquire);
        while (val == EMPTY_SENTINEL)
        {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            __nanosleep(16);
#endif
            val = slot.load(cuda::memory_order_acquire);
        }
        return val;
    }

    __device__ bool claim_batch_adaptive(
        uint32_t max_batch,
        uint32_t min_batch,
        uint32_t num_blocks,
        uint32_t& out_start,
        uint32_t& out_count) const
    {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> consumed_atomic(*consumed);
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> size_atomic(*m_size);
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> inflight_atomic(*in_flight);

        while (true)
        {
            // Load consumed before m_size so we never compare an old tail against a newer head.
            uint32_t head = consumed_atomic.load(cuda::memory_order_acquire);
            const uint32_t tail = size_atomic.load(cuda::memory_order_acquire);

            if (head < tail)
            {
                const uint32_t available = tail - head;
                const uint32_t fair_share = available / (num_blocks > 0u ? num_blocks : 1u);
                uint32_t target = (fair_share > min_batch) ? fair_share : min_batch;
                if (target > max_batch)
                {
                    target = max_batch;
                }
                const uint32_t take = (available < target) ? available : target;

                if (consumed_atomic.compare_exchange_weak(
                        head,
                        head + take,
                        cuda::memory_order_acq_rel,
                        cuda::memory_order_relaxed))
                {
                    out_start = head;
                    out_count = take;
                    return true;
                }
                continue;
            }

            if (inflight_atomic.load(cuda::memory_order_acquire) == 0u)
            {
                out_start = 0u;
                out_count = 0u;
                return false;
            }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            __nanosleep(32);
#endif
        }
    }

    __device__ void complete_batch(uint32_t count) const
    {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(*in_flight)
            .fetch_sub(count, cuda::memory_order_acq_rel);
    }

    __device__ T& operator[](uint32_t i) const
    {
        return data[i];
    }
};

template <typename T = uint32_t>
class DeviceQueue
{
public:
    explicit DeviceQueue(uint32_t capacity)
        : m_capacity(capacity)
    {
        allocate_storage();
        reset();
    }

    DeviceQueue(uint32_t capacity, T initial_item)
        : m_capacity(capacity)
    {
        allocate_storage();
        reset_with_initial(initial_item);
    }

    DeviceQueue(const DeviceQueue&) = delete;
    DeviceQueue& operator=(const DeviceQueue&) = delete;

    DeviceQueue(DeviceQueue&& other) noexcept
        : data(std::exchange(other.data, nullptr))
        , size_d(std::exchange(other.size_d, nullptr))
        , consumed_d(std::exchange(other.consumed_d, nullptr))
        , in_flight_d(std::exchange(other.in_flight_d, nullptr))
        , m_capacity(std::exchange(other.m_capacity, 0u))
    {
    }

    DeviceQueue& operator=(DeviceQueue&& other) noexcept
    {
        if (this != &other)
        {
            free_storage();
            data = std::exchange(other.data, nullptr);
            size_d = std::exchange(other.size_d, nullptr);
            consumed_d = std::exchange(other.consumed_d, nullptr);
            in_flight_d = std::exchange(other.in_flight_d, nullptr);
            m_capacity = std::exchange(other.m_capacity, 0u);
        }
        return *this;
    }

    ~DeviceQueue()
    {
        free_storage();
    }

    void reset()
    {
        if (m_capacity > 0u && data != nullptr)
        {
            CUDA_CHECK(cudaMemset(data, 0xFF, static_cast<size_t>(m_capacity) * sizeof(T)));
        }
        CUDA_CHECK(cudaMemset(size_d, 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(consumed_d, 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(in_flight_d, 0, sizeof(uint32_t)));
    }

    void reset_with_initial(const T* host_items, uint32_t count)
    {
        reset();
        assert(count <= m_capacity);
        if (count > 0u)
        {
            for (uint32_t i = 0u; i < count; ++i)
            {
                assert(host_items[i] != DeviceQueueView<T>::EMPTY_SENTINEL);
            }
            CUDA_CHECK(cudaMemcpy(data, host_items, static_cast<size_t>(count) * sizeof(T), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(size_d, &count, sizeof(uint32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(in_flight_d, &count, sizeof(uint32_t), cudaMemcpyHostToDevice));
        }
    }

    void reset_with_initial(T item)
    {
        reset_with_initial(&item, 1u);
    }

    DeviceQueueView<T> view() const
    {
        return DeviceQueueView<T>{
            .data = data,
            .capacity = m_capacity,
            .m_size = size_d,
            .consumed = consumed_d,
            .in_flight = in_flight_d
        };
    }

    uint32_t capacity() const { return m_capacity; }

private:
    void allocate_storage()
    {
        if (m_capacity > 0u)
        {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), static_cast<size_t>(m_capacity) * sizeof(T)));
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&size_d), sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&consumed_d), sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&in_flight_d), sizeof(uint32_t)));
    }

    void free_storage()
    {
        if (data != nullptr)
        {
            cudaFree(data);
            data = nullptr;
        }
        if (size_d != nullptr)
        {
            cudaFree(size_d);
            size_d = nullptr;
        }
        if (consumed_d != nullptr)
        {
            cudaFree(consumed_d);
            consumed_d = nullptr;
        }
        if (in_flight_d != nullptr)
        {
            cudaFree(in_flight_d);
            in_flight_d = nullptr;
        }
    }

    T* data = nullptr;
    uint32_t* size_d = nullptr;
    uint32_t* consumed_d = nullptr;
    uint32_t* in_flight_d = nullptr;
    uint32_t m_capacity = 0u;
};
