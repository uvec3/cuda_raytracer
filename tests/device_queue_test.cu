#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "../src/cuda_common.hpp"
#include "../src/queue/DeviceQueue.cuh"
#include "../src/cuda/scan.cuh"

namespace
{
    __host__ __device__ inline uint64_t mix64(uint64_t x)
    {
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return x;
    }

    __host__ __device__ inline uint64_t compute_payload(uint32_t depth, uint64_t path_code)
    {
        return mix64(path_code ^ (static_cast<uint64_t>(depth) * 0x9e3779b97f4a7c15ULL)) ^ 0x123456789abcdef0ULL;
    }

    __host__ __device__ inline bool should_subdivide_full(uint32_t depth, uint64_t /*path_code*/, uint32_t max_depth)
    {
        return depth < max_depth;
    }

    __host__ __device__ inline bool should_subdivide_irregular(uint32_t depth, uint64_t path_code, uint32_t max_depth)
    {
        if (depth >= max_depth)
        {
            return false;
        }
        if (depth < 3u)
        {
            return true;
        }
        const uint64_t h = mix64(path_code ^ (static_cast<uint64_t>(depth) << 32));
        return (h % 100ULL) < 18ULL;
    }

    uint32_t count_expected_nodes(bool irregular, uint32_t max_depth)
    {
        struct Frame
        {
            uint32_t depth;
            uint64_t path;
        };
        std::vector<Frame> stack;
        stack.push_back({0u, 0ULL});
        uint32_t count = 1u;
        while (!stack.empty())
        {
            const Frame f = stack.back();
            stack.pop_back();
            const bool sub = irregular
                ? should_subdivide_irregular(f.depth, f.path, max_depth)
                : should_subdivide_full(f.depth, f.path, max_depth);
            if (sub)
            {
                count += 8u;
                for (uint32_t b = 0u; b < 8u; ++b)
                {
                    stack.push_back({f.depth + 1u, (f.path << 3) | b});
                }
            }
        }
        return count;
    }

    __global__ void expand_test_tree_kernel(
        DeviceQueueView<uint32_t> queue,
        uint32_t* node_depth,
        uint64_t* node_path,
        uint64_t* node_payload,
        uint32_t* visit_counts,
        uint32_t* node_alloc_counter,
        uint32_t max_nodes,
        uint32_t max_depth,
        bool irregular,
        uint32_t* error_count)
    {
        __shared__ uint32_t s_batch_start;
        __shared__ uint32_t s_batch_count;
        __shared__ uint32_t s_node_base;
        __shared__ uint32_t s_queue_base;
        __shared__ bool s_has_work;

        constexpr uint32_t MIN_BATCH = 32u;

        while (true)
        {
            if (threadIdx.x == 0)
            {
                s_has_work = queue.claim_batch_adaptive(
                    blockDim.x, MIN_BATCH, gridDim.x, s_batch_start, s_batch_count);
            }
            __syncthreads();

            if (!s_has_work)
            {
                return;
            }

            const uint32_t batch_count = s_batch_count;
            const bool in_range = (threadIdx.x < batch_count);

            uint32_t node_id = 0u;
            uint32_t depth = 0u;
            uint64_t path = 0ULL;
            bool subdivides = false;

            if (in_range)
            {
                node_id = queue.load_wait(s_batch_start + threadIdx.x);
                if (node_id >= max_nodes)
                {
                    atomicAdd(error_count, 1u);
                }
                else
                {
                    atomicAdd(&visit_counts[node_id], 1u);
                    depth = node_depth[node_id];
                    path = node_path[node_id];
                    const uint64_t expected_payload = compute_payload(depth, path);
                    if (node_payload[node_id] != expected_payload)
                    {
                        atomicAdd(error_count, 1u);
                    }
                    subdivides = irregular
                        ? should_subdivide_irregular(depth, path, max_depth)
                        : should_subdivide_full(depth, path, max_depth);
                }
            }

            const int my_sub = subdivides ? 1 : 0;
            const int inc_sub = block_scan_kogge_stone(my_sub);
            const int exc_sub = inc_sub - my_sub;

            if (threadIdx.x == blockDim.x - 1)
            {
                const uint32_t total_sub = static_cast<uint32_t>(inc_sub);
                if (total_sub > 0u)
                {
                    s_node_base = atomicAdd(node_alloc_counter, total_sub * 8u);
                    assert(s_node_base + total_sub * 8u <= max_nodes);
                    s_queue_base = queue.reserve_push(total_sub * 8u);
                }
            }
            __syncthreads();

            if (in_range && subdivides)
            {
                const uint32_t child_start = s_node_base + static_cast<uint32_t>(exc_sub) * 8u;
                const uint32_t q_start = s_queue_base + static_cast<uint32_t>(exc_sub) * 8u;
                const uint32_t child_depth = depth + 1u;

                for (uint32_t b = 0u; b < 8u; ++b)
                {
                    const uint32_t child_id = child_start + b;
                    const uint64_t child_path = (path << 3) | b;
                    node_depth[child_id] = child_depth;
                    node_path[child_id] = child_path;
                    node_payload[child_id] = compute_payload(child_depth, child_path);
                }

                for (uint32_t b = 0u; b < 8u; ++b)
                {
                    queue.publish(q_start + b, child_start + b);
                }
            }

            __syncthreads();

            if (threadIdx.x == 0)
            {
                queue.complete_batch(batch_count);
            }
        }
    }

    uint32_t get_persistent_block_count(uint32_t block_size)
    {
        int device_id = 0, num_sms = 0, max_blocks_per_sm = 0;
        CUDA_CHECK(cudaGetDevice(&device_id));
        CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, expand_test_tree_kernel, static_cast<int>(block_size), 0));
        return static_cast<uint32_t>(std::max(1, num_sms * max_blocks_per_sm));
    }

    void run_tree_stress_once(
        DeviceQueue<uint32_t>& queue,
        uint32_t max_nodes,
        uint32_t max_depth,
        bool irregular)
    {
        constexpr uint32_t block_size = 256u;
        const uint32_t num_blocks = get_persistent_block_count(block_size);
        const uint32_t expected_nodes = count_expected_nodes(irregular, max_depth);
        ASSERT_LE(expected_nodes, max_nodes);

        uint32_t* d_depth = nullptr;
        uint64_t* d_path = nullptr;
        uint64_t* d_payload = nullptr;
        uint32_t* d_visits = nullptr;
        uint32_t* d_alloc_counter = nullptr;
        uint32_t* d_error_count = nullptr;

        CUDA_CHECK(cudaMalloc(&d_depth, static_cast<size_t>(max_nodes) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_path, static_cast<size_t>(max_nodes) * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_payload, static_cast<size_t>(max_nodes) * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_visits, static_cast<size_t>(max_nodes) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_alloc_counter, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_error_count, sizeof(uint32_t)));

        CUDA_CHECK(cudaMemset(d_depth, 0, static_cast<size_t>(max_nodes) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_path, 0, static_cast<size_t>(max_nodes) * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(d_payload, 0, static_cast<size_t>(max_nodes) * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(d_visits, 0, static_cast<size_t>(max_nodes) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_error_count, 0, sizeof(uint32_t)));

        const uint32_t init_alloc = 1u;
        const uint64_t root_payload = compute_payload(0u, 0ULL);
        CUDA_CHECK(cudaMemcpy(d_alloc_counter, &init_alloc, sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_payload, &root_payload, sizeof(uint64_t), cudaMemcpyHostToDevice));

        queue.reset_with_initial(0u);

        expand_test_tree_kernel<<<num_blocks, block_size>>>(
            queue.view(), d_depth, d_path, d_payload, d_visits,
            d_alloc_counter, max_nodes, max_depth, irregular, d_error_count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        uint32_t total_allocated = 0u, error_count = 0u;
        uint32_t q_size = 0u, q_consumed = 0u, q_in_flight = 0u;
        CUDA_CHECK(cudaMemcpy(&total_allocated, d_alloc_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&error_count, d_error_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        const auto qview = queue.view();
        CUDA_CHECK(cudaMemcpy(&q_size, qview.m_size, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&q_consumed, qview.consumed, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&q_in_flight, qview.in_flight, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        EXPECT_EQ(error_count, 0u);
        EXPECT_EQ(total_allocated, expected_nodes);
        EXPECT_EQ(q_size, expected_nodes);
        EXPECT_EQ(q_consumed, expected_nodes);
        EXPECT_EQ(q_in_flight, 0u);

        std::vector<uint32_t> host_visits(max_nodes, 0u);
        CUDA_CHECK(cudaMemcpy(host_visits.data(), d_visits, static_cast<size_t>(max_nodes) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for (uint32_t i = 0u; i < expected_nodes; ++i)
        {
            ASSERT_EQ(host_visits[i], 1u) << "Node " << i << " visit count mismatch";
        }
        for (uint32_t i = expected_nodes; i < max_nodes; ++i)
        {
            ASSERT_EQ(host_visits[i], 0u) << "Unexpected visit beyond allocated range at " << i;
        }

        cudaFree(d_depth);
        cudaFree(d_path);
        cudaFree(d_payload);
        cudaFree(d_visits);
        cudaFree(d_alloc_counter);
        cudaFree(d_error_count);
    }
} // namespace

TEST(DeviceQueue, SingleRootNoChildren)
{
    DeviceQueue<uint32_t> queue(256u);
    run_tree_stress_once(queue, 64u, 0u, false);
}

TEST(DeviceQueue, MultiBlockOctreeExpansionStress)
{
    // Full 8-ary tree to depth 5: 1 + 8 + 64 + 512 + 4096 + 32768 = 37,449 nodes.
    {
        constexpr uint32_t max_nodes = 40000u;
        DeviceQueue<uint32_t> queue(max_nodes);
        run_tree_stress_once(queue, max_nodes, 5u, false);
    }

    // Irregular sparse tree to depth 7.
    {
        constexpr uint32_t max_nodes = 200000u;
        DeviceQueue<uint32_t> queue(max_nodes);
        run_tree_stress_once(queue, max_nodes, 7u, true);
    }
}

TEST(DeviceQueue, RepeatedRunsNonDeterministicStress)
{
    constexpr uint32_t max_nodes = 200000u;
    DeviceQueue<uint32_t> queue(max_nodes);
    for (int run = 0; run < 10; ++run)
    {
        SCOPED_TRACE(::testing::Message() << "Run #" << run);
        run_tree_stress_once(queue, max_nodes, 7u, true);
    }
}
