#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <algorithm>
#include "../src/cuda_common.hpp"
#include "../src/soa_cuda.cuh"
#include "../src/soa/src/soa_pool.hpp"


struct View
{
    float *a;
    float *b;
    float *c;
};

TEST(SOA_TEST, Host_Basic)
{
    const int initial_size=250;
    soa_cuda_host_vector<View> soa{initial_size};
    EXPECT_EQ(soa.size(),initial_size);
    EXPECT_TRUE(soa.capacity()>=initial_size);


    for (int i=0;i<soa.size();++i)
    {
        soa.data().a[i]=i;
        soa.data().b[i]=i+0.5;
    }

    soa.resize(45);
    EXPECT_EQ(soa.size(),45);
    EXPECT_TRUE(soa.capacity()>=45);

    for (int i=0;i<soa.size();++i)
    {
        EXPECT_EQ(soa.data().a[i],i);
        EXPECT_EQ(soa.data().b[i],i+0.5);
    }

    //Set capacity
    soa.set_capacity(55);
    EXPECT_EQ(soa.size(),45);
    EXPECT_TRUE(soa.capacity()==55);

    //Expand with resize
    soa.resize(300);
    EXPECT_EQ(soa.size(),300);
    EXPECT_TRUE(soa.capacity()>=300);

    //view access
    for (int i=0;i<soa.size();++i)
    {
        auto element=soa[i];
        element[&View::c]=element[&View::a] + element[&View::b];
    }

    //old values are preserved
    for (int i=0;i<45;++i)
    {
        EXPECT_EQ(soa.data().a[i],i);
        EXPECT_EQ(soa.data().b[i],i+0.5);

        //printf("%f + %f = %f\n",soa.data().a[i],soa.data().b[i],soa.data().c[i]);
    }

    //addion performed correctly
    for (int i=0;i<soa.size();++i)
    {
        EXPECT_EQ(soa.data().c[i],soa.data().a[i]+soa.data().b[i]);
        //printf("%f + %f = %f\n",soa.data().a[i],soa.data().b[i],soa.data().c[i]);
    }
}


TEST(SOA_TEST, Unified_Basic)
{
    printf("Unified test\n");
    const int initial_size=250;
    soa_cuda_unified_vector<View> soa{initial_size};
    EXPECT_EQ(soa.size(),initial_size);
    EXPECT_TRUE(soa.capacity()>=initial_size);


    for (int i=0;i<soa.size();++i)
    {
        soa.data().a[i]=i;
        soa.data().b[i]=i+0.5;
    }

    soa.resize(45);
    EXPECT_EQ(soa.size(),45);
    EXPECT_TRUE(soa.capacity()>=45);

    for (int i=0;i<soa.size();++i)
    {
        EXPECT_EQ(soa.data().a[i],i);
        EXPECT_EQ(soa.data().b[i],i+0.5);
    }

    //Set capacity
    soa.set_capacity(55);
    EXPECT_EQ(soa.size(),45);
    EXPECT_TRUE(soa.capacity()==55);

    //Expand with resize
    soa.resize(300);
    EXPECT_EQ(soa.size(),300);
    EXPECT_TRUE(soa.capacity()>=300);

    //view access
    for (int i=0;i<soa.size();++i)
    {
        auto element=soa[i];
        element[&View::c]=element[&View::a] + element[&View::b];
    }

    //old values are preserved
    for (int i=0;i<45;++i)
    {
        EXPECT_EQ(soa.data().a[i],i);
        EXPECT_EQ(soa.data().b[i],i+0.5);

        //printf("%f + %f = %f\n",soa.data().a[i],soa.data().b[i],soa.data().c[i]);
    }

    //addition performed correctly
    for (int i=0;i<soa.size();++i)
    {
        EXPECT_EQ(soa.data().c[i],soa.data().a[i]+soa.data().b[i]);
        //printf("%f + %f = %f\n",soa.data().a[i],soa.data().b[i],soa.data().c[i]);
    }

    soa.clear();
}

__global__ void kernelAdd(View view,uint32_t size)
{
    auto i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<size)
    {
        view.c[i]=view.a[i]+view.b[i];
    }
}

TEST(SOA_TEST, Unified_Kernel_Add)
{
    printf("Unified test\n");
    const int initial_size=250;
    soa_cuda_unified_vector<View> soa{initial_size};

    for (int i=0;i<soa.size();++i)
    {
        soa.data().a[i]=i;
        soa.data().b[i]=i+0.5;
    }

    kernelAdd<<<soa.size(),1>>>(soa.data(),soa.size());

    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i=0;i<soa.size();++i)
    {
        EXPECT_EQ(soa.data().c[i],soa.data().a[i]+soa.data().b[i]);
        printf("%f + %f = %f\n",soa.data().a[i],soa.data().b[i],soa.data().c[i]);
    }

    soa.resize(500);
    for (int i=0;i<soa.size();++i)
    {
        // EXPECT_EQ(soa.data().c[i],soa.data().a[i]+soa.data().b[i]);
        printf("%f + %f = %f\n",soa.data().a[i],soa.data().b[i],soa.data().c[i]);
    }
}


__global__ void kernelAdd(soa_view<View> view)
{
    auto i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<view.size())
    {
        view.c[i]=view.a[i]+view.b[i];
    }
}

TEST(SOA_TEST, Unified_Kernel_Add_View)
{
    const int initial_size=250;
    soa_cuda_unified_vector<View> soa{initial_size};

    for (int i=0;i<soa.size();++i)
    {
        soa.data().a[i]=i;
        soa.data().b[i]=i+0.5;
    }

    kernelAdd<<<soa.size(),1>>>(soa.view());

    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i=0;i<soa.size();++i)
    {
        EXPECT_EQ(soa.data().c[i],soa.data().a[i]+soa.data().b[i]);
        printf("%f + %f = %f\n",soa.data().a[i],soa.data().b[i],soa.data().c[i]);
    }

    soa.resize(500);
    for (int i=0;i<soa.size();++i)
    {
        // EXPECT_EQ(soa.data().c[i],soa.data().a[i]+soa.data().b[i]);
        printf("v %f + %f = %f\n",soa.data().a[i],soa.data().b[i],soa.data().c[i]);
    }
}



TEST(SOA_TEST, Allocation)
{
    soa_cuda_unified_vector<View> soa;
    EXPECT_EQ(soa.size(),0);
    EXPECT_EQ(soa.capacity(),0);

    auto start=soa.allocate_range<2.f>(10);
    EXPECT_EQ(soa.size(),10);
    EXPECT_EQ(soa.capacity(),20);
    EXPECT_EQ(start,0);

    start=soa.allocate_range<2.f>(10);
    EXPECT_EQ(soa.size(),20);
    EXPECT_EQ(soa.capacity(),20);
    EXPECT_EQ(start,10);

    auto element= soa.allocate_element<2.f>();
    EXPECT_EQ(soa.size(),21);
    EXPECT_EQ(soa.capacity(),42);
    EXPECT_EQ(element.index(),20);

    element = soa.allocate_element<2.f>();
    EXPECT_EQ(soa.size(),22);
    EXPECT_EQ(soa.capacity(),42);
    EXPECT_EQ(element.index(),21);


    soa.allocate_range<3.f>(10000);
    EXPECT_EQ(soa.size(),10022);
    EXPECT_EQ(soa.capacity(),10022*3);

    //try to write into memory
    for (int i=0;i<soa.size();++i)
    {
        soa.data().a[i]=i;
        soa.data().b[i]=i+0.5;
    }

    //pass through device and back
    kernelAdd<<<soa.size(),1>>>(soa.view());
    CUDA_CHECK(cudaDeviceSynchronize());

    //check result on host
    for (int i=0;i<soa.size();++i)
    {
        EXPECT_EQ(soa.data().c[i],soa.data().a[i]+soa.data().b[i]);
    }

    soa.clear();
    EXPECT_EQ(soa.size(),0);
    EXPECT_EQ(soa.capacity(),0);
}


// ============================================================================
// SOA POOL TESTS
// ============================================================================

namespace
{
    struct TestChunk
    {
        size_t offset = 0;
        size_t size = 0;
        float seed = 0.0f;
    };

    bool chunks_overlap(size_t off1, size_t sz1, size_t off2, size_t sz2)
    {
        return !(off1 + sz1 <= off2 || off2 + sz2 <= off1);
    }

    void verify_no_chunk_overlaps(const std::vector<TestChunk>& chunks)
    {
        for (size_t i = 0; i < chunks.size(); ++i)
        {
            for (size_t j = i + 1; j < chunks.size(); ++j)
            {
                EXPECT_FALSE(chunks_overlap(chunks[i].offset, chunks[i].size, chunks[j].offset, chunks[j].size))
                    << "Chunk " << i << " [" << chunks[i].offset << ", " << chunks[i].offset + chunks[i].size
                    << ") overlaps with chunk " << j << " [" << chunks[j].offset << ", "
                    << chunks[j].offset + chunks[j].size << ")";
            }
        }
    }

    template<typename Pool>
    void fill_chunk(Pool& pool, const TestChunk& chunk)
    {
        for (size_t i = 0; i < chunk.size; ++i)
        {
            size_t idx = chunk.offset + i;
            pool.data().a[idx] = chunk.seed + static_cast<float>(i) * 1.5f;
            pool.data().b[idx] = chunk.seed * 2.0f + static_cast<float>(i) * 0.25f;
            pool.data().c[idx] = pool.data().a[idx] + pool.data().b[idx];
        }
    }

    template<typename Pool>
    void verify_chunk(const Pool& pool, const TestChunk& chunk)
    {
        for (size_t i = 0; i < chunk.size; ++i)
        {
            size_t idx = chunk.offset + i;
            float expected_a = chunk.seed + static_cast<float>(i) * 1.5f;
            float expected_b = chunk.seed * 2.0f + static_cast<float>(i) * 0.25f;
            float expected_c = expected_a + expected_b;

            EXPECT_FLOAT_EQ(pool.data().a[idx], expected_a)
                << "Mismatch in a at chunk offset " << chunk.offset << " index " << i;
            EXPECT_FLOAT_EQ(pool.data().b[idx], expected_b)
                << "Mismatch in b at chunk offset " << chunk.offset << " index " << i;
            EXPECT_FLOAT_EQ(pool.data().c[idx], expected_c)
                << "Mismatch in c at chunk offset " << chunk.offset << " index " << i;
        }
    }
}

TEST(SOA_POOL, BasicAllocFreeAndIntegrity)
{
    soa_pool<View> pool(256);
    EXPECT_GE(pool.capacity(), 256);

    const std::vector<size_t> chunk_sizes = {7, 13, 1, 64, 29, 3, 100, 17};
    std::vector<TestChunk> chunks;

    for (size_t i = 0; i < chunk_sizes.size(); ++i)
    {
        size_t off = pool.alloc(chunk_sizes[i]);
        TestChunk ch{off, chunk_sizes[i], static_cast<float>((i + 1) * 100)};
        fill_chunk(pool, ch);
        chunks.push_back(ch);
    }

    verify_no_chunk_overlaps(chunks);

    for (const auto& ch : chunks)
    {
        verify_chunk(pool, ch);
    }

    // Deallocate odd-indexed chunks in non-trivial manner (creating holes)
    std::vector<TestChunk> surviving_chunks;
    for (size_t i = 0; i < chunks.size(); ++i)
    {
        if (i % 2 == 1)
        {
            pool.free(chunks[i].offset);
        }
        else
        {
            surviving_chunks.push_back(chunks[i]);
        }
    }

    // Verify surviving chunks remain completely untouched and uncorrupted
    for (const auto& ch : surviving_chunks)
    {
        verify_chunk(pool, ch);
    }

    // Allocate new chunks of different sizes into the fragmented space
    const std::vector<size_t> new_sizes = {10, 5, 25, 8};
    for (size_t i = 0; i < new_sizes.size(); ++i)
    {
        size_t off = pool.alloc(new_sizes[i]);
        TestChunk new_ch{off, new_sizes[i], static_cast<float>((i + 50) * 10)};
        fill_chunk(pool, new_ch);
        surviving_chunks.push_back(new_ch);
    }

    verify_no_chunk_overlaps(surviving_chunks);

    // Verify all active chunks (both original surviving and newly allocated)
    for (const auto& ch : surviving_chunks)
    {
        verify_chunk(pool, ch);
    }

    // Free everything
    for (const auto& ch : surviving_chunks)
    {
        pool.free(ch.offset);
    }
}

TEST(SOA_POOL, DynamicGrowthAcrossExpansions)
{
    // Start with a small initial pool size of 16 to force multiple auto-expansions
    soa_pool<View, DefaultAllocator, 2> pool(16);
    EXPECT_GE(pool.capacity(), 16);

    std::vector<TestChunk> active_chunks;

    // Chunk 0 fits in initial pool (size 10 <= 16)
    size_t off0 = pool.alloc(10);
    TestChunk ch0{off0, 10, 10.0f};
    fill_chunk(pool, ch0);
    active_chunks.push_back(ch0);

    // Chunk 1 exceeds remaining space (6 left, need 20) -> triggers 1st auto-expansion
    size_t off1 = pool.alloc(20);
    EXPECT_GT(pool.capacity(), 16);
    TestChunk ch1{off1, 20, 20.0f};
    fill_chunk(pool, ch1);
    active_chunks.push_back(ch1);

    // Verify ch0 was safely copied to the new buffer during resize
    verify_chunk(pool, ch0);
    verify_chunk(pool, ch1);

    // Chunk 2 triggers 2nd auto-expansion
    size_t off2 = pool.alloc(80);
    TestChunk ch2{off2, 80, 80.0f};
    fill_chunk(pool, ch2);
    active_chunks.push_back(ch2);

    // Chunk 3 triggers 3rd auto-expansion
    size_t off3 = pool.alloc(300);
    TestChunk ch3{off3, 300, 300.0f};
    fill_chunk(pool, ch3);
    active_chunks.push_back(ch3);

    verify_no_chunk_overlaps(active_chunks);

    // Verify all previous chunks preserved across all 3 auto-expansions
    for (const auto& ch : active_chunks)
    {
        verify_chunk(pool, ch);
    }

    // Free chunks in non-sequential order: free 1 and 2, keep 0 and 3
    pool.free(active_chunks[1].offset);
    pool.free(active_chunks[2].offset);

    verify_chunk(pool, active_chunks[0]);
    verify_chunk(pool, active_chunks[3]);

    // Allocate a new chunk that can reuse the freed space
    size_t off4 = pool.alloc(50);
    TestChunk ch4{off4, 50, 500.0f};
    fill_chunk(pool, ch4);

    verify_chunk(pool, active_chunks[0]);
    verify_chunk(pool, active_chunks[3]);
    verify_chunk(pool, ch4);

    pool.free(active_chunks[0].offset);
    pool.free(active_chunks[3].offset);
    pool.free(ch4.offset);
}

TEST(SOA_POOL, CoalescingAndFragmentation)
{
    // Verify that freeing adjacent chunks coalesces them, allowing a larger chunk to fit
    soa_pool<View> pool(256);

    // Allocate 5 contiguous chunks: A (16), B (32), C (16), D (32), E (16)
    size_t offA = pool.alloc(16);
    size_t offB = pool.alloc(32);
    size_t offC = pool.alloc(16);
    size_t offD = pool.alloc(32);
    size_t offE = pool.alloc(16);

    TestChunk chA{offA, 16, 1.0f};
    TestChunk chB{offB, 32, 2.0f};
    TestChunk chC{offC, 16, 3.0f};
    TestChunk chD{offD, 32, 4.0f};
    TestChunk chE{offE, 16, 5.0f};

    fill_chunk(pool, chA);
    fill_chunk(pool, chB);
    fill_chunk(pool, chC);
    fill_chunk(pool, chD);
    fill_chunk(pool, chE);

    // Free B and D to create two separated holes of size 32
    pool.free(offB);
    pool.free(offD);

    verify_chunk(pool, chA);
    verify_chunk(pool, chC);
    verify_chunk(pool, chE);

    // Allocate two 16-element chunks: should fit into the 32-element hole left by B
    size_t offB1 = pool.alloc(16);
    size_t offB2 = pool.alloc(16);
    TestChunk chB1{offB1, 16, 21.0f};
    TestChunk chB2{offB2, 16, 22.0f};
    fill_chunk(pool, chB1);
    fill_chunk(pool, chB2);

    verify_chunk(pool, chA);
    verify_chunk(pool, chB1);
    verify_chunk(pool, chB2);
    verify_chunk(pool, chC);
    verify_chunk(pool, chE);

    // Now free B2 (16), C (16). Hole D (32) was already free.
    // Adjacent regions [B2 (16)] + [C (16)] + [D (32)] coalesce into a contiguous block of 64!
    pool.free(offB2);
    pool.free(offC);

    size_t offLarge = pool.alloc(64);
    TestChunk chLarge{offLarge, 64, 99.0f};
    fill_chunk(pool, chLarge);

    // Verify all active chunks are valid and no overlaps exist
    verify_chunk(pool, chA);
    verify_chunk(pool, chB1);
    verify_chunk(pool, chE);
    verify_chunk(pool, chLarge);

    std::vector<TestChunk> remaining = {chA, chB1, chE, chLarge};
    verify_no_chunk_overlaps(remaining);

    pool.free(offA);
    pool.free(offB1);
    pool.free(offE);
    pool.free(offLarge);
}

TEST(SOA_POOL, SingleElementAllocAndProxy)
{
    soa_pool<View> pool(64);

    std::vector<size_t> single_indices;
    for (int i = 0; i < 16; ++i)
    {
        auto element = pool.alloc_element();
        size_t idx = element.index();
        single_indices.push_back(idx);

        element[&View::a] = static_cast<float>(i * 10);
        element[&View::b] = static_cast<float>(i * 10 + 5);
        element[&View::c] = element[&View::a] + element[&View::b];
    }

    // Verify through pool[] operator
    for (size_t i = 0; i < single_indices.size(); ++i)
    {
        size_t idx = single_indices[i];
        auto element = pool[idx];
        EXPECT_FLOAT_EQ(element[&View::a], static_cast<float>(i * 10));
        EXPECT_FLOAT_EQ(element[&View::b], static_cast<float>(i * 10 + 5));
        EXPECT_FLOAT_EQ(element[&View::c], static_cast<float>(i * 20 + 5));
    }

    // Free alternate single elements
    std::vector<size_t> kept_indices;
    for (size_t i = 0; i < single_indices.size(); ++i)
    {
        if (i % 2 == 1)
        {
            pool.free(single_indices[i]);
        }
        else
        {
            kept_indices.push_back(single_indices[i]);
        }
    }

    // Allocate a multi-element chunk in between
    size_t chunk_off = pool.alloc(4);
    TestChunk ch{chunk_off, 4, 777.0f};
    fill_chunk(pool, ch);

    // Verify kept single elements
    for (size_t idx : kept_indices)
    {
        auto element = pool[idx];
        EXPECT_FLOAT_EQ(element[&View::c], element[&View::a] + element[&View::b]);
    }
    verify_chunk(pool, ch);

    for (size_t idx : kept_indices)
    {
        pool.free(idx);
    }
    pool.free(chunk_off);
}

TEST(SOA_POOL, ReallocGrowthAndRelocation)
{
    soa_pool<View> pool(128);

    // Allocate 3 sequential chunks: A (16), B (32), C (24)
    size_t offA = pool.alloc(16);
    size_t offB = pool.alloc(32);
    size_t offC = pool.alloc(24);

    TestChunk chA{offA, 16, 10.0f};
    TestChunk chB{offB, 32, 20.0f};
    TestChunk chC{offC, 24, 30.0f};

    fill_chunk(pool, chA);
    fill_chunk(pool, chB);
    fill_chunk(pool, chC);

    // Realloc chunk A from 16 to 48:
    // In-place expansion is blocked by chunk B, so realloc must allocate a new chunk,
    // copy the existing 16 elements, and free the old offset.
    size_t new_offA = pool.realloc(offA, 16, 48);
    EXPECT_NE(new_offA, offA);

    // Verify the original 16 elements are preserved at new_offA
    for (size_t i = 0; i < 16; ++i)
    {
        size_t idx = new_offA + i;
        float expected_a = 10.0f + static_cast<float>(i) * 1.5f;
        float expected_b = 20.0f + static_cast<float>(i) * 0.25f;
        EXPECT_FLOAT_EQ(pool.data().a[idx], expected_a);
        EXPECT_FLOAT_EQ(pool.data().b[idx], expected_b);
    }

    // Verify chunks B and C are untouched
    verify_chunk(pool, chB);
    verify_chunk(pool, chC);

    // Write into the newly expanded portion (16..47) of chunk A
    for (size_t i = 16; i < 48; ++i)
    {
        size_t idx = new_offA + i;
        pool.data().a[idx] = 100.0f + static_cast<float>(i);
        pool.data().b[idx] = 200.0f + static_cast<float>(i);
        pool.data().c[idx] = pool.data().a[idx] + pool.data().b[idx];
    }

    // Realloc chunk B shrinking from 32 to 10
    size_t new_offB = pool.realloc(offB, 32, 10);
    // Verify prefix 10 elements of B are preserved
    for (size_t i = 0; i < 10; ++i)
    {
        size_t idx = new_offB + i;
        float expected_a = 20.0f + static_cast<float>(i) * 1.5f;
        float expected_b = 40.0f + static_cast<float>(i) * 0.25f;
        EXPECT_FLOAT_EQ(pool.data().a[idx], expected_a);
        EXPECT_FLOAT_EQ(pool.data().b[idx], expected_b);
    }

    pool.free(new_offA);
    pool.free(new_offB);
    pool.free(offC);
}

TEST(SOA_POOL, NonTrivialChurnSimulation)
{
    // Deterministic pseudo-random simulation of allocate, free, and realloc cycles
    soa_pool<View> pool(128);

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> size_dist(1, 48);
    std::uniform_int_distribution<int> op_dist(0, 9);

    std::vector<TestChunk> active_chunks;
    float next_seed = 1.0f;

    for (int step = 0; step < 120; ++step)
    {
        int op = op_dist(rng);

        if (active_chunks.empty() || op < 5)
        {
            // Allocate new chunk (50% chance or when empty)
            size_t sz = size_dist(rng);
            size_t off = pool.alloc(sz);
            TestChunk ch{off, sz, next_seed};
            next_seed += 10.0f;
            fill_chunk(pool, ch);
            active_chunks.push_back(ch);
        }
        else if (op < 8)
        {
            // Reallocate an existing chunk to a new size (30% chance)
            std::uniform_int_distribution<size_t> idx_dist(0, active_chunks.size() - 1);
            size_t pick = idx_dist(rng);
            TestChunk& target = active_chunks[pick];

            size_t new_sz = size_dist(rng);
            size_t new_off = pool.realloc(target.offset, target.size, new_sz);

            // Verify prefix data preservation
            size_t min_sz = std::min(target.size, new_sz);
            for (size_t i = 0; i < min_sz; ++i)
            {
                float exp_a = target.seed + static_cast<float>(i) * 1.5f;
                float exp_b = target.seed * 2.0f + static_cast<float>(i) * 0.25f;
                EXPECT_FLOAT_EQ(pool.data().a[new_off + i], exp_a);
                EXPECT_FLOAT_EQ(pool.data().b[new_off + i], exp_b);
            }

            target.offset = new_off;
            target.size = new_sz;
            // Re-fill to establish clean state for the whole new chunk
            fill_chunk(pool, target);
        }
        else
        {
            // Free an existing chunk (20% chance)
            std::uniform_int_distribution<size_t> idx_dist(0, active_chunks.size() - 1);
            size_t pick = idx_dist(rng);
            pool.free(active_chunks[pick].offset);
            active_chunks.erase(active_chunks.begin() + pick);
        }

        // Verify invariant: active chunks never overlap
        verify_no_chunk_overlaps(active_chunks);

        // Every 15 steps: verify data integrity of all active chunks
        if (step % 15 == 0)
        {
            for (const auto& ch : active_chunks)
            {
                verify_chunk(pool, ch);
            }
        }
    }

    // Verify all remaining chunks after churn
    for (const auto& ch : active_chunks)
    {
        verify_chunk(pool, ch);
    }

    // Free all remaining chunks in reverse order
    for (auto it = active_chunks.rbegin(); it != active_chunks.rend(); ++it)
    {
        pool.free(it->offset);
    }

    // Verify pool is reusable and can allocate a large block
    size_t final_off = pool.alloc(64);
    TestChunk final_ch{final_off, 64, 999.0f};
    fill_chunk(pool, final_ch);
    verify_chunk(pool, final_ch);
    pool.free(final_off);
}

TEST(SOA_POOL, ClearAndReuse)
{
    soa_pool<View> pool(128);

    size_t off0 = pool.alloc(20);
    size_t off1 = pool.alloc(40);
    TestChunk ch0{off0, 20, 11.0f};
    TestChunk ch1{off1, 40, 22.0f};
    fill_chunk(pool, ch0);
    fill_chunk(pool, ch1);

    verify_chunk(pool, ch0);
    verify_chunk(pool, ch1);

    // Clear the pool
    pool.clear();

    // Allocate again from cleared pool
    size_t new_off0 = pool.alloc(30);
    size_t new_off1 = pool.alloc(50);
    TestChunk new_ch0{new_off0, 30, 33.0f};
    TestChunk new_ch1{new_off1, 50, 44.0f};
    fill_chunk(pool, new_ch0);
    fill_chunk(pool, new_ch1);

    verify_chunk(pool, new_ch0);
    verify_chunk(pool, new_ch1);

    pool.free(new_off0);
    pool.free(new_off1);
}

__global__ void kernelAddChunk(soa_view<View> view, size_t offset, size_t size)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        view.c[offset + i] = view.a[offset + i] + view.b[offset + i];
    }
}

TEST(SOA_POOL, CudaUnifiedPoolNonTrivialChunksAndKernel)
{
    soa_cuda_unified_pool<View> pool(64);

    const std::vector<size_t> initial_sizes = {32, 64, 16, 96, 48};
    std::vector<TestChunk> chunks;

    for (size_t i = 0; i < initial_sizes.size(); ++i)
    {
        size_t off = pool.alloc(initial_sizes[i]);
        TestChunk ch{off, initial_sizes[i], static_cast<float>((i + 1) * 50)};

        // Write input data a and b on host
        for (size_t j = 0; j < ch.size; ++j)
        {
            size_t idx = ch.offset + j;
            pool.data().a[idx] = ch.seed + static_cast<float>(j);
            pool.data().b[idx] = (ch.seed + static_cast<float>(j)) * 2.0f;
            pool.data().c[idx] = 0.0f; // To be computed by GPU kernel
        }
        chunks.push_back(ch);
    }

    // Non-trivial deallocation: free chunks 1 and 3 to fragment unified memory
    pool.free(chunks[1].offset);
    pool.free(chunks[3].offset);

    // Reallocate chunk 0 to larger size
    size_t new_off0 = pool.realloc(chunks[0].offset, chunks[0].size, 50);
    chunks[0].offset = new_off0;
    chunks[0].size = 50;
    for (size_t j = 0; j < chunks[0].size; ++j)
    {
        size_t idx = chunks[0].offset + j;
        pool.data().a[idx] = chunks[0].seed + static_cast<float>(j);
        pool.data().b[idx] = (chunks[0].seed + static_cast<float>(j)) * 2.0f;
        pool.data().c[idx] = 0.0f;
    }

    // Allocate a new chunk into the freed fragmented space
    size_t off_new = pool.alloc(40);
    TestChunk ch_new{off_new, 40, 300.0f};
    for (size_t j = 0; j < ch_new.size; ++j)
    {
        size_t idx = ch_new.offset + j;
        pool.data().a[idx] = ch_new.seed + static_cast<float>(j);
        pool.data().b[idx] = (ch_new.seed + static_cast<float>(j)) * 2.0f;
        pool.data().c[idx] = 0.0f;
    }

    std::vector<TestChunk> active_chunks = {chunks[0], chunks[2], chunks[4], ch_new};
    verify_no_chunk_overlaps(active_chunks);

    // Launch CUDA kernel for each active chunk in the unified pool
    for (const auto& ch : active_chunks)
    {
        int threads = 32;
        int blocks = static_cast<int>((ch.size + threads - 1) / threads);
        kernelAddChunk<<<blocks, threads>>>(pool.view(), ch.offset, ch.size);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify GPU results on host
    for (const auto& ch : active_chunks)
    {
        for (size_t j = 0; j < ch.size; ++j)
        {
            size_t idx = ch.offset + j;
            float expected_c = pool.data().a[idx] + pool.data().b[idx];
            EXPECT_FLOAT_EQ(pool.data().c[idx], expected_c)
                << "CUDA kernel computation mismatch at active chunk offset " << ch.offset << " index " << j;
        }
    }

    // Free remaining chunks
    for (const auto& ch : active_chunks)
    {
        pool.free(ch.offset);
    }
}
