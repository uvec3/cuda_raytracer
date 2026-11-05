#pragma once
#include "../cuda_common.hpp"
#include <cstdint>


__global__ void my_scan_kernel(int* output, int* input, uint32_t n,int* block_sums=nullptr)
{
    __shared__ int s[2048];
    int block_offset=blockIdx.x*blockDim.x*2;

    if (block_offset+threadIdx.x<n)
        s[threadIdx.x]=input[block_offset+threadIdx.x];
    if (block_offset+threadIdx.x+blockDim.x<n)
        s[threadIdx.x+blockDim.x]=input[block_offset+threadIdx.x+blockDim.x];

    int group_size=1;
    while (group_size<2*blockDim.x)
    {
        __syncthreads();
        auto t_group=threadIdx.x/group_size;
        auto t_off=threadIdx.x%group_size;

        auto add_id=group_size-1+t_group*2*group_size;
        auto element=group_size+t_off+t_group*2*group_size;

        s[element]+=s[add_id];

        group_size*=2;
    }

    if (block_offset+threadIdx.x<n)
        output[block_offset+threadIdx.x]=s[threadIdx.x];
    if (block_offset+threadIdx.x+blockDim.x<n)
        output[block_offset+threadIdx.x+blockDim.x]=s[threadIdx.x+blockDim.x];
    if (block_sums!=0 && threadIdx.x==0)
        block_sums[blockIdx.x]=s[2*blockDim.x-1];
}

__global__ void add_block_sums(int* partial_sums, int* block_sums, uint32_t block_size,uint32_t n)
{
    auto i=blockIdx.x*blockDim.x+threadIdx.x;
    //skip first block
    i+=block_size;

    if (i<n)
    {
        auto previous_block=i/block_size-1;
        // printf("element=%d, partial sum=%d, prev_block=%d, previous block sum=%d\n",i,partial_sums[i],previous_block,block_sums[previous_block]);
        partial_sums[i]+=block_sums[previous_block];
    }
}

void my_scan(int* output, int* input, uint32_t n)
{
    uint32_t block_size=512;
    uint32_t number_of_blocks=(n-1)/block_size*2+1;

    int* block_sums=nullptr;
    if (number_of_blocks>1)
        cudaMalloc(&block_sums,sizeof(int)*number_of_blocks);

    my_scan_kernel<<<number_of_blocks,block_size>>>(output,input,n,block_sums);

    if (number_of_blocks>1)
    {
        my_scan(block_sums,block_sums,number_of_blocks);
        add_block_sums<<<number_of_blocks-1,block_size>>>(output,block_sums,block_size*2,n);
        cudaFree(block_sums);
    }
}




#define WARP_SIZE 32

inline __device__ uint32_t warp_id()
{
    return threadIdx.x/WARP_SIZE;
}

inline __device__ uint32_t thread_lane()
{
    return threadIdx.x%WARP_SIZE;
}

inline int __device__ warp_scan_kogge_stone(int value)
{
    for (int stride=1;stride<WARP_SIZE;stride*=2)
    {
        int element_stride_away=__shfl_up_sync(0xffffffff,value,stride);
        if (thread_lane()>=stride)
            value+=element_stride_away;

    }


    return value;
}


inline __device__ int block_shfl_up(int value)
{
    __shared__ int s[32];
    if (thread_lane()==WARP_SIZE-1)
    {
        s[warp_id()]=value;
    }
    value=__shfl_up_sync(0xffffffff,value,1);

    __syncthreads();

    if (thread_lane()==0&&warp_id()>0)
    {
        value=s[warp_id()-1];
    }

    return value;
}


inline int __device__ block_scan_kogge_stone(int value)
{
    value=warp_scan_kogge_stone(value);

    __shared__ int buff_s[WARP_SIZE];

    if (thread_lane()==WARP_SIZE-1)
    {
        buff_s[warp_id()]=value;
    }

    __syncthreads();
    if (warp_id()==0)
    {
        buff_s[threadIdx.x]=warp_scan_kogge_stone(buff_s[threadIdx.x]);
    }
    __syncthreads();

    if (warp_id()>0)
    {
        value+=buff_s[warp_id()-1];
    }

    return value;
}


__global__ void scan_kogge_stone_warp_kernel(int* out, int* in, int n,int* block_sums=nullptr)
{
    auto i = blockDim.x*blockIdx.x+threadIdx.x;
    int res;
    if (i<n)
    {
        res =block_scan_kogge_stone(in[i]);
        out[i]=res;
    }
    if (block_sums!=nullptr && threadIdx.x==blockDim.x-1)
    {
        block_sums[blockIdx.x]=res;
    }
}

void scan_kogge_stone_warp(int* output, int* input, uint32_t n)
{
    uint32_t block_size=1024;
    uint32_t number_of_blocks=(n-1)/block_size+1;

    int* block_sums=nullptr;
    if (number_of_blocks>1)
        cudaMalloc(&block_sums,sizeof(int)*number_of_blocks);

    scan_kogge_stone_warp_kernel<<<number_of_blocks,block_size>>>(output,input,n,block_sums);

    if (number_of_blocks>1)
    {
        scan_kogge_stone_warp(block_sums,block_sums,number_of_blocks);
        add_block_sums<<<number_of_blocks-1,block_size>>>(output,block_sums,block_size,n);
        cudaFree(block_sums);
    }
}

#define COARSENING_FACTOR 11
__global__ void scan_kogge_stone_warp_coarse_reg_kernel(int* out, const int* in, int n, int* block_sums=nullptr)
{
    const auto block_offset = blockIdx.x*blockDim.x*COARSENING_FACTOR;

    //load from global to shared memory
    __shared__ int s[COARSENING_FACTOR*512];
    #pragma unroll
    for (int i=0;i<COARSENING_FACTOR;++i)
    {
        if (block_offset+i*blockDim.x+threadIdx.x<n)
            s[i*blockDim.x+threadIdx.x]=in[block_offset+i*blockDim.x+threadIdx.x];
    }

    __syncthreads();

    //load from shared memory to registers and perform segment scan
    int r[COARSENING_FACTOR];
    r[0]=s[threadIdx.x*COARSENING_FACTOR];
    #pragma unroll
    for (int i=1;i<COARSENING_FACTOR;++i)
    {
        r[i]=s[threadIdx.x*COARSENING_FACTOR+i]+r[i-1];
    }

    //do scan on the sums
    auto scan= block_scan_kogge_stone(r[COARSENING_FACTOR-1]);

    //get scan of previous thread
    auto prev_scan=block_shfl_up(scan);
    // printf("t=%d sum=%d scan=%d   prev_scan=%d\n",threadIdx.x,r[COARSENING_FACTOR-1], scan,prev_scan);


    //put back in shared memory and add previous thread scan
     #pragma unroll
     for (int i=0;i<COARSENING_FACTOR;++i)
     {
         if (threadIdx.x!=0)
            r[i]+=prev_scan;
         s[threadIdx.x*COARSENING_FACTOR+i]=r[i];
     }

    __syncthreads();
    // //upload from shared to global memory
    #pragma unroll
    for (int i=0;i<COARSENING_FACTOR;++i)
    {
        if (block_offset+i*blockDim.x+threadIdx.x<n)
            out[block_offset+i*blockDim.x+threadIdx.x]=s[i*blockDim.x+threadIdx.x];
    }


    if (block_sums!=nullptr&&threadIdx.x==blockDim.x-1)
    {
        block_sums[blockIdx.x]=scan;
    }
}

void scan_kogge_stone_warp_coarse_reg(int* output, int* input, uint32_t n)
{
    uint32_t block_size=512;
    uint32_t number_of_blocks=(n-1)/(block_size*COARSENING_FACTOR)+1;

    int* block_sums=nullptr;
    if (number_of_blocks>1)
        cudaMalloc(&block_sums,sizeof(int)*number_of_blocks*COARSENING_FACTOR);

    scan_kogge_stone_warp_coarse_reg_kernel<<<number_of_blocks,block_size>>>(output,input,n,block_sums);

    if (number_of_blocks>1)
    {
        scan_kogge_stone_warp_coarse_reg(block_sums,block_sums,number_of_blocks);
        add_block_sums<<<(n-1)/block_size,block_size>>>(output,block_sums,block_size*COARSENING_FACTOR,n);
        cudaFree(block_sums);
    }
}




#define COARSENING_FACTOR 7
__global__ void scan_kogge_stone_warp_coarse_kernel(int* out, const int* in, uint32_t n, int* block_sums=nullptr)
{
    const auto block_offset = blockIdx.x*blockDim.x*COARSENING_FACTOR;

    //load from global to shared memory
    __shared__ int s[COARSENING_FACTOR*512];
    #pragma unroll
    for (int i=0;i<COARSENING_FACTOR;++i)
    {
        if (block_offset+i*blockDim.x+threadIdx.x<n)
            s[i*blockDim.x+threadIdx.x]=in[block_offset+i*blockDim.x+threadIdx.x];
    }

    __syncthreads();

    //perform segment scan
    #pragma unroll
    for (int i=1;i<COARSENING_FACTOR;++i)
    {
        s[threadIdx.x*COARSENING_FACTOR+i]+=s[threadIdx.x*COARSENING_FACTOR+i-1];
    }

    //do scan on the sums across all threads in current block
    s[(threadIdx.x+1)*COARSENING_FACTOR-1]= block_scan_kogge_stone(s[(threadIdx.x+1)*COARSENING_FACTOR-1]);

    //previous thread scan
    #pragma unroll
    for (int i=0;i<COARSENING_FACTOR-1;++i)
    {
        if (threadIdx.x!=0)
            s[threadIdx.x*COARSENING_FACTOR+i]+=s[threadIdx.x*COARSENING_FACTOR-1];
    }

    __syncthreads();
    //upload from shared to global memory
    #pragma unroll
    for (int i=0;i<COARSENING_FACTOR;++i)
    {
        if (block_offset+i*blockDim.x+threadIdx.x<n)
            out[block_offset+i*blockDim.x+threadIdx.x]=s[i*blockDim.x+threadIdx.x];
    }


    if (block_sums!=nullptr&&threadIdx.x==blockDim.x-1)
    {
        block_sums[blockIdx.x]=s[COARSENING_FACTOR*blockDim.x-1];
    }
}

void scan_kogge_stone_warp_coarse(int* output, int* input, uint32_t n)
{
    uint32_t block_size=512;
    uint32_t number_of_blocks=(n-1)/(block_size*COARSENING_FACTOR)+1;

    int* block_sums=nullptr;
    if (number_of_blocks>1)
        cudaMalloc(&block_sums,sizeof(int)*number_of_blocks*COARSENING_FACTOR);

    cudaFuncSetAttribute(scan_kogge_stone_warp_coarse_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 64000);
    scan_kogge_stone_warp_coarse_kernel<<<number_of_blocks,block_size>>>(output,input,n,block_sums);

    if (number_of_blocks>1)
    {
        scan_kogge_stone_warp_coarse(block_sums,block_sums,number_of_blocks);
        add_block_sums<<<(n-1)/block_size,block_size>>>(output,block_sums,block_size*COARSENING_FACTOR,n);
        cudaFree(block_sums);
    }
}


enum FlagState: int
{
    READY=0,
    LOCAL_SUM_READY=1,
    SCAN_READY=2
};


__device__ inline  void block_vector_memcpy(int* dst,const int* src, uint32_t size)
{
    const float4* src4=reinterpret_cast<const float4*>(src);
    float4* dst4=reinterpret_cast<float4*>(dst);

    uint32_t i=threadIdx.x;

    for (;i<size/4;i+=blockDim.x)
    {
        dst4[i]=src4[i];
    }

    if (i*4+threadIdx.x<size)
    {
        dst[i*4+threadIdx.x]=src[i*4+threadIdx.x];
    }
}


#define BLOCK_DIM 512
#define COARSENING_FACTOR 15
__global__ void scan_unidirectional_kernel(int* out, const int* in, uint32_t n, int* flags, int* block_sums, int* block_scans, uint32_t* blockCounter)
{
    __shared__ uint32_t blockID;
    if (threadIdx.x==0)
    {
        blockID=cuda::atomic_ref<uint32_t,cuda::thread_scope_device>(*blockCounter)
            .fetch_add(1,cuda::memory_order_relaxed);

        // printf("block %d received index %d\n", blockIdx.x,blockID);
    }

    __syncthreads();

    const auto block_offset =blockID*blockDim.x*COARSENING_FACTOR;

    //load from global to shared memory
    __shared__ alignas(16) int s[COARSENING_FACTOR*BLOCK_DIM];
    block_vector_memcpy(s,in+block_offset,COARSENING_FACTOR*BLOCK_DIM);

    __syncthreads();

    //perform segment scan
    #pragma unroll
    for (int i=1;i<COARSENING_FACTOR;++i)
    {
        s[threadIdx.x*COARSENING_FACTOR+i]+=s[threadIdx.x*COARSENING_FACTOR+i-1];
    }


    //do scan on the sums across all threads in current block
    auto scan= block_scan_kogge_stone(s[(threadIdx.x+1)*COARSENING_FACTOR-1]);

    __shared__ int previous_block_scan;
    cuda::atomic_ref<int,cuda::thread_scope_device> flag(flags[blockID]);
    if (blockID==0)//last thread of first block
    {
        if (threadIdx.x==blockDim.x-1)
        {
            block_scans[blockID]=scan;
            flag.store(SCAN_READY,cuda::memory_order_release);
        }
    }
    else
    {
        if (threadIdx.x==blockDim.x-1)//last thread in the block
        {
            block_sums[blockID]=scan;
            flag.store(LOCAL_SUM_READY,cuda::memory_order_release);

            int prefix=0;
            auto bid=blockID-1;
            while (true)
            {
                cuda::atomic_ref<int,cuda::thread_scope_device> previousFlag(flags[bid]);
                auto flag_value=previousFlag.load(cuda::memory_order_acquire);
                if (flag_value==LOCAL_SUM_READY)
                {
                    prefix+=block_sums[bid];
                    bid-=1;
                }
                else if (flag_value==SCAN_READY)
                {
                    prefix+=block_scans[bid];
                    break;
                }
            };
            block_scans[blockID]=prefix+scan;
            flag.store(SCAN_READY,cuda::memory_order_release);

            previous_block_scan=prefix;
        }
        __syncthreads();
        scan+=previous_block_scan;
    }

    s[(threadIdx.x+1)*COARSENING_FACTOR-1]=scan;
    __syncthreads();
    //previous thread scan
    #pragma unroll
    for (int i=0;i<COARSENING_FACTOR-1;++i)
    {
        if (threadIdx.x!=0)
            s[threadIdx.x*COARSENING_FACTOR+i]+=s[threadIdx.x*COARSENING_FACTOR-1];
        else if (blockID!=0)
            s[threadIdx.x*COARSENING_FACTOR+i]+=previous_block_scan;
    }

    __syncthreads();
    //upload from shared to global memory
    block_vector_memcpy(out+block_offset,s,COARSENING_FACTOR*BLOCK_DIM);
}

void scan_unidirectional(int* output, int* input, uint32_t n)
{
    uint32_t block_size=BLOCK_DIM;
    uint32_t number_of_blocks=(n-1)/(block_size*COARSENING_FACTOR)+1;

    int* flags=nullptr;
    int* blockSums=nullptr;
    int* blockScans=nullptr;
    uint32_t* blockCounter=nullptr;

    {
        cudaMalloc(&flags,sizeof(int)*number_of_blocks);
        cudaMemset(flags,0,sizeof(int)*number_of_blocks);

        cudaMalloc(&blockSums,sizeof(int)*number_of_blocks);
        cudaMalloc(&blockScans,sizeof(int)*number_of_blocks);

        cudaMalloc(&blockCounter,sizeof(*blockCounter));
        cudaMemset(blockCounter,0,sizeof(*blockCounter));
    }

    scan_unidirectional_kernel<<<number_of_blocks,block_size>>>(output,input,n,flags,blockSums,blockScans,
        blockCounter);

    {
        cudaFree(flags);
        cudaFree(blockSums);
        cudaFree(blockScans);
        cudaFree(blockCounter);
    }
}