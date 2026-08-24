#include <stdexcept>
#include <string>

#define CHECK_CUDA_ERROR(e) if(e!=cudaSuccess)\
{ \
   throw std::runtime_error("Error:"+std::to_string(__LINE__)); \
   cudaError_t last_error = cudaGetLastError();\
}


__device__ __host__ float operation(float a, float b)
{
    return a+b;
}


__global__ void kernelAdd(float* result, float* a, float* b,uint64_t N)
{
    auto i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<N)
    {
        //printf("%d executed\n",i);
        result[i]=operation(a[i],b[i]);
    }
    else
    {
        //printf("%d skipped\n",i);
    }
}


void deviceAdd(float* result_h, float* a_h, float* b_h,uint64_t N)
{
    const uint64_t blockSize=128;

    auto vecSizeBytes=sizeof(result_h[0])*N;


    cudaError resultCode;
    float* a_d; resultCode = cudaMalloc(&a_d,vecSizeBytes);
    CHECK_CUDA_ERROR(resultCode)
    float* b_d; resultCode = cudaMalloc(&b_d,vecSizeBytes);
    CHECK_CUDA_ERROR(resultCode)
    float* result_d; resultCode = cudaMalloc(&result_d,vecSizeBytes);
    CHECK_CUDA_ERROR(resultCode)

    resultCode=cudaMemcpy(a_d,a_h,vecSizeBytes,cudaMemcpyHostToDevice);
    resultCode=cudaMemcpy(b_d,b_h,vecSizeBytes,cudaMemcpyHostToDevice);

    kernelAdd<<<(N-1)/blockSize+1,blockSize>>>(result_d,a_d,b_d,N);

    resultCode=cudaDeviceSynchronize();

    resultCode=cudaMemcpy(result_h,result_d,vecSizeBytes,cudaMemcpyDeviceToHost);

    resultCode=cudaFree(result_d);
    resultCode=cudaFree(a_d);
    resultCode=cudaFree(b_d);
}
