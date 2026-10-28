#pragma once

#ifdef __CUDACC__
    #define CUDA_HOST_DEVICE __host__ __device__
#elif defined(__cplusplus)
    #define CUDA_HOST_DEVICE
#endif

#define CUDA_CHECK(call)                                             \
do {                                                                 \
    cudaError_t err = call;                                          \
    if (err != cudaSuccess) {                                        \
        std::cerr << "CUDA Error:\n"                                 \
        << "  File: " << __FILE__ << ":" << __LINE__ << "\n"         \
        << "  Function: " << #call << "\n"                           \
        << "  Name: " << cudaGetErrorName(err) << "\n"               \
        << "  Message: " << cudaGetErrorString(err) << "\n";         \
        cudaDeviceReset();                                           \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while (0)