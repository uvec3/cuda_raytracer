#pragma once

#include <cuda_runtime.h>
#include <iostream>


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



#if defined(__CUDA_ARCH__)
#define DEVICE_ONLY(statement) statement;
#define HOST_ONLY(statement)
#else
#define DEVICE_ONLY(statement)
#define HOST_ONLY(statement) statement;
#endif