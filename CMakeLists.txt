CMAKE_MINIMUM_REQUIRED(VERSION 4.2)

SET(CMAKE_CXX_STANDARD 23)
SET(CMAKE_CUDA_STANDARD 20)
project(Raytracer CXX CUDA)


if (MSVC)
    add_compile_options(
            "$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>"
    )
endif()
add_compile_definitions(NOMINMAX)

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--threads=0>")


set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)
find_program(BIN2C bin2c HINTS ${CUDAToolkit_BIN_DIR})

#OPTIX

include(FetchContent)
FetchContent_Declare(
        optix_headers
        GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
        GIT_TAG        v8.0.0 # Or v7.7.0 / v8.1.0 / v9.0.0 depending on your driver
        GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(optix_headers)

# CLIP
set(CLIP_EXAMPLES OFF CACHE BOOL "Compile clip examples" FORCE)
set(CLIP_TESTS OFF CACHE BOOL "Compile clip tests" FORCE)
FetchContent_Declare(
        clip
        GIT_REPOSITORY https://github.com/dacap/clip.git
        GIT_TAG        main
        GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(clip)


add_executable(Raytracer main.cpp
        src/core/RaytracerController.cpp src/core/RaytracerController_cuda.cu src/core/RaytracerController.hpp
        src/raytracer/Raytracer.cu src/raytracer/Raytracer.cuh
        src/raytracer/Scene.cu src/raytracer/Scene.hpp
        src/acceleration_structures/octree/OctreeAS.cu src/acceleration_structures/octree/OctreeAS.cuh src/acceleration_structures/octree/OctreeGeometry.cuh
        src/acceleration_structures/uniform_grid/UniformGridAS.cu
        src/acceleration_structures/optix/OptixAS.cu


        external/tinygltf.cpp external/tiny_gltf_v3.h
        external/stb_image.cpp external/stb_image.h
        external/OffsetAllocator/offsetAllocator.cpp external/OffsetAllocator/offsetAllocator.hpp
)

set_source_files_properties(src/core/RaytracerController.cpp
        PROPERTIES CXX_STANDARD 26
)


target_include_directories(Raytracer PRIVATE
        "${optix_headers_SOURCE_DIR}/include"
)


set(VKBASE_ASSETS ${CMAKE_CURRENT_SOURCE_DIR}/assets/Scenes ${CMAKE_CURRENT_SOURCE_DIR}/assets/Fonts)
add_subdirectory(vkbase)
add_subdirectory(external/imgui_container_explorer)
vkbase_target_copy_assets(Raytracer)

set(OPTIX_SHADERS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/acceleration_structures/optix/shaders.cu")
set(OPTIX_IR_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shaders.optixir")
set(OPTIX_HEADER_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/shaders_optix_ir.h")

add_custom_command(
    OUTPUT "${OPTIX_HEADER_OUTPUT}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
    COMMAND "${CMAKE_CUDA_COMPILER}"
            -std=c++20
            --optix-ir
            --expt-relaxed-constexpr
            --extended-lambda
            -DGLM_FORCE_ALIGNED_GENTYPES
            -DGLM_FORCE_CUDA
            -DGLM_FORCE_PURE
            -Xcompiler=/Zc:preprocessor
            "-diag-suppress=20012,20015,174"
            "-I${optix_headers_SOURCE_DIR}/include"
            "-I${CMAKE_CURRENT_SOURCE_DIR}"
            "-I${glm_SOURCE_DIR}"
            "-I${CMAKE_CURRENT_SOURCE_DIR}/external/soa_vector/src"
            -c "${OPTIX_SHADERS_SRC}"
            -o "${OPTIX_IR_OUTPUT}"
    COMMAND ${CMAKE_COMMAND}
            -DBIN2C=${BIN2C}
            -DINPUT=${OPTIX_IR_OUTPUT}
            -DOUTPUT=${OPTIX_HEADER_OUTPUT}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_bin2c.cmake"
    DEPENDS "${OPTIX_SHADERS_SRC}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/acceleration_structures/optix/params.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/raytracer/RenderParameters.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/raytracer/rt_kernels.cuh"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/geometry_common.cuh"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/types.hpp"
    COMMENT "Compiling OptiX shaders to OptiX IR and generating C header"
)

add_custom_target(generate_optix_shaders DEPENDS "${OPTIX_HEADER_OUTPUT}")
add_dependencies(Raytracer generate_optix_shaders)

target_include_directories(Raytracer PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
)

target_link_libraries(Raytracer
        vkbase::core

        vkbase::backend::SDL::core
        vkbase::backend::SDL::imgui

        vkbase::assets
        vkbase::imgui
        vkbase::imgui::log
        vkbase::buffer
        vkbase::image
        vkbase::cuda
        imgui_container_explorer::imgui_container_explorer

        CUDA::cudart CUDA::cuda_driver
        clip
)

add_subdirectory(tests)



target_compile_options(Raytracer
        PRIVATE

        $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20012>
        $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=174>


        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>

        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>

        $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-fmad=true>
)

# Enables -rdc=true
#set_property(TARGET Raytracer PROPERTY CUDA_SEPARABLE_COMPILATION ON)
## Enables -dlto and the correct lto_86 code generation
#set_property(TARGET Raytracer PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
