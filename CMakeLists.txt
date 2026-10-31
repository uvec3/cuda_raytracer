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
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--threads=0>")


set(CMAKE_CUDA_ARCHITECTURES 86)


add_executable(Raytracer main.cpp
        src/core/RaytracerController.cpp src/core/RaytracerController_cuda.cu src/core/RaytracerController.hpp
        src/raytracer/Raytracer.cu src/raytracer/Raytracer.cuh
        src/raytracer/Scene.cpp src/raytracer/Scene.hpp
        src/acceleration_structures/octree/OctreeAS.cu src/acceleration_structures/octree/OctreeAS.cuh src/acceleration_structures/octree/OctreeGeometry.cuh
        src/ui_helpers/soa_ui_view.hpp

        external/tinygltf.cpp external/tiny_gltf_v3.h
        external/stb_image.cpp external/stb_image.h
        external/OffsetAllocator/offsetAllocator.cpp external/OffsetAllocator/offsetAllocator.hpp
)


set(VKBASE_ASSETS ${CMAKE_CURRENT_SOURCE_DIR}/assets/Scenes ${CMAKE_CURRENT_SOURCE_DIR}/assets/Fonts)
add_subdirectory(vkbase)
vkbase_target_copy_assets(Raytracer)

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
