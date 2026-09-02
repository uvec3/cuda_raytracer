CMAKE_MINIMUM_REQUIRED(VERSION 4.2)

SET(CMAKE_CXX_STANDARD 23)
SET(CMAKE_CUDA_STANDARD 20)

project(Raytracer CXX CUDA)


add_executable(Raytracer main.cpp
    src/kernels/kernel.cu
        src/kernels/kernel.h)


set(ASSETS  ${CMAKE_CURRENT_SOURCE_DIR}/assets/Scenes ${CMAKE_CURRENT_SOURCE_DIR}/assets/Fonts)

add_subdirectory(vkbase)

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

