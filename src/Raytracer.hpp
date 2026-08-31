#pragma once

#include <chrono>

#include "vkbase/core/EngineBase.h"
#include "vkbase/extensions/imgui/imgui.hpp"
#include "vkbase/extensions/imgui/Log.hpp"
#include "kernels/kernel.h"
#include "vkbase/extensions/cuda/Image.hpp"
#include "CudaVulkanSwapChain.hpp"


struct RenderParameters
{
    double t;
};

class Raytracer
{
public:
    CudaVulkanSwapChain swap_chain{2};


    std::chrono::high_resolution_clock::time_point start_time;

    Raytracer()
    {

        start_time=std::chrono::high_resolution_clock::now();
    }




    void draw_frame(RenderParameters parameters)
    {

        double t= (std::chrono::high_resolution_clock::now()-start_time).count() / 1000000000.0;


        auto image_id=swap_chain.acquireNextImage();
        auto& image=swap_chain.getImage(image_id);

        draw_device(image.cuda_surface,image.width,image.height, t);

        cudaDeviceSynchronize();

        swap_chain.present(image_id);
    }
};