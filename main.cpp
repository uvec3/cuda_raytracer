#include <iostream>
#include <vkbase_core.hpp>
#include "src/core/RaytracerController.hpp"
#include "src/core/CudaVulkanSwapChain.hpp"
#include "vkbase/extensions/imgui/imgui.hpp"
#include "vkbase/extensions/imgui/Log.hpp"

int main([[maybe_unused]] int argc, [[maybe_unused]] char** args)
{
    bool exitFlag=false;
    std::unique_ptr<RaytracerController> raytracer_controller;
    try
    {
        //register extensions
        vkbase::cuda::init();

        //target image count
        vkbase::config.min_img_in_swapchain=2;
        vkbase::config.preferred_present_mode=VK_PRESENT_MODE_IMMEDIATE_KHR;
        vkbase::sys::fullscreen(true);
        vkbase::init("Raytracer");//init engine
        vkbase::imgui::initIMGUI();//init imgui
        //vkbase::imgui::registerEngineMessagesLog();//copy engine messages to imgui log window

        //set up imgui
        vkbase::imgui::setDefaultFontName("JetBrainsMono-Regular");
        vkbase::imgui::setDefaultFontSize(23);
        ImGuiStyle& style = ImGui::GetStyle();
        style.Colors[ImGuiCol_WindowBg].w = 1.f;

        raytracer_controller=std::make_unique<RaytracerController>();

        vkbase::addBeforeSubmitCallback([&](uint32_t image_id)
        {
            auto dependencies=raytracer_controller->swap_chain->swap();
            for (auto wp:dependencies)
                vkbase::addNextSubmitWait(wp);
        },10);

        //main loop
        while (vkbase::handleEvents()&&!exitFlag)
        {
            vkbase::drawFrame();
        }

        raytracer_controller=nullptr;//explicitly destroy raytracer before vkbase::destroy call
        vkbase::destroy();
    }
    catch (const std::exception &e)
    {
        printf("Exception:%s", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
