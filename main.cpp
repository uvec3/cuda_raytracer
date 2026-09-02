#include <iostream>
#include "src/Raytracer.hpp"


static bool exitFlag=false;
static std::unique_ptr<Raytracer> raytracer;


int main([[maybe_unused]] int argc, [[maybe_unused]] char** args)
{
    try
    {
        //register extensions
        vkbase::cuda::init();


        //target image count
        vkbase::imageCount=2;
        vkbase::sys::fullscreen(true);
        vkbase::init("Raytracer");//init engine
        vkbase::imgui::initIMGUI();//init imgui
        vkbase::imgui::registerEngineMessagesLog();//copy engine messages to imgui log window


        //set up imgui
        vkbase::imgui::setDefaultFontName("JetBrainsMono-Regular");
        vkbase::imgui::setDefaultFontSize(23);
        ImGuiStyle& style = ImGui::GetStyle();
        style.Colors[ImGuiCol_WindowBg].w = 1.f;

        raytracer=std::make_unique<Raytracer>();

        vkbase::addBeforeSubmitCallback([](uint32_t image_id)
        {
            auto dependencies=raytracer->swap_chain.swap();
            for (auto wp:dependencies)
                vkbase::addNextSubmitWait(wp);
        },10);


        //main loop
        while (vkbase::handleEvents()&&!exitFlag)
        {
            vkbase::drawFrame();
        }

        raytracer=nullptr;//destroy raytracer before vkbase::destroy
        vkbase::destroy();
    }
    catch (const std::exception &e)
    {
        std::cout<< "Exception: "+ std::string(e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
