#include <iostream>
#include "src/Raytracer.hpp"


static bool exitFlag=false;
static std::unique_ptr<Raytracer> raytracer;


void ui()
{
    vkbase::imgui::showDebugWindow();

    if (ImGui::Begin("Test wind"))
    {
        auto available=ImGui::GetContentRegionAvail();
        raytracer->swap_chain.setImageSize(available.x,available.y);

        auto image_to_presen=raytracer->swap_chain.getImageToPresent();
        if (image_to_presen)
            ImGui::Image(image_to_presen->image_imgui,available);
    }

    ImGui::End();
}


int main([[maybe_unused]] int argc, [[maybe_unused]] char** args)
{
    try
    {
        //register extensions
        vkbase::cuda::init();


        //target image count
        vkbase::imageCount=2;
        // vkbase::sys::fullscreen(true);
        vkbase::init("Raytracer");//init engine
        vkbase::imgui::initIMGUI();//init imgui
        vkbase::imgui::registerEngineMessagesLog();//copy engine messages to imgui log window


        //set up imgui
        // vkbase::imgui::setDefaultFontName("JetBrainsMono-Regular");
        vkbase::imgui::setDefaultFontSize(16);
        ImGuiStyle& style = ImGui::GetStyle();
        style.Colors[ImGuiCol_WindowBg].w = 1.f;


        //register callbacks
        vkbase::imgui::addOnUiCallback(ui);

        raytracer=std::make_unique<Raytracer>();
        RenderParameters parameters{};


        //cuda loop
        bool stop_cuda=false;
        std::thread cuda_loop([&]()
        {
            while (!stop_cuda)
            {
                raytracer->draw_frame(parameters);
            }
        });


        vkbase::addBeforeSubmitCallback([](uint32_t image_id)
        {
            auto dependencies=raytracer->swap_chain.swap();
            for (auto wp:dependencies)
                vkbase::addNextSubmitWait(wp);
        },10);


        //main loop
        while (vkbase::handleEvents()&&!exitFlag)
        {
            vkbase::info("\n\nNew frame");

            vkbase::drawFrame();
        }

        stop_cuda=true;
        cuda_loop.join();
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
