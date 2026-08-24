#include <iostream>
#include "vkbase/core/EngineBase.h"
#include "vkbase/extensions/imgui/imgui.hpp"
#include "vkbase/extensions/imgui/Log.hpp"
#include "src/kernels/kernel.h"

static bool exitFlag=false;

void ui()
{
    vkbase::imgui::showDebugWindow();
}

void initVec(float* vec, uint64_t N)
{
    for (uint64_t i=0;i<N;++i)
    {
        vec[i]=i;
    }
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char** args)
{
    int N=111;
    float* a= (float*) malloc(N* sizeof(float));
    float* b= (float*) malloc(N* sizeof(float));
    float* result= (float*) malloc(N* sizeof(float));
    float* result_from_device = (float*) malloc(N* sizeof(float));

    initVec(a,N);
    initVec(b,N);
    memset(result,0,N*sizeof(float));



    deviceAdd(result_from_device,a,b,N);

    std::cout<<"\nd: ";
    for (auto i=0;i<N;++i)
    {
        std::cout<<result_from_device[i]<<"\n";
    }

    try
    {
        //target image count
        vkbase::imageCount=2;
        vkbase::init("IPTE");//init engine
        vkbase::imgui::initIMGUI();//init imgui
        vkbase::imgui::registerEngineMessagesLog();//copy engine messages to imgui log window

        // vkbase::imgui::setDefaultFontName("JetBrainsMono-Regular");
        vkbase::imgui::setDefaultFontSize(16);
        ImGuiStyle& style = ImGui::GetStyle();
        style.Colors[ImGuiCol_WindowBg].w = 0.5f;


        //register ui function
        vkbase::imgui::addOnUiCallback(ui);


        //main loop
        while (vkbase::handleEvents()&&!exitFlag)
        {
            vkbase::drawFrame();
        }


        vkbase::destroy();
    }
    catch (const std::exception &e)
    {
        std::cout<< "Exception: "+ std::string(e.what());
    }

    return EXIT_SUCCESS;
}
