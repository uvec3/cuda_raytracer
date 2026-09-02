#pragma once

#include <chrono>
#include <numeric>

#include "vkbase/core/EngineBase.h"
#include "vkbase/extensions/imgui/imgui.hpp"
#include "vkbase/extensions/imgui/Log.hpp"
#include "kernels/kernel.h"
#include "vkbase/extensions/cuda/Image.hpp"
#include "CudaVulkanSwapChain.hpp"
#include <glm/glm.hpp>
#include <cuda_profiler_api.h>


struct WindowInfo
{
    std::string name;
    std::function<bool()> draw;
    bool open=true;
};

class Raytracer
{
public:
    CudaVulkanSwapChain swap_chain{5};
    RenderParameters parameters;

    std::vector<WindowInfo> windows;


    std::chrono::high_resolution_clock::time_point start_time;

    Raytracer()
    {
        //register callbacks
        vkbase::imgui::addOnUiCallback(WRAP_MEMBER_FUNC(ui));
        start_time=std::chrono::high_resolution_clock::now();


        cuda_loop = std::thread([&]()
        {
            while (!stop_cuda)
            {
                if (!cuda_is_running)
                {
                    wake_cuda.wait(false);
                    wake_cuda = false;
                }

                next_frame();
            }
        });


        windows.push_back({"Debug info",vkbase::imgui::showDebugWindow});
        windows.push_back({"Frame info", WRAP_MEMBER_FUNC(frame_info)});
        windows.push_back({"Log", vkbase::imgui::drawEngineMessagesLog});
        windows.push_back({"Output image", [this]()
        {
            bool open=true;
            if (ImGui::Begin("Output",&open))
            {
                ui_image();
            }
            ImGui::End();

            return open;
        }});
    }

    ~Raytracer()
    {
        stop_cuda=true;
        wake_cuda=true;
        wake_cuda.notify_one();
        if (cuda_loop.joinable())
            cuda_loop.join();
        vkDeviceWaitIdle(vkbase::device);
    }


    //cuda loop
    std::atomic<bool> stop_cuda=false;
    std::atomic<bool> cuda_is_running=false;
    std::atomic<bool> wake_cuda=false;


    std::thread cuda_loop;

    uint64_t frame_counter=0;
    double last_frame_time=0;
    const int frames_to_track=500;
    std::deque<double> last_frames_time={};
    bool enable_profiler=false;

    bool fullscreen=false;
    bool hide_ui=false;


    std::mutex ui_mutex;
    void ui()
    {
        vkbase::imgui::scaleDefaultFontUI();
        ui_input();

        std::lock_guard lock(ui_mutex);


        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);

        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration |
                                        ImGuiWindowFlags_NoMove |
                                        ImGuiWindowFlags_NoResize |
                                        ImGuiWindowFlags_NoSavedSettings |
                                        ImGuiWindowFlags_NoBringToFrontOnFocus;


        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));

        if (ImGui::Begin("##FullscreenWindow", nullptr, window_flags))
        {
            if (hide_ui)
            {
                ui_image();
                ImGui::PopStyleVar(4);
            }
            else
            {
                // ui_menu();
                // ui_toolbar();
                ImGui::DockSpace(ImGui::GetID("DockSpace"), ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);
                ImGui::PopStyleVar(4);

                for (auto& window:windows)
                {
                    if (window.open)                    {
                        window.open=window.draw();
                    }
                }
            }
        }
        ImGui::End();
    }

    void ui_input()
    {
        if (!ImGui::GetIO().WantTextInput)
        {
            if (ImGui::IsKeyPressed(ImGuiKey_Space))
            {
                draw_one_frame();
            }

            if (ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
            {
                if (ImGui::IsKeyPressed(ImGuiKey_Space))
                {
                    if (cuda_is_running)
                        pause();
                    else
                        play();
                }
            }
        }


        if (ImGui::IsKeyPressed(ImGuiKey_F11))
        {
            vkbase::sys::fullscreen(!vkbase::sys::isFullscreen());
        }

        if (ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
        {
            if (ImGui::IsKeyPressed(ImGuiKey_GraveAccent))
            {
                hide_ui=!hide_ui;
            }
        }


    }

    void ui_surface_input()
    {
        if (ImGui::IsMouseDown(0))
            parameters.mouse_pos={ImGui::GetMousePos().x-ImGui::GetItemRectMin().x,ImGui::GetMousePos().y-ImGui::GetItemRectMin().y};
    }

    void ui_image()
    {
        auto available=ImGui::GetContentRegionAvail();
        swap_chain.setImageSize(available.x,available.y);

        auto image_to_present=swap_chain.getImageToPresent();
        if (image_to_present)
        {
            ImGui::Image(image_to_present->image_imgui,ImVec2(image_to_present->image.width,image_to_present->image.height));
            ui_surface_input();
        }
    }

    void ui_toolbar()
    {

        if (cuda_is_running)
        {
            if (ImGui::Button("||"))
            {
                pause();
            }
        }
        else
        {
            if (ImGui::Button("> "))
            {
                play();
            }
        }

        ImGui::SameLine();
        ImGui::BeginDisabled(cuda_is_running);
        if (ImGui::Button("|>"))
        {
            draw_one_frame();
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        if (ImGui::Button("[ ]"))
        {
            fullscreen=!fullscreen;
        }


        ImGui::SameLine();

        ImGui::Checkbox("Profile",&enable_profiler);
    }

    void ui_menu()
    {
        static bool showMenu = true;

        //show menu if mouse is over it for fullscreen mode when ui is hidden
        if(!hide_ui|| ImGui::GetIO().MousePos.y<ImGui::GetMainViewport()->Pos.y+vkbase::imgui::getDefaultFontSize()*2||
        ImGui::IsKeyDown(ImGui::GetKeyIndex(ImGuiKey_LeftCtrl))||ImGui::IsKeyDown(ImGui::GetKeyIndex(ImGuiKey_RightCtrl)))
        {
            showMenu = true;
        }


        if(!showMenu)//hide menu
            return;


        if(ImGui::BeginMainMenuBar())
        {

            if(ImGui::BeginMenu("View"))
            {

                for (auto& window:windows)
                {
                    if(ImGui::MenuItem(window.name.c_str(), "", window.open))
                    {
                        window.open=!window.open;
                    }
                }

                if(ImGui::MenuItem("Fullscreen", "F11", vkbase::sys::isFullscreen()))
                {
                    vkbase::sys::fullscreen(!vkbase::sys::isFullscreen());
                }

                if(ImGui::MenuItem("Hide UI", "Ctrl + `", hide_ui))
                {
                    hide_ui=!hide_ui;
                }
                ImGui::EndMenu();
            }

            if(ImGui::MenuItem("Exit"))
            {
                // exitFlag = true;
            }
            ImGui::EndMainMenuBar();
        }

        showMenu= ImGui::IsAnyItemFocused() || ImGui::IsAnyItemActive() || ImGui::IsAnyItemHovered() ;
    }

    void draw_one_frame()
    {
        wake_cuda=true;
        wake_cuda.notify_one();
    }

    bool frame_info()
    {
        static bool open=true;
        if (ImGui::Begin("Frame info",&open))
        {
            ImGui::Text("Frames rendered : %llu", frame_counter);

            ImGui::Text("Last frame time : %f ms", last_frame_time);
            auto avg= std::reduce(last_frames_time.begin(), last_frames_time.end())/frames_to_track;
            ImGui::Text("Avg frame time (last %d frames)   : %f ms", frames_to_track,avg);
            ImGui::Text("Estimated FPS  : %f", 1000.f/avg);


            auto w=swap_chain.get_width();
            auto h=swap_chain.get_height();

            ImGui::Text("Image width    : %d", w);
            ImGui::Text("Image height   : %d ", h);
            ImGui::Text("Total pixels   : %d", w*h);
        }
        ImGui::End();

        return open;
    }

    void play()
    {
        if (cuda_is_running)
            return;
        cuda_is_running=true;
        wake_cuda=true;
        wake_cuda.notify_one();
    }

    void pause()
    {
        cuda_is_running=false;
    }


    void next_frame()
    {
        const bool profiling_this_frame=enable_profiler;

        if (profiling_this_frame)
            cudaProfilerStart();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        auto image_id=swap_chain.acquireNextImage(stop_cuda);

        if (stop_cuda )
        {
            if (image_id!=-1)
                swap_chain.cancelImage(image_id);
            return;
        }

        auto& image=swap_chain.getImage(image_id);


        ui_mutex.lock();
        {
            parameters.t= (std::chrono::high_resolution_clock::now()-start_time).count() / 1000000000.0;
            cudaEventRecord(start);
            draw_device(image.cuda_surface,image.width,image.height,parameters);
            cudaEventRecord(stop);
        }
        ui_mutex.unlock();

        cudaEventSynchronize(stop);

        if (profiling_this_frame)
            cudaProfilerStop();

        swap_chain.present(image_id);

        ui_mutex.lock();
        {
            frame_counter+=1;
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            last_frame_time=milliseconds;
            last_frames_time.push_back(last_frame_time);
            if (last_frames_time.size()>=frames_to_track)
                last_frames_time.pop_front();

        }
        ui_mutex.unlock();
    }
};