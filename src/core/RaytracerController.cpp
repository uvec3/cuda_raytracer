#include "RaytracerController.hpp"

#include <cuda_profiler_api.h>
#include <numeric>
#include <vkbase_core.hpp>
#include "../../vkbase/extensions/imgui/imgui.hpp"
#include "../../vkbase/extensions/imgui/Log.hpp"
#include "../../vkbase/extensions/cuda/Image.hpp"
#include "CudaVulkanSwapChain.hpp"
#include "../cuda_common.hpp"
#include "../../vkbase/extensions/Assets/assets.h"
#include <ranges>


RaytracerController::RaytracerController()
{
    createRaytracer();
    swap_chain=new CudaVulkanSwapChain{5};

    //register callbacks
    vkbase::imgui::addOnUiCallback(WRAP_MEMBER_FUNC(ui));


    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&finish));

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
    windows.push_back({"Frame info", WRAP_MEMBER_FUNC(ui_frame_info)});
    windows.push_back({"Parameters", WRAP_MEMBER_FUNC(ui_show_parameters)});
    windows.push_back({"Log", vkbase::imgui::drawEngineMessagesLog});
    windows.push_back({"Table",WRAP_MEMBER_FUNC(ui_table)});
    windows.push_back({"Debug pixel",WRAP_MEMBER_FUNC(ui_debug_pixel)});
    windows.push_back({"Select scene",WRAP_MEMBER_FUNC(ui_scenes_manager)});
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


    initScenes();


    // soa_ui_view<decltype(scene.vertices)> table{scene.vertices};

    active_table="Acceleration structure";

    // play();
}

RaytracerController::~RaytracerController()
{
    stop_cuda=true;
    wake_cuda=true;
    wake_cuda.notify_one();
    if (cuda_loop.joinable())
        cuda_loop.join();
    vkDeviceWaitIdle(vkbase::device);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(finish));


    destroyRaytracer();
    delete swap_chain;
}

void RaytracerController::initScenes()
{
    auto scene_assets=vkbase::assets::assetsFrom("Scenes");
    for (auto [scene_name, scene_data]:scene_assets)
    {
        SceneData s;
        s.scene=std::make_unique<Scene>(scene_name);
        s.scene->load(scene_data);

        s.soa_tables["Vertices"]=std::make_unique<soa_ui_view<decltype(s.scene->as.vertices)>>(s.scene->as.vertices);
        s.soa_tables["Acceleration structure"]=std::make_unique<soa_ui_view<decltype(s.scene->as.as)>>(s.scene->as.as);
        s.soa_tables["Box contents"]=std::make_unique<soa_ui_view<decltype(s.scene->as.box_content)>>(s.scene->as.box_content);
        s.soa_tables["Primitives"]=std::make_unique<soa_ui_view<decltype(s.scene->as.primitives)>>(s.scene->as.primitives);
        s.soa_tables["Triangles"]=std::make_unique<soa_ui_view<decltype(s.scene->as.triangles)>>(s.scene->as.triangles);
        s.soa_tables["Spaces"]=std::make_unique<soa_ui_view<decltype(s.scene->as.spaces)>>(s.scene->as.spaces);

        scenes.push_back(std::move(s));
    }
}

void RaytracerController::ui()
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
                if (window.open)
                {
                    window.open=window.draw();
                }
            }

        }
    }
    ImGui::End();
}

void RaytracerController::ui_input()
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





void RaytracerController::ui_menu()
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

void RaytracerController::ui_image()
{
    ui_toolbar();
    auto available=ImGui::GetContentRegionAvail();
    swap_chain->setImageSize(available.x,available.y);
    current_image_size={available.x,available.y};

    auto image_to_present=swap_chain->getImageToPresent();
    if (image_to_present)
    {
        ImGui::Image(image_to_present->image_imgui,ImVec2(image_to_present->image.width,image_to_present->image.height));
        ui_surface_input();

    }
}

void RaytracerController::ui_toolbar()
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

bool RaytracerController::ui_frame_info()
{
    static bool open=true;
    if (ImGui::Begin("Frame info",&open))
    {
        ImGui::Text("Frames rendered : %llu", frame_counter);

        ImGui::Text("Last frame time : %f ms", last_frame_time);
        auto avg= std::reduce(last_frames_time.begin(), last_frames_time.end())/frames_to_track;
        ImGui::Text("Avg frame time (last %d frames)   : %f ms", frames_to_track,avg);
        ImGui::Text("Estimated FPS  : %f", 1000.f/avg);


        auto w=swap_chain->get_width();
        auto h=swap_chain->get_height();

        ImGui::Text("Image width    : %d", w);
        ImGui::Text("Image height   : %d ", h);
        ImGui::Text("Total pixels   : %d", w*h);
    }
    ImGui::End();

    return open;
}



bool RaytracerController::ui_scenes_manager()
{
    static bool open=true;
    if (ImGui::Begin("Active scene",&open))
    {
        auto& active_scene=*scenes[active_scene_number].scene;
        auto& selected_model=scenes[active_scene_number].active_space;

        if (ImGui::BeginCombo("Select table", active_scene.get_name().c_str()))
        {
            for (int i=0;i<scenes.size();++i)
            {

                auto& s=scenes[i];
                if (ImGui::Selectable(s.scene->get_name().c_str(),i==active_scene_number) )
                {
                    active_scene_number=i;
                }
            }
            ImGui::EndCombo();
        }


        ImGui::Text("Show");

        auto available=ImGui::GetContentRegionAvail();
        available.x=available.x/2;
        if (ImGui::BeginChild("model",available,1))
        {

            if (ImGui::Selectable("Scene root",selected_model==0))
            {
                selected_model=0;
            }

            for (int i=1;i<=active_scene.m_meshes.size();++i)
            {
                ImGui::PushID(i);
                auto& mesh=active_scene.m_meshes[i-1];
                if (ImGui::Selectable(mesh.name.c_str(),selected_model==i))
                {
                    selected_model=i;
                }
                ImGui::PopID();
            }
        }
        ImGui::EndChild();

        ImGui::SameLine();

        if (ImGui::BeginChild("model info",available,1))
        {
            ImGui::Text("Selected model id:%d", selected_model);


            static int min_depth=4;
            static int max_depth=10;
            static bool stretch=false;
            static float triangle_to_box_stop_ratio=1.f;
            static bool swapYZ=true;
            if (selected_model==0)
            {
                ImGui::Text("Root node selected");

                if (ImGui::Button("Select space"))
                {
                    active_scene.set_active_space(0);
                }

                if (ImGui::Button("build AS space"))
                {
                    pause();
                    device_data_mutex.lock();
                    // if (mesh.space_id!=UINT32_MAX)
                    //     active_scene.as.free_space(mesh.space_id);
                    auto new_id=active_scene.as.create_space_for_models(active_scene.get_mesh_transform(), stretch,swapYZ,min_depth,max_depth,triangle_to_box_stop_ratio);
                    active_scene.set_active_space(new_id);//select built mesh
                    device_data_mutex.unlock();
                    play();
                }
            }
            else
            {
                auto& mesh=active_scene.m_meshes[selected_model-1];
                ImGui::Text("Model id %d", selected_model);
                ImGui::Text("Model space id: %d",mesh.space_id);

                if (ImGui::Button("Select space"))
                {
                    device_data_mutex.lock();
                    active_scene.set_active_space(mesh.space_id);
                    device_data_mutex.unlock();
                }


                ImGui::Text("Number of triangles: %d",(uint32_t)mesh.triangles.size());
                ImGui::Text("First triangle: %d",0);

                ImGui::Text("Number of vertices: %d",(uint32_t)mesh.vertices.size());
                ImGui::Text("First vertex: %d",0);


                if (ImGui::Button("build AS space"))
                {
                    pause();
                    device_data_mutex.lock();
                    if (mesh.space_id!=UINT32_MAX)
                        active_scene.as.free_space(mesh.space_id);
                    mesh.space_id=active_scene.as.create_space_for_models(std::views::single(std::pair<glm::mat4,Mesh&>{glm::mat4(1),mesh}), stretch,swapYZ,min_depth,max_depth,triangle_to_box_stop_ratio);
                    active_scene.set_active_space(mesh.space_id);//select built mesh
                    device_data_mutex.unlock();
                    play();
                }
            }



            ImGui::InputInt("Min depth",&min_depth,1);
            ImGui::InputInt("Max depth",&max_depth,1);
            ImGui::InputFloat("Triangle to box stop ratio",&triangle_to_box_stop_ratio,0.1);
            ImGui::Checkbox("Stretch",&stretch);
            ImGui::Checkbox("Swap y and z",&swapYZ);
        }
        ImGui::EndChild();
    }
    ImGui::End();
    return open;
}


void RaytracerController::next_frame()
{
    const bool profiling_this_frame=enable_profiler;

    if (profiling_this_frame)
        CUDA_CHECK(cudaProfilerStart());

    //HOST WAIT
    auto image_id=swap_chain->acquireNextImage(stop_cuda);

    if (stop_cuda)
    {
        if (image_id!=-1)
            swap_chain->cancelImage(image_id);
        return;
    }

    auto& image=swap_chain->getImage(image_id);


    device_data_mutex.lock();
    CUDA_CHECK(cudaEventRecord(start));
    draw_cuda(image.cuda_surface,image.width,image.height);
    CUDA_CHECK(cudaEventRecord(finish));

    //HOST WAIT
    CUDA_CHECK(cudaEventSynchronize(finish));
    swap_chain->present(image_id);
    device_data_mutex.unlock();

    if (profiling_this_frame)
    {
        CUDA_CHECK(cudaProfilerStop());
    }


    {
        std::lock_guard lock(ui_mutex);
        frame_counter+=1;
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, finish);
        last_frame_time=milliseconds;
        last_frames_time.push_back(last_frame_time);
        if (last_frames_time.size()>=frames_to_track)
            last_frames_time.pop_front();
    }
}

bool RaytracerController::ui_table()
{
    static bool open=true;
    if (ImGui::Begin("Table",&open))
    {
        auto& soa_tables=scenes[active_scene_number].soa_tables;

        if (ImGui::BeginCombo("Select table", active_table.c_str()))
        {
            for (auto& [key,val]:soa_tables)
            {
                const bool is_selected = (key==active_table);

                if (ImGui::Selectable(key.c_str(), is_selected))
                {
                    active_table=key;
                }

                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        bool data_available=!cuda_is_running && device_data_mutex.try_lock();
        if (!active_table.empty())
            soa_tables.at(active_table)->draw(data_available);
        if (data_available)
            device_data_mutex.unlock();

    }
    ImGui::End();

    return open;
}
