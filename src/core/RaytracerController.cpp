#include "RaytracerController.hpp"
#include "../types.hpp"
#include "../raytracer/Scene.hpp"
#include <container_explorer.hpp>
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

SceneData::SceneData() : explorer(std::make_unique<ContainerExplorer>()) {}
SceneData::~SceneData() = default;
SceneData::SceneData(SceneData&&) noexcept = default;
SceneData& SceneData::operator=(SceneData&&) noexcept = default;

void OctreeAS::register_containers(ContainerExplorer& explorer)
{
    explorer.register_container("octree.spaces", spaces);
    explorer.register_container("octree.as", as);
    explorer.register_container("octree.primitives", primitives);
    explorer.register_container("octree.box_content", box_content);
    explorer.register_container("octree.vertices", vertices);

}


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
    windows.push_back({"Parameters", WRAP_MEMBER_FUNC(ui_rendering_parameters)});
    windows.push_back({"Log", vkbase::imgui::drawEngineMessagesLog});
    windows.push_back({"Table",WRAP_MEMBER_FUNC(ui_table)});
    windows.push_back({"Debug pixel",WRAP_MEMBER_FUNC(ui_debug_pixel)});
    windows.push_back({"Select scene",WRAP_MEMBER_FUNC(ui_scenes_manager)});
    windows.push_back({"Output image", [this]()
    {
        bool open=true;
        if (ImGui::Begin("Output",&open, ImGuiWindowFlags_HorizontalScrollbar))
        {
            ui_image();
        }
        ImGui::End();

        return open;
    }});


    initScenes();
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

        s.scene->as.register_containers(*s.explorer);
        s.explorer->register_container("scene.materials", s.scene->m_materials, [this, sc = s.scene.get()](std::size_t)
        {
            std::lock_guard lock(device_data_mutex);
            sc->sync_materials_to_device();
        });
        s.explorer->register_container("scene.textures", s.scene->m_textures);
        s.explorer->register_container("scene.meshes",s.scene->m_meshes);
        for (auto& mesh: s.scene->m_meshes)
        {
            s.explorer->register_container(mesh.name+".triangles",mesh.triangles);
            s.explorer->register_container(mesh.name+".vertices",mesh.vertices);
        }

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
    if (hide_ui && resolution_mode != 0 && !scale_display_to_window)
    {
        window_flags |= ImGuiWindowFlags_HorizontalScrollbar;
    }


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

static const ResolutionPreset s_resolution_presets[] = {
    {"Window (Dynamic)",        0,    0},
    {"1280 x 720 (720p HD)", 1280,  720},
    {"1920 x 1080 (1080p FHD)", 1920, 1080},
    {"2560 x 1440 (1440p QHD)", 2560, 1440},
    {"3440 x 1440 (UWQHD 21:9)", 3440, 1440},
    {"3840 x 2160 (4K UHD)", 3840, 2160},
    {"7680 x 4320 (8K UHD)", 7680, 4320},
    {"Custom",                  0,    0}
};
static const int s_resolution_preset_count = static_cast<int>(sizeof(s_resolution_presets) / sizeof(s_resolution_presets[0]));

const ResolutionPreset* get_resolution_presets()
{
    return s_resolution_presets;
}

int get_resolution_preset_count()
{
    return s_resolution_preset_count;
}

glm::uvec2 RaytracerController::get_target_resolution(glm::vec2 available_size) const
{
    if (resolution_mode == 0) // Window (Dynamic)
    {
        uint32_t w = available_size.x > 1.0f ? static_cast<uint32_t>(available_size.x) : 1u;
        uint32_t h = available_size.y > 1.0f ? static_cast<uint32_t>(available_size.y) : 1u;
        return {std::clamp(w, 1u, 10000u), std::clamp(h, 1u, 10000u)};
    }
    if (resolution_mode >= 1 && resolution_mode < s_resolution_preset_count - 1)
    {
        return {s_resolution_presets[resolution_mode].width, s_resolution_presets[resolution_mode].height};
    }
    // Custom
    uint32_t w = std::clamp(static_cast<uint32_t>(custom_resolution[0]), 1u, 10000u);
    uint32_t h = std::clamp(static_cast<uint32_t>(custom_resolution[1]), 1u, 10000u);
    return {w, h};
}

void RaytracerController::ui_image()
{
    ui_toolbar();
    auto available = ImGui::GetContentRegionAvail();
    auto target_res = get_target_resolution({available.x, available.y});

    swap_chain->setImageSize(target_res.x, target_res.y);
    current_image_size = target_res;

    auto image_to_present = swap_chain->getImageToPresent();
    if (image_to_present)
    {
        ImVec2 display_size(static_cast<float>(image_to_present->image.width),
                            static_cast<float>(image_to_present->image.height));

        if (resolution_mode != 0 && scale_display_to_window && available.x > 0.0f && available.y > 0.0f)
        {
            float aspect = static_cast<float>(image_to_present->image.width) / static_cast<float>(image_to_present->image.height);
            float disp_w = available.x;
            float disp_h = disp_w / aspect;
            if (disp_h > available.y)
            {
                disp_h = available.y;
                disp_w = disp_h * aspect;
            }
            display_size = ImVec2(disp_w, disp_h);
        }

        ImGui::Image(image_to_present->image_imgui, display_size);
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



static void render_node_tree(Node& node, Node*& selected_node, uint32_t active_space_id, int& id_counter)
{
    ImGui::PushID(++id_counter);

    bool is_selected = (selected_node == &node);
    bool is_active_render = (node.as_ref.is_valid() && node.as_ref.space_id == active_space_id);

    std::string label = node.name.empty() ? "Node" : node.name;
    if (node.as_ref.is_valid())
    {
        label += (node.as_ref.type == ASType::Octree)
            ? (" [Octree #" + std::to_string(node.as_ref.space_id) + "]")
            : (" [BVH #" + std::to_string(node.as_ref.space_id) + "]");
    }
    if (is_active_render)
    {
        label += " [Active]";
    }

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
    if (is_selected)
    {
        flags |= ImGuiTreeNodeFlags_Selected;
    }

    if (node.children.empty())
    {
        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
        if (is_active_render)
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 1.0f, 0.4f, 1.0f));
        ImGui::TreeNodeEx((void*)&node, flags, "%s", label.c_str());
        if (is_active_render)
            ImGui::PopStyleColor();

        if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
        {
            selected_node = &node;
        }
    }
    else
    {
        if (is_active_render)
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 1.0f, 0.4f, 1.0f));
        bool node_open = ImGui::TreeNodeEx((void*)&node, flags, "%s", label.c_str());
        if (is_active_render)
            ImGui::PopStyleColor();

        if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
        {
            selected_node = &node;
        }

        if (node_open)
        {
            for (auto& child : node.children)
            {
                render_node_tree(child, selected_node, active_space_id, id_counter);
            }
            ImGui::TreePop();
        }
    }

    ImGui::PopID();
}

bool RaytracerController::ui_scenes_manager()
{
    static bool open=true;
    if (ImGui::Begin("Active scene",&open))
    {
        auto& scene_data = scenes[active_scene_number];
        auto& active_scene = *scene_data.scene;

        if (ImGui::BeginCombo("Select scene", active_scene.get_name().c_str()))
        {
            for (int i=0;i<scenes.size();++i)
            {
                auto& s=scenes[i];
                if (ImGui::Selectable(s.scene->get_name().c_str(), i==active_scene_number))
                {
                    active_scene_number=i;
                }
            }
            ImGui::EndCombo();
        }

        auto available = ImGui::GetContentRegionAvail();
        float left_width = available.x * 0.45f;
        float right_width = available.x - left_width - 8.0f;

        if (ImGui::BeginChild("scene_tree", ImVec2(left_width, available.y), true))
        {
            int id_counter = 0;
            render_node_tree(active_scene.root_node, scene_data.selected_node, active_scene.get_active_space(), id_counter);
        }
        ImGui::EndChild();

        ImGui::SameLine();

        if (ImGui::BeginChild("node_inspector", ImVec2(right_width, available.y), true))
        {
            static bool auto_select = true;
            static bool auto_play = true;

            if (scene_data.selected_node == nullptr)
            {
                scene_data.selected_node = &active_scene.root_node;
            }

            Node& node = *scene_data.selected_node;

            ImGui::Text("Node: %s", node.name.c_str());
            ImGui::Separator();

            auto trans = node.getTranslation();
            auto scale = node.getScale();
            ImGui::Text("Translation: (%.2f, %.2f, %.2f)", trans.x, trans.y, trans.z);
            ImGui::Text("Scale:       (%.2f, %.2f, %.2f)", scale.x, scale.y, scale.z);

            if (node.meshId > 0 && node.meshId - 1 < active_scene.m_meshes.size())
            {
                auto& mesh = active_scene.m_meshes[node.meshId - 1];
                ImGui::Text("Direct Mesh: %s", mesh.name.c_str());
                ImGui::Text("  Triangles: %u", (uint32_t)mesh.triangles.size());
                ImGui::Text("  Vertices:  %u", (uint32_t)mesh.vertices.size());
            }
            else
            {
                ImGui::Text("Direct Mesh: None");
            }

            size_t sub_meshes = node.count_subtree_meshes();
            size_t sub_triangles = node.count_subtree_triangles(active_scene.m_meshes);
            size_t sub_vertices = node.count_subtree_vertices(active_scene.m_meshes);
            ImGui::Text("Subtree Meshes:    %zu", sub_meshes);
            ImGui::Text("Subtree Triangles: %zu", sub_triangles);
            ImGui::Text("Subtree Vertices:  %zu", sub_vertices);

            ImGui::Separator();
            ImGui::Text("Acceleration Structure");

            if (node.as_ref.is_valid())
            {
                const char* type_str = (node.as_ref.type == ASType::Octree) ? "Octree" : "BVH";
                ImGui::Text("Status: Built (%s, Space #%u)", type_str, node.as_ref.space_id);
                if (active_scene.as.spaces.is_allocated(node.as_ref.space_id))
                {
                    auto sp = active_scene.as.spaces[node.as_ref.space_id];
                    ImGui::Text("  Boxes:      %u", (uint32_t)sp.as_size);
                    ImGui::Text("  Contents:   %u", (uint32_t)sp.box_content_size);
                    ImGui::Text("  Primitives: %u", (uint32_t)sp.primitives_size);
                }
                if (active_scene.get_active_space() == node.as_ref.space_id)
                {
                    ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.4f, 1.0f), "Currently Active in Renderer");
                }
            }
            else
            {
                ImGui::Text("Status: Not built");
            }

            static int selected_as_type = 0; // 0: Octree, 1: BVH
            const char* as_type_names[] = {"Octree", "BVH (Coming soon)"};
            ImGui::Combo("AS Type", &selected_as_type, as_type_names, 2);

            if (ImGui::Button("Build AS"))
            {
                pause();
                device_data_mutex.lock();
                if (node.as_ref.is_valid())
                {
                    active_scene.as.free_space(node.as_ref.space_id);
                    node.as_ref = {};
                }

                auto relative_meshes = active_scene.get_mesh_transform(&node);
                if (!relative_meshes.empty())
                {
                    uint32_t new_space_id = active_scene.as.create_space_for_models(
                        relative_meshes);
                    node.as_ref = {new_space_id, ASType::Octree};

                    if (auto_select)
                    {
                        active_scene.set_active_space(new_space_id);
                    }
                }
                device_data_mutex.unlock();
                if (auto_play)
                    play();
            }

            active_scene.as.config_ui();


            ImGui::Separator();



            ImGui::Checkbox("Auto select for render after build", &auto_select);
            ImGui::Checkbox("Auto play after build", &auto_play);
            ImGui::BeginDisabled(sub_triangles == 0 || selected_as_type != 0);

            ImGui::EndDisabled();

            ImGui::SameLine();

            ImGui::BeginDisabled(!node.as_ref.is_valid());
            if (ImGui::Button("Select for Render"))
            {
                device_data_mutex.lock();
                active_scene.set_active_space(node.as_ref.space_id);
                device_data_mutex.unlock();
            }
            ImGui::EndDisabled();

            ImGui::SameLine();

            ImGui::BeginDisabled(!node.as_ref.is_valid());
            if (ImGui::Button("Free AS"))
            {
                pause();
                device_data_mutex.lock();
                if (node.as_ref.is_valid())
                {
                    active_scene.as.free_space(node.as_ref.space_id);
                    node.as_ref = {};
                }
                device_data_mutex.unlock();
            }
            ImGui::EndDisabled();
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

    draw_cuda(image.cuda_surface,image.width,image.height);


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
    return scenes[active_scene_number].explorer->draw("Table");
}
