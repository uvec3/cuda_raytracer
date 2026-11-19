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
#include <clip.h>

SceneData::SceneData() : explorer(std::make_unique<ContainerExplorer>()) {}
SceneData::~SceneData() = default;
SceneData::SceneData(SceneData&&) noexcept = default;
SceneData& SceneData::operator=(SceneData&&) noexcept = default;

void OctreeAS::register_containers(ContainerExplorer& explorer)
{
    explorer.register_container("octree.as", as);
    explorer.register_container("octree.box_content", box_content);
}

void UniformGridAS::register_containers(ContainerExplorer& explorer)
{
    explorer.register_container("uniform_grid.grids", grids);
    explorer.register_container("uniform_grid.as", as);
    explorer.register_container("uniform_grid.box_content", box_content);
}

void KDTreeAS::register_containers(ContainerExplorer& explorer)
{
    explorer.register_container("kdtree.trees", trees);
    explorer.register_container("kdtree.as", as);
    explorer.register_container("kdtree.box_content", box_content);
}

void BVHAS::register_containers(ContainerExplorer& explorer)
{
    explorer.register_container("bvh.trees", trees);
    explorer.register_container("bvh.as", as);
    explorer.register_container("bvh.as16", as16);
    explorer.register_container("bvh.box_content", box_content);
}

void HeterogeneousAS::register_containers(ContainerExplorer& explorer)
{
    explorer.register_container("heterogeneous.trees", trees);
    explorer.register_container("heterogeneous.instances", instances);
}


RaytracerController::RaytracerController()
{
    createRaytracer();
    swap_chain=new CudaVulkanSwapChain{5};

    //register callbacks
    vkbase::imgui::addOnUiCallback(WRAP_MEMBER_FUNC(ui));


    CUDA_CHECK(cudaStreamCreateWithFlags(&render_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&start, cudaEventDefault));
    CUDA_CHECK(cudaEventCreateWithFlags(&finish, cudaEventBlockingSync));

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
    if (render_stream)
        CUDA_CHECK(cudaStreamDestroy(render_stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(finish));


    destroyRaytracer();
    delete swap_chain;
}

static void collect_scene_asset_keys(const std::string& dir, std::vector<std::string>& out_keys)
{
    for (const auto& entry : vkbase::assets::directory(dir))
    {
        const std::string filename = entry.path().filename().generic_string();
        if (filename.empty())
            continue;

        const std::string asset_key = dir.empty() ? filename : (dir + "/" + filename);
        if (entry.is_directory())
        {
            collect_scene_asset_keys(asset_key, out_keys);
        }
        else if (entry.is_regular_file())
        {
            out_keys.push_back(asset_key);
        }
    }
}

void RaytracerController::initScenes()
{
    std::vector<std::string> scene_keys;
    collect_scene_asset_keys("Scenes", scene_keys);
    std::sort(scene_keys.begin(), scene_keys.end());

    scenes.reserve(scene_keys.size());
    for (const auto& scene_name : scene_keys)
    {
        SceneData s;
        s.scene = std::make_unique<Scene>(scene_name);
        scenes.push_back(std::move(s));
    }
}

bool RaytracerController::ensure_scene_loaded(int scene_index)
{
    if (scene_index < 0 || scene_index >= static_cast<int>(scenes.size()))
        return false;

    auto& s = scenes[scene_index];
    if (!s.scene)
        return false;
    if (s.scene->is_loaded())
        return true;

    const std::string_view scene_data = vkbase::assets::get(s.scene->get_name());
    if (scene_data.empty())
        return false;

    s.scene->load(scene_data, load_scene_textures);
    if (!s.scene->is_loaded())
        return false;

    s.explorer->register_container("nodes", s.scene->m_nodes);

    s.scene->as_octree.register_containers(*s.explorer);
    s.scene->as_uniform_grid.register_containers(*s.explorer);
    s.scene->as_kdtree.register_containers(*s.explorer);
    s.scene->as_bvh.register_containers(*s.explorer);
    s.scene->as_heterogeneous.register_containers(*s.explorer);
    s.explorer->register_container("scene.materials", s.scene->m_materials, [this, sc = s.scene.get()](std::size_t)
    {
        std::lock_guard lock(device_data_mutex);
        sc->sync_materials_to_device();
    });

    s.explorer->register_container("scene.nodes", s.scene->m_nodes);

    s.explorer->register_container("scene.textures", s.scene->m_textures);
    s.explorer->register_container("scene.meshes", s.scene->m_meshes);

    return true;
}

bool RaytracerController::has_active_loaded_scene() const
{
    return active_scene_number >= 0 &&
           active_scene_number < static_cast<int>(scenes.size()) &&
           scenes[active_scene_number].scene &&
           scenes[active_scene_number].scene->is_loaded();
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

        if ((ImGui::IsKeyDown(ImGuiKey_LeftAlt) || ImGui::IsKeyDown(ImGuiKey_RightAlt)) &&
            !ImGui::IsKeyDown(ImGuiKey_LeftCtrl) &&
            !ImGui::IsKeyDown(ImGuiKey_RightCtrl) &&
            ImGui::IsKeyPressed(ImGuiKey_A, false))
        {
            build_as();
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
    {"2560 x 2560", 2560, 2560},
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
        ui_keys();

        if (ImGui::BeginPopupContextItem("##ImageContextMenu"))
        {
            if (ImGui::MenuItem("Copy Image"))
            {
                copy_image_to_clipboard();
            }
            ImGui::EndPopup();
        }
    }
}

void RaytracerController::copy_image_to_clipboard()
{
    if (!swap_chain)
        return;

    std::vector<uint8_t> pixels;
    uint32_t width = 0;
    uint32_t height = 0;

    if (!swap_chain->copyPresentedImageToHost(pixels, width, height) || pixels.empty())
        return;

    if (!copy_image_with_alpha)
    {
        // Keep 32bpp RGBA spec because clip's Windows backend only supports 32bpp
        // and premultiplies RGB by the masked alpha value.
        for (size_t i = 3; i < pixels.size(); i += 4)
        {
            pixels[i] = 255;
        }
    }

    clip::image_spec spec;
    spec.width = width;
    spec.height = height;
    spec.bits_per_pixel = 32;
    spec.bytes_per_row = width * 4;
    spec.red_mask = 0x000000ff;
    spec.green_mask = 0x0000ff00;
    spec.blue_mask = 0x00ff0000;
    spec.alpha_mask = 0xff000000;
    spec.red_shift = 0;
    spec.green_shift = 8;
    spec.blue_shift = 16;
    spec.alpha_shift = 24;

    clip::image img(reinterpret_cast<const char*>(pixels.data()), spec);
    clip::set_image(img);
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



static void render_node_tree(Scene& scene, uint32_t node_id, uint32_t& selected_node_id, uint32_t active_space_id, int& id_counter, uint32_t& out_activate_space_id)
{
    if (node_id >= scene.m_nodes.size())
        return;

    const Node& node = scene.m_nodes[node_id];
    ImGui::PushID(++id_counter);

    bool is_selected = (selected_node_id == node_id);
    bool is_active_render = (node.as_ref.is_valid() && node.as_ref.space_id == active_space_id);
    const bool is_shared = scene.is_mesh_node_shared(node_id);

    std::string label = node.name.empty() ? "Node" : node.name;
    if (node.as_ref.is_valid())
    {
        std::string type_name = "AS";
        if (node.as_ref.type == ASType::Optix) type_name = "OptiX";
        else if (node.as_ref.type == ASType::Octree) type_name = "Octree";
        else if (node.as_ref.type == ASType::UniformGrid) type_name = "UniformGrid";
        else if (node.as_ref.type == ASType::KDTree) type_name = "KDTree";
        else if (node.as_ref.type == ASType::BVH) type_name = "BVH";
        else if (node.as_ref.type == ASType::Heterogeneous) type_name = "Heterogeneous";
        label += " [" + type_name + " #" + std::to_string(node.as_ref.space_id) + (is_shared ? " (shared)" : "") + "]";
    }
    else if (is_shared)
    {
        label += " [shared]";
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

    const bool is_leaf = node.children.empty();
    if (is_leaf)
    {
        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    }

    if (is_active_render)
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 1.0f, 0.4f, 1.0f));
    const bool node_open = ImGui::TreeNodeEx(
        reinterpret_cast<void*>(static_cast<uintptr_t>(node_id)), flags, "%s", label.c_str());
    if (is_active_render)
        ImGui::PopStyleColor();

    if (!ImGui::IsItemToggledOpen())
    {
        if (ImGui::IsItemClicked())
        {
            selected_node_id = node_id;
        }
        if (node.as_ref.is_valid() &&
            scene.spaces.is_allocated(node.as_ref.space_id) &&
            ImGui::IsItemHovered() &&
            ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
        {
            selected_node_id = node_id;
            out_activate_space_id = node.as_ref.space_id;
        }
    }

    if (!is_leaf && node_open)
    {
        for (uint32_t child_id : node.children)
        {
            render_node_tree(scene, child_id, selected_node_id, active_space_id, id_counter, out_activate_space_id);
        }
        ImGui::TreePop();
    }

    ImGui::PopID();
}

void RaytracerController::build_as()
{
    if (!has_active_loaded_scene())
        return;

    auto& scene_data = scenes[active_scene_number];
    if (scene_data.scene->m_nodes.empty())
        return;

    auto& active_scene = *scene_data.scene;
    if (scene_data.selected_node_id >= active_scene.m_nodes.size())
    {
        scene_data.selected_node_id = active_scene.root_node_id;
    }
    if (scene_data.selected_node_id >= active_scene.m_nodes.size())
        return;

    const Node& node = active_scene.m_nodes[scene_data.selected_node_id];
    if (node.count_subtree_triangles(active_scene.m_nodes, active_scene.m_meshes) == 0)
        return;

    static constexpr ASType as_types[] = {
        ASType::Octree,
        ASType::UniformGrid,
        ASType::KDTree,
        ASType::BVH,
        ASType::Optix,
        ASType::Heterogeneous
    };
    const int idx = std::clamp(selected_as_type, 0, static_cast<int>(std::size(as_types)) - 1);
    const ASType as_type = as_types[idx];

    pause();
    {
        std::lock_guard lock(device_data_mutex);
        const uint32_t new_space_id = active_scene.build_node_as(scene_data.selected_node_id, as_type);
        if (new_space_id != UINT32_MAX && auto_select_after_build)
        {
            active_scene.set_active_space(new_space_id);
        }
    }
    if (auto_play_after_build)
    {
        play();
    }
}

bool RaytracerController::ui_scenes_manager()
{
    static bool open=true;
    if (ImGui::Begin("Active scene",&open))
    {
        const char* preview_name = "Select a scene...";
        if (active_scene_number >= 0 &&
            active_scene_number < static_cast<int>(scenes.size()) &&
            scenes[active_scene_number].scene)
        {
            preview_name = scenes[active_scene_number].scene->get_name().c_str();
        }

        if (ImGui::BeginCombo("Select scene", preview_name))
        {
            for (int i = 0; i < static_cast<int>(scenes.size()); ++i)
            {
                auto& s = scenes[i];
                if (!s.scene)
                    continue;

                const std::string item_label =
                    s.scene->get_name() + (s.scene->is_loaded() ? " [loaded]" : "") + "###" + s.scene->get_name();
                const bool is_selected = (i == active_scene_number);
                if (ImGui::Selectable(item_label.c_str(), is_selected))
                {
                    if (ensure_scene_loaded(i))
                    {
                        active_scene_number = i;
                    }
                }
                if (is_selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        ImGui::Checkbox("Load textures", &load_scene_textures);

        if (!has_active_loaded_scene())
        {
            ImGui::TextUnformatted(scenes.empty() ? "No scenes found." : "Select a scene above to load it.");
        }
        else
        {
            auto& scene_data = scenes[active_scene_number];
            auto& active_scene = *scene_data.scene;

            const auto available = ImGui::GetContentRegionAvail();
            constexpr float splitter_width = 8.0f;
            const float child_height = std::max(1.0f, available.y);
            const float usable_width = std::max(0.0f, available.x - splitter_width);
            const float min_pane_width = std::min(120.0f, usable_width * 0.5f);
            const float max_left_width = std::max(min_pane_width, usable_width - min_pane_width);

            static float scene_tree_split_ratio = 0.35f;
            static float scene_splitter_drag_offset = 0.0f;
            const float split_area_start_x = ImGui::GetCursorScreenPos().x;
            const float left_width = std::clamp(usable_width * scene_tree_split_ratio, min_pane_width, max_left_width);
            const float right_width = std::max(0.0f, usable_width - left_width);

            uint32_t activate_space_id = UINT32_MAX;
            if (ImGui::BeginChild("scene_tree", ImVec2(std::max(1.0f, left_width), child_height), true))
            {
                int id_counter = 0;
                render_node_tree(active_scene, active_scene.root_node_id, scene_data.selected_node_id, active_scene.get_active_space(), id_counter, activate_space_id);
            }
            ImGui::EndChild();

            if (activate_space_id != UINT32_MAX)
            {
                std::lock_guard lock(device_data_mutex);
                active_scene.set_active_space(activate_space_id);
            }

            ImGui::SameLine(0.0f, 0.0f);
            ImGui::InvisibleButton("##scene_splitter", ImVec2(splitter_width, child_height));
            const bool splitter_hovered = ImGui::IsItemHovered();
            const bool splitter_active = ImGui::IsItemActive();
            if (splitter_hovered || splitter_active)
            {
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
            }
            if (ImGui::IsItemActivated())
            {
                scene_splitter_drag_offset = ImGui::GetIO().MousePos.x - ImGui::GetItemRectMin().x;
            }
            if (splitter_active && usable_width > 2.0f * min_pane_width)
            {
                const float new_left = std::clamp(
                    ImGui::GetIO().MousePos.x - split_area_start_x - scene_splitter_drag_offset,
                    min_pane_width,
                    max_left_width);
                scene_tree_split_ratio = new_left / usable_width;
            }

            {
                const ImVec2 rect_min = ImGui::GetItemRectMin();
                const ImVec2 rect_max = ImGui::GetItemRectMax();
                const float center_x = 0.5f * (rect_min.x + rect_max.x);
                const ImGuiCol col_idx = splitter_active
                    ? ImGuiCol_SeparatorActive
                    : (splitter_hovered ? ImGuiCol_SeparatorHovered : ImGuiCol_Separator);
                const float line_thickness = (splitter_hovered || splitter_active) ? 3.0f : 1.0f;
                const float pad_y = std::min(12.0f, (rect_max.y - rect_min.y) * 0.25f);
                ImGui::GetWindowDrawList()->AddLine(
                    ImVec2(center_x, rect_min.y + pad_y),
                    ImVec2(center_x, rect_max.y - pad_y),
                    ImGui::GetColorU32(col_idx),
                    line_thickness);
            }

            ImGui::SameLine(0.0f, 0.0f);

            if (ImGui::BeginChild("node_inspector", ImVec2(std::max(1.0f, right_width), child_height), true))
            {
                ui_node_inspector();
            }
            ImGui::EndChild();
        }
    }
    ImGui::End();
    return open;
}

void RaytracerController::ui_node_inspector()
{
    if (!has_active_loaded_scene())
        return;

    auto& scene_data = scenes[active_scene_number];
    if (scene_data.scene->m_nodes.empty())
        return;

    auto& active_scene = *scene_data.scene;
    if (scene_data.selected_node_id >= active_scene.m_nodes.size())
    {
        scene_data.selected_node_id = active_scene.root_node_id;
    }

    if (scene_data.selected_node_id >= active_scene.m_nodes.size())
        return;

    Node& node = active_scene.m_nodes[scene_data.selected_node_id];

    ImGui::Text("Node: %s (#%u)%s",
                node.name.c_str(),
                scene_data.selected_node_id,
                active_scene.is_mesh_node_shared(scene_data.selected_node_id) ? " [shared]" : "");
    ImGui::Separator();

    auto trans = node.getTranslation();
    auto scale = node.getScale();
    ImGui::Text("Translation: (%.2f, %.2f, %.2f)", trans.x, trans.y, trans.z);
    ImGui::Text("Scale:       (%.2f, %.2f, %.2f)", scale.x, scale.y, scale.z);

    if (node.meshId > 0 && node.meshId - 1 < active_scene.m_meshes.size())
    {
        auto& mesh = active_scene.m_meshes[node.meshId - 1];
        ImGui::Text("Direct Mesh: %s", mesh.name.c_str());
        ImGui::Text("  Triangles: %u", (uint32_t)mesh.d_triangles.size());
        ImGui::Text("  Vertices:  %u", (uint32_t)mesh.d_vertices.size());
    }
    else
    {
        ImGui::Text("Direct Mesh: None");
    }

    size_t sub_meshes = node.count_subtree_meshes(active_scene.m_nodes);
    size_t sub_triangles = node.count_subtree_triangles(active_scene.m_nodes, active_scene.m_meshes);
    size_t sub_vertices = node.count_subtree_vertices(active_scene.m_nodes, active_scene.m_meshes);
    ImGui::Text("Subtree Meshes:    %zu", sub_meshes);
    ImGui::Text("Subtree Triangles: %zu", sub_triangles);
    ImGui::Text("Subtree Vertices:  %zu", sub_vertices);

    ImGui::Separator();
    ImGui::Text("Acceleration Structure");

    if (node.as_ref.is_valid())
    {
        const char* type_str = "Unknown";
        if (node.as_ref.type == ASType::Optix) type_str = "OptiX";
        else if (node.as_ref.type == ASType::Octree) type_str = "Octree";
        else if (node.as_ref.type == ASType::UniformGrid) type_str = "UniformGrid";
        else if (node.as_ref.type == ASType::KDTree) type_str = "KDTree";
        else if (node.as_ref.type == ASType::BVH) type_str = "BVH";
        else if (node.as_ref.type == ASType::Heterogeneous) type_str = "Heterogeneous";
        ImGui::Text("Status: Built (%s, Space #%u)", type_str, node.as_ref.space_id);
        if (active_scene.spaces.is_allocated(node.as_ref.space_id))
        {
            auto sp = active_scene.spaces[node.as_ref.space_id];
            if (node.as_ref.type == ASType::Octree)
                active_scene.as_octree.info_ui(node.as_ref.space_id, sp);
            else if (node.as_ref.type == ASType::UniformGrid)
                active_scene.as_uniform_grid.info_ui(node.as_ref.space_id, sp);
            else if (node.as_ref.type == ASType::KDTree)
                active_scene.as_kdtree.info_ui(node.as_ref.space_id, sp);
            else if (node.as_ref.type == ASType::BVH)
                active_scene.as_bvh.info_ui(node.as_ref.space_id, sp);
            else if (node.as_ref.type == ASType::Heterogeneous)
                active_scene.as_heterogeneous.info_ui(node.as_ref.space_id, sp);
            else if (node.as_ref.type == ASType::Optix)
                active_scene.as_optix.info_ui(node.as_ref.space_id, sp);
        }
        if (const MeshBuildRecord* mesh_rec = active_scene.as_heterogeneous.find_mesh_record_by_space(node.as_ref.space_id))
        {
            if (ImGui::TreeNode("Heterogeneous Candidate Benchmarks"))
            {
                ImGui::Text("  Instances:        %u", mesh_rec->instance_count);
                ImGui::Text("  Selected Variant: %s", mesh_rec->selected_variant_name.c_str());
                ImGui::Text("  BLAS Build Time:  %.2f ms", mesh_rec->selected_build_ms);
                for (const auto& cand : mesh_rec->candidates)
                {
                    ImGui::Text("    %-24s | %6.3f ms | %6.2f ms build | %6.1f B/tri %s",
                                cand.variant_name.c_str(),
                                cand.probe_trace_ms,
                                cand.build_ms,
                                cand.bytes_per_triangle,
                                cand.within_memory_limit ? "" : "[over]");
                }
                ImGui::TreePop();
            }
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

    const char* as_type_names[] = {"Octree", "Uniform Grid", "KD-Tree", "BVH", "OptiX (Hardware RT)", "Heterogeneous"};
    ImGui::Combo("AS Type", &selected_as_type, as_type_names, 6);

    ImGui::BeginDisabled(sub_triangles == 0);
    if (ImGui::Button("Build AS"))
    {
        build_as();
    }
    ImGui::EndDisabled();

    active_scene.config_ui();

    if (selected_as_type == 0)
    {
        active_scene.as_octree.config_ui();
    }
    else if (selected_as_type == 1)
    {
        active_scene.as_uniform_grid.config_ui();
    }
    else if (selected_as_type == 2)
    {
        active_scene.as_kdtree.config_ui();
    }
    else if (selected_as_type == 3)
    {
        active_scene.as_bvh.config_ui();
    }
    else if (selected_as_type == 5)
    {
        active_scene.as_heterogeneous.config_ui();
    }

    ImGui::Separator();

    ImGui::Checkbox("Auto select for render after build", &auto_select_after_build);
    ImGui::Checkbox("Auto play after build", &auto_play_after_build);

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
        active_scene.free_node_as(scene_data.selected_node_id);
        device_data_mutex.unlock();
    }
    ImGui::EndDisabled();
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
    if (!has_active_loaded_scene())
    {
        bool open = true;
        if (ImGui::Begin("Table", &open))
        {
            ImGui::TextUnformatted("No scene loaded.");
        }
        ImGui::End();
        return open;
    }
    return scenes[active_scene_number].explorer->draw("Table");
}
