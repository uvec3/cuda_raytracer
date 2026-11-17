#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../types.hpp"

#include "../../vkbase/extensions/cuda/cuda.hpp"


struct WindowInfo
{
    std::string name;
    std::function<bool()> draw;
    bool open=true;
};

struct ResolutionPreset
{
    const char* label;
    uint32_t width;
    uint32_t height;
};

const ResolutionPreset* get_resolution_presets();
int get_resolution_preset_count();

class Raytracer;
class CudaVulkanSwapChain;
class Scene;
class Node;
class ContainerExplorer;


struct SceneData
{
    std::unique_ptr<Scene> scene;
    std::unique_ptr<ContainerExplorer> explorer;
    uint32_t selected_node_id = 0;

    SceneData();
    ~SceneData();
    SceneData(SceneData&&) noexcept;
    SceneData& operator=(SceneData&&) noexcept;
};

class RaytracerController
{
public:
    CudaVulkanSwapChain* swap_chain=nullptr;
    Raytracer* raytracer=nullptr;

    std::vector<WindowInfo> windows;
    std::vector<SceneData> scenes;
    int active_scene_number=-1;


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
    std::mutex device_data_mutex;

    glm::ivec2 debug_pixel{0,0};
    glm::vec4 debug_pixel_result{0};
    glm::uvec2 current_image_size{0};

    int resolution_mode = 0; // 0: Window (Dynamic), 1..(count-2): Presets, (count-1): Custom
    int last_fixed_mode = 3; // Default fixed preset
    int custom_resolution[2] = {1920, 1080};
    bool scale_display_to_window = true;
    int selected_as_type = 3; // 0: Octree, 1: Uniform Grid, 2: KD-Tree, 3: BVH, 4: OptiX, 5: Heterogeneous
    bool auto_select_after_build = true;
    bool auto_play_after_build = true;
    bool copy_image_with_alpha = false;
    bool load_scene_textures = true;

    //frame start finish events
    cudaEvent_t start, finish;
    cudaStream_t render_stream = nullptr;

    float movement_speed=1;

    RaytracerController();
    ~RaytracerController();

    void initScenes();

    bool ensure_scene_loaded(int scene_index);

    bool has_active_loaded_scene() const;

    void createRaytracer();
    void destroyRaytracer();

    void ui();

    void ui_input();

    void ui_surface_input();

    void ui_menu();

    void ui_keys();

    void ui_image();

    void copy_image_to_clipboard();

    void ui_toolbar();

    bool ui_rendering_parameters();

    bool ui_frame_info();

    bool ui_debug_pixel();

    bool ui_scenes_manager();

    void ui_node_inspector();

    void build_as();

    void draw_one_frame();

    void play();

    void pause();

    void next_frame();

    bool ui_table();

    void draw_cuda(cudaSurfaceObject_t surface, uint32_t width, uint32_t height);

    glm::uvec2 get_target_resolution(glm::vec2 available_size) const;
};