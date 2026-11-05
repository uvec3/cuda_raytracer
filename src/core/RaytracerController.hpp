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
#include <glm/glm.hpp>

#include "../../vkbase/extensions/cuda/cuda.hpp"
#include <container_explorer.hpp>


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

static const ResolutionPreset* get_resolution_presets();
static int get_resolution_preset_count();

class Raytracer;
class CudaVulkanSwapChain;
class Scene;
class Node;


struct SceneData
{
    std::unique_ptr<Scene> scene;
    ContainerExplorer explorer;
    Node* selected_node = nullptr;

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
    int active_scene_number=0;


    //cuda loop
    std::atomic<bool> stop_cuda=false;
    std::atomic<bool> cuda_is_running=false;
    std::atomic<bool> wake_cuda=false;

    std::thread cuda_loop;

    uint64_t frame_counter=0;
    double last_frame_time=0;
    const int frames_to_track=500;
    std::deque<double> last_frames_time={};
    bool enable_profiler=true;

    bool fullscreen=false;
    bool hide_ui=false;
    std::mutex ui_mutex;
    std::mutex device_data_mutex;

    glm::ivec2 debug_pixel{0,0};
    glm::vec4 debug_pixel_result{0};
    glm::uvec2 current_image_size{0};

    int resolution_mode = 0; // 0: Window (Dynamic), 1: 720p, 2: 1080p, 3: 1440p, 4: UWQHD, 5: 4K, 6: 8K, 7: Custom
    int last_fixed_mode = 2; // Default fixed preset: 1080p
    int custom_resolution[2] = {1920, 1080};
    bool scale_display_to_window = true;

    glm::uvec2 get_target_resolution(glm::vec2 available_size) const;

    cudaEvent_t start, finish;

    float movement_speed=1;

    RaytracerController();
    ~RaytracerController();

    void initScenes();

    void createRaytracer();
    void destroyRaytracer();


    void ui();

    void ui_input();

    void ui_surface_input();

    void ui_menu();

    void ui_image();

    void ui_toolbar();

    bool ui_show_parameters();

    bool ui_frame_info();

    bool ui_debug_pixel();

    bool ui_scenes_manager();

    void draw_one_frame();

    void play();

    void pause();

    void next_frame();

    bool ui_table();

    void draw_cuda(cudaSurfaceObject_t surface, uint32_t width, uint32_t height);
};