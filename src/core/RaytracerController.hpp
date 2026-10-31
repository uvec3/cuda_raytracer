#pragma once

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


struct WindowInfo
{
    std::string name;
    std::function<bool()> draw;
    bool open=true;
};

class Raytracer;
class CudaVulkanSwapChain;
class Scene;
class soa_ui_table_interface;


struct SceneData
{
    std::unique_ptr<Scene> scene;
    std::map<std::string,std::unique_ptr<soa_ui_table_interface>> soa_tables;
    uint32_t active_space=0;

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



    std::string active_table;

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