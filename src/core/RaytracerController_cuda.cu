#include "RaytracerController.hpp"
#include "../raytracer/Raytracer.cuh"
#include "../raytracer/Scene.hpp"
#include  <imgui.h>
#include <cuda_profiler_api.h>
#include "../cuda_common.hpp"


void RaytracerController::createRaytracer()
{
    raytracer=new Raytracer{};
}

void RaytracerController::destroyRaytracer()
{
    delete raytracer;
}


bool RaytracerController::ui_show_parameters()
{
    bool open=true;

    if (ImGui::Begin("Parameters",&open))
    {
        RenderParameters& p=raytracer->parameters;

        ImGui::DragFloat("AX",&p.ax);
        ImGui::DragFloat("AZ",&p.az);
        ImGui::DragFloat3("Pos",&p.view_pos[0],0.01);
        ImGui::InputFloat("Near",&p.near_,0.01);
        ImGui::Checkbox("Show time",&p.measure_time);
        ImGui::Checkbox("Show iterations",&p.show_iterations);
        ImGui::Checkbox("Show triangles",&p.show_triangles);
        ImGui::Checkbox("Show boxes",&p.show_boxes);
        ImGui::InputInt("Time max",&p.time_max);
        ImGui::Checkbox("Fake normals",&p.use_fake_normals);
        ImGui::Checkbox("Fixed light source",&p.fixed_light_source);
        ImGui::DragFloat3("Light source",&p.light_source[0],0.01);
        ImGui::DragFloat("Diffuse lightning",&p.diffuse_light,0.01);
        ImGui::DragFloat("Ambient lightning",&p.ambient_light,0.01);

        ImGui::Separator();
        ImGui::Text("Resolution");

        bool resolution_changed = false;
        bool is_fixed = (resolution_mode != 0);
        if (ImGui::Checkbox("Fixed resolution", &is_fixed))
        {
            if (is_fixed)
                resolution_mode = (last_fixed_mode > 0 ? last_fixed_mode : 2); // Default to 1080p
            else
                resolution_mode = 0; // Window (Dynamic)
            resolution_changed = true;
        }

        static const char* const preset_names[] = {
            "Window (Dynamic)",
            "1280 x 720 (720p HD)",
            "1920 x 1080 (1080p FHD)",
            "2560 x 1440 (1440p QHD)",
            "3440 x 1440 (UWQHD 21:9)",
            "3840 x 2160 (4K UHD)",
            "7680 x 4320 (8K UHD)",
            "Custom"
        };
        constexpr int preset_count = static_cast<int>(sizeof(preset_names) / sizeof(preset_names[0]));

        if (ImGui::Combo("Preset", &resolution_mode, preset_names, preset_count))
        {
            if (resolution_mode != 0)
                last_fixed_mode = resolution_mode;
            resolution_changed = true;
        }

        if (resolution_mode == preset_count - 1) // Custom
        {
            if (ImGui::InputInt2("Custom size (W x H)", custom_resolution))
            {
                if (custom_resolution[0] < 1) custom_resolution[0] = 1;
                if (custom_resolution[0] > 10000) custom_resolution[0] = 10000;
                if (custom_resolution[1] < 1) custom_resolution[1] = 1;
                if (custom_resolution[1] > 10000) custom_resolution[1] = 10000;
                resolution_changed = true;
            }
        }

        if (resolution_mode != 0)
        {
            ImGui::Checkbox("Scale display to window", &scale_display_to_window);
        }

        ImGui::Text("Current render size: %u x %u", current_image_size.x, current_image_size.y);

        if (resolution_changed && !cuda_is_running)
        {
            draw_one_frame();
        }
    }
    ImGui::End();


    return  open;
}



void RaytracerController::draw_one_frame()
{
    wake_cuda=true;
    wake_cuda.notify_one();
}

void RaytracerController::play()
{
    if (cuda_is_running)
        return;
    cuda_is_running=true;
    wake_cuda=true;
    wake_cuda.notify_one();
}

void RaytracerController::pause()
{
    cuda_is_running=false;
}


void RaytracerController::draw_cuda(cudaSurfaceObject_t surface, uint32_t width, uint32_t height)
{
    //lock access to the parameters
    std::lock_guard lock(ui_mutex);
    auto& scene=scenes[active_scene_number].scene;

    CUDA_CHECK(cudaEventRecord(start));
    raytracer->draw(*scene,surface,width, height);
    CUDA_CHECK(cudaEventRecord(finish));
}


void RaytracerController::ui_surface_input()
{
    const float sensitivity=0.01;
    const float speed=0.2;

    ImGuiIO& io = ImGui::GetIO();
    auto& p=raytracer->parameters;
    float dt=io.DeltaTime;


    if (ImGui::IsItemHovered())
    {
        if (ImGui::IsMouseDown(0))
        {
            auto delta= io.MouseDelta;
            p.ax+=-delta.y*sensitivity;
            p.az+=-delta.x*sensitivity;
        }

        if (ImGui::IsMouseDown(2))
        {
            auto pos=ImGui::GetMousePos();
            auto corner=ImGui::GetItemRectMin();
            auto item_size=ImGui::GetItemRectSize();
            float scale_x = (item_size.x > 0.0f) ? (static_cast<float>(current_image_size.x) / item_size.x) : 1.0f;
            float scale_y = (item_size.y > 0.0f) ? (static_cast<float>(current_image_size.y) / item_size.y) : 1.0f;
            int px = static_cast<int>((pos.x - corner.x) * scale_x);
            int py = static_cast<int>((pos.y - corner.y) * scale_y);
            int max_x = static_cast<int>(current_image_size.x);
            int max_y = static_cast<int>(current_image_size.y);
            if (px < 0) px = 0;
            if (px >= max_x) px = max_x > 0 ? max_x - 1 : 0;
            if (py < 0) py = 0;
            if (py >= max_y) py = max_y > 0 ? max_y - 1 : 0;
            debug_pixel = {px, py};
        }

        if (ImGui::IsKeyDown(ImGuiKey_LeftAlt))
        {
            p.near_+=io.MouseWheel*0.01f;
        }
        else
        {
            movement_speed+=movement_speed*io.MouseWheel*0.1;
            if (movement_speed<0.000001)
                movement_speed=0.000001;
        }
    }


    if (ImGui::IsWindowFocused())
    {
        glm::vec3 move{0};
        if (ImGui::IsKeyDown(ImGuiKey_E))
            move+=glm::vec3(0,1,0);
        if (ImGui::IsKeyDown(ImGuiKey_D))
            move+=glm::vec3(0,-1,0);
        if (ImGui::IsKeyDown(ImGuiKey_S))
            move+=glm::vec3(-1,0,0);
        if (ImGui::IsKeyDown(ImGuiKey_F))
            move+=glm::vec3(1,0,0);
        if (ImGui::IsKeyDown(ImGuiKey_R))
            move+=glm::vec3(0,0,1);
        if (ImGui::IsKeyDown(ImGuiKey_W))
            move+=glm::vec3(0,0,-1);

        if (ImGui::IsKeyDown(ImGuiKey_A))
            move*=5;

        if (ImGui::IsKeyDown(ImGuiKey_LeftShift))
            move/=10;

        auto sin_az=sin(p.az);
        auto cos_az=cos(p.az);

        move={ move.x*cos_az-move.y*sin_az, move.x*sin_az+move.y*cos_az, move.z};
        p.view_pos+=move*speed*movement_speed*dt;
    }
}


bool RaytracerController::ui_debug_pixel()
{
    static bool open=true;
    if (ImGui::Begin("Debug pixel",&open))
    {
        ImGui::Text("Pixel to debug: x=%d y=%d",debug_pixel.x,debug_pixel.y);
        ImGui::Text("width=%d height=%d",current_image_size.x,current_image_size.y);

        if (!cuda_is_running&&device_data_mutex.try_lock())
        {
            if (ImGui::Button("Evaluate"))
            {
                auto& scene=scenes[active_scene_number].scene;
                raytracer->update_parameters(*scene,current_image_size.x, current_image_size.y);
                RenderParameters host_params = raytracer->parameters;
                host_params.vertices = scene->as.vertices.view();
                host_params.triangles = scene->as.triangles.view();
                host_params.as = scene->as.as.view();
                host_params.containers = scene->as.box_content.view();
                host_params.primitives = scene->as.primitives.view();
                debug_pixel_result=draw_pixel(debug_pixel.x,debug_pixel.y,host_params);
            }
            device_data_mutex.unlock();
        }
        ImGui::ColorEdit4("result", &debug_pixel_result[0]);
    }
    ImGui::End();

    return open;
}
