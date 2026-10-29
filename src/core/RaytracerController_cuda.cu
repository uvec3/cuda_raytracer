#include "RaytracerController.hpp"
#include "../raytracer/Raytracer.cuh"
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
        ImGui::DragFloat3("Pos",&p.view_pos[0]);
        ImGui::InputFloat("Near",&p.near_);
        ImGui::Checkbox("Show time",&p.measure_time);
        ImGui::Checkbox("Show iterations",&p.show_iterations);
        ImGui::Checkbox("Show triangles",&p.show_triangles);
        ImGui::Checkbox("Show boxes",&p.show_boxes);
        ImGui::InputInt("Time max",&p.time_max);
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
            debug_pixel={pos.x-corner.x,pos.y-corner.y};
        }

        p.near_+=io.MouseWheel*0.01f;
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
        p.view_pos+=move*speed*dt;
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
