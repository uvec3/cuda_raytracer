#include <container_explorer.hpp>

#include <array>
#include <list>
#include <map>
#include <string>
#include <vector>

#if defined(_MSC_VER) && !defined(__CUDACC__) && !defined(__forceinline__)
#define __forceinline__ __forceinline
#endif

#include "soa_vector.hpp"
#include "aosoa_vector.hpp"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <tchar.h>

#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"

// Forward declare message handler from imgui_impl_win32.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace demo_types
{
    template <typename T>
    class DemoPool
    {
    public:
        DemoPool(std::size_t size = 0) : m_storage(size), m_used_size(0) {}
        std::size_t size() const { return m_storage.size(); }
        std::size_t capacity() const { return m_storage.capacity(); }
        std::size_t used_size() const { return m_used_size; }
        T& operator[](std::size_t i) { return m_storage[i]; }
        const T& operator[](std::size_t i) const { return m_storage[i]; }
        std::size_t alloc_element()
        {
            std::size_t idx = m_storage.size();
            m_storage.emplace_back();
            ++m_used_size;
            return idx;
        }
    private:
        std::vector<T> m_storage;
        std::size_t m_used_size{0};
    };

    enum class MaterialKind : int
    {
        Diffuse = 0,
        Specular = 1,
        Dielectric = 2,
        Emissive = 3
    };

    struct Transform2D
    {
        float x{0.0f};
        float y{0.0f};
        float rotation{0.0f};
        float scale{1.0f};
    };

    struct SceneEntity
    {
        int id{0};
        std::string label{"Entity"};
        MaterialKind material{MaterialKind::Diffuse};
        Transform2D transform{};
#if CONTAINER_EXPLORER_HAS_GLM
        glm::vec3 color{1.0f, 0.5f, 0.2f};
#endif
        bool active{true};
    };

    struct SoaSubDataValue
    {
        float weight;
        double timestamp;
    };

    struct SoaParticleValue
    {
#if CONTAINER_EXPLORER_HAS_GLM
        glm::vec3 position;
        glm::vec3 velocity;
#else
        float px;
        float py;
#endif
        int particle_id;
        SoaSubDataValue meta;
    };

    struct SoaSubData
    {
        using value_type = SoaSubDataValue;
        float& weight;
        double& timestamp;
    };

    struct SoaParticleRef
    {
        using value_type = SoaParticleValue;
#if CONTAINER_EXPLORER_HAS_GLM
        glm::vec3& position;
        glm::vec3& velocity;
#else
        float& px;
        float& py;
#endif
        int& particle_id;
        SoaSubData meta;
    };
} // namespace demo_types

using namespace demo_types;

namespace
{
    ID3D11Device* g_pd3dDevice = nullptr;
    ID3D11DeviceContext* g_pd3dDeviceContext = nullptr;
    IDXGISwapChain* g_pSwapChain = nullptr;
    ID3D11RenderTargetView* g_mainRenderTargetView = nullptr;

    void CreateRenderTarget()
    {
        ID3D11Texture2D* pBackBuffer = nullptr;
        g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
        if (pBackBuffer != nullptr)
        {
            g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_mainRenderTargetView);
            pBackBuffer->Release();
        }
    }

    void CleanupRenderTarget()
    {
        if (g_mainRenderTargetView)
        {
            g_mainRenderTargetView->Release();
            g_mainRenderTargetView = nullptr;
        }
    }

    bool CreateDeviceD3D(HWND hWnd)
    {
        DXGI_SWAP_CHAIN_DESC sd{};
        sd.BufferCount = 2;
        sd.BufferDesc.Width = 0;
        sd.BufferDesc.Height = 0;
        sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        sd.BufferDesc.RefreshRate.Numerator = 60;
        sd.BufferDesc.RefreshRate.Denominator = 1;
        sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.OutputWindow = hWnd;
        sd.SampleDesc.Count = 1;
        sd.SampleDesc.Quality = 0;
        sd.Windowed = TRUE;
        sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

        UINT createDeviceFlags = 0;
        D3D_FEATURE_LEVEL featureLevel;
        const D3D_FEATURE_LEVEL featureLevelArray[2] = {D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0};
        if (D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags,
                                          featureLevelArray, 2, D3D11_SDK_VERSION, &sd, &g_pSwapChain,
                                          &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext) != S_OK)
        {
            return false;
        }

        CreateRenderTarget();
        return true;
    }

    void CleanupDeviceD3D()
    {
        CleanupRenderTarget();
        if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = nullptr; }
        if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
        if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = nullptr; }
    }

    LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
    {
        if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
            return true;

        switch (msg)
        {
        case WM_SIZE:
            if (g_pd3dDevice != nullptr && wParam != SIZE_MINIMIZED)
            {
                CleanupRenderTarget();
                g_pSwapChain->ResizeBuffers(0, static_cast<UINT>(LOWORD(lParam)), static_cast<UINT>(HIWORD(lParam)),
                                            DXGI_FORMAT_UNKNOWN, 0);
                CreateRenderTarget();
            }
            return 0;
        case WM_SYSCOMMAND:
            if ((wParam & 0xfff0) == SC_KEYMENU)
                return 0;
            break;
        case WM_DESTROY:
            ::PostQuitMessage(0);
            return 0;
        }
        return ::DefWindowProcW(hWnd, msg, wParam, lParam);
    }
} // namespace

int main(int, char**)
{
    WNDCLASSEXW wc = {
        sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(nullptr),
        nullptr, nullptr, nullptr, nullptr, L"ContainerExplorerDemo", nullptr
    };
    ::RegisterClassExW(&wc);
    HWND hwnd = ::CreateWindowW(wc.lpszClassName, L"ImGui ContainerExplorer Demo",
                                WS_OVERLAPPEDWINDOW, 100, 100, 1400, 850,
                                nullptr, nullptr, wc.hInstance, nullptr);

    if (!CreateDeviceD3D(hwnd))
    {
        CleanupDeviceD3D();
        ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
        return 1;
    }

    ::ShowWindow(hwnd, SW_SHOWDEFAULT);
    ::UpdateWindow(hwnd);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui::StyleColorsDark();
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    // Populate sample containers
    std::vector<SceneEntity> entities(25);
    for (std::size_t i = 0; i < entities.size(); ++i)
    {
        entities[i].id = static_cast<int>(i);
        entities[i].label = "Object_" + std::to_string(i);
        entities[i].material = static_cast<MaterialKind>(i % 4);
        entities[i].transform = {static_cast<float>(i) * 1.5f, static_cast<float>(i) * -0.5f, 0.0f, 1.0f};
        entities[i].active = (i % 2 == 0);
    }

    soa::soa_vector<SoaParticleRef> soa_particles(50);
    for (std::size_t i = 0; i < soa_particles.size(); ++i)
    {
#if CONTAINER_EXPLORER_HAS_GLM
        soa_particles[i].position = glm::vec3(static_cast<float>(i), static_cast<float>(i * 2), 0.0f);
        soa_particles[i].velocity = glm::vec3(0.1f * static_cast<float>(i), 1.0f, -0.5f);
#else
        soa_particles[i].px = static_cast<float>(i);
        soa_particles[i].py = static_cast<float>(i * 2);
#endif
        soa_particles[i].particle_id = static_cast<int>(1000 + i);
        soa_particles[i].meta.weight = 1.0f + static_cast<float>(i) * 0.25f;
        soa_particles[i].meta.timestamp = static_cast<double>(i) * 0.016;
    }

    DemoPool<SceneEntity> entity_pool(16);
    for (int i = 0; i < 6; ++i)
    {
        auto idx = entity_pool.alloc_element();
        entity_pool[idx].id = 500 + i;
        entity_pool[idx].label = "PoolItem_" + std::to_string(i);
    }

    std::list<Transform2D> waypoints{{0.f, 0.f, 0.f, 1.f}, {10.f, 5.f, 45.f, 1.f}, {25.f, -10.f, 90.f, 2.f}};
    std::map<std::string, double> metrics{{"frame_ms", 16.67}, {"gpu_ms", 4.12}, {"triangles", 125000.0}};

    std::string last_modified_msg = "No element modified yet.";

    ContainerExplorer explorer;
    explorer.register_container("Scene Entities (std::vector)", entities,
        [&](std::size_t id) {
            last_modified_msg = "Scene Entities: element #" + std::to_string(id) + " modified manually";
        });
    explorer.register_container("Particles (soa::soa_vector)", soa_particles,
        [&](std::size_t id) {
            last_modified_msg = "Particles (SOA): element #" + std::to_string(id) + " modified manually";
        });
    explorer.register_container("Entity Pool (DemoPool)", entity_pool,
        [&](std::size_t id) {
            last_modified_msg = "Entity Pool: element #" + std::to_string(id) + " modified manually";
        });
    explorer.register_container("Waypoints (std::list)", waypoints,
        [&](std::size_t id) {
            last_modified_msg = "Waypoints: element #" + std::to_string(id) + " modified manually";
        });
    explorer.register_container("Metrics (std::map)", metrics,
        [&](std::size_t id) {
            last_modified_msg = "Metrics: entry #" + std::to_string(id) + " modified manually";
        });

    bool show_data = true;
    bool running = true;
    while (running)
    {
        MSG msg;
        while (::PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE))
        {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            if (msg.message == WM_QUIT)
                running = false;
        }
        if (!running)
            break;

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        if (ImGui::Begin("Demo Controls"))
        {
            ImGui::Checkbox("show_data (allow table data access)", &show_data);
            ImGui::Separator();
            ImGui::Text("Registered containers: %zu", explorer.container_count());

            if (!explorer.has_container("Waypoints (std::list)"))
            {
                if (ImGui::Button("Re-register Waypoints (std::list)"))
                {
                    explorer.register_container("Waypoints (std::list)", waypoints,
                        [&](std::size_t id) {
                            last_modified_msg = "Waypoints: element #" + std::to_string(id) + " modified manually";
                        });
                }
            }
            else
            {
                if (ImGui::Button("Remove Waypoints (std::list)"))
                    explorer.remove_container("Waypoints (std::list)");
            }

            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Modification Callback Signal:");
            ImGui::TextUnformatted(last_modified_msg.c_str());
        }
        ImGui::End();

        explorer.draw("Container Explorer", show_data);

        ImGui::Render();
        const float clear_color[4] = {0.10f, 0.11f, 0.13f, 1.00f};
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        g_pSwapChain->Present(1, 0);
    }

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    ::DestroyWindow(hwnd);
    ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
    return 0;
}
