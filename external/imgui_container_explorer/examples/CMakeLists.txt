if (WIN32 AND TARGET imgui_core)
    add_executable(container_explorer_demo
        demo_main.cpp
        "${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp"
        "${imgui_SOURCE_DIR}/backends/imgui_impl_dx11.cpp"
    )

    target_link_libraries(container_explorer_demo PRIVATE
        imgui_container_explorer::imgui_container_explorer
        soa_vector
        d3d11
        dxgi
    )
endif()
