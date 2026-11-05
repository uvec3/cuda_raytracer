cmake_minimum_required(VERSION 3.24)

if (POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

project(imgui_container_explorer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>")
endif()

option(CONTAINER_EXPLORER_BUILD_TESTS "Build unit tests for imgui_container_explorer" ${PROJECT_IS_TOP_LEVEL})
option(CONTAINER_EXPLORER_BUILD_EXAMPLES "Build demo application for imgui_container_explorer" ${PROJECT_IS_TOP_LEVEL})

include(FetchContent)

# -----------------------------------------------------------------------------
# Dependencies: Boost.PFR (from git)
# -----------------------------------------------------------------------------
if (NOT TARGET Boost::pfr)
    FetchContent_Declare(
        boost_pfr
        GIT_REPOSITORY https://github.com/boostorg/pfr.git
        GIT_TAG master
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(boost_pfr)
endif()

# -----------------------------------------------------------------------------
# Optional GLM discovery (from Vulkan SDK or system)
# -----------------------------------------------------------------------------
find_package(Vulkan QUIET)

# -----------------------------------------------------------------------------
# Header-only INTERFACE library target
# -----------------------------------------------------------------------------
add_library(imgui_container_explorer INTERFACE)
add_library(imgui_container_explorer::imgui_container_explorer ALIAS imgui_container_explorer)

target_include_directories(imgui_container_explorer
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(imgui_container_explorer INTERFACE cxx_std_20)
target_link_libraries(imgui_container_explorer INTERFACE Boost::pfr)

if (Vulkan_FOUND)
    target_include_directories(imgui_container_explorer INTERFACE ${Vulkan_INCLUDE_DIRS})
endif()

# -----------------------------------------------------------------------------
# Dependencies: ImGui (from git)
# -----------------------------------------------------------------------------
if ((CONTAINER_EXPLORER_BUILD_TESTS OR CONTAINER_EXPLORER_BUILD_EXAMPLES) AND NOT TARGET imgui_core)
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG docking
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(imgui)

    add_library(imgui_core STATIC
        "${imgui_SOURCE_DIR}/imgui.cpp"
        "${imgui_SOURCE_DIR}/imgui_draw.cpp"
        "${imgui_SOURCE_DIR}/imgui_tables.cpp"
        "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
        "${imgui_SOURCE_DIR}/imgui_demo.cpp"
    )
    target_include_directories(imgui_core PUBLIC
        "${imgui_SOURCE_DIR}"
        "${imgui_SOURCE_DIR}/backends"
    )
    target_link_libraries(imgui_container_explorer INTERFACE imgui_core)
endif()

# -----------------------------------------------------------------------------
# Dependencies: soa_vector (from git)
# -----------------------------------------------------------------------------
if (CONTAINER_EXPLORER_BUILD_TESTS OR CONTAINER_EXPLORER_BUILD_EXAMPLES)
    FetchContent_Declare(
        soa_vector
        GIT_REPOSITORY https://github.com/uvec3/soa_vector.git
        GIT_TAG main
        GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(soa_vector)
    if (NOT soa_vector_POPULATED)
        FetchContent_Populate(soa_vector)
    endif()

    if (NOT TARGET soa_vector)
        add_library(soa_vector INTERFACE)
        add_library(soa::soa_vector ALIAS soa_vector)
        target_include_directories(soa_vector INTERFACE "${soa_vector_SOURCE_DIR}/src")
    endif()
endif()

# -----------------------------------------------------------------------------
# Tests & Examples
# -----------------------------------------------------------------------------
if (CONTAINER_EXPLORER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if (CONTAINER_EXPLORER_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
