include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(container_explorer_tests
    container_explorer_tests.cpp
)

target_link_libraries(container_explorer_tests PRIVATE
    imgui_container_explorer::imgui_container_explorer
    soa_vector
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(container_explorer_tests)
