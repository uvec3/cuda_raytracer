include(FetchContent)
FetchContent_Declare(   googletest   URL https://github.com/google/googletest/archive/refs/heads/main.zip )
FetchContent_MakeAvailable(googletest)
enable_testing()


add_executable(soa_test soa_test.cu
        ../external/OffsetAllocator/offsetAllocator.hpp ../external/OffsetAllocator/offsetAllocator.cpp
        ../src/pool/LogicalAllocatorOffset.hpp ../src/pool/LogicalAllocatorOffset.cpp)

set_target_properties(soa_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
target_link_libraries(soa_test PRIVATE gtest_main Boost::pfr)


add_executable(traversal_test traversal.cu)
set_target_properties(traversal_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
target_link_libraries(traversal_test PRIVATE gtest_main glm::glm)


add_executable(scene_test scene_test.cpp
        ../src/raytracer/Scene.cpp ../src/raytracer/Scene.hpp
        ../src/acceleration_structures/octree/OctreeAS.cu
        ../external/tinygltf.cpp ../external/tiny_gltf_v3.h
        ../external/stb_image.cpp ../external/stb_image.h
        ../external/OffsetAllocator/offsetAllocator.hpp ../external/OffsetAllocator/offsetAllocator.cpp
        ../src/pool/LogicalAllocatorOffset.hpp ../src/pool/LogicalAllocatorOffset.cpp)
set_target_properties(scene_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
target_link_libraries(scene_test PRIVATE vkbase::assets glm::glm gtest_main Boost::pfr vkbase::cuda)


add_executable(allocation allocation.cpp
        ../src/pool/LogicalAllocator.hpp ../src/pool/LogicalAllocator.cpp
        ../external/OffsetAllocator/offsetAllocator.hpp ../external/OffsetAllocator/offsetAllocator.cpp
        ../src/pool/LogicalAllocatorOffset.hpp ../src/pool/LogicalAllocatorOffset.cpp
            )
target_link_libraries(allocation PRIVATE gtest_main)