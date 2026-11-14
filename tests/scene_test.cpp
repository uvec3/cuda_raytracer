#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include "../src/raytracer/Scene.hpp"
#include "../src/raytracer/Raytracer.cuh"
#include "../src/raytracer/RenderParameters.hpp"
#include "../src/acceleration_structures/octree/OctreeGeometry.cuh"
#include "../src/acceleration_structures/uniform_grid/UniformGridGeometry.cuh"
#include "../src/acceleration_structures/kdtree/KDTreeGeometry.cuh"
#include "../src/acceleration_structures/bvh/BVHGeometry.cuh"
#include "../vkbase/extensions/Assets/assets.h"

TEST(Scene, Load)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);
}

TEST(Scene, NodeHierarchyAndASRef)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    EXPECT_FALSE(scene.root_node.name.empty());
    EXPECT_FALSE(scene.root_node.children.empty());

    Node& subscene = scene.root_node.children[0];
    EXPECT_FALSE(subscene.name.empty());
    EXPECT_GT(subscene.count_subtree_meshes(), 0);
    EXPECT_GT(subscene.count_subtree_triangles(scene.m_meshes), 0);

    EXPECT_FALSE(subscene.as_ref.is_valid());
    EXPECT_EQ(subscene.as_ref.type, ASType::None);

    auto subscene_meshes = scene.get_mesh_transform(&subscene);
    ASSERT_FALSE(subscene_meshes.empty());

    scene.as_octree.config.stretch = false;
    scene.as_octree.config.swapYZ = true;
    scene.as_octree.config.min_depth = 1;
    scene.as_octree.config.max_depth = 10;
    scene.as_octree.config.triangle_to_box_stop_ratio = 1.1f;
    scene.as_octree.config.use_gpu = true;
    scene.as_octree.config.sort_space = true;
    uint32_t space_id = scene.build_as(subscene_meshes, ASType::Octree);
    subscene.as_ref = {space_id, ASType::Octree};

    EXPECT_TRUE(subscene.as_ref.is_valid());
    EXPECT_EQ(subscene.as_ref.type, ASType::Octree);
    EXPECT_EQ(subscene.as_ref.space_id, space_id);

    scene.set_active_space(subscene.as_ref.space_id);
    EXPECT_EQ(scene.get_active_space(), space_id);
    EXPECT_GE(scene.get_root_box(), 0);
}

TEST(OctreeAS, BuildAndTraverseTeapotGPU)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    scene.as_octree.config.stretch = false;
    scene.as_octree.config.swapYZ = true;
    scene.as_octree.config.min_depth = 1;
    scene.as_octree.config.max_depth = 10;
    scene.as_octree.config.triangle_to_box_stop_ratio = 1.1f;
    uint32_t space_id = scene.build_as(meshes, ASType::Octree);

    std::cout << "primitives_size = " << scene.spaces[space_id].primitives_size << std::endl;
    std::cout << "as_size = " << scene.spaces[space_id].as_size << std::endl;
    std::cout << "box_content_size = " << scene.spaces[space_id].box_content_size << std::endl;

    EXPECT_GT(scene.spaces[space_id].as_size, 0);
    EXPECT_GT(scene.spaces[space_id].box_content_size, 0);
    EXPECT_GT(scene.spaces[space_id].primitives_size, 0);

    // Verify full ray traversal and triangle hit
    int root_box = scene.spaces[space_id].root_box;
    EXPECT_GE(root_box, 0);

    soa::soa_vector<PrimitiveChunk16> host_primitives = scene.geometry_context.primitives;
    soa::soa_vector<Vertex> host_vertices = scene.geometry_context.vertices;

    // Cast a bundle of rays towards the teapot center (0.5, 0.5, 0.5)
    int hit_count = 0;
    for (float x = 0.3f; x <= 0.7f; x += 0.05f)
    {
        for (float z = 0.3f; z <= 0.7f; z += 0.05f)
        {
            glm::vec3 origin(x, -0.5f, z);
            glm::vec3 current_source = origin;
            glm::vec3 dir(0.0f, 1.0f, 0.0f);

            CurrentBox current_box{.id = root_box};
            int it = 0;
            int32_t container_id = INT32_MAX;
            bool hit_triangle = false;

            while (container_id != 0 && !hit_triangle && it < 500)
            {
                auto result = traverse_stackless(&scene.as_octree.as[0], current_box, origin, current_source, dir, it);
                container_id = result.content;
                if (container_id > 0)
                {
                    for (; !hit_triangle; ++container_id)
                    {
                        auto bc = scene.as_octree.box_content[container_id];
                        PrimitiveChunk16 chunk0 = host_primitives[bc.id()];
                        if (chunk0.type() == PrimitiveType::Triangle)
                        {
                            const Triangle& tri = reinterpret_cast<const Triangle&>(chunk0);
                            auto v1 = glm::vec3(host_vertices[tri.v0].position);
                            auto v2 = glm::vec3(host_vertices[tri.v1].position);
                            auto v3 = glm::vec3(host_vertices[tri.v2].position);

                            float t, u, v;
                            if (intersectRayTriangle(origin, dir, current_box.t_min, current_box.t_max, v1, v2, v3, t, u, v))
                            {
                                hit_triangle = true;
                                break;
                            }
                        }
                        if (bc.is_last()) break;
                    }
                }
                if (current_box.stack.depth() == 0) break;
            }
            if (hit_triangle) ++hit_count;
        }
    }

    std::cout << "Rays hitting teapot triangles: " << hit_count << " / 81" << std::endl;
    EXPECT_GT(hit_count, 15);
}

TEST(OctreeAS, BuildAndTraverseTeapotFullTriangleGPU)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    scene.as_octree.config.stretch = false;
    scene.as_octree.config.swapYZ = true;
    scene.as_octree.config.min_depth = 1;
    scene.as_octree.config.max_depth = 10;
    scene.as_octree.config.triangle_to_box_stop_ratio = 1.1f;
    scene.as_octree.config.triangle_type = PrimitiveType::FullTriangle;
    uint32_t space_id = scene.build_as(meshes, ASType::Octree);

    EXPECT_GT(scene.spaces[space_id].as_size, 0);
    EXPECT_GT(scene.spaces[space_id].box_content_size, 0);
    EXPECT_GT(scene.spaces[space_id].primitives_size, 0);
    EXPECT_EQ(scene.spaces[space_id].vertices_size, 0u); // vertices freed for FullTriangle

    int root_box = scene.spaces[space_id].root_box;
    EXPECT_GE(root_box, 0);

    soa::soa_vector<PrimitiveChunk16> host_primitives = scene.geometry_context.primitives;

    int hit_count = 0;
    for (float x = 0.3f; x <= 0.7f; x += 0.05f)
    {
        for (float z = 0.3f; z <= 0.7f; z += 0.05f)
        {
            glm::vec3 origin(x, -0.5f, z);
            glm::vec3 current_source = origin;
            glm::vec3 dir(0.0f, 1.0f, 0.0f);

            CurrentBox current_box{.id = root_box};
            int it = 0;
            int32_t container_id = INT32_MAX;
            bool hit_triangle = false;

            while (container_id != 0 && !hit_triangle && it < 500)
            {
                auto result = traverse_stackless(&scene.as_octree.as[0], current_box, origin, current_source, dir, it);
                container_id = result.content;
                if (container_id > 0)
                {
                    for (; !hit_triangle; ++container_id)
                    {
                        auto bc = scene.as_octree.box_content[container_id];
                        PrimitiveChunk16 chunk0 = host_primitives[bc.id()];
                        if (chunk0.type() == PrimitiveType::FullTriangle)
                        {
                            PrimitiveChunk16 chunks[4] = {
                                chunk0,
                                host_primitives[bc.id() + 1],
                                host_primitives[bc.id() + 2],
                                host_primitives[bc.id() + 3]
                            };
                            const InlinedTriangle& ftri = reinterpret_cast<const InlinedTriangle&>(chunks);

                            float t, u, v;
                            if (intersectRayTriangle(origin, dir, current_box.t_min, current_box.t_max, ftri.v0, ftri.v1, ftri.v2, t, u, v))
                            {
                                hit_triangle = true;
                                break;
                            }
                        }
                        if (bc.is_last()) break;
                    }
                }
                if (current_box.stack.depth() == 0) break;
            }
            if (hit_triangle) ++hit_count;
        }
    }

    std::cout << "Rays hitting FullTriangle teapot triangles: " << hit_count << " / 81" << std::endl;
    EXPECT_GT(hit_count, 15);
}

TEST(OctreeAS, BuildAndTraverseTeapotNagataPatchGPU)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    scene.as_octree.config.stretch = false;
    scene.as_octree.config.swapYZ = true;
    scene.as_octree.config.min_depth = 1;
    scene.as_octree.config.max_depth = 10;
    scene.as_octree.config.triangle_to_box_stop_ratio = 1.1f;
    scene.as_octree.config.triangle_type = PrimitiveType::NagataPatch;
    uint32_t space_id = scene.build_as(meshes, ASType::Octree);

    EXPECT_GT(scene.spaces[space_id].as_size, 0);
    EXPECT_GT(scene.spaces[space_id].box_content_size, 0);
    EXPECT_GT(scene.spaces[space_id].primitives_size, 0);
    EXPECT_GT(scene.spaces[space_id].vertices_size, 0u);

    int root_box = scene.spaces[space_id].root_box;
    EXPECT_GE(root_box, 0);

    soa::soa_vector<PrimitiveChunk16> host_primitives = scene.geometry_context.primitives;
    soa::soa_vector<Vertex> host_vertices = scene.geometry_context.vertices;

    int hit_count = 0;
    for (float x = 0.3f; x <= 0.7f; x += 0.05f)
    {
        for (float z = 0.3f; z <= 0.7f; z += 0.05f)
        {
            glm::vec3 origin(x, -0.5f, z);
            glm::vec3 current_source = origin;
            glm::vec3 dir(0.0f, 1.0f, 0.0f);

            CurrentBox current_box{.id = root_box};
            int it = 0;
            int32_t container_id = INT32_MAX;
            bool hit_triangle = false;

            while (container_id != 0 && !hit_triangle && it < 500)
            {
                auto result = traverse_stackless(&scene.as_octree.as[0], current_box, origin, current_source, dir, it);
                container_id = result.content;
                if (container_id > 0)
                {
                    for (; !hit_triangle; ++container_id)
                    {
                        auto bc = scene.as_octree.box_content[container_id];
                        PrimitiveChunk16 chunk0 = host_primitives[bc.id()];
                        if (chunk0.type() == PrimitiveType::NagataPatch)
                        {
                            NagataPatchPrimitive patch = load_primitive<NagataPatchPrimitive>(host_primitives, bc.id(), chunk0);

                            float t, u, v;
                            if (intersectNagataPrecomputed(origin, dir, current_box.t_min, current_box.t_max,
                                                           glm::vec3(host_vertices[patch.v0].position),
                                                           glm::vec3(host_vertices[patch.v1].position),
                                                           glm::vec3(host_vertices[patch.v2].position),
                                                           patch.C12, patch.C23, patch.C31,
                                                           t, u, v))
                            {
                                hit_triangle = true;
                                break;
                            }
                        }
                        if (bc.is_last()) break;
                    }
                }
                if (current_box.stack.depth() == 0) break;
            }
            if (hit_triangle) ++hit_count;
        }
    }

    std::cout << "Rays hitting NagataPatch teapot patches: " << hit_count << " / 81" << std::endl;
    EXPECT_GT(hit_count, 15);
}

TEST(OctreeAS, BuildAndTraverseTeapotCPU)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    scene.as_octree.config.stretch = false;
    scene.as_octree.config.swapYZ = true;
    scene.as_octree.config.min_depth = 1;
    scene.as_octree.config.max_depth = 10;
    scene.as_octree.config.triangle_to_box_stop_ratio = 1.1f;
    scene.as_octree.config.use_gpu = false;
    uint32_t space_id = scene.build_as(meshes, ASType::Octree);

    EXPECT_GT(scene.spaces[space_id].as_size, 0);
    EXPECT_GT(scene.spaces[space_id].box_content_size, 0);
    EXPECT_GT(scene.spaces[space_id].primitives_size, 0);

    int root_box = scene.spaces[space_id].root_box;
    EXPECT_GE(root_box, 0);

    soa::soa_vector<PrimitiveChunk16> host_primitives = scene.geometry_context.primitives;
    soa::soa_vector<Vertex> host_vertices = scene.geometry_context.vertices;

    int hit_count = 0;
    for (float x = 0.3f; x <= 0.7f; x += 0.05f)
    {
        for (float z = 0.3f; z <= 0.7f; z += 0.05f)
        {
            glm::vec3 origin(x, -0.5f, z);
            glm::vec3 current_source = origin;
            glm::vec3 dir(0.0f, 1.0f, 0.0f);

            CurrentBox current_box{.id = root_box};
            int it = 0;
            int32_t container_id = INT32_MAX;
            bool hit_triangle = false;

            while (container_id != 0 && !hit_triangle && it < 500)
            {
                auto result = traverse_stackless(&scene.as_octree.as[0], current_box, origin, current_source, dir, it);
                container_id = result.content;
                if (container_id > 0)
                {
                    for (; !hit_triangle; ++container_id)
                    {
                        auto bc = scene.as_octree.box_content[container_id];
                        PrimitiveChunk16 chunk0 = host_primitives[bc.id()];
                        if (chunk0.type() == PrimitiveType::Triangle)
                        {
                            const Triangle& tri = reinterpret_cast<const Triangle&>(chunk0);
                            auto v1 = glm::vec3(host_vertices[tri.v0].position);
                            auto v2 = glm::vec3(host_vertices[tri.v1].position);
                            auto v3 = glm::vec3(host_vertices[tri.v2].position);

                            float t, u, v;
                            if (intersectRayTriangle(origin, dir, current_box.t_min, current_box.t_max, v1, v2, v3, t, u, v))
                            {
                                hit_triangle = true;
                                break;
                            }
                        }
                        if (bc.is_last()) break;
                    }
                }
                if (current_box.stack.depth() == 0) break;
            }
            if (hit_triangle) ++hit_count;
        }
    }

    std::cout << "Rays hitting CPU-built teapot triangles: " << hit_count << " / 81" << std::endl;
    EXPECT_GT(hit_count, 15);
}

TEST(OctreeAS, BuildAndTraverseTeapotWithoutSort)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    // Build AS without sort_space (sort = false)
    scene.as_octree.config.stretch = false;
    scene.as_octree.config.swapYZ = true;
    scene.as_octree.config.min_depth = 1;
    scene.as_octree.config.max_depth = 10;
    scene.as_octree.config.triangle_to_box_stop_ratio = 1.1f;
    scene.as_octree.config.use_gpu = true;
    scene.as_octree.config.sort_space = false;
    uint32_t space_id = scene.build_as(meshes, ASType::Octree);

    EXPECT_GT(scene.spaces[space_id].as_size, 0);
    EXPECT_GT(scene.spaces[space_id].box_content_size, 0);
    EXPECT_GT(scene.spaces[space_id].primitives_size, 0);

    int root_box = scene.spaces[space_id].root_box;
    EXPECT_GE(root_box, 0);

    soa::soa_vector<PrimitiveChunk16> host_primitives = scene.geometry_context.primitives;
    soa::soa_vector<Vertex> host_vertices = scene.geometry_context.vertices;

    int hit_count = 0;
    for (float x = 0.3f; x <= 0.7f; x += 0.05f)
    {
        for (float z = 0.3f; z <= 0.7f; z += 0.05f)
        {
            glm::vec3 origin(x, -0.5f, z);
            glm::vec3 current_source = origin;
            glm::vec3 dir(0.0f, 1.0f, 0.0f);

            CurrentBox current_box{.id = root_box};
            int it = 0;
            int32_t container_id = INT32_MAX;
            bool hit_triangle = false;

            while (container_id != 0 && !hit_triangle && it < 500)
            {
                auto result = traverse_stackless(&scene.as_octree.as[0], current_box, origin, current_source, dir, it);
                container_id = result.content;
                if (container_id > 0)
                {
                    for (; !hit_triangle; ++container_id)
                    {
                        auto bc = scene.as_octree.box_content[container_id];
                        PrimitiveChunk16 chunk0 = host_primitives[bc.id()];
                        if (chunk0.type() == PrimitiveType::Triangle)
                        {
                            const Triangle& tri = reinterpret_cast<const Triangle&>(chunk0);
                            auto v1 = glm::vec3(host_vertices[tri.v0].position);
                            auto v2 = glm::vec3(host_vertices[tri.v1].position);
                            auto v3 = glm::vec3(host_vertices[tri.v2].position);

                            float t, u, v;
                            if (intersectRayTriangle(origin, dir, current_box.t_min, current_box.t_max, v1, v2, v3, t, u, v))
                            {
                                hit_triangle = true;
                                break;
                            }
                        }
                        if (bc.is_last()) break;
                    }
                }
                if (current_box.stack.depth() == 0) break;
            }
            if (hit_triangle) ++hit_count;
        }
    }

    std::cout << "Rays hitting unsorted-space teapot triangles: " << hit_count << " / 81" << std::endl;
    EXPECT_GT(hit_count, 15);
}

TEST(OctreeAS, BuildAndTraverseHairballGPU)
{
    Scene scene("Hairball");
    auto hairball_view = vkbase::assets::get("Scenes/hairball2.glb");
    if (hairball_view.empty())
    {
        GTEST_SKIP() << "hairball2.glb not found";
    }
    scene.load(hairball_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    scene.as_octree.config.stretch = false;
    scene.as_octree.config.swapYZ = true;
    scene.as_octree.config.min_depth = 1;
    scene.as_octree.config.max_depth = 8;
    scene.as_octree.config.triangle_to_box_stop_ratio = 1.1f;
    uint32_t space_id = scene.build_as(meshes, ASType::Octree);
    EXPECT_GT(scene.spaces[space_id].as_size, 0);
    EXPECT_GT(scene.spaces[space_id].box_content_size, 0);
    EXPECT_GT(scene.spaces[space_id].primitives_size, 0);

    std::cout << "Hairball AS built on GPU: boxes=" << scene.spaces[space_id].as_size
              << ", contents=" << scene.spaces[space_id].box_content_size
              << ", primitives=" << scene.spaces[space_id].primitives_size << std::endl;
}

TEST(Assets, AssetsFrom)
{
    auto scenes = vkbase::assets::assetsFrom("Scenes");
    EXPECT_GE(scenes.size(), 2);
    bool found_teapot = false;
    for (const auto& [name, data] : scenes)
    {
        if (name.find("utah_teapot.glb") != std::string::npos)
        {
            found_teapot = true;
            EXPECT_FALSE(data.empty());
        }
    }
    EXPECT_TRUE(found_teapot);
}

TEST(Assets, DirectoryIterator)
{
    int count = 0;
    bool found_teapot = false;
    for (const auto& entry : vkbase::assets::directory("Scenes"))
    {
        count++;
        EXPECT_TRUE(entry.is_regular_file());
        EXPECT_FALSE(entry.is_directory());
        if (entry.path().filename().string().find("utah_teapot.glb") != std::string::npos)
        {
            found_teapot = true;
        }
    }
    EXPECT_GE(count, 2);
    EXPECT_TRUE(found_teapot);
}

TEST(Scene, LoadTexturesAndMaterials)
{
    Scene scene("ABeautifulGame");
    auto game_view = vkbase::assets::get("Scenes/ABeautifulGame.glb");
    if (game_view.empty())
    {
        GTEST_SKIP() << "ABeautifulGame.glb not found";
    }
    scene.load(game_view);

    EXPECT_TRUE(scene.is_loaded());
    EXPECT_GT(scene.m_textures.size(), 0u);
    EXPECT_GT(scene.m_materials.size(), 1u); // slot 0 default + loaded materials
    EXPECT_EQ(scene.d_materials.size(), scene.m_materials.size());

    // Verify at least one texture was decoded and uploaded to CUDA
    bool has_valid_cuda_tex = false;
    for (const auto& tex : scene.m_textures)
    {
        if (tex.texObj != 0 && tex.cudaArray != nullptr && tex.width > 0 && tex.height > 0)
        {
            has_valid_cuda_tex = true;
            break;
        }
    }
    EXPECT_TRUE(has_valid_cuda_tex);

    // Verify at least one material references a valid baseColorTexture or normalTexture
    bool has_textured_mat = false;
    for (size_t i = 1; i < scene.m_materials.size(); ++i)
    {
        auto mat = scene.m_materials[i];
        if (mat.baseColorTextureIndex > 0 && mat.baseColorTexture != 0)
        {
            has_textured_mat = true;
            break;
        }
    }
    EXPECT_TRUE(has_textured_mat);
}

TEST(OptixAS, BuildTeapot)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    uint32_t space_id = scene.build_as(meshes, ASType::Optix);

    EXPECT_GT(scene.spaces[space_id].primitives_size, 0);
    EXPECT_TRUE(scene.as_optix.is_valid());
    EXPECT_NE(scene.as_optix.get_handle(), 0);
    EXPECT_EQ(scene.spaces[space_id].as_type, ASType::Optix);
}

TEST(OptixAS, RenderTeapot)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    uint32_t space_id = scene.build_as(meshes, ASType::Optix);
    ASSERT_TRUE(scene.as_optix.is_valid());

    const uint32_t width = 64;
    const uint32_t height = 64;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t cuArray = nullptr;
    ASSERT_EQ(cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArraySurfaceLoadStore), cudaSuccess);

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    cudaSurfaceObject_t surface = 0;
    ASSERT_EQ(cudaCreateSurfaceObject(&surface, &resDesc), cudaSuccess);

    RenderParameters params{};
    params.width = width;
    params.height = height;
    params.view_pos = glm::vec3(0.5f, 0.0f, 0.3f);
    params.ax = 0.0f;
    params.az = 0.0f;
    params.ax_c = glm::vec2(1.0f, 0.0f);
    params.az_c = glm::vec2(1.0f, 0.0f);
    params.near_ = 1.0f;
    params.ambient_light = 0.2f;
    params.diffuse_light = 0.8f;
    params.fixed_light_source = false;
    params.light_source = glm::vec3(0.5f, 0.5f, 1.5f);
    params.use_fake_normals = true;
    params.background_color = glm::vec4(0.1f, 0.2f, 0.3f, 1.0f);
    params.primitives = scene.geometry_context.primitives.view();
    params.vertices = scene.geometry_context.vertices.view();
    params.materials = scene.geometry_context.materials.view();

    EXPECT_NO_THROW(scene.as_optix.render(surface, params, 0));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uchar4> host_pixels(width * height);
    EXPECT_EQ(cudaMemcpy2DFromArray(host_pixels.data(), width * sizeof(uchar4), cuArray, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToHost), cudaSuccess);

    // Verify pixels were written (alpha should be 255 and colors non-zero)
    bool has_non_zero = false;
    for (const auto& px : host_pixels)
    {
        if (px.w > 0 && (px.x > 0 || px.y > 0 || px.z > 0))
        {
            has_non_zero = true;
            break;
        }
    }
    EXPECT_TRUE(has_non_zero);

    cudaDestroySurfaceObject(surface);
    cudaFreeArray(cuArray);
}

TEST(OptixAS, RenderWithTimeMeasurement)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    uint32_t space_id = scene.build_as(meshes, ASType::Optix);
    ASSERT_TRUE(scene.as_optix.is_valid());

    const uint32_t width = 64;
    const uint32_t height = 64;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t cuArray = nullptr;
    ASSERT_EQ(cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArraySurfaceLoadStore), cudaSuccess);

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    cudaSurfaceObject_t surface = 0;
    ASSERT_EQ(cudaCreateSurfaceObject(&surface, &resDesc), cudaSuccess);

    RenderParameters params{};
    params.width = width;
    params.height = height;
    params.view_pos = glm::vec3(0.5f, 0.0f, 0.3f);
    params.ax = 0.0f;
    params.az = 0.0f;
    params.ax_c = glm::vec2(1.0f, 0.0f);
    params.az_c = glm::vec2(1.0f, 0.0f);
    params.near_ = 1.0f;
    params.ambient_light = 0.2f;
    params.diffuse_light = 0.8f;
    params.fixed_light_source = false;
    params.light_source = glm::vec3(0.5f, 0.5f, 1.5f);
    params.use_fake_normals = true;
    params.background_color = glm::vec4(0.1f, 0.2f, 0.3f, 1.0f);
    params.primitives = scene.geometry_context.primitives.view();
    params.vertices = scene.geometry_context.vertices.view();
    params.materials = scene.geometry_context.materials.view();
    params.measure_time = true;
    params.time_max = 5000;

    EXPECT_NO_THROW(scene.as_optix.render(surface, params, 0));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uchar4> host_pixels(width * height);
    EXPECT_EQ(cudaMemcpy2DFromArray(host_pixels.data(), width * sizeof(uchar4), cuArray, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToHost), cudaSuccess);

    bool has_timing_values = false;
    for (const auto& px : host_pixels)
    {
        if (px.w == 255 && px.x > 0)
        {
            has_timing_values = true;
            break;
        }
    }
    EXPECT_TRUE(has_timing_values);

    cudaDestroySurfaceObject(surface);
    cudaFreeArray(cuArray);
}

TEST(OptixAS, RenderWithRefraction)
{
    Scene scene("Refraction");
    auto refraction_view = vkbase::assets::get("Scenes/Refraction.glb");
    if (refraction_view.empty())
    {
        refraction_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    }
    ASSERT_FALSE(refraction_view.empty());
    scene.load(refraction_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    uint32_t space_id = scene.build_as(meshes, ASType::Optix);
    ASSERT_TRUE(scene.as_optix.is_valid());

    const uint32_t width = 64;
    const uint32_t height = 64;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t cuArray = nullptr;
    ASSERT_EQ(cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArraySurfaceLoadStore), cudaSuccess);

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    cudaSurfaceObject_t surface = 0;
    ASSERT_EQ(cudaCreateSurfaceObject(&surface, &resDesc), cudaSuccess);

    RenderParameters params{};
    params.width = width;
    params.height = height;
    params.view_pos = glm::vec3(0.0f, 0.0f, 2.0f);
    params.ax = 0.0f;
    params.az = 0.0f;
    params.ax_c = glm::vec2(1.0f, 0.0f);
    params.az_c = glm::vec2(1.0f, 0.0f);
    params.near_ = 1.0f;
    params.ambient_light = 0.2f;
    params.diffuse_light = 0.8f;
    params.fixed_light_source = false;
    params.light_source = glm::vec3(0.5f, 0.5f, 1.5f);
    params.use_fake_normals = true;
    params.background_color = glm::vec4(0.1f, 0.2f, 0.3f, 1.0f);
    params.primitives = scene.geometry_context.primitives.view();
    params.vertices = scene.geometry_context.vertices.view();
    params.materials = scene.geometry_context.materials.view();
    params.enableRefraction = true;

    EXPECT_NO_THROW(scene.as_optix.render(surface, params, 0));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uchar4> host_pixels(width * height);
    EXPECT_EQ(cudaMemcpy2DFromArray(host_pixels.data(), width * sizeof(uchar4), cuArray, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToHost), cudaSuccess);

    bool has_rendered_pixels = false;
    for (const auto& px : host_pixels)
    {
        if (px.w == 255 && (px.x > 0 || px.y > 0 || px.z > 0))
        {
            has_rendered_pixels = true;
            break;
        }
    }
    EXPECT_TRUE(has_rendered_pixels);

    cudaDestroySurfaceObject(surface);
    cudaFreeArray(cuArray);
}

TEST(OptixAS, BuildMultipleSpacesAndFree)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes1 = scene.get_mesh_transform();
    ASSERT_FALSE(meshes1.empty());

    uint32_t space1 = scene.build_as(meshes1, ASType::Optix);
    EXPECT_TRUE(scene.as_optix.is_valid(space1));
    OptixTraversableHandle handle1 = scene.as_optix.get_handle(space1);
    EXPECT_NE(handle1, 0);

    auto meshes2 = scene.get_mesh_transform();
    ASSERT_FALSE(meshes2.empty());

    uint32_t space2 = scene.build_as(meshes2, ASType::Optix);
    EXPECT_NE(space1, space2);
    EXPECT_TRUE(scene.as_optix.is_valid(space2));
    OptixTraversableHandle handle2 = scene.as_optix.get_handle(space2);
    EXPECT_NE(handle2, 0);

    // Both spaces must be valid simultaneously
    EXPECT_TRUE(scene.as_optix.is_valid(space1));
    EXPECT_TRUE(scene.as_optix.is_valid(space2));

    const uint32_t width = 32;
    const uint32_t height = 32;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t cuArray = nullptr;
    ASSERT_EQ(cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArraySurfaceLoadStore), cudaSuccess);

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    cudaSurfaceObject_t surface = 0;
    ASSERT_EQ(cudaCreateSurfaceObject(&surface, &resDesc), cudaSuccess);

    RenderParameters params{};
    params.width = width;
    params.height = height;
    params.view_pos = glm::vec3(0.5f, 0.0f, 0.3f);
    params.ax = 0.0f;
    params.az = 0.0f;
    params.ax_c = glm::vec2(1.0f, 0.0f);
    params.az_c = glm::vec2(1.0f, 0.0f);
    params.near_ = 1.0f;
    params.ambient_light = 0.2f;
    params.diffuse_light = 0.8f;
    params.background_color = glm::vec4(0.1f, 0.2f, 0.3f, 1.0f);
    params.primitives = scene.geometry_context.primitives.view();
    params.vertices = scene.geometry_context.vertices.view();
    params.materials = scene.geometry_context.materials.view();

    // Render space 1 and space 2 independently
    EXPECT_NO_THROW(scene.as_optix.render(space1, surface, params, 0));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_NO_THROW(scene.as_optix.render(space2, surface, params, 0));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Free space 1 only
    scene.free_space(space1);
    EXPECT_FALSE(scene.as_optix.is_valid(space1));
    EXPECT_TRUE(scene.as_optix.is_valid(space2));

    // Space 2 must still render cleanly
    EXPECT_NO_THROW(scene.as_optix.render(space2, surface, params, 0));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Free space 2
    scene.free_space(space2);
    EXPECT_FALSE(scene.as_optix.is_valid(space2));
    EXPECT_FALSE(scene.as_optix.is_valid());

    cudaDestroySurfaceObject(surface);
    cudaFreeArray(cuArray);
}

namespace
{
    int count_uniform_grid_teapot_hits(Scene& scene, uint32_t space_id, PrimitiveType expected_type)
    {
        const int grid_id = scene.spaces[space_id].root_box;
        EXPECT_GE(grid_id, 0);
        const auto grid = scene.as_uniform_grid.grids[static_cast<size_t>(grid_id)];
        const glm::vec3 grid_max = glm::vec3(grid.dims) * grid.cell_size;

        soa::soa_vector<PrimitiveChunk16> host_primitives = scene.geometry_context.primitives;
        soa::soa_vector<Vertex> host_vertices = scene.geometry_context.vertices;

        int hit_count = 0;
        for (int ix = 0; ix < 9; ++ix)
        {
            for (int iz = 0; iz < 9; ++iz)
            {
                const float x = grid_max.x * (0.1f + 0.8f * (static_cast<float>(ix) / 8.0f));
                const float z = grid_max.z * (0.1f + 0.8f * (static_cast<float>(iz) / 8.0f));
                const glm::vec3 origin(x, -0.5f, z);
                const glm::vec3 dir(0.0f, 1.0f, 0.0f);

                CurrentGridCell state{};
                int it = 0;
                bool hit_triangle = false;

                while (!hit_triangle && it < 500)
                {
                    uint32_t container_id = traverse_uniform_grid(
                        &scene.as_uniform_grid.as[0],
                        grid.dims,
                        grid.cell_size,
                        grid.as_start,
                        state,
                        origin,
                        dir,
                        it);
                    if (container_id == 0)
                        break;

                    for (; !hit_triangle; ++container_id)
                    {
                        auto bc = scene.as_uniform_grid.box_content[container_id];
                        PrimitiveChunk16 chunk0 = host_primitives[bc.id()];
                        EXPECT_EQ(chunk0.type(), expected_type);

                        float t = 0.0f, u = 0.0f, v = 0.0f;
                        bool hit = false;
                        if (chunk0.type() == PrimitiveType::Triangle)
                        {
                            const Triangle tri = chunk0.as<Triangle>();
                            hit = intersectRayTriangle(
                                origin, dir, state.t_min, state.t_max,
                                glm::vec3(host_vertices[tri.v0].position),
                                glm::vec3(host_vertices[tri.v1].position),
                                glm::vec3(host_vertices[tri.v2].position),
                                t, u, v);
                        }
                        else if (chunk0.type() == PrimitiveType::FullTriangle)
                        {
                            const InlinedTriangle ftri = load_primitive<InlinedTriangle>(host_primitives, bc.id(), chunk0);
                            hit = intersectRayTriangle(
                                origin, dir, state.t_min, state.t_max,
                                ftri.v0, ftri.v1, ftri.v2,
                                t, u, v);
                        }
                        else if (chunk0.type() == PrimitiveType::NagataPatch)
                        {
                            const NagataPatchPrimitive patch = load_primitive<NagataPatchPrimitive>(host_primitives, bc.id(), chunk0);
                            hit = intersectNagataPrecomputed(
                                origin, dir, state.t_min, state.t_max,
                                glm::vec3(host_vertices[patch.v0].position),
                                glm::vec3(host_vertices[patch.v1].position),
                                glm::vec3(host_vertices[patch.v2].position),
                                patch.C12, patch.C23, patch.C31,
                                t, u, v);
                        }

                        if (hit)
                        {
                            hit_triangle = true;
                            break;
                        }
                        if (bc.is_last())
                            break;
                    }
                    if (state.exited)
                        break;
                }

                if (hit_triangle)
                    ++hit_count;
            }
        }
        return hit_count;
    }
} // namespace

TEST(UniformGridAS, BuildAndTraverseTeapotGPU)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    scene.as_uniform_grid.config.resolution = 20;
    scene.as_uniform_grid.config.swapYZ = true;
    scene.as_uniform_grid.config.sort_space = true;
    scene.as_uniform_grid.config.triangle_type = PrimitiveType::Triangle;
    uint32_t space_id = scene.build_as(meshes, ASType::UniformGrid);

    EXPECT_EQ(scene.spaces[space_id].as_type, ASType::UniformGrid);
    EXPECT_GT(scene.spaces[space_id].as_size, 0u);
    EXPECT_GT(scene.spaces[space_id].box_content_size, 0u);
    EXPECT_GT(scene.spaces[space_id].primitives_size, 0u);

    const int grid_id = scene.spaces[space_id].root_box;
    ASSERT_GE(grid_id, 0);
    const auto grid = scene.as_uniform_grid.grids[static_cast<size_t>(grid_id)];
    EXPECT_EQ(std::max({grid.dims.x, grid.dims.y, grid.dims.z}), 20u);
    EXPECT_EQ(grid.dims.x * grid.dims.y * grid.dims.z, scene.spaces[space_id].as_size);

    // Verify tight normalization: vertices start at (0,0,0), max extent is 1.0, and dims match resolution along longest axis
    soa::soa_vector<Vertex> host_vertices = scene.geometry_context.vertices;
    const uint32_t v_start = scene.spaces[space_id].vertices_start;
    const uint32_t v_size = scene.spaces[space_id].vertices_size;
    glm::vec3 v_min(std::numeric_limits<float>::infinity());
    glm::vec3 v_max(-std::numeric_limits<float>::infinity());
    for (uint32_t i = 0; i < v_size; ++i)
    {
        const glm::vec3 p = host_vertices[v_start + i].position;
        v_min = glm::min(v_min, p);
        v_max = glm::max(v_max, p);
    }
    EXPECT_NEAR(v_min.x, 0.0f, 1e-5f);
    EXPECT_NEAR(v_min.y, 0.0f, 1e-5f);
    EXPECT_NEAR(v_min.z, 0.0f, 1e-5f);
    const float max_extent = glm::max(glm::max(v_max.x, v_max.y), v_max.z);
    EXPECT_NEAR(max_extent, 1.0f, 1e-5f);
    EXPECT_NEAR(grid.cell_size, max_extent / 20.0f, 1e-5f);
    EXPECT_EQ(grid.dims.x, std::clamp(static_cast<uint32_t>(std::ceil((v_max.x / grid.cell_size) - 1e-5f)), 1u, 20u));
    EXPECT_EQ(grid.dims.y, std::clamp(static_cast<uint32_t>(std::ceil((v_max.y / grid.cell_size) - 1e-5f)), 1u, 20u));
    EXPECT_EQ(grid.dims.z, std::clamp(static_cast<uint32_t>(std::ceil((v_max.z / grid.cell_size) - 1e-5f)), 1u, 20u));

    const int hit_count = count_uniform_grid_teapot_hits(scene, space_id, PrimitiveType::Triangle);
    std::cout << "UniformGrid rays hitting teapot triangles: " << hit_count << " / 81" << std::endl;
    EXPECT_GT(hit_count, 15);
}

TEST(UniformGridAS, BuildAndTraverseTeapotFullTriangleGPU)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    scene.as_uniform_grid.config.resolution = 20;
    scene.as_uniform_grid.config.swapYZ = true;
    scene.as_uniform_grid.config.sort_space = true;
    scene.as_uniform_grid.config.triangle_type = PrimitiveType::FullTriangle;
    uint32_t space_id = scene.build_as(meshes, ASType::UniformGrid);

    EXPECT_GT(scene.spaces[space_id].as_size, 0u);
    EXPECT_GT(scene.spaces[space_id].box_content_size, 0u);
    EXPECT_GT(scene.spaces[space_id].primitives_size, 0u);
    EXPECT_EQ(scene.spaces[space_id].vertices_size, 0u);

    const int hit_count = count_uniform_grid_teapot_hits(scene, space_id, PrimitiveType::FullTriangle);
    std::cout << "UniformGrid rays hitting FullTriangle teapot: " << hit_count << " / 81" << std::endl;
    EXPECT_GT(hit_count, 15);
}

TEST(UniformGridAS, BuildAndTraverseTeapotNagataPatchGPU)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    scene.as_uniform_grid.config.resolution = 20;
    scene.as_uniform_grid.config.swapYZ = true;
    scene.as_uniform_grid.config.sort_space = true;
    scene.as_uniform_grid.config.triangle_type = PrimitiveType::NagataPatch;
    uint32_t space_id = scene.build_as(meshes, ASType::UniformGrid);

    EXPECT_GT(scene.spaces[space_id].as_size, 0u);
    EXPECT_GT(scene.spaces[space_id].box_content_size, 0u);
    EXPECT_GT(scene.spaces[space_id].primitives_size, 0u);
    EXPECT_GT(scene.spaces[space_id].vertices_size, 0u);

    const int hit_count = count_uniform_grid_teapot_hits(scene, space_id, PrimitiveType::NagataPatch);
    std::cout << "UniformGrid rays hitting NagataPatch teapot: " << hit_count << " / 81" << std::endl;
    EXPECT_GT(hit_count, 15);
}

TEST(UniformGridAS, BuildAndTraverseWithoutSortAndFree)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    scene.as_uniform_grid.config.resolution = 20;
    scene.as_uniform_grid.config.swapYZ = true;
    scene.as_uniform_grid.config.sort_space = false;
    scene.as_uniform_grid.config.triangle_type = PrimitiveType::Triangle;
    uint32_t space_id = scene.build_as(meshes, ASType::UniformGrid);

    EXPECT_GT(scene.spaces[space_id].as_size, 0u);
    EXPECT_GT(scene.spaces[space_id].box_content_size, 0u);
    EXPECT_GT(scene.spaces[space_id].primitives_size, 0u);

    const int hit_count = count_uniform_grid_teapot_hits(scene, space_id, PrimitiveType::Triangle);
    std::cout << "UniformGrid rays hitting unsorted teapot: " << hit_count << " / 81" << std::endl;
    EXPECT_GT(hit_count, 15);

    const uint32_t grid_id = static_cast<uint32_t>(scene.spaces[space_id].root_box);
    const uint32_t as_start = scene.spaces[space_id].as_start;
    const uint32_t bc_start = scene.spaces[space_id].box_content_start;

    scene.free_space(space_id);
    EXPECT_FALSE(scene.spaces.is_allocated(space_id));
    EXPECT_FALSE(scene.as_uniform_grid.grids.is_allocated(grid_id));
    EXPECT_FALSE(scene.as_uniform_grid.as.is_allocated(as_start));
    EXPECT_FALSE(scene.as_uniform_grid.box_content.is_allocated(bc_start));
}

TEST(UniformGridAS, RenderTeapotGPU)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    scene.as_uniform_grid.config.resolution = 20;
    scene.as_uniform_grid.config.swapYZ = true;
    scene.as_uniform_grid.config.sort_space = true;
    scene.as_uniform_grid.config.triangle_type = PrimitiveType::Triangle;
    uint32_t space_id = scene.build_as(meshes, ASType::UniformGrid);
    scene.set_active_space(space_id);

    const uint32_t width = 64;
    const uint32_t height = 64;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t cuArray = nullptr;
    ASSERT_EQ(cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArraySurfaceLoadStore), cudaSuccess);

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    cudaSurfaceObject_t surface = 0;
    ASSERT_EQ(cudaCreateSurfaceObject(&surface, &resDesc), cudaSuccess);

    Raytracer raytracer;
    raytracer.parameters.view_pos = glm::vec3(0.5f, -1.2f, 0.3f);
    raytracer.parameters.ax = 0.0f;
    raytracer.parameters.az = 0.0f;
    raytracer.parameters.background_color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    raytracer.update_parameters(scene, width, height);

    EXPECT_EQ(raytracer.parameters.active_as_type, ASType::UniformGrid);
    EXPECT_NO_THROW(raytracer.draw_first_hit(scene, surface, width, height, 0));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uchar4> host_pixels(width * height);
    EXPECT_EQ(cudaMemcpy2DFromArray(host_pixels.data(), width * sizeof(uchar4), cuArray, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToHost), cudaSuccess);

    uint32_t non_black_pixels = 0;
    for (const auto& px : host_pixels)
    {
        EXPECT_EQ(px.w, 255);
        if (px.x > 0 || px.y > 0 || px.z > 0)
        {
            ++non_black_pixels;
        }
    }
    std::cout << "UniformGrid rendered non-black pixels: " << non_black_pixels << " / " << (width * height) << std::endl;
    EXPECT_GT(non_black_pixels, 0u);
    EXPECT_LT(non_black_pixels, width * height);

    cudaDestroySurfaceObject(surface);
    cudaFreeArray(cuArray);
}

namespace
{
    int count_kdtree_teapot_hits(Scene& scene, uint32_t space_id, PrimitiveType expected_type)
    {
        const int tree_id = scene.spaces[space_id].root_box;
        EXPECT_GE(tree_id, 0);
        const auto tree = scene.as_kdtree.trees[static_cast<size_t>(tree_id)];
        const glm::vec3 bounds_min = tree.bounds_min;
        const glm::vec3 bounds_max = tree.bounds_max;

        soa::soa_vector<PrimitiveChunk16> host_primitives = scene.geometry_context.primitives;
        soa::soa_vector<Vertex> host_vertices = scene.geometry_context.vertices;

        int hit_count = 0;
        for (int ix = 0; ix < 9; ++ix)
        {
            for (int iz = 0; iz < 9; ++iz)
            {
                const float x = bounds_min.x + (bounds_max.x - bounds_min.x) * (0.1f + 0.8f * (static_cast<float>(ix) / 8.0f));
                const float z = bounds_min.z + (bounds_max.z - bounds_min.z) * (0.1f + 0.8f * (static_cast<float>(iz) / 8.0f));
                const glm::vec3 origin(x, -0.5f, z);
                const glm::vec3 dir(0.0f, 1.0f, 0.0f);

                CurrentKDNode state{};
                KDStackEntry stack[CurrentKDNode::SHORT_STACK_SIZE];
                int it = 0;
                bool hit_triangle = false;

                while (!hit_triangle && it < 500)
                {
                    uint32_t container_id = traverse_kdtree(
                        &scene.as_kdtree.as[0],
                        bounds_min,
                        bounds_max,
                        tree.as_start,
                        state,
                        stack,
                        origin,
                        dir,
                        it);
                    if (container_id == 0)
                        break;

                    for (; !hit_triangle; ++container_id)
                    {
                        auto bc = scene.as_kdtree.box_content[container_id];
                        PrimitiveChunk16 chunk0 = host_primitives[bc.id()];
                        EXPECT_EQ(chunk0.type(), expected_type);

                        float t = 0.0f, u = 0.0f, v = 0.0f;
                        bool hit = false;
                        if (chunk0.type() == PrimitiveType::Triangle)
                        {
                            const Triangle tri = chunk0.as<Triangle>();
                            hit = intersectRayTriangle(
                                origin, dir, state.t_min, state.t_max,
                                glm::vec3(host_vertices[tri.v0].position),
                                glm::vec3(host_vertices[tri.v1].position),
                                glm::vec3(host_vertices[tri.v2].position),
                                t, u, v);
                        }
                        else if (chunk0.type() == PrimitiveType::FullTriangle)
                        {
                            const InlinedTriangle ftri = load_primitive<InlinedTriangle>(host_primitives, bc.id(), chunk0);
                            hit = intersectRayTriangle(
                                origin, dir, state.t_min, state.t_max,
                                ftri.v0, ftri.v1, ftri.v2,
                                t, u, v);
                        }
                        else if (chunk0.type() == PrimitiveType::NagataPatch)
                        {
                            const NagataPatchPrimitive patch = load_primitive<NagataPatchPrimitive>(host_primitives, bc.id(), chunk0);
                            hit = intersectNagataPrecomputed(
                                origin, dir, state.t_min, state.t_max,
                                glm::vec3(host_vertices[patch.v0].position),
                                glm::vec3(host_vertices[patch.v1].position),
                                glm::vec3(host_vertices[patch.v2].position),
                                patch.C12, patch.C23, patch.C31,
                                t, u, v);
                        }

                        if (hit)
                        {
                            hit_triangle = true;
                            break;
                        }
                        if (bc.is_last())
                            break;
                    }
                    if (state.exited)
                        break;
                }

                if (hit_triangle)
                    ++hit_count;
            }
        }
        return hit_count;
    }
} // namespace

TEST(KDTreeAS, BuildAndTraverseTeapotGPU)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    scene.as_kdtree.config.split_mode = KDSplitMode::SAH;
    scene.as_kdtree.config.use_gpu = true;
    scene.as_kdtree.config.swapYZ = true;
    scene.as_kdtree.config.sort_space = true;
    scene.as_kdtree.config.triangle_type = PrimitiveType::Triangle;
    uint32_t space_id = scene.build_as(meshes, ASType::KDTree);

    EXPECT_EQ(scene.spaces[space_id].as_type, ASType::KDTree);
    EXPECT_GT(scene.spaces[space_id].as_size, 0u);
    EXPECT_GT(scene.spaces[space_id].box_content_size, 0u);
    EXPECT_GT(scene.spaces[space_id].primitives_size, 0u);
    EXPECT_GE(scene.spaces[space_id].root_box, 0);

    const int hit_count = count_kdtree_teapot_hits(scene, space_id, PrimitiveType::Triangle);
    std::cout << "KDTree (SAH GPU) rays hitting teapot triangles: " << hit_count << " / 81" << std::endl;
    EXPECT_GT(hit_count, 15);
}

TEST(KDTreeAS, BuildAndTraverseTeapotFullTriangleGPU)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    scene.as_kdtree.config.split_mode = KDSplitMode::SAH;
    scene.as_kdtree.config.use_gpu = true;
    scene.as_kdtree.config.swapYZ = true;
    scene.as_kdtree.config.sort_space = true;
    scene.as_kdtree.config.triangle_type = PrimitiveType::FullTriangle;
    uint32_t space_id = scene.build_as(meshes, ASType::KDTree);

    EXPECT_GT(scene.spaces[space_id].as_size, 0u);
    EXPECT_GT(scene.spaces[space_id].box_content_size, 0u);
    EXPECT_GT(scene.spaces[space_id].primitives_size, 0u);
    EXPECT_EQ(scene.spaces[space_id].vertices_size, 0u);

    const int hit_count = count_kdtree_teapot_hits(scene, space_id, PrimitiveType::FullTriangle);
    std::cout << "KDTree (FullTriangle GPU) rays hitting teapot: " << hit_count << " / 81" << std::endl;
    EXPECT_GT(hit_count, 15);
}

TEST(KDTreeAS, BuildAndTraverseTeapotNagataPatchGPU)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    scene.as_kdtree.config.split_mode = KDSplitMode::SAH;
    scene.as_kdtree.config.use_gpu = true;
    scene.as_kdtree.config.swapYZ = true;
    scene.as_kdtree.config.sort_space = true;
    scene.as_kdtree.config.triangle_type = PrimitiveType::NagataPatch;
    uint32_t space_id = scene.build_as(meshes, ASType::KDTree);

    EXPECT_GT(scene.spaces[space_id].as_size, 0u);
    EXPECT_GT(scene.spaces[space_id].box_content_size, 0u);
    EXPECT_GT(scene.spaces[space_id].primitives_size, 0u);
    EXPECT_GT(scene.spaces[space_id].vertices_size, 0u);

    const int hit_count = count_kdtree_teapot_hits(scene, space_id, PrimitiveType::NagataPatch);
    std::cout << "KDTree (NagataPatch GPU) rays hitting teapot: " << hit_count << " / 81" << std::endl;
    EXPECT_GT(hit_count, 15);
}

TEST(KDTreeAS, BuildAndTraverseTeapotCPU)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    scene.as_kdtree.config.split_mode = KDSplitMode::SAH;
    scene.as_kdtree.config.use_gpu = false;
    scene.as_kdtree.config.swapYZ = true;
    scene.as_kdtree.config.sort_space = false;
    scene.as_kdtree.config.triangle_type = PrimitiveType::Triangle;
    uint32_t space_id = scene.build_as(meshes, ASType::KDTree);

    EXPECT_GT(scene.spaces[space_id].as_size, 0u);
    EXPECT_GT(scene.spaces[space_id].box_content_size, 0u);
    EXPECT_GT(scene.spaces[space_id].primitives_size, 0u);

    const int hit_count = count_kdtree_teapot_hits(scene, space_id, PrimitiveType::Triangle);
    std::cout << "KDTree (SAH CPU) rays hitting teapot triangles: " << hit_count << " / 81" << std::endl;
    EXPECT_GT(hit_count, 15);
}

TEST(KDTreeAS, BuildAndTraverseSpatialMedianGPU)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    scene.as_kdtree.config.split_mode = KDSplitMode::SpatialMedian;
    scene.as_kdtree.config.use_gpu = true;
    scene.as_kdtree.config.swapYZ = true;
    scene.as_kdtree.config.sort_space = true;
    scene.as_kdtree.config.triangle_type = PrimitiveType::Triangle;
    uint32_t space_id = scene.build_as(meshes, ASType::KDTree);

    EXPECT_GT(scene.spaces[space_id].as_size, 0u);
    EXPECT_GT(scene.spaces[space_id].box_content_size, 0u);

    const int hit_count = count_kdtree_teapot_hits(scene, space_id, PrimitiveType::Triangle);
    std::cout << "KDTree (SpatialMedian GPU) rays hitting teapot triangles: " << hit_count << " / 81" << std::endl;
    EXPECT_GT(hit_count, 15);
}

TEST(KDTreeAS, RenderTeapotGPU)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    scene.as_kdtree.config.split_mode = KDSplitMode::SAH;
    scene.as_kdtree.config.use_gpu = true;
    scene.as_kdtree.config.swapYZ = true;
    scene.as_kdtree.config.sort_space = true;
    scene.as_kdtree.config.triangle_type = PrimitiveType::Triangle;
    uint32_t space_id = scene.build_as(meshes, ASType::KDTree);
    scene.set_active_space(space_id);

    const uint32_t width = 64;
    const uint32_t height = 64;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t cuArray = nullptr;
    ASSERT_EQ(cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArraySurfaceLoadStore), cudaSuccess);

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    cudaSurfaceObject_t surface = 0;
    ASSERT_EQ(cudaCreateSurfaceObject(&surface, &resDesc), cudaSuccess);

    Raytracer raytracer;
    raytracer.parameters.view_pos = glm::vec3(0.5f, -1.2f, 0.3f);
    raytracer.parameters.ax = 0.0f;
    raytracer.parameters.az = 0.0f;
    raytracer.parameters.background_color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    raytracer.update_parameters(scene, width, height);

    EXPECT_EQ(raytracer.parameters.active_as_type, ASType::KDTree);
    EXPECT_NO_THROW(raytracer.draw_first_hit(scene, surface, width, height, 0));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uchar4> host_pixels(width * height);
    EXPECT_EQ(cudaMemcpy2DFromArray(host_pixels.data(), width * sizeof(uchar4), cuArray, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToHost), cudaSuccess);

    uint32_t non_black_pixels = 0;
    for (const auto& px : host_pixels)
    {
        EXPECT_EQ(px.w, 255);
        if (px.x > 0 || px.y > 0 || px.z > 0)
        {
            ++non_black_pixels;
        }
    }
    std::cout << "KDTree rendered non-black pixels: " << non_black_pixels << " / " << (width * height) << std::endl;
    EXPECT_GT(non_black_pixels, 0u);
    EXPECT_LT(non_black_pixels, width * height);

    cudaDestroySurfaceObject(surface);
    cudaFreeArray(cuArray);

    const uint32_t tree_id = static_cast<uint32_t>(scene.spaces[space_id].root_box);
    const uint32_t as_start = scene.spaces[space_id].as_start;
    const uint32_t bc_start = scene.spaces[space_id].box_content_start;

    scene.free_space(space_id);
    EXPECT_FALSE(scene.spaces.is_allocated(space_id));
    EXPECT_FALSE(scene.as_kdtree.trees.is_allocated(tree_id));
    EXPECT_FALSE(scene.as_kdtree.as.is_allocated(as_start));
    EXPECT_FALSE(scene.as_kdtree.box_content.is_allocated(bc_start));
}


namespace
{
    int count_bvh_teapot_hits(Scene& scene, uint32_t space_id, PrimitiveType expected_type)
    {
        const int tree_id = scene.spaces[space_id].root_box;
        EXPECT_GE(tree_id, 0);
        const auto tree = scene.as_bvh.trees[static_cast<size_t>(tree_id)];
        const glm::vec3 bounds_min = tree.bounds_min;
        const glm::vec3 bounds_max = tree.bounds_max;

        soa::soa_vector<PrimitiveChunk16> host_primitives = scene.geometry_context.primitives;
        soa::soa_vector<Vertex> host_vertices = scene.geometry_context.vertices;

        int hit_count = 0;
        for (int ix = 0; ix < 9; ++ix)
        {
            for (int iz = 0; iz < 9; ++iz)
            {
                const float x = bounds_min.x + (bounds_max.x - bounds_min.x) * (0.1f + 0.8f * (static_cast<float>(ix) / 8.0f));
                const float z = bounds_min.z + (bounds_max.z - bounds_min.z) * (0.1f + 0.8f * (static_cast<float>(iz) / 8.0f));
                const glm::vec3 origin(x, -0.5f, z);
                const glm::vec3 dir(0.0f, 1.0f, 0.0f);

                CurrentBVHNode state{};
                BVHStackEntry stack[CurrentBVHNode::SHORT_STACK_SIZE];
                int it = 0;
                bool hit_triangle = false;

                while (!hit_triangle && it < 500)
                {
                    uint32_t container_id = (tree.use_bvh16 != 0u)
                        ? traverse_bvh16_impl(
                            &scene.as_bvh.as16[0],
                            tree.as_start,
                            bounds_min,
                            bounds_max,
                            state,
                            stack,
                            origin,
                            dir,
                            it)
                        : traverse_bvh_impl(
                            &scene.as_bvh.as[0],
                            tree.as_start,
                            state,
                            stack,
                            origin,
                            dir,
                            it);
                    if (container_id == 0)
                        break;

                    for (; !hit_triangle; ++container_id)
                    {
                        auto bc = scene.as_bvh.box_content[container_id];
                        PrimitiveChunk16 chunk0 = host_primitives[bc.id()];
                        EXPECT_EQ(chunk0.type(), expected_type);

                        float t = 0.0f, u = 0.0f, v = 0.0f;
                        bool hit = false;
                        if (chunk0.type() == PrimitiveType::Triangle)
                        {
                            const Triangle tri = chunk0.as<Triangle>();
                            hit = intersectRayTriangle(
                                origin, dir, state.t_min, state.t_max,
                                glm::vec3(host_vertices[tri.v0].position),
                                glm::vec3(host_vertices[tri.v1].position),
                                glm::vec3(host_vertices[tri.v2].position),
                                t, u, v);
                        }
                        else if (chunk0.type() == PrimitiveType::FullTriangle)
                        {
                            const InlinedTriangle ftri = load_primitive<InlinedTriangle>(host_primitives, bc.id(), chunk0);
                            hit = intersectRayTriangle(
                                origin, dir, state.t_min, state.t_max,
                                ftri.v0, ftri.v1, ftri.v2,
                                t, u, v);
                        }
                        else if (chunk0.type() == PrimitiveType::NagataPatch)
                        {
                            const NagataPatchPrimitive patch = load_primitive<NagataPatchPrimitive>(host_primitives, bc.id(), chunk0);
                            hit = intersectNagataPrecomputed(
                                origin, dir, state.t_min, state.t_max,
                                glm::vec3(host_vertices[patch.v0].position),
                                glm::vec3(host_vertices[patch.v1].position),
                                glm::vec3(host_vertices[patch.v2].position),
                                patch.C12, patch.C23, patch.C31,
                                t, u, v);
                        }

                        if (hit)
                        {
                            hit_triangle = true;
                            break;
                        }
                        if (bc.is_last())
                            break;
                    }
                    if (state.exited)
                        break;
                }

                if (hit_triangle)
                    ++hit_count;
            }
        }
        return hit_count;
    }
} // namespace

TEST(BVHAS, BuildAndTraverseTeapotGPU)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    scene.as_bvh.config.split_mode = BVHSplitMode::SAH;
    scene.as_bvh.config.use_gpu = true;
    scene.as_bvh.config.swapYZ = true;
    scene.as_bvh.config.sort_space = true;
    scene.as_bvh.config.triangle_type = PrimitiveType::Triangle;
    uint32_t space_id = scene.build_as(meshes, ASType::BVH);

    EXPECT_EQ(scene.spaces[space_id].as_type, ASType::BVH);
    EXPECT_GT(scene.spaces[space_id].as_size, 0u);
    EXPECT_GT(scene.spaces[space_id].box_content_size, 0u);
    EXPECT_GT(scene.spaces[space_id].primitives_size, 0u);
    EXPECT_GE(scene.spaces[space_id].root_box, 0);

    const int hit_count = count_bvh_teapot_hits(scene, space_id, PrimitiveType::Triangle);
    std::cout << "BVH (SAH GPU) rays hitting teapot triangles: " << hit_count << " / 81" << std::endl;
    EXPECT_GT(hit_count, 15);
}

TEST(BVHAS, BuildAndTraverseTeapotFullTriangleGPU)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    scene.as_bvh.config.split_mode = BVHSplitMode::SAH;
    scene.as_bvh.config.use_gpu = true;
    scene.as_bvh.config.swapYZ = true;
    scene.as_bvh.config.sort_space = true;
    scene.as_bvh.config.triangle_type = PrimitiveType::FullTriangle;
    uint32_t space_id = scene.build_as(meshes, ASType::BVH);

    EXPECT_GT(scene.spaces[space_id].as_size, 0u);
    EXPECT_GT(scene.spaces[space_id].box_content_size, 0u);
    EXPECT_GT(scene.spaces[space_id].primitives_size, 0u);
    EXPECT_EQ(scene.spaces[space_id].vertices_size, 0u);

    const int hit_count = count_bvh_teapot_hits(scene, space_id, PrimitiveType::FullTriangle);
    std::cout << "BVH (FullTriangle GPU) rays hitting teapot: " << hit_count << " / 81" << std::endl;
    EXPECT_GT(hit_count, 15);
}

TEST(BVHAS, BuildAndTraverseTeapotNagataPatchGPU)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    scene.as_bvh.config.split_mode = BVHSplitMode::SAH;
    scene.as_bvh.config.use_gpu = true;
    scene.as_bvh.config.swapYZ = true;
    scene.as_bvh.config.sort_space = true;
    scene.as_bvh.config.triangle_type = PrimitiveType::NagataPatch;
    uint32_t space_id = scene.build_as(meshes, ASType::BVH);

    EXPECT_GT(scene.spaces[space_id].as_size, 0u);
    EXPECT_GT(scene.spaces[space_id].box_content_size, 0u);
    EXPECT_GT(scene.spaces[space_id].primitives_size, 0u);
    EXPECT_GT(scene.spaces[space_id].vertices_size, 0u);

    const int hit_count = count_bvh_teapot_hits(scene, space_id, PrimitiveType::NagataPatch);
    std::cout << "BVH (NagataPatch GPU) rays hitting teapot: " << hit_count << " / 81" << std::endl;
    EXPECT_GT(hit_count, 15);
}

TEST(BVHAS, BuildAndTraverseTeapotCPU)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    scene.as_bvh.config.split_mode = BVHSplitMode::SAH;
    scene.as_bvh.config.use_gpu = false;
    scene.as_bvh.config.swapYZ = true;
    scene.as_bvh.config.sort_space = false;
    scene.as_bvh.config.triangle_type = PrimitiveType::Triangle;
    uint32_t space_id = scene.build_as(meshes, ASType::BVH);

    EXPECT_GT(scene.spaces[space_id].as_size, 0u);
    EXPECT_GT(scene.spaces[space_id].box_content_size, 0u);
    EXPECT_GT(scene.spaces[space_id].primitives_size, 0u);

    const int hit_count = count_bvh_teapot_hits(scene, space_id, PrimitiveType::Triangle);
    std::cout << "BVH (SAH CPU) rays hitting teapot triangles: " << hit_count << " / 81" << std::endl;
    EXPECT_GT(hit_count, 15);
}

TEST(BVHAS, BuildAndTraverseSpatialAndObjectMedianGPU)
{
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());

    {
        Scene scene("TeapotSpatial");
        scene.load(teapot_view);
        auto meshes = scene.get_mesh_transform();
        ASSERT_FALSE(meshes.empty());

        scene.as_bvh.config.split_mode = BVHSplitMode::SpatialMedian;
        scene.as_bvh.config.use_gpu = true;
        scene.as_bvh.config.swapYZ = true;
        scene.as_bvh.config.sort_space = true;
        scene.as_bvh.config.triangle_type = PrimitiveType::Triangle;
        uint32_t space_id = scene.build_as(meshes, ASType::BVH);

        EXPECT_GT(scene.spaces[space_id].as_size, 0u);
        EXPECT_GT(scene.spaces[space_id].box_content_size, 0u);

        const int hit_count = count_bvh_teapot_hits(scene, space_id, PrimitiveType::Triangle);
        std::cout << "BVH (SpatialMedian GPU) rays hitting teapot triangles: " << hit_count << " / 81" << std::endl;
        EXPECT_GT(hit_count, 15);
    }

    {
        Scene scene("TeapotObject");
        scene.load(teapot_view);
        auto meshes = scene.get_mesh_transform();
        ASSERT_FALSE(meshes.empty());

        scene.as_bvh.config.split_mode = BVHSplitMode::ObjectMedian;
        scene.as_bvh.config.use_gpu = true;
        scene.as_bvh.config.swapYZ = true;
        scene.as_bvh.config.sort_space = true;
        scene.as_bvh.config.triangle_type = PrimitiveType::Triangle;
        uint32_t space_id = scene.build_as(meshes, ASType::BVH);

        EXPECT_GT(scene.spaces[space_id].as_size, 0u);
        EXPECT_GT(scene.spaces[space_id].box_content_size, 0u);

        const int hit_count = count_bvh_teapot_hits(scene, space_id, PrimitiveType::Triangle);
        std::cout << "BVH (ObjectMedian GPU) rays hitting teapot triangles: " << hit_count << " / 81" << std::endl;
        EXPECT_GT(hit_count, 15);
    }
}

TEST(BVHAS, RenderTeapotGPU)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    auto meshes = scene.get_mesh_transform();
    ASSERT_FALSE(meshes.empty());

    scene.as_bvh.config.split_mode = BVHSplitMode::SAH;
    scene.as_bvh.config.use_gpu = true;
    scene.as_bvh.config.swapYZ = true;
    scene.as_bvh.config.sort_space = true;
    scene.as_bvh.config.triangle_type = PrimitiveType::Triangle;
    uint32_t space_id = scene.build_as(meshes, ASType::BVH);
    scene.set_active_space(space_id);

    const uint32_t width = 64;
    const uint32_t height = 64;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t cuArray = nullptr;
    ASSERT_EQ(cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArraySurfaceLoadStore), cudaSuccess);

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    cudaSurfaceObject_t surface = 0;
    ASSERT_EQ(cudaCreateSurfaceObject(&surface, &resDesc), cudaSuccess);

    Raytracer raytracer;
    raytracer.parameters.view_pos = glm::vec3(0.5f, -1.2f, 0.3f);
    raytracer.parameters.ax = 0.0f;
    raytracer.parameters.az = 0.0f;
    raytracer.parameters.background_color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    raytracer.update_parameters(scene, width, height);

    EXPECT_EQ(raytracer.parameters.active_as_type, ASType::BVH);
    EXPECT_NO_THROW(raytracer.draw_first_hit(scene, surface, width, height, 0));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uchar4> host_pixels(width * height);
    EXPECT_EQ(cudaMemcpy2DFromArray(host_pixels.data(), width * sizeof(uchar4), cuArray, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToHost), cudaSuccess);

    uint32_t non_black_pixels = 0;
    for (const auto& px : host_pixels)
    {
        EXPECT_EQ(px.w, 255);
        if (px.x > 0 || px.y > 0 || px.z > 0)
        {
            ++non_black_pixels;
        }
    }
    std::cout << "BVH rendered non-black pixels: " << non_black_pixels << " / " << (width * height) << std::endl;
    EXPECT_GT(non_black_pixels, 0u);
    EXPECT_LT(non_black_pixels, width * height);

    cudaDestroySurfaceObject(surface);
    cudaFreeArray(cuArray);

    const uint32_t tree_id = static_cast<uint32_t>(scene.spaces[space_id].root_box);
    const uint32_t as_start = scene.spaces[space_id].as_start;
    const uint32_t bc_start = scene.spaces[space_id].box_content_start;
    const bool was_bvh16 = (scene.as_bvh.trees[tree_id].use_bvh16 != 0u);

    scene.free_space(space_id);
    EXPECT_FALSE(scene.spaces.is_allocated(space_id));
    EXPECT_FALSE(scene.as_bvh.trees.is_allocated(tree_id));
    if (was_bvh16)
    {
        EXPECT_FALSE(scene.as_bvh.as16.is_allocated(as_start));
    }
    else
    {
        EXPECT_FALSE(scene.as_bvh.as.is_allocated(as_start));
    }
    EXPECT_FALSE(scene.as_bvh.box_content.is_allocated(bc_start));
}

TEST(BVHAS, CompareBVH16AndBVH32TeapotGPU)
{
    static_assert(sizeof(BVHNode16) == 16);
    static_assert(sizeof(BVHNode) == 32);
    EXPECT_EQ(sizeof(BVHNode16), 16u);
    EXPECT_EQ(sizeof(BVHNode), 32u);

    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());

    Scene scene16("TeapotBVH16");
    scene16.load(teapot_view);
    auto meshes16 = scene16.get_mesh_transform();
    ASSERT_FALSE(meshes16.empty());
    scene16.as_bvh.config.split_mode = BVHSplitMode::SAH;
    scene16.as_bvh.config.use_gpu = true;
    scene16.as_bvh.config.swapYZ = true;
    scene16.as_bvh.config.sort_space = true;
    scene16.as_bvh.config.triangle_type = PrimitiveType::Triangle;
    scene16.as_bvh.config.use_bvh16 = true;
    const uint32_t space16 = scene16.build_as(meshes16, ASType::BVH);
    scene16.set_active_space(space16);

    Scene scene32("TeapotBVH32");
    scene32.load(teapot_view);
    auto meshes32 = scene32.get_mesh_transform();
    ASSERT_FALSE(meshes32.empty());
    scene32.as_bvh.config.split_mode = BVHSplitMode::SAH;
    scene32.as_bvh.config.use_gpu = true;
    scene32.as_bvh.config.swapYZ = true;
    scene32.as_bvh.config.sort_space = true;
    scene32.as_bvh.config.triangle_type = PrimitiveType::Triangle;
    scene32.as_bvh.config.use_bvh16 = false;
    const uint32_t space32 = scene32.build_as(meshes32, ASType::BVH);
    scene32.set_active_space(space32);

    const uint32_t nodes16 = static_cast<uint32_t>(scene16.spaces[space16].as_size);
    const uint32_t nodes32 = static_cast<uint32_t>(scene32.spaces[space32].as_size);
    EXPECT_GT(nodes16, 0u);
    EXPECT_EQ(nodes16, nodes32);

    const size_t bytes16 = static_cast<size_t>(nodes16) * sizeof(BVHNode16);
    const size_t bytes32 = static_cast<size_t>(nodes32) * sizeof(BVHNode);
    EXPECT_EQ(bytes32, 2u * bytes16);

    const int tree16_id = scene16.spaces[space16].root_box;
    const int tree32_id = scene32.spaces[space32].root_box;
    ASSERT_GE(tree16_id, 0);
    ASSERT_GE(tree32_id, 0);
    EXPECT_EQ(scene16.as_bvh.trees[static_cast<size_t>(tree16_id)].use_bvh16, 1u);
    EXPECT_EQ(scene32.as_bvh.trees[static_cast<size_t>(tree32_id)].use_bvh16, 0u);

    const int hits16 = count_bvh_teapot_hits(scene16, space16, PrimitiveType::Triangle);
    const int hits32 = count_bvh_teapot_hits(scene32, space32, PrimitiveType::Triangle);
    EXPECT_GT(hits16, 15);
    EXPECT_EQ(hits16, hits32);

    const uint32_t width = 64;
    const uint32_t height = 64;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t cuArray = nullptr;
    ASSERT_EQ(cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArraySurfaceLoadStore), cudaSuccess);
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    cudaSurfaceObject_t surface = 0;
    ASSERT_EQ(cudaCreateSurfaceObject(&surface, &resDesc), cudaSuccess);

    auto render_scene = [&](Scene& s) {
        Raytracer raytracer;
        raytracer.parameters.view_pos = glm::vec3(0.5f, -1.2f, 0.3f);
        raytracer.parameters.ax = 0.0f;
        raytracer.parameters.az = 0.0f;
        raytracer.parameters.background_color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        raytracer.update_parameters(s, width, height);
        EXPECT_NO_THROW(raytracer.draw_first_hit(s, surface, width, height, 0));
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<uchar4> pixels(width * height);
        EXPECT_EQ(cudaMemcpy2DFromArray(pixels.data(), width * sizeof(uchar4), cuArray, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToHost), cudaSuccess);
        return pixels;
    };

    const auto pixels16 = render_scene(scene16);
    const auto pixels32 = render_scene(scene32);

    uint32_t diff_pixels = 0;
    uint32_t non_black_16 = 0;
    uint32_t non_black_32 = 0;
    for (size_t i = 0; i < pixels16.size(); ++i)
    {
        if (pixels16[i].x > 0 || pixels16[i].y > 0 || pixels16[i].z > 0) ++non_black_16;
        if (pixels32[i].x > 0 || pixels32[i].y > 0 || pixels32[i].z > 0) ++non_black_32;
        if (pixels16[i].x != pixels32[i].x || pixels16[i].y != pixels32[i].y || pixels16[i].z != pixels32[i].z || pixels16[i].w != pixels32[i].w)
        {
            ++diff_pixels;
        }
    }
    EXPECT_GT(non_black_16, 0u);
    EXPECT_EQ(non_black_16, non_black_32);
    EXPECT_EQ(diff_pixels, 0u);

    cudaDestroySurfaceObject(surface);
    cudaFreeArray(cuArray);
}

TEST(BVHAS, DISABLED_BenchmarkBVH16VsBVH32)
{
    const char* scenes[] = {"Scenes/utah_teapot.glb", "Scenes/hairball.glb", "Scenes/ABeautifulGame.glb"};
    const uint32_t width = 1920;
    const uint32_t height = 1080;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t cuArray = nullptr;
    ASSERT_EQ(cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArraySurfaceLoadStore), cudaSuccess);
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    cudaSurfaceObject_t surface = 0;
    ASSERT_EQ(cudaCreateSurfaceObject(&surface, &resDesc), cudaSuccess);

    cudaEvent_t ev_start = nullptr, ev_stop = nullptr;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    for (const char* asset_name : scenes)
    {
        auto view = vkbase::assets::get(asset_name);
        if (view.empty())
            continue;

        for (bool use16 : {false, true})
        {
            Scene scene(asset_name);
            scene.load(view);
            auto meshes = scene.get_mesh_transform();
            if (meshes.empty())
                continue;

            scene.as_bvh.config.split_mode = BVHSplitMode::SAH;
            scene.as_bvh.config.use_gpu = true;
            scene.as_bvh.config.swapYZ = true;
            scene.as_bvh.config.sort_space = true;
            scene.as_bvh.config.triangle_type = PrimitiveType::Triangle;
            scene.as_bvh.config.use_bvh16 = use16;
            const uint32_t space_id = scene.build_as(meshes, ASType::BVH);
            scene.set_active_space(space_id);

            const auto tree = scene.as_bvh.trees[static_cast<size_t>(scene.spaces[space_id].root_box)];
            const glm::vec3 center = (tree.bounds_min + tree.bounds_max) * 0.5f;
            const float radius = glm::length(tree.bounds_max - tree.bounds_min) * 0.5f;

            Raytracer raytracer;
            raytracer.parameters.view_pos = center + glm::vec3(0.0f, -1.6f * radius, 0.25f * radius);
            raytracer.parameters.ax = 0.0f;
            raytracer.parameters.az = 0.0f;
            raytracer.parameters.background_color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            raytracer.update_parameters(scene, width, height);

            for (int i = 0; i < 5; ++i)
                raytracer.draw_first_hit(scene, surface, width, height, 0);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

            const int iters = 30;
            cudaEventRecord(ev_start, 0);
            for (int i = 0; i < iters; ++i)
                raytracer.draw_first_hit(scene, surface, width, height, 0);
            cudaEventRecord(ev_stop, 0);
            ASSERT_EQ(cudaEventSynchronize(ev_stop), cudaSuccess);
            float total_ms = 0.0f;
            cudaEventElapsedTime(&total_ms, ev_start, ev_stop);

            // Also measure debug stats (show_iterations + show_boxes)
            raytracer.parameters.show_iterations = true;
            raytracer.parameters.show_boxes = true;
            raytracer.parameters.iterations_max = 255.0f;
            raytracer.parameters.boxes_max = 255.0f;
            raytracer.draw_first_hit(scene, surface, width, height, 0);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            std::vector<uchar4> dbg_px(width * height);
            ASSERT_EQ(cudaMemcpy2DFromArray(dbg_px.data(), width * sizeof(uchar4), cuArray, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToHost), cudaSuccess);

            uint64_t sum_it = 0, sum_box = 0, hit_px = 0;
            for (const auto& px : dbg_px)
            {
                sum_it += px.x;
                sum_box += px.y;
                if (px.z > 0) ++hit_px;
            }

            const size_t node_bytes = static_cast<size_t>(scene.spaces[space_id].as_size) * (use16 ? sizeof(BVHNode16) : sizeof(BVHNode));
            std::cout << "[BENCH] " << asset_name
                      << " mode=" << (use16 ? "BVH16" : "BVH32")
                      << " nodes=" << scene.spaces[space_id].as_size
                      << " nodeMB=" << (static_cast<double>(node_bytes) / (1024.0 * 1024.0))
                      << " avgMs=" << (total_ms / iters)
                      << " avgIt=" << (static_cast<double>(sum_it) / (width * height))
                      << " avgBoxHits=" << (static_cast<double>(sum_box) / (width * height))
                      << " hitPx=" << hit_px
                      << std::endl;
        }
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaDestroySurfaceObject(surface);
    cudaFreeArray(cuArray);
}
