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

    EXPECT_FALSE(scene.get_root_node().name.empty());
    EXPECT_FALSE(scene.get_root_node().children.empty());

    const uint32_t subscene_id = scene.get_root_node().children[0]; ASSERT_LT(subscene_id, scene.m_nodes.size()); Node& subscene = scene.m_nodes[subscene_id];
    EXPECT_FALSE(subscene.name.empty());
    EXPECT_GT(subscene.count_subtree_meshes(scene.m_nodes), 0u);
    EXPECT_GT(subscene.count_subtree_triangles(scene.m_nodes, scene.m_meshes), 0u); EXPECT_GT(subscene.count_subtree_vertices(scene.m_nodes, scene.m_meshes), 0u);

    EXPECT_FALSE(subscene.as_ref.is_valid());
    EXPECT_EQ(subscene.as_ref.type, ASType::None);

    auto subscene_meshes = scene.get_mesh_transform(&subscene);
    ASSERT_FALSE(subscene_meshes.empty());

    scene.as_octree.config.stretch = false;
    scene.config.swapYZ = true;
    scene.as_octree.config.min_depth = 1;
    scene.as_octree.config.max_depth = 10;
    scene.as_octree.config.triangle_to_box_stop_ratio = 1.1f;
    scene.as_octree.config.use_gpu = true;
    scene.config.sort_space = true;
    uint32_t space_id = scene.build_node_as(subscene_id, ASType::Octree);
    ASSERT_NE(space_id, UINT32_MAX);

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


    scene.as_octree.config.stretch = false;
    scene.config.swapYZ = true;
    scene.as_octree.config.min_depth = 1;
    scene.as_octree.config.max_depth = 10;
    scene.as_octree.config.triangle_to_box_stop_ratio = 1.1f;
    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::Octree);

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


    scene.as_octree.config.stretch = false;
    scene.config.swapYZ = true;
    scene.as_octree.config.min_depth = 1;
    scene.as_octree.config.max_depth = 10;
    scene.as_octree.config.triangle_to_box_stop_ratio = 1.1f;
    scene.config.triangle_type = PrimitiveType::FullTriangle;
    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::Octree);

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


    scene.as_octree.config.stretch = false;
    scene.config.swapYZ = true;
    scene.as_octree.config.min_depth = 1;
    scene.as_octree.config.max_depth = 10;
    scene.as_octree.config.triangle_to_box_stop_ratio = 1.1f;
    scene.config.triangle_type = PrimitiveType::NagataPatch;
    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::Octree);

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


    scene.as_octree.config.stretch = false;
    scene.config.swapYZ = true;
    scene.as_octree.config.min_depth = 1;
    scene.as_octree.config.max_depth = 10;
    scene.as_octree.config.triangle_to_box_stop_ratio = 1.1f;
    scene.as_octree.config.use_gpu = false;
    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::Octree);

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


    // Build AS without sort_space (sort = false)
    scene.as_octree.config.stretch = false;
    scene.config.swapYZ = true;
    scene.as_octree.config.min_depth = 1;
    scene.as_octree.config.max_depth = 10;
    scene.as_octree.config.triangle_to_box_stop_ratio = 1.1f;
    scene.as_octree.config.use_gpu = true;
    scene.config.sort_space = false;
    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::Octree);

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


    scene.as_octree.config.stretch = false;
    scene.config.swapYZ = true;
    scene.as_octree.config.min_depth = 1;
    scene.as_octree.config.max_depth = 8;
    scene.as_octree.config.triangle_to_box_stop_ratio = 1.1f;
    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::Octree);
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


    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::Optix);

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


    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::Optix);
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


    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::Optix);
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


    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::Optix);
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

TEST(UniformGridAS, ComputeAdaptiveGridDimensions)
{
    EXPECT_FLOAT_EQ(UniformGridAS::Config{}.voxels_per_triangle, 1.0f);

    // Cubic AABB: 1000 triangles at 1.0 vox/tri -> 10x10x10
    {
        const auto layout = UniformGridAS::compute_grid_dimensions(glm::vec3(1.0f, 1.0f, 1.0f), 1000u, 1.0f);
        EXPECT_EQ(layout.dims, glm::uvec3(10u, 10u, 10u));
        EXPECT_NEAR(layout.cell_size, 0.1f, 1e-6f);
    }

    // Anisotropic AABB: (1.0, 0.5, 0.5), 250 tris -> N_x*N_y*N_z = 250 with t = 10 -> 10x5x5
    {
        const auto layout = UniformGridAS::compute_grid_dimensions(glm::vec3(1.0f, 0.5f, 0.5f), 250u, 1.0f);
        EXPECT_EQ(layout.dims, glm::uvec3(10u, 5u, 5u));
        EXPECT_NEAR(layout.cell_size, 0.1f, 1e-6f);
    }

    // Planar AABB (one zero extent): 100 tris -> fixed to 1 on flat axis, 10x1x10
    {
        const auto layout = UniformGridAS::compute_grid_dimensions(glm::vec3(1.0f, 0.0f, 1.0f), 100u, 1.0f);
        EXPECT_EQ(layout.dims, glm::uvec3(10u, 1u, 10u));
        EXPECT_NEAR(layout.cell_size, 0.1f, 1e-6f);
    }

    // Thin 1D-like AABB: (1.0, 0.01, 0.01), 50 tris -> Y,Z clamp to 1, X solves to 50
    {
        const auto layout = UniformGridAS::compute_grid_dimensions(glm::vec3(1.0f, 0.01f, 0.01f), 50u, 1.0f);
        EXPECT_EQ(layout.dims, glm::uvec3(50u, 1u, 1u));
        EXPECT_GE(static_cast<float>(layout.dims.x) * layout.cell_size, 1.0f);
        EXPECT_GE(static_cast<float>(layout.dims.y) * layout.cell_size, 0.01f);
        EXPECT_GE(static_cast<float>(layout.dims.z) * layout.cell_size, 0.01f);
    }

    // Floor rounding + cell_size adjustment guarantees full AABB enclosure
    {
        const glm::vec3 aabb(1.0f, 1.0f, 1.0f);
        const auto layout = UniformGridAS::compute_grid_dimensions(aabb, 2000u, 1.0f);
        EXPECT_EQ(layout.dims, glm::uvec3(12u, 12u, 12u));
        EXPECT_GE(static_cast<float>(layout.dims.x) * layout.cell_size, aabb.x);
        EXPECT_GE(static_cast<float>(layout.dims.y) * layout.cell_size, aabb.y);
        EXPECT_GE(static_cast<float>(layout.dims.z) * layout.cell_size, aabb.z);
    }

    // Per-axis cap at 1024 (10-bit CurrentGridCell coordinates) still encloses elongated AABB
    {
        const glm::vec3 aabb(1.0f, 0.1f, 0.1f);
        const auto layout = UniformGridAS::compute_grid_dimensions(aabb, 100000000u, 10.0f);
        EXPECT_LE(layout.dims.x, UniformGridAS::kMaxAxisResolution);
        EXPECT_LE(layout.dims.y, UniformGridAS::kMaxAxisResolution);
        EXPECT_LE(layout.dims.z, UniformGridAS::kMaxAxisResolution);
        EXPECT_EQ(layout.dims.x, UniformGridAS::kMaxAxisResolution);
        EXPECT_GE(static_cast<float>(layout.dims.x) * layout.cell_size, aabb.x);
        EXPECT_GE(static_cast<float>(layout.dims.y) * layout.cell_size, aabb.y);
        EXPECT_GE(static_cast<float>(layout.dims.z) * layout.cell_size, aabb.z);
    }

    // Degenerate zero-size AABB falls back to 1x1x1
    {
        const auto layout = UniformGridAS::compute_grid_dimensions(glm::vec3(0.0f), 10u, 1.0f);
        EXPECT_EQ(layout.dims, glm::uvec3(1u, 1u, 1u));
        EXPECT_GT(layout.cell_size, 0.0f);
    }
}

TEST(UniformGridAS, DensityScalingAndPlanarGeometryGPU)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);

    scene.config.swapYZ = true;
    scene.config.sort_space = true;
    scene.config.triangle_type = PrimitiveType::Triangle;

    scene.as_uniform_grid.config.voxels_per_triangle = 0.5f;
    const uint32_t coarse_space = scene.build_node_as(scene.root_node_id, ASType::UniformGrid);
    ASSERT_NE(coarse_space, UINT32_MAX);
    const uint32_t coarse_as_size = scene.spaces[coarse_space].as_size;
    const float coarse_cell_size = scene.as_uniform_grid.grids[static_cast<size_t>(scene.spaces[coarse_space].root_box)].cell_size;
    EXPECT_GT(count_uniform_grid_teapot_hits(scene, coarse_space, PrimitiveType::Triangle), 15);

    scene.as_uniform_grid.config.voxels_per_triangle = 4.0f;
    const uint32_t fine_space = scene.build_node_as(scene.root_node_id, ASType::UniformGrid);
    ASSERT_NE(fine_space, UINT32_MAX);
    const auto fine_grid = scene.as_uniform_grid.grids[static_cast<size_t>(scene.spaces[fine_space].root_box)];

    EXPECT_GT(scene.spaces[fine_space].as_size, coarse_as_size);
    EXPECT_LT(fine_grid.cell_size, coarse_cell_size);
    EXPECT_GT(count_uniform_grid_teapot_hits(scene, fine_space, PrimitiveType::Triangle), 15);
}

TEST(UniformGridAS, BuildAndTraverseTeapotGPU)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);


    scene.as_uniform_grid.config.voxels_per_triangle = 1.0f;
    scene.config.swapYZ = true;
    scene.config.sort_space = true;
    scene.config.triangle_type = PrimitiveType::Triangle;
    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::UniformGrid);

    EXPECT_EQ(scene.spaces[space_id].as_type, ASType::UniformGrid);
    EXPECT_GT(scene.spaces[space_id].as_size, 0u);
    EXPECT_GT(scene.spaces[space_id].box_content_size, 0u);
    EXPECT_GT(scene.spaces[space_id].primitives_size, 0u);

    const int grid_id = scene.spaces[space_id].root_box;
    ASSERT_GE(grid_id, 0);
    const auto grid = scene.as_uniform_grid.grids[static_cast<size_t>(grid_id)];
    EXPECT_GE(std::min({grid.dims.x, grid.dims.y, grid.dims.z}), 1u);
    EXPECT_LE(std::max({grid.dims.x, grid.dims.y, grid.dims.z}), UniformGridAS::kMaxAxisResolution);
    EXPECT_EQ(grid.dims.x * grid.dims.y * grid.dims.z, scene.spaces[space_id].as_size);

    // Verify tight normalization: vertices start at (0,0,0), max extent is 1.0, and adaptive dims match solver
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
    const uint32_t num_triangles = scene.spaces[space_id].primitives_size / Triangle::occupied_slots;
    const auto expected_layout = UniformGridAS::compute_grid_dimensions(v_max, num_triangles, 1.0f);
    EXPECT_EQ(grid.dims, expected_layout.dims);
    EXPECT_NEAR(grid.cell_size, expected_layout.cell_size, 1e-6f);
    EXPECT_GE(static_cast<float>(grid.dims.x) * grid.cell_size, v_max.x - 1e-5f);
    EXPECT_GE(static_cast<float>(grid.dims.y) * grid.cell_size, v_max.y - 1e-5f);
    EXPECT_GE(static_cast<float>(grid.dims.z) * grid.cell_size, v_max.z - 1e-5f);

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


    scene.as_uniform_grid.config.voxels_per_triangle = 1.0f;
    scene.config.swapYZ = true;
    scene.config.sort_space = true;
    scene.config.triangle_type = PrimitiveType::FullTriangle;
    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::UniformGrid);

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


    scene.as_uniform_grid.config.voxels_per_triangle = 1.0f;
    scene.config.swapYZ = true;
    scene.config.sort_space = true;
    scene.config.triangle_type = PrimitiveType::NagataPatch;
    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::UniformGrid);

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


    scene.as_uniform_grid.config.voxels_per_triangle = 1.0f;
    scene.config.swapYZ = true;
    scene.config.sort_space = false;
    scene.config.triangle_type = PrimitiveType::Triangle;
    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::UniformGrid);

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


    scene.as_uniform_grid.config.voxels_per_triangle = 1.0f;
    scene.config.swapYZ = true;
    scene.config.sort_space = true;
    scene.config.triangle_type = PrimitiveType::Triangle;
    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::UniformGrid);
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


    scene.as_kdtree.config.split_mode = KDSplitMode::SAH;
    scene.as_kdtree.config.use_gpu = true;
    scene.config.swapYZ = true;
    scene.config.sort_space = true;
    scene.config.triangle_type = PrimitiveType::Triangle;
    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::KDTree);

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


    scene.as_kdtree.config.split_mode = KDSplitMode::SAH;
    scene.as_kdtree.config.use_gpu = true;
    scene.config.swapYZ = true;
    scene.config.sort_space = true;
    scene.config.triangle_type = PrimitiveType::FullTriangle;
    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::KDTree);

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


    scene.as_kdtree.config.split_mode = KDSplitMode::SAH;
    scene.as_kdtree.config.use_gpu = true;
    scene.config.swapYZ = true;
    scene.config.sort_space = true;
    scene.config.triangle_type = PrimitiveType::NagataPatch;
    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::KDTree);

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


    scene.as_kdtree.config.split_mode = KDSplitMode::SAH;
    scene.as_kdtree.config.use_gpu = false;
    scene.config.swapYZ = true;
    scene.config.sort_space = false;
    scene.config.triangle_type = PrimitiveType::Triangle;
    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::KDTree);

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


    scene.as_kdtree.config.split_mode = KDSplitMode::SpatialMedian;
    scene.as_kdtree.config.use_gpu = true;
    scene.config.swapYZ = true;
    scene.config.sort_space = true;
    scene.config.triangle_type = PrimitiveType::Triangle;
    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::KDTree);

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


    scene.as_kdtree.config.split_mode = KDSplitMode::SAH;
    scene.as_kdtree.config.use_gpu = true;
    scene.config.swapYZ = true;
    scene.config.sort_space = true;
    scene.config.triangle_type = PrimitiveType::Triangle;
    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::KDTree);
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


    scene.as_bvh.config.split_mode = BVHSplitMode::SAH;
    scene.as_bvh.config.use_gpu = true;
    scene.config.swapYZ = true;
    scene.config.sort_space = true;
    scene.config.triangle_type = PrimitiveType::Triangle;
    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::BVH);

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


    scene.as_bvh.config.split_mode = BVHSplitMode::SAH;
    scene.as_bvh.config.use_gpu = true;
    scene.config.swapYZ = true;
    scene.config.sort_space = true;
    scene.config.triangle_type = PrimitiveType::FullTriangle;
    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::BVH);

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


    scene.as_bvh.config.split_mode = BVHSplitMode::SAH;
    scene.as_bvh.config.use_gpu = true;
    scene.config.swapYZ = true;
    scene.config.sort_space = true;
    scene.config.triangle_type = PrimitiveType::NagataPatch;
    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::BVH);

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


    scene.as_bvh.config.split_mode = BVHSplitMode::SAH;
    scene.as_bvh.config.use_gpu = false;
    scene.config.swapYZ = true;
    scene.config.sort_space = false;
    scene.config.triangle_type = PrimitiveType::Triangle;
    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::BVH);

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

        scene.as_bvh.config.split_mode = BVHSplitMode::SpatialMedian;
        scene.as_bvh.config.use_gpu = true;
        scene.config.swapYZ = true;
        scene.config.sort_space = true;
        scene.config.triangle_type = PrimitiveType::Triangle;
        uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::BVH);

        EXPECT_GT(scene.spaces[space_id].as_size, 0u);
        EXPECT_GT(scene.spaces[space_id].box_content_size, 0u);

        const int hit_count = count_bvh_teapot_hits(scene, space_id, PrimitiveType::Triangle);
        std::cout << "BVH (SpatialMedian GPU) rays hitting teapot triangles: " << hit_count << " / 81" << std::endl;
        EXPECT_GT(hit_count, 15);
    }

    {
        Scene scene("TeapotObject");
        scene.load(teapot_view);

        scene.as_bvh.config.split_mode = BVHSplitMode::ObjectMedian;
        scene.as_bvh.config.use_gpu = true;
        scene.config.swapYZ = true;
        scene.config.sort_space = true;
        scene.config.triangle_type = PrimitiveType::Triangle;
        uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::BVH);

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


    scene.as_bvh.config.split_mode = BVHSplitMode::SAH;
    scene.as_bvh.config.use_gpu = true;
    scene.config.swapYZ = true;
    scene.config.sort_space = true;
    scene.config.triangle_type = PrimitiveType::Triangle;
    uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::BVH);
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
    scene16.as_bvh.config.split_mode = BVHSplitMode::SAH;
    scene16.as_bvh.config.use_gpu = true;
    scene16.config.swapYZ = true;
    scene16.config.sort_space = true;
    scene16.config.triangle_type = PrimitiveType::Triangle;
    scene16.as_bvh.config.use_bvh16 = true;
    const uint32_t space16 = scene16.build_node_as(scene16.root_node_id, ASType::BVH);
    scene16.set_active_space(space16);

    Scene scene32("TeapotBVH32");
    scene32.load(teapot_view);
    scene32.as_bvh.config.split_mode = BVHSplitMode::SAH;
    scene32.as_bvh.config.use_gpu = true;
    scene32.config.swapYZ = true;
    scene32.config.sort_space = true;
    scene32.config.triangle_type = PrimitiveType::Triangle;
    scene32.as_bvh.config.use_bvh16 = false;
    const uint32_t space32 = scene32.build_node_as(scene32.root_node_id, ASType::BVH);
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

            scene.as_bvh.config.split_mode = BVHSplitMode::SAH;
            scene.as_bvh.config.use_gpu = true;
            scene.config.swapYZ = true;
            scene.config.sort_space = true;
            scene.config.triangle_type = PrimitiveType::Triangle;
            scene.as_bvh.config.use_bvh16 = use16;
            const uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::BVH);
            if (space_id == UINT32_MAX)
                continue;
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

TEST(HeterogeneousAS, BuildAndRenderTeapotGPU)
{
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());

    Scene scene("HeterogeneousTeapot");
    scene.load(teapot_view);

    for (const auto& variant : scene.as_heterogeneous.variants)
    {
        if (variant.as_type == ASType::BVH)
        {
            EXPECT_TRUE(variant.bvh_config.use_bvh16);
        }
    }

    scene.as_heterogeneous.config.tlas_type = ASType::BVH;
    scene.as_heterogeneous.config.max_memory_bytes_per_triangle = 64.0f;
    scene.as_heterogeneous.config.num_probe_rays = 8192;

    const uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::Heterogeneous);
    ASSERT_TRUE(scene.spaces.is_allocated(space_id));
    EXPECT_EQ(scene.spaces[space_id].as_type, ASType::Heterogeneous);
    EXPECT_GE(scene.spaces[space_id].root_box, 0);

    const auto* records = scene.as_heterogeneous.get_mesh_records(space_id);
    ASSERT_NE(records, nullptr);
    ASSERT_EQ(records->size(), 1u);
    EXPECT_EQ((*records)[0].candidates.size(), scene.as_heterogeneous.variants.size());
    EXPECT_FALSE((*records)[0].selected_variant_name.empty());

    const uint32_t owned_blas_id = (*records)[0].space_id;
    EXPECT_TRUE(scene.spaces.is_allocated(owned_blas_id));

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
    raytracer.parameters.view_pos = glm::vec3(0.5f, -1.2f, 0.45f);
    raytracer.parameters.ax = 0.0f;
    raytracer.parameters.az = 0.0f;
    raytracer.parameters.background_color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    raytracer.update_parameters(scene, width, height);

    raytracer.draw_first_hit(scene, surface, width, height, 0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uchar4> host_pixels(width * height);
    ASSERT_EQ(cudaMemcpy2DFromArray(host_pixels.data(), width * sizeof(uchar4), cuArray, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToHost), cudaSuccess);

    uint32_t hit_pixels = 0;
    for (const auto& px : host_pixels)
    {
        if (px.x > 0 || px.y > 0 || px.z > 0)
            ++hit_pixels;
    }
    EXPECT_GT(hit_pixels, 0u);

    cudaDestroySurfaceObject(surface);
    cudaFreeArray(cuArray);

    scene.free_space(space_id);
    EXPECT_FALSE(scene.spaces.is_allocated(space_id));
    EXPECT_FALSE(scene.spaces.is_allocated(owned_blas_id));
    EXPECT_EQ(scene.as_heterogeneous.get_mesh_records(space_id), nullptr);
}

TEST(HeterogeneousAS, MultiMeshSelectionAndThresholdGPU)
{
    auto view = vkbase::assets::get("Scenes/ABeautifulGame.glb");
    ASSERT_FALSE(view.empty());

    Scene scene("HeterogeneousMultiMesh");
    scene.load(view);

    scene.as_heterogeneous.config.tlas_type = ASType::BVH;
    scene.as_heterogeneous.config.num_probe_rays = 4096;

    // Pass 1: Generous memory budget (128 B/tri)
    scene.as_heterogeneous.config.max_memory_bytes_per_triangle = 128.0f;
    const uint32_t space_generous = scene.build_node_as(scene.root_node_id, ASType::Heterogeneous);
    ASSERT_NE(space_generous, UINT32_MAX);
    ASSERT_TRUE(scene.spaces.is_allocated(space_generous));

    const auto* records_gen = scene.as_heterogeneous.get_mesh_records(space_generous);
    ASSERT_NE(records_gen, nullptr);
    ASSERT_GT(records_gen->size(), 1u);
    const size_t generous_records_count = records_gen->size();

    auto verify_selection_rule = [](const std::vector<MeshBuildRecord>& recs) -> double
    {
        double total_bytes = 0.0;
        double total_tris = 0.0;
        for (const auto& rec : recs)
        {
            EXPECT_FALSE(rec.candidates.empty());
            if (rec.candidates.empty())
                continue;
            bool any_feasible = false;
            float min_feasible_time = FLT_MAX;
            double min_bpt = DBL_MAX;
            const MeshCandidateResult* chosen = nullptr;

            for (const auto& cand : rec.candidates)
            {
                if (cand.within_memory_limit)
                {
                    any_feasible = true;
                    min_feasible_time = std::min(min_feasible_time, cand.probe_trace_ms);
                }
                min_bpt = std::min(min_bpt, cand.bytes_per_triangle);
                if (cand.variant_name == rec.selected_variant_name)
                    chosen = &cand;
            }
            EXPECT_NE(chosen, nullptr);
            if (chosen == nullptr)
                continue;
            if (any_feasible)
            {
                EXPECT_TRUE(chosen->within_memory_limit);
                EXPECT_LE(chosen->probe_trace_ms, min_feasible_time + 1e-5f);
            }
            else
            {
                EXPECT_LE(chosen->bytes_per_triangle, min_bpt + 1e-6);
            }

            total_bytes += static_cast<double>(chosen->as_nodes_bytes + chosen->box_content_bytes);
            total_tris += static_cast<double>(rec.num_triangles);
        }
        return (total_tris > 0.0) ? (total_bytes / total_tris) : 0.0;
    };

    const double avg_bpt_generous = verify_selection_rule(*records_gen);

    // Render Pass 1 (128x128)
    const uint32_t width = 128;
    const uint32_t height = 128;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t cuArray = nullptr;
    ASSERT_EQ(cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArraySurfaceLoadStore), cudaSuccess);

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    cudaSurfaceObject_t surface = 0;
    ASSERT_EQ(cudaCreateSurfaceObject(&surface, &resDesc), cudaSuccess);

    scene.set_active_space(space_generous);
    Raytracer raytracer;
    raytracer.parameters.view_pos = glm::vec3(0.5f, -1.2f, 0.5f);
    raytracer.parameters.ax = 0.0f;
    raytracer.parameters.az = 0.0f;
    raytracer.parameters.background_color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    raytracer.update_parameters(scene, width, height);

    raytracer.draw_first_hit(scene, surface, width, height, 0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uchar4> host_pixels(width * height);
    ASSERT_EQ(cudaMemcpy2DFromArray(host_pixels.data(), width * sizeof(uchar4), cuArray, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToHost), cudaSuccess);

    uint32_t hit_pixels = 0;
    for (const auto& px : host_pixels)
    {
        if (px.x > 0 || px.y > 0 || px.z > 0)
            ++hit_pixels;
    }
    EXPECT_GT(hit_pixels, 0u);

    // Pass 2: Tight memory budget (4 B/tri — strictly tighter than BVH16 so minimum-memory selection runs deterministically)
    scene.as_heterogeneous.config.max_memory_bytes_per_triangle = 4.0f;
    const uint32_t space_tight = scene.build_node_as(scene.root_node_id, ASType::Heterogeneous);
    ASSERT_NE(space_tight, UINT32_MAX);
    ASSERT_TRUE(scene.spaces.is_allocated(space_tight));

    const auto* records_tight = scene.as_heterogeneous.get_mesh_records(space_tight);
    ASSERT_NE(records_tight, nullptr);
    EXPECT_EQ(records_tight->size(), generous_records_count);

    const double avg_bpt_tight = verify_selection_rule(*records_tight);
    EXPECT_LE(avg_bpt_tight, avg_bpt_generous + 1e-6);

    scene.set_active_space(space_tight);
    raytracer.update_parameters(scene, width, height);
    raytracer.draw_first_hit(scene, surface, width, height, 0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uchar4> tight_pixels(width * height);
    ASSERT_EQ(cudaMemcpy2DFromArray(tight_pixels.data(), width * sizeof(uchar4), cuArray, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToHost), cudaSuccess);

    uint32_t tight_hits = 0;
    for (const auto& px : tight_pixels)
    {
        if (px.x > 0 || px.y > 0 || px.z > 0)
            ++tight_hits;
    }
    EXPECT_GT(tight_hits, 0u);

    cudaDestroySurfaceObject(surface);
    cudaFreeArray(cuArray);

    scene.free_node_as(scene.root_node_id);
    EXPECT_FALSE(scene.spaces.is_allocated(space_tight));
}

TEST(HeterogeneousAS, InstancingAndLiveNodeOverrideGPU)
{
    auto view = vkbase::assets::get("Scenes/ABeautifulGame.glb");
    ASSERT_FALSE(view.empty());

    Scene scene("HeterogeneousInstancing");
    scene.load(view);

    ASSERT_EQ(scene.m_mesh_node_ids.size(), scene.m_meshes.size());
    bool any_shared_mesh_node = false;
    for (uint32_t mesh_nid : scene.m_mesh_node_ids)
    {
        ASSERT_LT(mesh_nid, scene.m_nodes.size());
        if (scene.is_mesh_node_shared(mesh_nid))
        {
            any_shared_mesh_node = true;
        }
    }
    EXPECT_TRUE(any_shared_mesh_node);

    scene.as_heterogeneous.config.tlas_type = ASType::BVH;
    scene.as_heterogeneous.config.num_probe_rays = 2048;
    scene.as_heterogeneous.config.max_memory_bytes_per_triangle = 128.0f;

    const uint32_t tlas_space = scene.build_node_as(scene.root_node_id, ASType::Heterogeneous);
    ASSERT_NE(tlas_space, UINT32_MAX);
    ASSERT_TRUE(scene.spaces.is_allocated(tlas_space));
    scene.set_active_space(tlas_space);

    const auto* mesh_records = scene.as_heterogeneous.get_mesh_records(tlas_space);
    const auto* inst_records = scene.as_heterogeneous.get_instance_records(tlas_space);
    ASSERT_NE(mesh_records, nullptr);
    ASSERT_NE(inst_records, nullptr);
    EXPECT_EQ(mesh_records->size(), scene.m_meshes.size());
    EXPECT_GT(inst_records->size(), mesh_records->size());

    uint32_t shared_mesh_node_id = UINT32_MAX;
    uint32_t old_blas_space_id = UINT32_MAX;
    for (const auto& rec : *mesh_records)
    {
        ASSERT_LT(rec.mesh_node_id, scene.m_nodes.size());
        const Node& mesh_node = scene.m_nodes[rec.mesh_node_id];
        EXPECT_TRUE(mesh_node.as_ref.is_valid());
        EXPECT_EQ(mesh_node.as_ref.space_id, rec.space_id);
        EXPECT_EQ(mesh_node.as_ref.type, rec.selected_as_type);
        if (rec.instance_count > 1 && shared_mesh_node_id == UINT32_MAX)
        {
            shared_mesh_node_id = rec.mesh_node_id;
            old_blas_space_id = rec.space_id;
        }
    }
    ASSERT_NE(shared_mesh_node_id, UINT32_MAX);

    // Rebuild shared leaf mesh node in-place -> Heterogeneous TLAS automatically sees the rebuilt child BLAS
    const uint32_t rebuilt_space = scene.build_node_as(shared_mesh_node_id, ASType::UniformGrid);
    EXPECT_EQ(rebuilt_space, old_blas_space_id);
    EXPECT_EQ(scene.spaces[rebuilt_space].as_type, ASType::UniformGrid);
    const MeshBuildRecord* updated_mesh_rec = scene.as_heterogeneous.find_mesh_record_by_space(rebuilt_space);
    ASSERT_NE(updated_mesh_rec, nullptr);
    EXPECT_EQ(updated_mesh_rec->selected_as_type, ASType::UniformGrid);
    EXPECT_EQ(scene.m_nodes[shared_mesh_node_id].as_ref.type, ASType::UniformGrid);

    // Find an intermediate single-mesh instance node sharing that BLAS (or any instanced node) and override it
    uint32_t inst_node_id = UINT32_MAX;
    uint32_t default_blas_space = UINT32_MAX;
    for (const auto& irec : *inst_records)
    {
        if (irec.instance_node_id < scene.m_nodes.size() &&
            scene.m_nodes[irec.instance_node_id].meshId == 0 &&
            scene.m_nodes[irec.instance_node_id].count_subtree_meshes(scene.m_nodes) == 1)
        {
            inst_node_id = irec.instance_node_id;
            default_blas_space = irec.default_space_id;
            break;
        }
    }
    ASSERT_NE(inst_node_id, UINT32_MAX);

    const uint32_t override_space = scene.build_node_as(inst_node_id, ASType::KDTree);
    ASSERT_NE(override_space, UINT32_MAX);
    EXPECT_NE(override_space, default_blas_space);

    inst_records = scene.as_heterogeneous.get_instance_records(tlas_space);
    ASSERT_NE(inst_records, nullptr);
    bool found_override = false;
    for (const auto& irec : *inst_records)
    {
        if (irec.instance_node_id == inst_node_id)
        {
            EXPECT_EQ(irec.active_space_id, override_space);
            EXPECT_EQ(irec.active_as_type, ASType::KDTree);
            found_override = true;
            break;
        }
    }
    EXPECT_TRUE(found_override);

    // Render with active Heterogeneous TLAS (containing both in-place rebuilt BLAS and instance override)
    const uint32_t width = 128;
    const uint32_t height = 128;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t cuArray = nullptr;
    ASSERT_EQ(cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArraySurfaceLoadStore), cudaSuccess);

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    cudaSurfaceObject_t surface = 0;
    ASSERT_EQ(cudaCreateSurfaceObject(&surface, &resDesc), cudaSuccess);

    Raytracer raytracer;
    raytracer.parameters.view_pos = glm::vec3(0.5f, -1.2f, 0.5f);
    raytracer.parameters.ax = 0.0f;
    raytracer.parameters.az = 0.0f;
    raytracer.parameters.background_color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    raytracer.update_parameters(scene, width, height);

    raytracer.draw_first_hit(scene, surface, width, height, 0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uchar4> host_pixels(width * height);
    ASSERT_EQ(cudaMemcpy2DFromArray(host_pixels.data(), width * sizeof(uchar4), cuArray, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToHost), cudaSuccess);

    uint32_t hit_pixels = 0;
    for (const auto& px : host_pixels)
    {
        if (px.x > 0 || px.y > 0 || px.z > 0)
            ++hit_pixels;
    }
    EXPECT_GT(hit_pixels, 0u);

    cudaDestroySurfaceObject(surface);
    cudaFreeArray(cuArray);

    // Free instance override -> binds back to shared default BLAS space
    scene.free_node_as(inst_node_id);
    EXPECT_FALSE(scene.spaces.is_allocated(override_space));

    inst_records = scene.as_heterogeneous.get_instance_records(tlas_space);
    ASSERT_NE(inst_records, nullptr);
    bool found_reverted = false;
    for (const auto& irec : *inst_records)
    {
        if (irec.instance_node_id == inst_node_id)
        {
            EXPECT_EQ(irec.active_space_id, default_blas_space);
            found_reverted = true;
            break;
        }
    }
    EXPECT_TRUE(found_reverted);

    scene.free_node_as(scene.root_node_id);
    EXPECT_FALSE(scene.spaces.is_allocated(tlas_space));
    EXPECT_FALSE(scene.spaces.is_allocated(rebuilt_space));
}

TEST(HeterogeneousAS, RebuildTlasOnlyAndTimingsGPU)
{
    auto view = vkbase::assets::get("Scenes/ABeautifulGame.glb");
    ASSERT_FALSE(view.empty());

    Scene scene("HeterogeneousRebuildTlasOnly");
    scene.load(view);

    EXPECT_TRUE(scene.as_heterogeneous.config.rebuild_blas);
    scene.as_heterogeneous.config.tlas_type = ASType::BVH;
    scene.as_heterogeneous.config.num_probe_rays = 2048;
    scene.as_heterogeneous.config.max_memory_bytes_per_triangle = 128.0f;

    uint32_t tlas_space = scene.build_node_as(scene.root_node_id, ASType::Heterogeneous);
    ASSERT_NE(tlas_space, UINT32_MAX);
    ASSERT_TRUE(scene.spaces.is_allocated(tlas_space));
    scene.set_active_space(tlas_space);

    const int init_tree_id = scene.spaces[tlas_space].root_box;
    ASSERT_GE(init_tree_id, 0);
    EXPECT_EQ(scene.as_heterogeneous.trees[static_cast<size_t>(init_tree_id)].tlas_as_type, static_cast<uint32_t>(ASType::BVH));

    const double init_tlas_ms = scene.as_heterogeneous.get_tlas_build_time_ms(tlas_space);
    const double init_blas_ms = scene.as_heterogeneous.get_blas_build_time_ms(tlas_space);
    const double init_eval_ms = scene.as_heterogeneous.get_blas_eval_prep_time_ms(tlas_space);
    const double init_total_ms = scene.as_heterogeneous.get_build_time_ms(tlas_space);
    EXPECT_GT(init_tlas_ms, 0.0);
    EXPECT_GT(init_blas_ms, 0.0);
    EXPECT_GE(init_eval_ms, 0.0);
    EXPECT_NEAR(init_total_ms, init_tlas_ms + init_blas_ms + init_eval_ms, 1e-6);

    const auto* mesh_records = scene.as_heterogeneous.get_mesh_records(tlas_space);
    const auto* inst_records = scene.as_heterogeneous.get_instance_records(tlas_space);
    ASSERT_NE(mesh_records, nullptr);
    ASSERT_NE(inst_records, nullptr);
    ASSERT_FALSE(mesh_records->empty());

    std::vector<uint32_t> original_blas_spaces;
    for (const auto& rec : *mesh_records)
    {
        EXPECT_GT(rec.selected_build_ms, 0.0);
        EXPECT_FALSE(rec.candidates.empty());
        original_blas_spaces.push_back(rec.space_id);
    }

    uint32_t inst_node_id = UINT32_MAX;
    for (const auto& irec : *inst_records)
    {
        if (irec.instance_node_id < scene.m_nodes.size() &&
            scene.m_nodes[irec.instance_node_id].meshId == 0 &&
            scene.m_nodes[irec.instance_node_id].count_subtree_meshes(scene.m_nodes) == 1)
        {
            inst_node_id = irec.instance_node_id;
            break;
        }
    }
    ASSERT_NE(inst_node_id, UINT32_MAX);
    const uint32_t override_space = scene.build_node_as(inst_node_id, ASType::KDTree);
    ASSERT_NE(override_space, UINT32_MAX);

    const uint32_t width = 128;
    const uint32_t height = 128;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t cuArray = nullptr;
    ASSERT_EQ(cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArraySurfaceLoadStore), cudaSuccess);

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    cudaSurfaceObject_t surface = 0;
    ASSERT_EQ(cudaCreateSurfaceObject(&surface, &resDesc), cudaSuccess);

    Raytracer raytracer;
    raytracer.parameters.view_pos = glm::vec3(0.5f, -1.2f, 0.5f);
    raytracer.parameters.ax = 0.0f;
    raytracer.parameters.az = 0.0f;
    raytracer.parameters.background_color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

    std::vector<uchar4> host_pixels(width * height);
    const ASType target_tlas_types[] = {ASType::KDTree, ASType::UniformGrid, ASType::Octree, ASType::BVH};

    scene.as_heterogeneous.config.rebuild_blas = false;
    for (ASType target_tlas : target_tlas_types)
    {
        scene.as_heterogeneous.config.tlas_type = target_tlas;
        tlas_space = scene.build_node_as(scene.root_node_id, ASType::Heterogeneous);
        ASSERT_NE(tlas_space, UINT32_MAX);

        EXPECT_TRUE(scene.spaces.is_allocated(tlas_space));
        EXPECT_EQ(scene.get_active_space(), tlas_space);
        EXPECT_EQ(scene.m_nodes[scene.root_node_id].as_ref.space_id, tlas_space);
        EXPECT_EQ(scene.spaces[tlas_space].as_type, ASType::Heterogeneous);
        const int tree_id = scene.spaces[tlas_space].root_box;
        ASSERT_GE(tree_id, 0);
        EXPECT_EQ(scene.as_heterogeneous.trees[static_cast<size_t>(tree_id)].tlas_as_type, static_cast<uint32_t>(target_tlas));

        const double tlas_ms = scene.as_heterogeneous.get_tlas_build_time_ms(tlas_space);
        EXPECT_GT(tlas_ms, 0.0);
        EXPECT_DOUBLE_EQ(scene.as_heterogeneous.get_blas_build_time_ms(tlas_space), 0.0);
        EXPECT_DOUBLE_EQ(scene.as_heterogeneous.get_blas_eval_prep_time_ms(tlas_space), 0.0);
        EXPECT_NEAR(scene.as_heterogeneous.get_build_time_ms(tlas_space), tlas_ms, 1e-6);

        const auto* post_meshes = scene.as_heterogeneous.get_mesh_records(tlas_space);
        ASSERT_NE(post_meshes, nullptr);
        ASSERT_EQ(post_meshes->size(), original_blas_spaces.size());
        for (size_t i = 0; i < original_blas_spaces.size(); ++i)
        {
            EXPECT_EQ((*post_meshes)[i].space_id, original_blas_spaces[i]);
            EXPECT_TRUE(scene.spaces.is_allocated(original_blas_spaces[i]));
            EXPECT_TRUE((*post_meshes)[i].candidates.empty());
            EXPECT_DOUBLE_EQ((*post_meshes)[i].selected_build_ms, 0.0);
            EXPECT_EQ((*post_meshes)[i].selected_variant_name.rfind("Existing (", 0), 0u);
        }

        const auto* post_insts = scene.as_heterogeneous.get_instance_records(tlas_space);
        ASSERT_NE(post_insts, nullptr);
        bool override_preserved = false;
        for (const auto& irec : *post_insts)
        {
            if (irec.instance_node_id == inst_node_id)
            {
                EXPECT_EQ(irec.active_space_id, override_space);
                EXPECT_EQ(irec.active_as_type, ASType::KDTree);
                override_preserved = true;
                break;
            }
        }
        EXPECT_TRUE(override_preserved);

        raytracer.update_parameters(scene, width, height);
        raytracer.draw_first_hit(scene, surface, width, height, 0);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaMemcpy2DFromArray(host_pixels.data(), width * sizeof(uchar4), cuArray, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToHost), cudaSuccess);

        uint32_t hit_pixels = 0;
        for (const auto& px : host_pixels)
        {
            if (px.x > 0 || px.y > 0 || px.z > 0)
                ++hit_pixels;
        }
        EXPECT_GT(hit_pixels, 0u);
    }

    // Re-enabling rebuild_blas samples and rebuilds all BLAS spaces cleanly
    scene.as_heterogeneous.config.rebuild_blas = true;
    tlas_space = scene.build_node_as(scene.root_node_id, ASType::Heterogeneous);
    ASSERT_NE(tlas_space, UINT32_MAX);
    EXPECT_FALSE(scene.m_nodes[inst_node_id].as_ref.is_valid());
    const auto* rebuilt_insts = scene.as_heterogeneous.get_instance_records(tlas_space);
    ASSERT_NE(rebuilt_insts, nullptr);
    for (const auto& irec : *rebuilt_insts)
    {
        EXPECT_EQ(irec.active_space_id, irec.default_space_id);
        if (irec.instance_node_id == inst_node_id)
        {
            EXPECT_NE(irec.active_space_id, override_space);
        }
    }
    EXPECT_GT(scene.as_heterogeneous.get_blas_build_time_ms(tlas_space), 0.0);
    const auto* rebuilt_meshes = scene.as_heterogeneous.get_mesh_records(tlas_space);
    ASSERT_NE(rebuilt_meshes, nullptr);
    ASSERT_EQ(rebuilt_meshes->size(), original_blas_spaces.size());
    std::vector<uint32_t> rebuilt_blas_spaces;
    for (const auto& rec : *rebuilt_meshes)
    {
        EXPECT_GT(rec.selected_build_ms, 0.0);
        EXPECT_FALSE(rec.candidates.empty());
        rebuilt_blas_spaces.push_back(rec.space_id);
    }

    raytracer.update_parameters(scene, width, height);
    raytracer.draw_first_hit(scene, surface, width, height, 0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy2DFromArray(host_pixels.data(), width * sizeof(uchar4), cuArray, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToHost), cudaSuccess);
    uint32_t rebuilt_hits = 0;
    for (const auto& px : host_pixels)
    {
        if (px.x > 0 || px.y > 0 || px.z > 0)
            ++rebuilt_hits;
    }
    EXPECT_GT(rebuilt_hits, 0u);

    cudaDestroySurfaceObject(surface);
    cudaFreeArray(cuArray);

    scene.free_node_as(scene.root_node_id);
    EXPECT_FALSE(scene.spaces.is_allocated(tlas_space));
    for (uint32_t blas_id : rebuilt_blas_spaces)
    {
        EXPECT_FALSE(scene.spaces.is_allocated(blas_id));
    }
}

TEST(HeterogeneousAS, ReusePrebuiltStandaloneAndBuildMissingBlasGPU)
{
    auto view = vkbase::assets::get("Scenes/ABeautifulGame.glb");
    ASSERT_FALSE(view.empty());

    Scene scene("HeterogeneousReusePartial");
    scene.load(view);
    ASSERT_GE(scene.m_mesh_node_ids.size(), 2u);

    uint32_t inst_override_node = UINT32_MAX;
    uint32_t mesh_node_0 = UINT32_MAX;
    for (uint32_t nid = 0; nid < static_cast<uint32_t>(scene.m_nodes.size()); ++nid)
    {
        const Node& n = scene.m_nodes[nid];
        if (n.meshId == 0 && n.children.size() == 1 && n.count_subtree_meshes(scene.m_nodes) == 1)
        {
            const uint32_t child_id = n.children[0];
            if (child_id < scene.m_nodes.size() && scene.m_nodes[child_id].meshId > 0)
            {
                inst_override_node = nid;
                mesh_node_0 = child_id;
                break;
            }
        }
    }
    ASSERT_NE(inst_override_node, UINT32_MAX);
    ASSERT_NE(mesh_node_0, UINT32_MAX);

    uint32_t mesh_node_1 = UINT32_MAX;
    for (uint32_t candidate_mesh_nid : scene.m_mesh_node_ids)
    {
        if (candidate_mesh_nid != mesh_node_0)
        {
            mesh_node_1 = candidate_mesh_nid;
            break;
        }
    }
    ASSERT_NE(mesh_node_1, UINT32_MAX);

    const uint32_t prebuilt_bvh_space = scene.build_node_as(mesh_node_0, ASType::BVH);
    const uint32_t prebuilt_optix_space = scene.build_node_as(mesh_node_1, ASType::Optix);
    ASSERT_NE(prebuilt_bvh_space, UINT32_MAX);
    ASSERT_NE(prebuilt_optix_space, UINT32_MAX);

    const uint32_t prebuilt_override_space = scene.build_node_as(inst_override_node, ASType::KDTree);
    ASSERT_NE(prebuilt_override_space, UINT32_MAX);

    scene.as_heterogeneous.config.rebuild_blas = false;
    scene.as_heterogeneous.config.tlas_type = ASType::BVH;
    scene.as_heterogeneous.config.num_probe_rays = 2048;
    scene.as_heterogeneous.config.max_memory_bytes_per_triangle = 128.0f;

    uint32_t tlas_space = scene.build_node_as(scene.root_node_id, ASType::Heterogeneous);
    ASSERT_NE(tlas_space, UINT32_MAX);
    EXPECT_GT(scene.as_heterogeneous.get_blas_build_time_ms(tlas_space), 0.0);
    EXPECT_TRUE(!scene.spaces.is_allocated(prebuilt_optix_space) ||
                scene.spaces[prebuilt_optix_space].as_type != ASType::Optix);
    EXPECT_NE(scene.m_nodes[mesh_node_1].as_ref.type, ASType::Optix);

    const auto* mesh_recs = scene.as_heterogeneous.get_mesh_records(tlas_space);
    ASSERT_NE(mesh_recs, nullptr);
    bool checked_0 = false;
    bool checked_1 = false;
    uint32_t rebuilt_space_1 = UINT32_MAX;
    for (const auto& rec : *mesh_recs)
    {
        if (rec.mesh_node_id == mesh_node_0)
        {
            EXPECT_EQ(rec.space_id, prebuilt_bvh_space);
            EXPECT_EQ(rec.selected_as_type, ASType::BVH);
            EXPECT_EQ(rec.selected_variant_name, "Existing (BVH16)");
            EXPECT_TRUE(rec.candidates.empty());
            EXPECT_DOUBLE_EQ(rec.selected_build_ms, 0.0);
            checked_0 = true;
        }
        else
        {
            EXPECT_FALSE(rec.candidates.empty());
            EXPECT_GT(rec.selected_build_ms, 0.0);
            if (rec.mesh_node_id == mesh_node_1)
            {
                rebuilt_space_1 = rec.space_id;
                checked_1 = true;
            }
        }
    }
    EXPECT_TRUE(checked_0);
    EXPECT_TRUE(checked_1);
    ASSERT_NE(rebuilt_space_1, UINT32_MAX);

    const auto* inst_recs = scene.as_heterogeneous.get_instance_records(tlas_space);
    ASSERT_NE(inst_recs, nullptr);
    bool found_override = false;
    for (const auto& irec : *inst_recs)
    {
        if (irec.instance_node_id == inst_override_node)
        {
            EXPECT_EQ(irec.default_space_id, prebuilt_bvh_space);
            EXPECT_EQ(irec.active_space_id, prebuilt_override_space);
            EXPECT_EQ(irec.active_as_type, ASType::KDTree);
            found_override = true;
        }
    }
    EXPECT_TRUE(found_override);

    // Free only mesh_node_0's BLAS and rebuild with rebuild_blas = false -> only mesh_node_0 is re-sampled
    scene.free_node_as(mesh_node_0);
    tlas_space = scene.build_node_as(scene.root_node_id, ASType::Heterogeneous);
    ASSERT_NE(tlas_space, UINT32_MAX);
    mesh_recs = scene.as_heterogeneous.get_mesh_records(tlas_space);
    ASSERT_NE(mesh_recs, nullptr);
    for (const auto& rec : *mesh_recs)
    {
        if (rec.mesh_node_id == mesh_node_0)
        {
            EXPECT_FALSE(rec.candidates.empty());
            EXPECT_GT(rec.selected_build_ms, 0.0);
        }
        else
        {
            EXPECT_TRUE(rec.candidates.empty());
            EXPECT_DOUBLE_EQ(rec.selected_build_ms, 0.0);
            if (rec.mesh_node_id == mesh_node_1)
            {
                EXPECT_EQ(rec.space_id, rebuilt_space_1);
            }
        }
    }

    scene.free_node_as(inst_override_node);
    EXPECT_FALSE(scene.spaces.is_allocated(prebuilt_override_space));
    scene.free_node_as(scene.root_node_id);
    EXPECT_FALSE(scene.spaces.is_allocated(tlas_space));
    EXPECT_FALSE(scene.spaces.is_allocated(rebuilt_space_1));
}

TEST(BVHAS, RefractionBoxEdgePlanesMatchOctreeGPU)
{
    auto view = vkbase::assets::get("Scenes/Refraction.glb");
    ASSERT_FALSE(view.empty());

    Scene scene_oct("RefractionOct");
    scene_oct.load(view);
    scene_oct.as_octree.config.max_depth = 10;
    const uint32_t sp_oct = scene_oct.build_node_as(scene_oct.root_node_id, ASType::Octree);
    scene_oct.set_active_space(sp_oct);

    Scene scene_bvh32("RefractionBVH32");
    scene_bvh32.load(view);
    scene_bvh32.as_bvh.config.use_bvh16 = false;
    scene_bvh32.as_bvh.config.split_mode = BVHSplitMode::SAH;
    const uint32_t sp_32 = scene_bvh32.build_node_as(scene_bvh32.root_node_id, ASType::BVH);
    scene_bvh32.set_active_space(sp_32);

    Scene scene_bvh16("RefractionBVH16");
    scene_bvh16.load(view);
    scene_bvh16.as_bvh.config.use_bvh16 = true;
    scene_bvh16.as_bvh.config.split_mode = BVHSplitMode::SAH;
    const uint32_t sp_16 = scene_bvh16.build_node_as(scene_bvh16.root_node_id, ASType::BVH);
    scene_bvh16.set_active_space(sp_16);

    Scene scene_het("RefractionHetBVH16");
    scene_het.load(view);
    scene_het.as_heterogeneous.config.tlas_type = ASType::BVH;
    scene_het.as_heterogeneous.variants.clear();
    ASConfigVariant bvh16_only{};
    bvh16_only.name = "BVH16 SAH";
    bvh16_only.as_type = ASType::BVH;
    bvh16_only.bvh_config.use_bvh16 = true;
    bvh16_only.bvh_config.split_mode = BVHSplitMode::SAH;
    bvh16_only.bvh_config.use_gpu = true;
    scene_het.as_heterogeneous.variants.push_back(bvh16_only);
    const uint32_t sp_het = scene_het.build_node_as(scene_het.root_node_id, ASType::Heterogeneous);
    ASSERT_NE(sp_het, UINT32_MAX);
    scene_het.set_active_space(sp_het);

    const uint32_t width = 256;
    const uint32_t height = 256;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t cuArray = nullptr;
    ASSERT_EQ(cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArraySurfaceLoadStore), cudaSuccess);
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    cudaSurfaceObject_t surface = 0;
    ASSERT_EQ(cudaCreateSurfaceObject(&surface, &resDesc), cudaSuccess);

    auto render = [&](Scene& s, glm::vec3 vpos, float ax, float az) {
        Raytracer rt;
        rt.parameters.view_pos = vpos;
        rt.parameters.ax = ax;
        rt.parameters.az = az;
        rt.parameters.fov = 45.0f;
        rt.parameters.enableRefraction = true;
        rt.parameters.background_color = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
        rt.update_parameters(s, width, height);
        rt.draw_first_hit(s, surface, width, height, 0);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<uchar4> px(width * height);
        EXPECT_EQ(cudaMemcpy2DFromArray(px.data(), width * sizeof(uchar4), cuArray, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToHost), cudaSuccess);
        return px;
    };

    auto count_diff = [&](const std::vector<uchar4>& a, const std::vector<uchar4>& b) {
        uint32_t diffs = 0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            if (std::abs(int(a[i].x) - int(b[i].x)) > 2 ||
                std::abs(int(a[i].y) - int(b[i].y)) > 2 ||
                std::abs(int(a[i].z) - int(b[i].z)) > 2)
            {
                ++diffs;
            }
        }
        return diffs;
    };

    struct CameraView
    {
        glm::vec3 pos;
        float ax;
        float az;
    };
    const CameraView views[] = {
        {glm::vec3(0.25f, 0.25f, 0.35f), -0.15f, -0.45f},
        {glm::vec3(0.45f, 0.20f, 0.55f), 0.22f, -1.15f},
    };

    for (const auto& cam : views)
    {
        const auto px_oct = render(scene_oct, cam.pos, cam.ax, cam.az);
        const auto px_32 = render(scene_bvh32, cam.pos, cam.ax, cam.az);
        const auto px_16 = render(scene_bvh16, cam.pos, cam.ax, cam.az);
        const auto px_het = render(scene_het, cam.pos, cam.ax, cam.az);

        EXPECT_EQ(count_diff(px_oct, px_32), 0u);
        EXPECT_EQ(count_diff(px_oct, px_16), 0u);
        EXPECT_EQ(count_diff(px_32, px_16), 0u);
        EXPECT_EQ(count_diff(px_oct, px_het), 0u);
    }

    // Parallel/near-parallel ray on box boundary edge checks (BVH16 and BVH32 leaf slab robustness)
    {
        const soa::soa_vector<BVHTreeInfo> h_trees16 = scene_bvh16.as_bvh.trees;
        const soa::soa_vector<BVHNode16> h_as16 = scene_bvh16.as_bvh.as16;
        const soa::soa_vector<BVHNode> h_as32 = scene_bvh32.as_bvh.as;
        const auto& t16 = h_trees16[static_cast<size_t>(scene_bvh16.spaces[sp_16].root_box)];
        const auto& t32 = scene_bvh32.as_bvh.trees[static_cast<size_t>(scene_bvh32.spaces[sp_32].root_box)];

        const glm::vec3 parallel_dirs[] = {
            glm::normalize(glm::vec3(1.0f, 0.0f, 0.2f)),
            glm::normalize(glm::vec3(1.0f, 1e-8f, 0.2f)),
            glm::normalize(glm::vec3(1.0f, -1e-8f, 0.2f)),
        };
        for (const glm::vec3& pdir : parallel_dirs)
        {
            const glm::vec3 porig(0.25f, 1.0f, 0.35f);
            BVHStackEntry stack16[CurrentBVHNode::SHORT_STACK_SIZE];
            CurrentBVHNode state16{};
            int it16 = 0;
            uint32_t hits16 = 0;
            while (!state16.exited && it16 < 500)
            {
                const uint32_t cid = traverse_bvh16_impl<1>(&h_as16[0], t16.as_start, t16.bounds_min, t16.bounds_max, state16, stack16, porig, pdir, it16);
                if (cid > 0) ++hits16;
            }
            EXPECT_GT(hits16, 0u);

            BVHStackEntry stack32[CurrentBVHNode::SHORT_STACK_SIZE];
            CurrentBVHNode state32{};
            int it32 = 0;
            uint32_t hits32 = 0;
            while (!state32.exited && it32 < 500)
            {
                const uint32_t cid = traverse_bvh_impl<1>(&h_as32[0], t32.as_start, state32, stack32, porig, pdir, it32);
                if (cid > 0) ++hits32;
            }
            EXPECT_GT(hits32, 0u);
        }
    }

    cudaDestroySurfaceObject(surface);
    cudaFreeArray(cuArray);
}

TEST(Scene, SkipTexturesWhenDisabled)
{
    auto game_view = vkbase::assets::get("Scenes/ABeautifulGame.glb");
    ASSERT_FALSE(game_view.empty());

    Scene scene_with_textures("WithTextures");
    scene_with_textures.load(game_view, true);
    ASSERT_TRUE(scene_with_textures.is_loaded());
    ASSERT_FALSE(scene_with_textures.m_textures.empty());

    Scene scene_without_textures("WithoutTextures");
    scene_without_textures.load(game_view, false);
    ASSERT_TRUE(scene_without_textures.is_loaded());
    EXPECT_TRUE(scene_without_textures.m_textures.empty());
    EXPECT_EQ(scene_without_textures.m_meshes.size(), scene_with_textures.m_meshes.size());
    EXPECT_EQ(scene_without_textures.m_nodes.size(), scene_with_textures.m_nodes.size());
    EXPECT_EQ(scene_without_textures.m_materials.size(), scene_with_textures.m_materials.size());
    for (size_t i = 0; i < scene_without_textures.m_materials.size(); ++i)
    {
        auto&& mat = scene_without_textures.m_materials[i];
        EXPECT_EQ(mat.baseColorTextureIndex, 0u);
        EXPECT_EQ(mat.normalTextureIndex, 0u);
        EXPECT_EQ(mat.transmissionTextureIndex, 0u);
        EXPECT_EQ(mat.thicknessTextureIndex, 0u);
        EXPECT_EQ(mat.baseColorTexture, 0ull);
        EXPECT_EQ(mat.normalTexture, 0ull);
        EXPECT_EQ(mat.transmissionTexture, 0ull);
        EXPECT_EQ(mat.thicknessTexture, 0ull);
    }
}

TEST(Scene, DeviceSourceBuffersAndMeshViews)
{
    auto game_view = vkbase::assets::get("Scenes/ABeautifulGame.glb");
    ASSERT_FALSE(game_view.empty());

    Scene scene("DeviceSourceTest");
    scene.load(game_view, false);
    ASSERT_TRUE(scene.is_loaded());
    ASSERT_FALSE(scene.m_meshes.empty());
    ASSERT_GT(scene.d_source_vertices.size(), 0u);
    ASSERT_GT(scene.d_source_triangles.size(), 0u);

    size_t sum_verts = 0;
    size_t sum_tris = 0;
    for (const auto& mesh : scene.m_meshes)
    {
        sum_verts += mesh.d_vertices.size();
        sum_tris += mesh.d_triangles.size();
        EXPECT_LE(mesh.local_bounds_min.x, mesh.local_bounds_max.x);
        EXPECT_LE(mesh.local_bounds_min.y, mesh.local_bounds_max.y);
        EXPECT_LE(mesh.local_bounds_min.z, mesh.local_bounds_max.z);
        EXPECT_TRUE(std::isfinite(mesh.local_bounds_min.x) && std::isfinite(mesh.local_bounds_max.x));
        EXPECT_TRUE(std::isfinite(mesh.local_bounds_min.y) && std::isfinite(mesh.local_bounds_max.y));
        EXPECT_TRUE(std::isfinite(mesh.local_bounds_min.z) && std::isfinite(mesh.local_bounds_max.z));
    }
    EXPECT_EQ(sum_verts, scene.d_source_vertices.size());
    EXPECT_EQ(sum_tris, scene.d_source_triangles.size());

    // Moving Scene must preserve device source buffers and mesh views
    const size_t expected_verts = scene.d_source_vertices.size();
    const size_t expected_tris = scene.d_source_triangles.size();
    Scene moved = std::move(scene);
    EXPECT_EQ(moved.d_source_vertices.size(), expected_verts);
    EXPECT_EQ(moved.d_source_triangles.size(), expected_tris);

    size_t moved_verts = 0;
    size_t moved_tris = 0;
    for (const auto& mesh : moved.m_meshes)
    {
        moved_verts += mesh.d_vertices.size();
        moved_tris += mesh.d_triangles.size();
    }
    EXPECT_EQ(moved_verts, expected_verts);
    EXPECT_EQ(moved_tris, expected_tris);

    const uint32_t sp = moved.build_node_as(moved.root_node_id, ASType::BVH);
    ASSERT_NE(sp, UINT32_MAX);
    EXPECT_GT(moved.spaces[sp].primitives_size, 0u);
    EXPECT_GT(moved.spaces[sp].vertices_size, 0u);
}

TEST(Scene, GpuTransformDirectPackMatchesSortedPack)
{
    auto view = vkbase::assets::get("Scenes/ABeautifulGame.glb");
    ASSERT_FALSE(view.empty());

    const uint32_t width = 128;
    const uint32_t height = 128;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t cuArray = nullptr;
    ASSERT_EQ(cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArraySurfaceLoadStore), cudaSuccess);
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    cudaSurfaceObject_t surface = 0;
    ASSERT_EQ(cudaCreateSurfaceObject(&surface, &resDesc), cudaSuccess);

    auto render = [&](Scene& s) {
        Raytracer rt;
        rt.parameters.view_pos = glm::vec3(0.5f, -0.8f, 0.5f);
        rt.parameters.ax = 0.0f;
        rt.parameters.az = -0.3f;
        rt.parameters.fov = 45.0f;
        rt.parameters.background_color = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
        rt.update_parameters(s, width, height);
        rt.draw_first_hit(s, surface, width, height, 0);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<uchar4> px(width * height);
        EXPECT_EQ(cudaMemcpy2DFromArray(px.data(), width * sizeof(uchar4), cuArray, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToHost), cudaSuccess);
        return px;
    };

    const PrimitiveType prim_types[] = {
        PrimitiveType::Triangle,
        PrimitiveType::FullTriangle,
        PrimitiveType::NagataPatch,
    };

    for (PrimitiveType ptype : prim_types)
    {
        Scene scene_direct("DirectPack");
        scene_direct.load(view, false);
        scene_direct.config.swapYZ = true;
        scene_direct.config.sort_space = false;
        scene_direct.config.triangle_type = ptype;
        const uint32_t sp_direct = scene_direct.build_node_as(scene_direct.root_node_id, ASType::BVH);
        ASSERT_NE(sp_direct, UINT32_MAX);
        scene_direct.set_active_space(sp_direct);

        Scene scene_sorted("SortedPack");
        scene_sorted.load(view, false);
        scene_sorted.config.swapYZ = true;
        scene_sorted.config.sort_space = true;
        scene_sorted.config.triangle_type = ptype;
        const uint32_t sp_sorted = scene_sorted.build_node_as(scene_sorted.root_node_id, ASType::BVH);
        ASSERT_NE(sp_sorted, UINT32_MAX);
        scene_sorted.set_active_space(sp_sorted);

        EXPECT_EQ(scene_direct.spaces[sp_direct].primitives_size, scene_sorted.spaces[sp_sorted].primitives_size);
        EXPECT_EQ(scene_direct.spaces[sp_direct].vertices_size, scene_sorted.spaces[sp_sorted].vertices_size);
        EXPECT_EQ(scene_direct.spaces[sp_direct].has_nagata, ptype == PrimitiveType::NagataPatch);
        EXPECT_NEAR(scene_direct.spaces[sp_direct].normalization_translate.x, scene_sorted.spaces[sp_sorted].normalization_translate.x, 1e-5f);
        EXPECT_NEAR(scene_direct.spaces[sp_direct].normalization_translate.y, scene_sorted.spaces[sp_sorted].normalization_translate.y, 1e-5f);
        EXPECT_NEAR(scene_direct.spaces[sp_direct].normalization_translate.z, scene_sorted.spaces[sp_sorted].normalization_translate.z, 1e-5f);
        if (ptype == PrimitiveType::FullTriangle)
        {
            EXPECT_EQ(scene_direct.spaces[sp_direct].vertices_size, 0u);
        }

        const auto px_direct = render(scene_direct);
        const auto px_sorted = render(scene_sorted);
        uint32_t non_bg = 0;
        uint32_t diffs = 0;
        for (size_t i = 0; i < px_direct.size(); ++i)
        {
            if (px_direct[i].x != 255 || px_direct[i].y != 0 || px_direct[i].z != 255)
                ++non_bg;
            if (std::abs(int(px_direct[i].x) - int(px_sorted[i].x)) > 2 ||
                std::abs(int(px_direct[i].y) - int(px_sorted[i].y)) > 2 ||
                std::abs(int(px_direct[i].z) - int(px_sorted[i].z)) > 2)
            {
                ++diffs;
            }
        }
        EXPECT_GT(non_bg, 0u);
        EXPECT_EQ(diffs, 0u);
    }

    // Verify swapYZ = false produces distinct normalization bounds and vertices stay in [0, 1]
    {
        Scene scene_no_swap("NoSwapYZ");
        scene_no_swap.load(view, false);
        scene_no_swap.config.swapYZ = false;
        scene_no_swap.config.sort_space = false;
        scene_no_swap.config.triangle_type = PrimitiveType::Triangle;
        const uint32_t sp = scene_no_swap.build_node_as(scene_no_swap.root_node_id, ASType::BVH);
        ASSERT_NE(sp, UINT32_MAX);
        const auto& space = scene_no_swap.spaces[sp];
        ASSERT_GT(space.vertices_size, 0u);

        soa::soa_vector<Vertex> host_verts = scene_no_swap.geometry_context.vertices;
        glm::vec3 vmin(std::numeric_limits<float>::infinity());
        glm::vec3 vmax(-std::numeric_limits<float>::infinity());
        for (uint32_t i = 0; i < space.vertices_size; ++i)
        {
            glm::vec3 p = host_verts[space.vertices_start + i].position;
            vmin = glm::min(vmin, p);
            vmax = glm::max(vmax, p);
        }
        EXPECT_GE(vmin.x, -1e-4f);
        EXPECT_GE(vmin.y, -1e-4f);
        EXPECT_GE(vmin.z, -1e-4f);
        EXPECT_LE(vmax.x, 1.0f + 1e-4f);
        EXPECT_LE(vmax.y, 1.0f + 1e-4f);
        EXPECT_LE(vmax.z, 1.0f + 1e-4f);
    }

    cudaDestroySurfaceObject(surface);
    cudaFreeArray(cuArray);
}

TEST(HeterogeneousAS, AllTlasTypesRenderIdenticalAndStopOnHitGPU)
{
    auto view = vkbase::assets::get("Scenes/ABeautifulGame.glb");
    ASSERT_FALSE(view.empty());

    Scene scene("AllTlasTypesTest");
    scene.load(view, false);
    scene.as_heterogeneous.config.rebuild_blas = true;
    scene.as_heterogeneous.config.tlas_type = ASType::BVH;
    scene.as_heterogeneous.variants.clear();
    ASConfigVariant bvh16{};
    bvh16.name = "BVH16";
    bvh16.as_type = ASType::BVH;
    bvh16.bvh_config.use_bvh16 = true;
    bvh16.bvh_config.split_mode = BVHSplitMode::SAH;
    bvh16.bvh_config.use_gpu = true;
    scene.as_heterogeneous.variants.push_back(bvh16);

    uint32_t sp = scene.build_node_as(scene.root_node_id, ASType::Heterogeneous);
    ASSERT_NE(sp, UINT32_MAX);
    scene.as_heterogeneous.config.rebuild_blas = false;

    const ASType types[] = {ASType::BVH, ASType::KDTree, ASType::UniformGrid, ASType::Octree};
    const uint32_t width = 128, height = 128;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t arr = nullptr;
    ASSERT_EQ(cudaMallocArray(&arr, &ch, width, height, cudaArraySurfaceLoadStore), cudaSuccess);
    cudaResourceDesc rd{};
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = arr;
    cudaSurfaceObject_t surf = 0;
    ASSERT_EQ(cudaCreateSurfaceObject(&surf, &rd), cudaSuccess);

    std::vector<uchar4> ref_pixels;
    for (int ti = 0; ti < 4; ++ti)
    {
        scene.as_heterogeneous.config.tlas_type = types[ti];
        sp = scene.build_node_as(scene.root_node_id, ASType::Heterogeneous);
        ASSERT_NE(sp, UINT32_MAX);
        scene.set_active_space(sp);

        const auto& s = scene.spaces[sp];
        const auto htree = scene.as_heterogeneous.trees[static_cast<size_t>(s.root_box)];
        const uint32_t num_subspaces = s.primitives_size / SubspacePrimitive::occupied_slots;
        ASSERT_GT(num_subspaces, 0u);

        soa::soa_vector<PrimitiveChunk16> hp_chk = scene.geometry_context.primitives;

        if (types[ti] == ASType::UniformGrid)
        {
            const auto g = scene.as_uniform_grid.grids[static_cast<size_t>(htree.tlas_root_box)];
            uint32_t bad_nonoverlap = 0;
            for (uint32_t iz = 0; iz < g.dims.z; ++iz)
            for (uint32_t iy = 0; iy < g.dims.y; ++iy)
            for (uint32_t ix = 0; ix < g.dims.x; ++ix)
            {
                const uint32_t cid = scene.as_uniform_grid.as[g.as_start + uniform_grid_cell_index({ix, iy, iz}, g.dims)];
                if (cid == 0) continue;
                const glm::vec3 bmin = glm::vec3(ix, iy, iz) * g.cell_size;
                const glm::vec3 bmax = bmin + glm::vec3(g.cell_size);
                for (uint32_t c = cid;; ++c)
                {
                    const auto pr = scene.as_uniform_grid.box_content[c];
                    const auto sub = load_primitive<SubspacePrimitive>(hp_chk.view(), pr.id());
                    const glm::vec3 omin = glm::max(bmin, sub.bounds_min);
                    const glm::vec3 omax = glm::min(bmax, sub.bounds_max);
                    if (omin.x > omax.x + 1e-4f || omin.y > omax.y + 1e-4f || omin.z > omax.z + 1e-4f)
                        ++bad_nonoverlap;
                    if (pr.is_last()) break;
                }
            }
            EXPECT_EQ(bad_nonoverlap, 0u);
        }
        else if (types[ti] == ASType::KDTree)
        {
            const auto t = scene.as_kdtree.trees[static_cast<size_t>(htree.tlas_root_box)];
            uint32_t nonempty = 0, bad_nonoverlap = 0;
            std::function<void(uint32_t, glm::vec3, glm::vec3)> walk_kd = [&](uint32_t nid, glm::vec3 bmin, glm::vec3 bmax) {
                const auto n = scene.as_kdtree.as[nid];
                if (n.is_leaf())
                {
                    if (n.content() > 0)
                    {
                        ++nonempty;
                        for (uint32_t c = n.content();; ++c)
                        {
                            const auto pr = scene.as_kdtree.box_content[c];
                            const auto sub = load_primitive<SubspacePrimitive>(hp_chk.view(), pr.id());
                            const glm::vec3 omin = glm::max(bmin, sub.bounds_min);
                            const glm::vec3 omax = glm::min(bmax, sub.bounds_max);
                            if (omin.x > omax.x + 1e-4f || omin.y > omax.y + 1e-4f || omin.z > omax.z + 1e-4f)
                                ++bad_nonoverlap;
                            if (pr.is_last()) break;
                        }
                    }
                    return;
                }
                glm::vec3 lmax = bmax; lmax[n.axis()] = n.split_pos;
                glm::vec3 rmin = bmin; rmin[n.axis()] = n.split_pos;
                walk_kd(n.left_child(), bmin, lmax);
                walk_kd(n.left_child() + 1u, rmin, bmax);
            };
            walk_kd(t.as_start, t.bounds_min, t.bounds_max);
            EXPECT_EQ(bad_nonoverlap, 0u);
            EXPECT_LT(nonempty, num_subspaces * 4u);
        }
        else if (types[ti] == ASType::Octree)
        {
            uint32_t bad_nonoverlap = 0;
            std::function<void(int, BoxStack)> walk_oct = [&](int node_id, BoxStack stk) {
                const int val = scene.as_octree.as[node_id];
                if (val > 0)
                {
                    for (int b = 0; b < 8; ++b)
                    {
                        BoxStack ch = stk;
                        ch.descend(b);
                        walk_oct(val + b, ch);
                    }
                }
                else if (val < 0)
                {
                    const glm::vec3 c = stk.get_center();
                    const float h = stk.half_box_size();
                    const glm::vec3 bmin = c - glm::vec3(h);
                    const glm::vec3 bmax = c + glm::vec3(h);
                    for (uint32_t cid = static_cast<uint32_t>(-val);; ++cid)
                    {
                        const auto pr = scene.as_octree.box_content[cid];
                        const auto sub = load_primitive<SubspacePrimitive>(hp_chk.view(), pr.id());
                        const glm::vec3 omin = glm::max(bmin, sub.bounds_min);
                        const glm::vec3 omax = glm::min(bmax, sub.bounds_max);
                        if (omin.x > omax.x + 1e-4f || omin.y > omax.y + 1e-4f || omin.z > omax.z + 1e-4f)
                            ++bad_nonoverlap;
                        if (pr.is_last()) break;
                    }
                }
            };
            walk_oct(htree.tlas_root_box, BoxStack{});
            EXPECT_EQ(bad_nonoverlap, 0u);
        }

        Raytracer rt;
        rt.parameters.view_pos = glm::vec3(0.5f, -0.4f, 0.25f);
        rt.parameters.ax = -0.25f;
        rt.parameters.az = 0.0f;
        rt.parameters.fov = 45.0f;
        rt.parameters.background_color = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
        rt.update_parameters(scene, width, height);
        rt.draw_first_hit(scene, surf, width, height, 0);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        std::vector<uchar4> px(width * height);
        ASSERT_EQ(cudaMemcpy2DFromArray(px.data(), width * sizeof(uchar4), arr, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToHost), cudaSuccess);

        if (ti == 0)
        {
            ref_pixels = std::move(px);
            uint32_t non_bg = 0;
            for (const auto& c : ref_pixels)
                if (c.x != 255 || c.y != 0 || c.z != 255) ++non_bg;
            EXPECT_GT(non_bg, 0u);
        }
        else
        {
            uint32_t diffs = 0;
            for (size_t i = 0; i < px.size(); ++i)
            {
                if (std::abs(int(px[i].x) - int(ref_pixels[i].x)) > 2 ||
                    std::abs(int(px[i].y) - int(ref_pixels[i].y)) > 2 ||
                    std::abs(int(px[i].z) - int(ref_pixels[i].z)) > 2)
                {
                    ++diffs;
                }
            }
            EXPECT_EQ(diffs, 0u) << "TLAS type index " << ti << " differed from BVH TLAS";
        }
    }

    cudaDestroySurfaceObject(surf);
    cudaFreeArray(arr);
}

TEST(OctreeAS, BenchmarkBuildScenes)
{
    // Warm up CUDA context/allocations using utah_teapot.glb so the first measured scene is not skewed.
    {
        auto warmup_view = vkbase::assets::get("Scenes/utah_teapot.glb");
        ASSERT_FALSE(warmup_view.empty());
        Scene warmup("Warmup");
        warmup.load(warmup_view);
        warmup.as_octree.config.stretch = false;
        warmup.config.swapYZ = true;
        warmup.as_octree.config.min_depth = 1;
        warmup.as_octree.config.max_depth = 10;
        warmup.as_octree.config.triangle_to_box_stop_ratio = 1.1f;
        warmup.as_octree.config.use_gpu = true;
        uint32_t warmup_sp = warmup.build_node_as(warmup.root_node_id, ASType::Octree);
        ASSERT_NE(warmup_sp, UINT32_MAX);
    }

    struct BenchCase
    {
        const char* label;
        const char* asset;
        int max_depth;
    };

    const BenchCase cases[] = {
        {"Teapot", "Scenes/utah_teapot.glb", 10},
        {"Hairball", "Scenes/hairball.glb", 8},
        {"Sponza", "Scenes/Sponza.glb", 10},
    };

    for (const auto& c : cases)
    {
        auto view = vkbase::assets::get(c.asset);
        ASSERT_FALSE(view.empty()) << "Missing asset: " << c.asset;

        Scene scene(c.label);
        scene.load(view);
        scene.as_octree.config.stretch = false;
        scene.config.swapYZ = true;
        scene.as_octree.config.min_depth = 1;
        scene.as_octree.config.max_depth = c.max_depth;
        scene.as_octree.config.triangle_to_box_stop_ratio = 1.1f;
        scene.as_octree.config.use_gpu = true;

        const uint32_t space_id = scene.build_node_as(scene.root_node_id, ASType::Octree);
        ASSERT_NE(space_id, UINT32_MAX);

        const Space& sp = scene.spaces[space_id];
        EXPECT_GT(sp.as_size, 0u);
        EXPECT_GT(sp.box_content_size, 0u);
        EXPECT_GT(sp.primitives_size, 0u);

        // Verify valid parent/child linkage and leaf contents across the tree.
        std::vector<std::pair<int, int>> stack;
        stack.emplace_back(sp.root_box, 0);
        while (!stack.empty())
        {
            const auto [node_id, depth] = stack.back();
            stack.pop_back();
            const int val = scene.as_octree.as[node_id];
            if (val > 0)
            {
                const int expected_slot8 = (depth == 0) ? 0 : node_id;
                EXPECT_EQ(scene.as_octree.as[val + 8], expected_slot8);
                for (int b = 0; b < 8; ++b)
                {
                    stack.emplace_back(val + b, depth + 1);
                }
            }
            else if (val < 0)
            {
                uint32_t cid = static_cast<uint32_t>(-val);
                while (true)
                {
                    const auto bc = scene.as_octree.box_content[cid];
                    EXPECT_LT(bc.id(), scene.geometry_context.primitives.size());
                    if (bc.is_last()) break;
                    ++cid;
                }
            }
        }

        std::cout << "[BENCH] " << c.label
                  << " | as_size=" << sp.as_size
                  << " | box_content_size=" << sp.box_content_size
                  << " | create_as_ms=" << scene.as_octree.last_create_as_ms()
                  << " | total_build_ms=" << scene.as_octree.build_time_ms(space_id)
                  << std::endl;
    }
}

TEST(OctreeAS, GPUvsCPUEquivalenceTeapot)
{
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());

    auto build_teapot = [&](bool use_gpu, Scene& scene) -> uint32_t {
        scene.load(teapot_view);
        scene.as_octree.config.stretch = false;
        scene.config.swapYZ = true;
        scene.as_octree.config.min_depth = 1;
        scene.as_octree.config.max_depth = 10;
        scene.as_octree.config.triangle_to_box_stop_ratio = 1.1f;
        scene.as_octree.config.use_gpu = use_gpu;
        return scene.build_node_as(scene.root_node_id, ASType::Octree);
    };

    Scene scene_gpu("TeapotGPU");
    Scene scene_cpu("TeapotCPU");
    const uint32_t sp_gpu_id = build_teapot(true, scene_gpu);
    const uint32_t sp_cpu_id = build_teapot(false, scene_cpu);
    ASSERT_NE(sp_gpu_id, UINT32_MAX);
    ASSERT_NE(sp_cpu_id, UINT32_MAX);

    const Space& sp_gpu = scene_gpu.spaces[sp_gpu_id];
    const Space& sp_cpu = scene_cpu.spaces[sp_cpu_id];
    EXPECT_EQ(sp_gpu.as_size, sp_cpu.as_size);
    EXPECT_EQ(sp_gpu.box_content_size, sp_cpu.box_content_size);
    EXPECT_EQ(sp_gpu.primitives_size, sp_cpu.primitives_size);

    // Lockstep structural walk comparing node types, parent slot-8 links, and leaf primitive sets.
    struct WalkEntry
    {
        int gpu_node;
        int cpu_node;
        BoxStack stack;
    };
    std::vector<WalkEntry> walk;
    walk.push_back({sp_gpu.root_box, sp_cpu.root_box, BoxStack{}});

    auto collect_leaf_prims = [](const OctreeAS& oct, int val) {
        std::vector<uint32_t> prims;
        uint32_t cid = static_cast<uint32_t>(-val);
        while (true)
        {
            const auto bc = oct.box_content[cid];
            prims.push_back(bc.id());
            if (bc.is_last()) break;
            ++cid;
        }
        std::sort(prims.begin(), prims.end());
        return prims;
    };

    while (!walk.empty())
    {
        const WalkEntry cur = walk.back();
        walk.pop_back();

        const int gval = scene_gpu.as_octree.as[cur.gpu_node];
        const int cval = scene_cpu.as_octree.as[cur.cpu_node];

        ASSERT_EQ(gval > 0, cval > 0) << "Mismatch internal/leaf at depth " << cur.stack.depth();
        ASSERT_EQ(gval == 0, cval == 0) << "Mismatch empty leaf at depth " << cur.stack.depth();
        ASSERT_EQ(gval < 0, cval < 0) << "Mismatch non-empty leaf at depth " << cur.stack.depth();

        if (gval > 0)
        {
            if (cur.stack.depth() == 0)
            {
                EXPECT_EQ(scene_gpu.as_octree.as[gval + 8], 0);
                EXPECT_EQ(scene_cpu.as_octree.as[cval + 8], 0);
            }
            else
            {
                EXPECT_EQ(scene_gpu.as_octree.as[gval + 8], cur.gpu_node);
                EXPECT_EQ(scene_cpu.as_octree.as[cval + 8], cur.cpu_node);
            }

            for (int b = 0; b < 8; ++b)
            {
                BoxStack child = cur.stack;
                child.descend(b);
                walk.push_back({gval + b, cval + b, child});
            }
        }
        else if (gval < 0)
        {
            const auto g_prims = collect_leaf_prims(scene_gpu.as_octree, gval);
            const auto c_prims = collect_leaf_prims(scene_cpu.as_octree, cval);
            EXPECT_EQ(g_prims, c_prims);
        }
    }

    // Raycast comparison on a 21x21 ray grid.
    soa::soa_vector<PrimitiveChunk16> gpu_prims = scene_gpu.geometry_context.primitives;
    soa::soa_vector<Vertex> gpu_verts = scene_gpu.geometry_context.vertices;
    soa::soa_vector<PrimitiveChunk16> cpu_prims = scene_cpu.geometry_context.primitives;
    soa::soa_vector<Vertex> cpu_verts = scene_cpu.geometry_context.vertices;

    auto trace_ray = [](const OctreeAS& oct,
                        int root_box,
                        const soa::soa_vector<PrimitiveChunk16>& prims,
                        const soa::soa_vector<Vertex>& verts,
                        glm::vec3 origin,
                        glm::vec3 dir,
                        float& out_t) -> bool {
        CurrentBox current_box{.id = root_box};
        glm::vec3 current_source = origin;
        int it = 0;
        int32_t container_id = INT32_MAX;
        float best_t = std::numeric_limits<float>::infinity();
        bool hit = false;

        while (container_id != 0 && !hit && it < 500)
        {
            auto result = traverse_stackless(&oct.as[0], current_box, origin, current_source, dir, it);
            container_id = result.content;
            if (container_id > 0)
            {
                for (;; ++container_id)
                {
                    auto bc = oct.box_content[container_id];
                    PrimitiveChunk16 chunk0 = prims[bc.id()];
                    if (chunk0.type() == PrimitiveType::Triangle)
                    {
                        const Triangle& tri = reinterpret_cast<const Triangle&>(chunk0);
                        auto v1 = glm::vec3(verts[tri.v0].position);
                        auto v2 = glm::vec3(verts[tri.v1].position);
                        auto v3 = glm::vec3(verts[tri.v2].position);

                        float t, u, v;
                        if (intersectRayTriangle(origin, dir, current_box.t_min, current_box.t_max, v1, v2, v3, t, u, v))
                        {
                            if (t < best_t)
                            {
                                best_t = t;
                            }
                            hit = true;
                        }
                    }
                    if (bc.is_last()) break;
                }
            }
            if (current_box.stack.depth() == 0) break;
        }
        out_t = best_t;
        return hit;
    };

    int hit_count = 0;
    for (int ix = 0; ix <= 20; ++ix)
    {
        for (int iz = 0; iz <= 20; ++iz)
        {
            const float x = 0.25f + 0.5f * (static_cast<float>(ix) / 20.0f);
            const float z = 0.25f + 0.5f * (static_cast<float>(iz) / 20.0f);
            const glm::vec3 origin(x, -0.5f, z);
            const glm::vec3 dir(0.0f, 1.0f, 0.0f);

            float t_gpu = 0.0f, t_cpu = 0.0f;
            const bool h_gpu = trace_ray(scene_gpu.as_octree, sp_gpu.root_box, gpu_prims, gpu_verts, origin, dir, t_gpu);
            const bool h_cpu = trace_ray(scene_cpu.as_octree, sp_cpu.root_box, cpu_prims, cpu_verts, origin, dir, t_cpu);

            EXPECT_EQ(h_gpu, h_cpu) << "Hit mismatch at ray (" << ix << ", " << iz << ")";
            if (h_gpu && h_cpu)
            {
                EXPECT_NEAR(t_gpu, t_cpu, 1e-5f) << "Distance mismatch at ray (" << ix << ", " << iz << ")";
                ++hit_count;
            }
        }
    }
    EXPECT_GT(hit_count, 50);
}

TEST(HeterogeneousAS, BenchmarkTlasRebuildChess)
{
    auto view = vkbase::assets::get("Scenes/ABeautifulGame.glb");
    ASSERT_FALSE(view.empty());

    Scene scene("ChessTlasBench");
    scene.load(view);

    // Initial full build to populate BLASes in the shared pools (realistic steady-state).
    scene.as_heterogeneous.config.rebuild_blas = true;
    scene.as_heterogeneous.config.tlas_type = ASType::BVH;
    scene.as_heterogeneous.config.num_probe_rays = 2048;
    scene.as_heterogeneous.config.max_memory_bytes_per_triangle = 128.0f;

    uint32_t tlas_space = scene.build_node_as(scene.root_node_id, ASType::Heterogeneous);
    ASSERT_NE(tlas_space, UINT32_MAX);
    const auto* insts = scene.as_heterogeneous.get_instance_records(tlas_space);
    ASSERT_NE(insts, nullptr);
    const size_t num_instances = insts->size();

    struct TlasCase
    {
        const char* name;
        ASType type;
    };
    const TlasCase cases[] = {
        {"BVH16", ASType::BVH},
        {"KDTree", ASType::KDTree},
        {"UniformGrid", ASType::UniformGrid},
        {"Octree", ASType::Octree},
    };

    scene.as_heterogeneous.config.rebuild_blas = false;
    constexpr int kIters = 5;

    for (const auto& c : cases)
    {
        scene.as_heterogeneous.config.tlas_type = c.type;
        // Warmup
        tlas_space = scene.build_node_as(scene.root_node_id, ASType::Heterogeneous);
        ASSERT_NE(tlas_space, UINT32_MAX);

        double sum_tlas_ms = 0.0;
        double min_tlas_ms = std::numeric_limits<double>::infinity();
        double sum_wall_ms = 0.0;
        double min_wall_ms = std::numeric_limits<double>::infinity();

        for (int it = 0; it < kIters; ++it)
        {
            const auto t0 = std::chrono::high_resolution_clock::now();
            tlas_space = scene.build_node_as(scene.root_node_id, ASType::Heterogeneous);
            const auto t1 = std::chrono::high_resolution_clock::now();
            ASSERT_NE(tlas_space, UINT32_MAX);

            const double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            const double tlas_ms = scene.as_heterogeneous.get_tlas_build_time_ms(tlas_space);
            sum_tlas_ms += tlas_ms;
            min_tlas_ms = std::min(min_tlas_ms, tlas_ms);
            sum_wall_ms += wall_ms;
            min_wall_ms = std::min(min_wall_ms, wall_ms);
        }

        std::cout << "[TLAS BENCH] Chess (" << num_instances << " instances) | " << c.name
                  << " | tlas_ms avg=" << (sum_tlas_ms / kIters)
                  << " min=" << min_tlas_ms
                  << " | wall_ms avg=" << (sum_wall_ms / kIters)
                  << " min=" << min_wall_ms
                  << std::endl;
    }

    scene.free_node_as(scene.root_node_id);
}
