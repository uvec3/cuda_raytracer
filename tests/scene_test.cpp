#include <gtest/gtest.h>
#include "../src/raytracer/Scene.hpp"
#include "../src/acceleration_structures/octree/OctreeGeometry.cuh"
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

    scene.as.config.stretch = false;
    scene.as.config.swapYZ = true;
    scene.as.config.min_depth = 1;
    scene.as.config.max_depth = 10;
    scene.as.config.triangle_to_box_stop_ratio = 1.1f;
    scene.as.config.use_gpu = true;
    scene.as.config.sort_space = true;
    uint32_t space_id = scene.as.create_space_for_models(subscene_meshes);
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

    scene.as.config.stretch = false;
    scene.as.config.swapYZ = true;
    scene.as.config.min_depth = 1;
    scene.as.config.max_depth = 10;
    scene.as.config.triangle_to_box_stop_ratio = 1.1f;
    uint32_t space_id = scene.as.create_space_for_models(meshes);

    std::cout << "primitives_size = " << scene.as.spaces[space_id].primitives_size << std::endl;
    std::cout << "as_size = " << scene.as.spaces[space_id].as_size << std::endl;
    std::cout << "box_content_size = " << scene.as.spaces[space_id].box_content_size << std::endl;

    EXPECT_GT(scene.as.spaces[space_id].as_size, 0);
    EXPECT_GT(scene.as.spaces[space_id].box_content_size, 0);
    EXPECT_GT(scene.as.spaces[space_id].primitives_size, 0);

    // Verify full ray traversal and triangle hit
    int root_box = scene.as.spaces[space_id].root_box;
    EXPECT_GE(root_box, 0);

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
                auto result = traverse_stackless(&scene.as.as[0], current_box, origin, current_source, dir, it);
                container_id = result.content;
                if (container_id > 0)
                {
                    for (; !hit_triangle; ++container_id)
                    {
                        auto bc = scene.as.box_content[container_id];
                        PrimitiveChunk16 chunk0 = scene.as.primitives[bc.id()];
                        if (chunk0.type() == PrimitiveType::Triangle)
                        {
                            const Triangle& tri = reinterpret_cast<const Triangle&>(chunk0);
                            auto v1 = scene.as.vertices[tri.v0].position;
                            auto v2 = scene.as.vertices[tri.v1].position;
                            auto v3 = scene.as.vertices[tri.v2].position;

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

    scene.as.config.stretch = false;
    scene.as.config.swapYZ = true;
    scene.as.config.min_depth = 1;
    scene.as.config.max_depth = 10;
    scene.as.config.triangle_to_box_stop_ratio = 1.1f;
    scene.as.config.triangle_type = PrimitiveType::FullTriangle;
    uint32_t space_id = scene.as.create_space_for_models(meshes);

    EXPECT_GT(scene.as.spaces[space_id].as_size, 0);
    EXPECT_GT(scene.as.spaces[space_id].box_content_size, 0);
    EXPECT_GT(scene.as.spaces[space_id].primitives_size, 0);
    EXPECT_EQ(scene.as.spaces[space_id].vertices_size, 0u); // vertices freed for FullTriangle

    int root_box = scene.as.spaces[space_id].root_box;
    EXPECT_GE(root_box, 0);

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
                auto result = traverse_stackless(&scene.as.as[0], current_box, origin, current_source, dir, it);
                container_id = result.content;
                if (container_id > 0)
                {
                    for (; !hit_triangle; ++container_id)
                    {
                        auto bc = scene.as.box_content[container_id];
                        PrimitiveChunk16 chunk0 = scene.as.primitives[bc.id()];
                        if (chunk0.type() == PrimitiveType::FullTriangle)
                        {
                            PrimitiveChunk16 chunks[4] = {
                                chunk0,
                                scene.as.primitives[bc.id() + 1],
                                scene.as.primitives[bc.id() + 2],
                                scene.as.primitives[bc.id() + 3]
                            };
                            const FullTriangle& ftri = reinterpret_cast<const FullTriangle&>(chunks);

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

    scene.as.config.stretch = false;
    scene.as.config.swapYZ = true;
    scene.as.config.min_depth = 1;
    scene.as.config.max_depth = 10;
    scene.as.config.triangle_to_box_stop_ratio = 1.1f;
    scene.as.config.triangle_type = PrimitiveType::NagataPatch;
    uint32_t space_id = scene.as.create_space_for_models(meshes);

    EXPECT_GT(scene.as.spaces[space_id].as_size, 0);
    EXPECT_GT(scene.as.spaces[space_id].box_content_size, 0);
    EXPECT_GT(scene.as.spaces[space_id].primitives_size, 0);
    EXPECT_EQ(scene.as.spaces[space_id].vertices_size, 0u); // vertices freed for NagataPatch

    int root_box = scene.as.spaces[space_id].root_box;
    EXPECT_GE(root_box, 0);

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
                auto result = traverse_stackless(&scene.as.as[0], current_box, origin, current_source, dir, it);
                container_id = result.content;
                if (container_id > 0)
                {
                    for (; !hit_triangle; ++container_id)
                    {
                        auto bc = scene.as.box_content[container_id];
                        PrimitiveChunk16 chunk0 = scene.as.primitives[bc.id()];
                        if (chunk0.type() == PrimitiveType::NagataPatch)
                        {
                            NagataPatchPrimitive patch = load_primitive<NagataPatchPrimitive>(scene.as.primitives, bc.id(), chunk0);

                            float t, u, v;
                            if (intersectNagataPrecomputed(origin, dir, current_box.t_min, current_box.t_max,
                                                           patch.X1, patch.X2, patch.X3, patch.C12, patch.C23, patch.C31,
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

    scene.as.config.stretch = false;
    scene.as.config.swapYZ = true;
    scene.as.config.min_depth = 1;
    scene.as.config.max_depth = 10;
    scene.as.config.triangle_to_box_stop_ratio = 1.1f;
    scene.as.config.use_gpu = false;
    uint32_t space_id = scene.as.create_space_for_models(meshes);

    EXPECT_GT(scene.as.spaces[space_id].as_size, 0);
    EXPECT_GT(scene.as.spaces[space_id].box_content_size, 0);
    EXPECT_GT(scene.as.spaces[space_id].primitives_size, 0);

    int root_box = scene.as.spaces[space_id].root_box;
    EXPECT_GE(root_box, 0);

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
                auto result = traverse_stackless(&scene.as.as[0], current_box, origin, current_source, dir, it);
                container_id = result.content;
                if (container_id > 0)
                {
                    for (; !hit_triangle; ++container_id)
                    {
                        auto bc = scene.as.box_content[container_id];
                        PrimitiveChunk16 chunk0 = scene.as.primitives[bc.id()];
                        if (chunk0.type() == PrimitiveType::Triangle)
                        {
                            const Triangle& tri = reinterpret_cast<const Triangle&>(chunk0);
                            auto v1 = scene.as.vertices[tri.v0].position;
                            auto v2 = scene.as.vertices[tri.v1].position;
                            auto v3 = scene.as.vertices[tri.v2].position;

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
    scene.as.config.stretch = false;
    scene.as.config.swapYZ = true;
    scene.as.config.min_depth = 1;
    scene.as.config.max_depth = 10;
    scene.as.config.triangle_to_box_stop_ratio = 1.1f;
    scene.as.config.use_gpu = true;
    scene.as.config.sort_space = false;
    uint32_t space_id = scene.as.create_space_for_models(meshes);

    EXPECT_GT(scene.as.spaces[space_id].as_size, 0);
    EXPECT_GT(scene.as.spaces[space_id].box_content_size, 0);
    EXPECT_GT(scene.as.spaces[space_id].primitives_size, 0);

    int root_box = scene.as.spaces[space_id].root_box;
    EXPECT_GE(root_box, 0);

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
                auto result = traverse_stackless(&scene.as.as[0], current_box, origin, current_source, dir, it);
                container_id = result.content;
                if (container_id > 0)
                {
                    for (; !hit_triangle; ++container_id)
                    {
                        auto bc = scene.as.box_content[container_id];
                        PrimitiveChunk16 chunk0 = scene.as.primitives[bc.id()];
                        if (chunk0.type() == PrimitiveType::Triangle)
                        {
                            const Triangle& tri = reinterpret_cast<const Triangle&>(chunk0);
                            auto v1 = scene.as.vertices[tri.v0].position;
                            auto v2 = scene.as.vertices[tri.v1].position;
                            auto v3 = scene.as.vertices[tri.v2].position;

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

    scene.as.config.stretch = false;
    scene.as.config.swapYZ = true;
    scene.as.config.min_depth = 1;
    scene.as.config.max_depth = 8;
    scene.as.config.triangle_to_box_stop_ratio = 1.1f;
    uint32_t space_id = scene.as.create_space_for_models(meshes);
    EXPECT_GT(scene.as.spaces[space_id].as_size, 0);
    EXPECT_GT(scene.as.spaces[space_id].box_content_size, 0);
    EXPECT_GT(scene.as.spaces[space_id].primitives_size, 0);

    std::cout << "Hairball AS built on GPU: boxes=" << scene.as.spaces[space_id].as_size
              << ", contents=" << scene.as.spaces[space_id].box_content_size
              << ", primitives=" << scene.as.spaces[space_id].primitives_size << std::endl;
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


