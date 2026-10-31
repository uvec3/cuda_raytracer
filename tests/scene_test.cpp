#include <gtest/gtest.h>
#include "../src/raytracer/Scene.hpp"
#include "../vkbase/extensions/Assets/assets.h"

TEST(Scene, Load)
{
    Scene scene("Teapot");
    auto teapot_view = vkbase::assets::get("Scenes/utah_teapot.glb");
    ASSERT_FALSE(teapot_view.empty());
    scene.load(teapot_view);
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
