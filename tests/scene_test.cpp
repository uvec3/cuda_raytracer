#include <gtest/gtest.h>
#include "../src/raytracer/Scene.hpp"
#include "../vkbase/extensions/Assets/assets.h"

TEST(Scene, Load)
{
    Scene scene;
    scene.load(vkbase::assets::get("Scenes/utah_teapot.glb"));


    // std::cout<<glm::vec4(0);
}