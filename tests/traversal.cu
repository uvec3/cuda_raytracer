#include <gtest/gtest.h>
#include "../src/geometry_common.cuh"



TEST(Traversal, AS_Stack)
{
    BoxStack stack;
    stack.print();

    stack.descend(0);
    stack.print();
    stack.set_outside(true);

    stack.move(1);
    stack.print();
    stack.move(2);
    stack.print();
    stack.move(3);
    stack.print();
    stack.move(4);
    stack.print();
    stack.move(5);
    stack.print();
    stack.move(6);
    stack.print();
    stack.move(7);
    stack.print();
    stack.set_invalid(true);


    stack.descend(7);
    stack.print();

    stack.set_invalid(false);
    stack.set_outside(false);

    stack.descend(6);
    stack.print();

    stack.descend(5);
    stack.print();

    stack.move(4);
    stack.print();

    stack.move(3);
    stack.print();

    stack.ascend();
    stack.print();

    stack.ascend();
    stack.print();

    stack.ascend();
    stack.print();

    stack.ascend();
    stack.print();

    stack.descend(1);
    stack.descend(2);
    stack.descend(3);
    stack.descend(4);
    stack.descend(5);
    stack.descend(6);
    stack.descend(7);
    stack.descend(0);

    stack.descend(1);
    stack.descend(2);
    stack.descend(3);
    stack.descend(4);
    stack.descend(5);
    stack.descend(6);
    stack.descend(7);
    stack.descend(0);

    stack.descend(2);
    stack.descend(3);
    stack.descend(4);
    stack.print();

    ASSERT_EQ(stack.branch(1),1);
    ASSERT_EQ(stack.branch(2),2);
    ASSERT_EQ(stack.branch(3),3);
    ASSERT_EQ(stack.branch(4),4);
    ASSERT_EQ(stack.branch(5),5);
    ASSERT_EQ(stack.branch(6),6);
    ASSERT_EQ(stack.branch(7),7);
    ASSERT_EQ(stack.branch(8),0);

    ASSERT_EQ(stack.branch(9),1);
    ASSERT_EQ(stack.branch(10),2);
    ASSERT_EQ(stack.branch(11),3);
    ASSERT_EQ(stack.branch(12),4);
    ASSERT_EQ(stack.branch(13),5);
    ASSERT_EQ(stack.branch(14),6);
    ASSERT_EQ(stack.branch(15),7);
    ASSERT_EQ(stack.branch(16),0);


    ASSERT_EQ(stack.branch(17),2);
    ASSERT_EQ(stack.branch(18),3);
    ASSERT_EQ(stack.branch(19),4);

    stack.ascend();
    stack.print();

    std::cout<<bits_division_to_float(0b0000'0000'0000'0000'0000'0000'0000'0000)<<"\n";
    std::cout<<bits_division_to_float(0b0100'0000'0000'0000'0000'0000'0000'0000)<<"\n";

    std::cout<<bits_division_to_float(0b1000'0000'0000'0000'0000'0000'0000'0000)<<"\n";
    std::cout<<bits_division_to_float(0b1100'0000'0000'0000'0000'0000'0000'0000)<<"\n";
}


TEST(Traversal, NagataKnobNeckMisses)
{
    struct Case {
        glm::vec3 cam, dir;
        glm::vec3 v1, v2, v3;
        glm::vec3 n1, n2, n3;
    };
    Case cases[] = {
        // NEWT MISS #1 (tid=999)
        {
            {0.5f, 0.235238f, 0.463108f}, glm::normalize(glm::vec3(-0.0237692f, 0.915167f, -0.402373f)),
            {0.494438f, 0.361116f, 0.411794f}, {0.488902f, 0.336482f, 0.421532f}, {0.505234f, 0.352815f, 0.411794f},
            glm::normalize(glm::vec3(0.146095f, 0.252394f, 0.956532f)),
            glm::normalize(glm::vec3(0.506558f, 0.506528f, 0.697731f)),
            glm::normalize(glm::vec3(0.206093f, 0.206093f, 0.956583f))
        },
        // NEWT MISS #2 (tid=1000)
        {
            {0.5f, 0.235238f, 0.463108f}, glm::normalize(glm::vec3(-0.0716524f, 0.921978f, -0.380555f)),
            {0.48162f, 0.366445f, 0.411794f}, {0.482748f, 0.341213f, 0.421532f}, {0.494438f, 0.361116f, 0.411794f},
            glm::normalize(glm::vec3(0.0756868f, 0.282086f, 0.956399f)),
            glm::normalize(glm::vec3(0.358848f, 0.620064f, 0.697674f)),
            glm::normalize(glm::vec3(0.146095f, 0.252394f, 0.956532f))
        },
        // NEWT MISS #3 (tid=1061)
        {
            {0.5f, 0.235238f, 0.463108f}, glm::normalize(glm::vec3(-0.0729757f, 0.946291f, -0.314972f)),
            {0.493367f, 0.314839f, 0.434552f}, {0.502121f, 0.305114f, 0.449476f}, {0.49245f, 0.307821f, 0.434552f},
            glm::normalize(glm::vec3(0.990389f, 6.10001e-05f, -0.13831f)),
            glm::normalize(glm::vec3(0.759538f, -0.202863f, -0.61802f)),
            glm::normalize(glm::vec3(0.956719f, -0.255813f, -0.138741f))
        },
        // NEWT MISS #4 (tid=999)
        {
            {0.5f, 0.234839f, 0.455121f}, glm::normalize(glm::vec3(-0.0471405f, 0.942809f, -0.329983f)),
            {0.494438f, 0.361116f, 0.411794f}, {0.488902f, 0.336482f, 0.421532f}, {0.505234f, 0.352815f, 0.411794f},
            glm::normalize(glm::vec3(0.146095f, 0.252394f, 0.956532f)),
            glm::normalize(glm::vec3(0.506558f, 0.506528f, 0.697731f)),
            glm::normalize(glm::vec3(0.206093f, 0.206093f, 0.956583f))
        },
        // NEWT MISS #5 (tid=995)
        {
            {0.555441f, 0.259398f, 0.471015f}, glm::normalize(glm::vec3(-0.616594f, 0.685848f, -0.386554f)),
            {0.452895f, 0.366445f, 0.411794f}, {0.467258f, 0.345323f, 0.421532f}, {0.467258f, 0.368327f, 0.411794f},
            glm::normalize(glm::vec3(-0.0756868f, 0.282086f, 0.956399f)),
            glm::normalize(glm::vec3(0.0f, 0.717076f, 0.696995f)),
            glm::normalize(glm::vec3(0.0f, 0.292527f, 0.956257f))
        },
        // NEWT MISS #6 (tid=1851)
        {
            {0.555441f, 0.259398f, 0.471015f}, glm::normalize(glm::vec3(-0.499836f, 0.85248f, 0.15311f)),
            {0.520644f, 0.329741f, 0.47742f}, {0.508251f, 0.326282f, 0.486628f}, {0.509738f, 0.314839f, 0.486628f},
            glm::normalize(glm::vec3(0.935433f, 0.250163f, 0.249766f)),
            glm::normalize(glm::vec3(0.351454f, 0.0935098f, 0.931524f)),
            glm::normalize(glm::vec3(0.363844f, 0.0f, 0.93146f))
        },
        // NEWT MISS #7 (tid=1001)
        {
            {0.556286f, 0.258553f, 0.463108f}, glm::normalize(glm::vec3(-0.588762f, 0.724491f, -0.358429f)),
            {0.467258f, 0.368327f, 0.411794f}, {0.475444f, 0.34425f, 0.421532f}, {0.48162f, 0.366445f, 0.411794f},
            glm::normalize(glm::vec3(0.0f, 0.292527f, 0.956257f)),
            glm::normalize(glm::vec3(0.185588f, 0.692147f, 0.697488f)),
            glm::normalize(glm::vec3(0.0756868f, 0.282086f, 0.956399f))
        },
    };

    for (int idx = 0; idx < 7; ++idx) {
        const auto& c = cases[idx];
        float tA, uA, vA;
        bool hitA = intersectNagataOnTheFlyAnalytical(c.cam, c.dir, 1e-5f, 1.0f, c.v1, c.v2, c.v3, c.n1, c.n2, c.n3, tA, uA, vA);
        float tN, uN, vN;
        bool hitN = intersectNagataOnTheFly(c.cam, c.dir, 1e-5f, 1.0f, c.v1, c.v2, c.v3, c.n1, c.n2, c.n3, tN, uN, vN);
        EXPECT_TRUE(hitA) << "Analytical missed case #" << idx;
        EXPECT_TRUE(hitN) << "Newton missed case #" << idx;
        EXPECT_NEAR(tN, tA, 1e-5f) << "Depth mismatch on case #" << idx;
    }
}
