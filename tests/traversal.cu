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