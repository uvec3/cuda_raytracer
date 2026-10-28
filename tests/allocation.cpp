#include <gtest/gtest.h>
#include "../src/soa/LogicalAllocator.hpp"
#include "../src/soa/LogicalAllocatorOffset.hpp"

TEST(Allocation, LogicalAllocator)
{
    LogicalAllocator a(20);

    std::function<void(size_t)> resize_callback=[&a](size_t new_size)
    {
        printf("Resize callback: old = %lld new = %lld\n",a.size(),new_size);
        a.resize(new_size*3/2);
    };



    a.set_resize_callback(resize_callback);
    printf("%s\n",a.free_regions_string().c_str());

    a.alloc(5);
    printf("%s\n",a.free_regions_string().c_str());
    a.alloc(10);
    printf("%s\n",a.free_regions_string().c_str());

    a.alloc(10);
    printf("%s\n",a.free_regions_string().c_str());

    a.free(0,10);
    printf("%s\n",a.free_regions_string().c_str());

    a.alloc(1);
    printf("%s\n",a.free_regions_string().c_str());

    a.alloc(10);
    printf("%s\n",a.free_regions_string().c_str());

    auto id=a.alloc(100);
    printf("%s\n",a.free_regions_string().c_str());

    a.free(id,99);
    printf("%s\n",a.free_regions_string().c_str());


    id=a.alloc(20);
    printf("%s\n",a.free_regions_string().c_str());

    id=a.alloc(80);
    printf("%s\n",a.free_regions_string().c_str());

    id=a.alloc(1);
    printf("%s\n",a.free_regions_string().c_str());


    a.free(id,1);
    printf("%s\n",a.free_regions_string().c_str());


    a.free(0,1);
    printf("%s\n",a.free_regions_string().c_str());


    a.alloc(100);
    printf("%s\n",a.free_regions_string().c_str());


    a.alloc(160);
    printf("%s\n",a.free_regions_string().c_str());

    a.free(471,1);
    printf("%s\n",a.free_regions_string().c_str());

    a.free(150,10);
    printf("%s\n",a.free_regions_string().c_str());

    //extend
    bool result=a.try_realloc(50,5,20);
    printf("Reallocation success: %s\n",result?"true":"false");
    printf("%s\n",a.free_regions_string().c_str());

    result=a.try_realloc(470,1,2);
    printf("Reallocation success: %s\n",result?"true":"false");
    printf("%s\n",a.free_regions_string().c_str());


    result=a.try_realloc(470,5,1);
    printf("Reallocation success: %s\n",result?"true":"false");
    printf("%s\n",a.free_regions_string().c_str());

    a.clear(0);
    printf("%s\n",a.free_regions_string().c_str());


    a.alloc(10);
    printf("%s\n",a.free_regions_string().c_str());
}


TEST(Allocation, LogicalAllocatorOffset)
{
    LogicalAllocatorOffset a(20);

    std::function<void(size_t)> resize_callback=[&a](size_t new_size)
    {
        printf("Resize callback: old = %lld new = %lld\n",a.size(),new_size);
        a.resize(new_size*3/2);
    };



    a.set_resize_callback(resize_callback);
    printf("%s\n",a.free_regions_string().c_str());

    auto start=a.alloc(5);
    printf("%s\n",a.free_regions_string().c_str());
    start=a.alloc(10);
    printf("%s\n",a.free_regions_string().c_str());

    start=a.alloc(10);
    printf("%s\n",a.free_regions_string().c_str());

    a.free(0,10);
    printf("%s\n",a.free_regions_string().c_str());

    start=a.alloc(1);
    printf("%s\n",a.free_regions_string().c_str());

    start=a.alloc(10);
    printf("%s\n",a.free_regions_string().c_str());

    auto id=a.alloc(100);
    printf("%s\n",a.free_regions_string().c_str());

    a.free(id,99);
    printf("%s\n",a.free_regions_string().c_str());


    id=a.alloc(20);
    printf("%s\n",a.free_regions_string().c_str());

    id=a.alloc(80);
    printf("%s\n",a.free_regions_string().c_str());

    id=a.alloc(1);
    printf("%s\n",a.free_regions_string().c_str());


    a.free(id,1);
    printf("%s\n",a.free_regions_string().c_str());


    a.free(0,1);
    printf("%s\n",a.free_regions_string().c_str());


    a.alloc(100);
    printf("%s\n",a.free_regions_string().c_str());


    a.alloc(160);
    printf("%s\n",a.free_regions_string().c_str());

    a.free(471,1);
    printf("%s\n",a.free_regions_string().c_str());

    a.free(150,10);
    printf("%s\n",a.free_regions_string().c_str());

    //extend
    bool result=a.try_realloc(50,5,20);
    printf("Reallocation success: %s\n",result?"true":"false");
    printf("%s\n",a.free_regions_string().c_str());

    result=a.try_realloc(470,1,2);
    printf("Reallocation success: %s\n",result?"true":"false");
    printf("%s\n",a.free_regions_string().c_str());


    result=a.try_realloc(470,5,1);
    printf("Reallocation success: %s\n",result?"true":"false");
    printf("%s\n",a.free_regions_string().c_str());

    a.clear(0);
    printf("%s\n",a.free_regions_string().c_str());


    a.alloc(10);
    printf("%s\n",a.free_regions_string().c_str());
}