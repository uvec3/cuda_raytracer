#include "cuda_common.hpp"
#include <thrust/tuple.h>
#include "geometry_common.cuh"
#include "types.hpp"
#include "../external/soa_vector/src/soa_vector.hpp"

inline __global__ void map_to_first_entry(uint32_t* list, uint32_t list_size, uint32_t* map, uint32_t offset)
{
    auto i=blockDim.x*blockIdx.x+threadIdx.x;
    if (i<list_size)
    {
        auto element_id=list[i]-offset;
        cuda::atomic_ref<uint32_t,cuda::thread_scope_device> map_ref{map[element_id]};
        (void)map_ref.fetch_min(i,cuda::memory_order::relaxed);
    }
}

inline __global__ void map_triangles_to_first_entry(soa::soa_view<Triangle> triangles, uint32_t list_size, uint32_t* map, uint32_t offset)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < list_size)
    {
        auto&& tri = triangles[i];
        uint32_t ids[3] = {tri.v0 - offset, tri.v1 - offset, tri.v2 - offset};
        for (uint32_t element_id : ids)
        {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> map_ref{map[element_id]};
            (void)map_ref.fetch_min(i, cuda::memory_order::relaxed);
        }
    }
}

inline __global__ void remap_triangle_vertices(soa::soa_view<Triangle> triangles, const uint32_t* map, uint32_t vertices_start, uint32_t size)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
    {
        auto&& tri = triangles[i];
        tri.v0 = map[tri.v0 - vertices_start];
        tri.v1 = map[tri.v1 - vertices_start];
        tri.v2 = map[tri.v2 - vertices_start];
    }
}


inline __global__ void inverse(const uint32_t* src, uint32_t* dst,uint32_t offset, uint32_t size)
{
    auto i=blockDim.x*blockIdx.x+threadIdx.x;
    if (i<size)
    {
        dst[src[i]-offset]=i+offset;
    }
}

thrust::device_vector<uint32_t>  sort_by_first_appearance_in_lists(std::vector<uint32_t*> lists, uint32_t list_size,uint32_t elements_offset, uint32_t number_of_elements, auto data_iterator)
{
    thrust::device_vector<uint32_t> first_entry(number_of_elements,list_size);

    for (const auto& list :lists)
    {
        map_to_first_entry<<<(list_size-1)/256+1,256>>>(list,list_size,first_entry.data().get(),elements_offset);
    }

    thrust::device_vector<uint32_t> map_new_to_old(number_of_elements,thrust::no_init);
    thrust::sequence(map_new_to_old.begin(), map_new_to_old.end(),elements_offset);

    auto combined_iterator=thrust::zip_iterator(cuda::std::tuple{data_iterator,map_new_to_old.begin()});
    thrust::sort_by_key(thrust::device,first_entry.begin(),first_entry.end(),combined_iterator);

    thrust::device_vector<uint32_t> map_old_to_new(number_of_elements,thrust::no_init);
    inverse<<<(number_of_elements-1)/128+1,128>>>(map_new_to_old.data().get(),map_old_to_new.data().get(),elements_offset,number_of_elements);

    return map_old_to_new;
}

thrust::device_vector<uint32_t> sort_by_first_appearance_in_triangles(soa::soa_view<Triangle> triangles, uint32_t list_size, uint32_t elements_offset, uint32_t number_of_elements, auto data_iterator)
{
    thrust::device_vector<uint32_t> first_entry(number_of_elements, list_size);

    map_triangles_to_first_entry<<<(list_size - 1) / 256 + 1, 256>>>(triangles, list_size, first_entry.data().get(), elements_offset);

    thrust::device_vector<uint32_t> map_new_to_old(number_of_elements, thrust::no_init);
    thrust::sequence(map_new_to_old.begin(), map_new_to_old.end(), elements_offset);

    auto combined_iterator = thrust::zip_iterator(cuda::std::tuple{data_iterator, map_new_to_old.begin()});
    thrust::sort_by_key(thrust::device, first_entry.begin(), first_entry.end(), combined_iterator);

    thrust::device_vector<uint32_t> map_old_to_new(number_of_elements, thrust::no_init);
    inverse<<<(number_of_elements - 1) / 128 + 1, 128>>>(map_new_to_old.data().get(), map_old_to_new.data().get(), elements_offset, number_of_elements);

    return map_old_to_new;
}



inline __global__ void calculate_morton_codes(soa::soa_view<Vertex> vertices, soa::soa_view<Triangle> triangles,
    uint64_t* code_result, uint32_t size)
{
    auto i=blockDim.x*blockIdx.x+threadIdx.x;
    if (i<size)
    {
        auto&& tri = triangles[i];
        glm::vec3 center=(vertices[tri.v0].position+vertices[tri.v1].position+vertices[tri.v2].position)/3.f;
        auto x_bits=bits_division_from_float(center.x);
        auto y_bits=bits_division_from_float(center.y);
        auto z_bits=bits_division_from_float(center.z);
        code_result[i] = spread_left_aligned(z_bits) | spread_left_aligned(y_bits)>>1 | spread_left_aligned(x_bits)>>2;
    }
}