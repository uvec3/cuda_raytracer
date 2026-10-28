#include "cuda_common.hpp"
#include <thrust/tuple.h>

__global__ void map_to_first_entry(uint32_t* list, uint32_t* map, uint32_t list_size)
{
    auto i=blockDim.x*blockIdx.x+threadIdx.x;
    if (i<list_size)
    {
        auto element_id=list[i];
        cuda::atomic_ref<uint32_t,cuda::thread_scope_device> map_ref{map[element_id]};
        (void)map_ref.fetch_min(i,cuda::memory_order::relaxed);
    }
}

void sort_by_first_appearance_in_list(uint32_t* list, uint32_t list_size,uint32_t number_of_elements, auto data_iterator)
{
    thrust::device_vector<uint32_t> map(number_of_elements,list_size);
    map_to_first_entry<<<(list_size-1)/256+1,256>>>(list,map.data().get(),list_size);
    thrust::sort_by_key(thrust::device,map.begin(),map.end(),data_iterator);
}
