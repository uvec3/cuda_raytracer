#include "OctreeAS.cuh"
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/sort.h>

#include "imgui.h"
#include "../../device_algorithms.cuh"
#include "../../queue/DeviceQueue.cuh"
#include "../../cuda/scan.cuh"

struct InsertionContext
{
    int depth_min;
    int depth_max;
    float triangle_box_stop_ratio;
    soa::soa_view<TriangleNumberId> num_id;

    soa::soa_vector<int> __as;
    soa::soa_vector<BoxContent> __box_content;
};

__host__ __device__ inline void get_primitive_triangle_vertices(
    soa::soa_view<PrimitiveChunk16> primitives,
    soa::soa_view<Vertex> vertices,
    uint32_t prim_slot_id,
    glm::vec3& v0, glm::vec3& v1, glm::vec3& v2,
    glm::vec3& n0, glm::vec3& n1, glm::vec3& n2)
{
    PrimitiveChunk16 chunk0 = primitives[prim_slot_id];
    if (chunk0.type() == PrimitiveType::Triangle)
    {
        Triangle tri = chunk0.as<Triangle>();
        v0 = vertices[tri.v0].position;
        v1 = vertices[tri.v1].position;
        v2 = vertices[tri.v2].position;
        n0 = vertices[tri.v0].normal;
        n1 = vertices[tri.v1].normal;
        n2 = vertices[tri.v2].normal;
    }
    else if (chunk0.type() == PrimitiveType::FullTriangle)
    {
        FullTriangle ftri = load_primitive<FullTriangle>(primitives, prim_slot_id, chunk0);
        v0 = ftri.v0;
        v1 = ftri.v1;
        v2 = ftri.v2;
        n0 = ftri.n0_compressed.unpack();
        n1 = ftri.n1_compressed.unpack();
        n2 = ftri.n2_compressed.unpack();
    }
    else if (chunk0.type() == PrimitiveType::NagataPatch)
    {
        NagataPatchPrimitive patch = load_primitive<NagataPatchPrimitive>(primitives, prim_slot_id, chunk0);
        v0 = patch.X1;
        v1 = patch.X2;
        v2 = patch.X3;
        n0 = patch.evaluateNormal(1.0f, 0.0f);
        n1 = patch.evaluateNormal(0.0f, 1.0f);
        n2 = patch.evaluateNormal(0.0f, 0.0f);
    }
}

void OctreeAS::config_ui()
{
    ImGui::InputInt("Min depth", &config.min_depth, 1);
    ImGui::InputInt("Max depth", &config.max_depth, 1);
    ImGui::InputFloat("Triangle to box stop ratio", &config.triangle_to_box_stop_ratio, 0.1f);
    ImGui::Checkbox("Stretch", &config.stretch);
    ImGui::Checkbox("Swap y and z", &config.swapYZ);
    ImGui::Checkbox("Build on GPU", &config.use_gpu);
    ImGui::Checkbox("Sort space", &config.sort_space);

    const char* tri_type_names[] = {"Triangle (16B Indexed)", "FullTriangle (64B Inline)", "NagataPatch (96B Inline Curved)"};
    int tri_type_idx = 0;
    if (config.triangle_type == PrimitiveType::FullTriangle)
        tri_type_idx = 1;
    else if (config.triangle_type == PrimitiveType::NagataPatch)
        tri_type_idx = 2;

    if (ImGui::Combo("Triangle storage type", &tri_type_idx, tri_type_names, 3))
    {
        if (tri_type_idx == 0) config.triangle_type = PrimitiveType::Triangle;
        else if (tri_type_idx == 1) config.triangle_type = PrimitiveType::FullTriangle;
        else if (tri_type_idx == 2) config.triangle_type = PrimitiveType::NagataPatch;
    }
}

OctreeAS::OctreeAS()
{
    //reserve element 0 for empty value and don't use it
    as.alloc(1);
    as[0] = 0;
    box_content.alloc(1);
    sync_to_device();
}

uint32_t OctreeAS::create_space()
{
    auto new_space_id=spaces.alloc_element();
    auto&& new_space=spaces[new_space_id];
    auto root_box_id=as.alloc_element();
    as[root_box_id]=0;

    new_space.root_box =new_space_id;
    new_space.vertices_start =0;
    new_space.vertices_size =0;
    new_space.primitives_start =0;
    new_space.primitives_size =0;
    new_space.normalization_scale =glm::vec3(1);
    new_space.normalization_translate =glm::vec3(0);

    sync_to_device();

    return new_space_id;
}

void OctreeAS::free_space(uint32_t space_id)
{
    auto vertices_start = spaces[space_id].vertices_start;
    auto vertices_size = spaces[space_id].vertices_size;
    if (vertices_size > 0)
        vertices.free(vertices_start);

    auto as_start=spaces[space_id].as_start;
    auto as_size=spaces[space_id].as_size;
    if (as_size>0)
        as.free(as_start);

    auto box_content_start=spaces[space_id].box_content_start;
    auto box_content_size=spaces[space_id].box_content_size;
    if (box_content_size>0)
        box_content.free(box_content_start);

    auto primitives_start=spaces[space_id].primitives_start;
    auto primitives_size=spaces[space_id].primitives_size;
    if (primitives_size>0)
        primitives.free(primitives_start);

    spaces.free(space_id);
}

int OctreeAS::divide_box(InsertionContext& insertion_context, BoxInfo box_info)
{
    auto content = insertion_context.__as[box_info.id];
    const int32_t child_start = insertion_context.__as.allocate(9);
    for (int i = 0; i < 8; ++i)
    {
        insertion_context.__as[child_start + i] = 0;
    }
    insertion_context.__as[child_start + 8] = static_cast<int>(box_info.id);//place parent id as 9-th element

    if (content > 0)
        throw std::runtime_error("ain't suppose to divide if already split");

    if (content < 0)//pass data to child if has
    {
        exit(1);
    }

    insertion_context.__as[box_info.id] = child_start;
    return child_start;
}


void OctreeAS::add_box_contents(InsertionContext& ic, uint32_t box_id, std::span<const uint32_t> prim_ids)
{
    if (prim_ids.empty())
        return;

    auto content_id = ic.__as[box_id];
    if (content_id != 0)
        throw std::runtime_error("can't add content to non-empty box");

    auto start = ic.__box_content.allocate(prim_ids.size());
    ic.__as[box_id] = -static_cast<int>(start);

    for (size_t i = 0; i < prim_ids.size(); ++i)
    {
        ic.__box_content[start + i] = BoxContent(prim_ids[i], (i + 1 == prim_ids.size()));
    }
}



void OctreeAS::normalize_space(uint32_t space_id,bool stretch)
{
    auto vertices_start = &d_vertices[spaces[space_id].vertices_start].position;
    auto vertices_size = spaces[space_id].vertices_size;

    constexpr float inf = std::numeric_limits<float>::infinity();
    auto aabb_min=thrust::reduce(thrust::device,vertices_start,vertices_start+vertices_size,glm::vec3(inf),
        []__device__(const glm::vec3& a, const glm::vec3& b)->glm::vec3 {return glm::min(a,b);});
    auto aabb_max=thrust::reduce(thrust::device,vertices_start,vertices_start+vertices_size,glm::vec3(-inf),
        []__device__(const glm::vec3& a, const glm::vec3& b)->glm::vec3 {return glm::max(a,b);});
    auto aabb_size=aabb_max-aabb_min;

    glm::vec3 normalization_translate=-aabb_min;
    glm::vec3 normalization_scale;
    if (stretch)
    {
        normalization_scale=glm::vec3(1) /(aabb_size)*0.98f;
    }
    else
    {
        normalization_scale=glm::vec3(1) / glm::vec3( glm::max(glm::max(aabb_size.x,aabb_size.y),aabb_size.z)) * 0.98f;
    }

    thrust::transform(thrust::device,vertices_start,vertices_start+ vertices_size,vertices_start, [=]__device__(const glm::vec3& v)
    {
        return (v+normalization_translate)*normalization_scale+0.01f;
    });

    spaces[space_id].normalization_translate = normalization_translate;
    spaces[space_id].normalization_scale = normalization_scale;
}


#define STACK_SIZE 3000
__global__
void evaluate_triangle_box(soa::soa_view<PrimitiveChunk16> primitives, soa::soa_view<Vertex> vertices, uint32_t primitives_start,
    uint32_t number_of_triangles, uint32_t occupied_slots, soa::soa_view<TriangleNumberId> result, uint32_t* result_counter,
    int DEPTH_MIN, int DEPTH_MAX, float TRIANGLE_BOX_STOP_RATIO,
    auto find_triangle_cube_relation)
{
    struct StackElement
    {
        uint32_t id;
        BoxStack box{};
    };

    __shared__ cuda::atomic<int> stack_size;
    __shared__ StackElement stack[STACK_SIZE];
    __shared__ int batch_start;

    auto push_to_stack=[&](uint32_t id, BoxStack number)
    {
        auto stack_start=stack_size.fetch_add(1);
        assert(stack_start<STACK_SIZE);
        stack[stack_start].id=id;
        stack[stack_start].box=number;
    };


    //initiate_stack
    if (threadIdx.x==0)
    {
        stack_size.store(0);
    }

    __syncthreads();


    auto block_first=blockIdx.x*blockDim.x;
    uint32_t local_idx=block_first+threadIdx.x;
    uint32_t t_id=primitives_start+local_idx*occupied_slots;
    BoxStack box{.m_data = 0};

    if (local_idx<number_of_triangles)
    {
        glm::vec3 v1, v2, v3, n1, n2, n3;
        get_primitive_triangle_vertices(primitives, vertices, t_id, v1, v2, v3, n1, n2, n3);
        auto rel=find_triangle_cube_relation(box.get_center(),box.half_box_size(),v1,v2,v3,n1,n2,n3);

        if (rel!=TriangleCubeRelation::Outside)
        {
            push_to_stack(t_id,BoxStack{.m_data=0});
        }
    }


    while (true)
    {
        __syncthreads();//read-after-write from stack

        if (threadIdx.x==0)//update stack size
        {
            batch_start=stack_size.load()-blockDim.x;

            stack_size-=blockDim.x;
            if (stack_size.load()<0)
                stack_size.store(0);
        }


        __syncthreads();//write-after-read into stack


        if (batch_start<=-static_cast<int>(blockDim.x))
        {
            // printf("Exit: batch_start=%d\n",batch_strart);
            return;
        }


        int element= batch_start+static_cast<int>(threadIdx.x);


        if (element>=0)
        {
            t_id=stack[element].id;
            box=stack[element].box;
        }

        __syncthreads();

        if (element>=0)
        {
            glm::vec3 v0, v1, v2, n0, n1, n2;
            get_primitive_triangle_vertices(primitives, vertices, t_id, v0, v1, v2, n0, n1, n2);

            bool descend=box.depth()<DEPTH_MAX;
            if (descend && box.depth()>=DEPTH_MIN)
            {
                glm::vec3 aabb_max=glm::max(glm::max(v0,v1),v2);
                glm::vec3 aabb_min=glm::min(glm::min(v0,v1),v2);
                glm::vec3 dim=aabb_max-aabb_min;
                float triangle_size=glm::max(glm::max(dim.x,dim.y),dim.z);

                if (triangle_size>box.box_size()*TRIANGLE_BOX_STOP_RATIO)
                {
                    descend=false;
                }
            }


            if (descend)//recursive check
            {
                box.descend(0);
                for (int branch=0;branch<8;++branch)
                {
                    box.move(branch);
                    auto rel=find_triangle_cube_relation(box.get_center(),box.half_box_size(),v0,v1,v2,n0,n1,n2);
                    if (rel!=TriangleCubeRelation::Outside)
                    {
                        push_to_stack(t_id,box);
                    }
                }
            }
            else//add to the results
            {
                cuda::atomic_ref<uint32_t> result_counter_a(*result_counter);
                auto result_start=result_counter_a.fetch_add(1);
                assert(result_start < result.size());
                result[result_start].id=t_id;
                result[result_start].number=box.m_data;
            }
        }
    }
}


template <typename T>
CUDA_HOST_DEVICE uint32_t find_greater_or_equal_index(T value, const T* sorted_list, uint32_t search_start, uint32_t search_end)
{
    while (search_start < search_end)
    {
        auto middle_pos = search_start + (search_end - search_start) / 2;
        auto number = sorted_list[middle_pos];

        if (number < value)
        {
            search_start = middle_pos + 1;
        }
        else
        {
            search_end = middle_pos;
        }
    }

    return search_start;
}

struct ExecutionState
{
    uint32_t current_layer_begin=0;
    uint32_t current_layer_pos=0;
    uint32_t next_layer_begin=1;
    uint32_t next_layer_pos=1;
};

__global__ void create_as(const uint64_t* used_boxes, uint32_t used_boxes_size, int* as, uint64_t* as_box, uint32_t max_capacity, uint32_t* out_as_size)
{
    __shared__ ExecutionState state;
    __shared__ uint32_t block_base_offset;

    // init
    if (threadIdx.x == 0)
    {
        state.current_layer_begin = 0;
        state.current_layer_pos = 0;
        state.next_layer_begin = 1;
        state.next_layer_pos = 1;
        as[0] = 0;
        as_box[0] = 0;
    }
    __syncthreads();

    while (state.current_layer_begin != state.next_layer_begin) // while current layer not empty
    {
        while (state.current_layer_pos < state.next_layer_begin) // process all elements in current depth level
        {
            uint32_t this_box_id = state.current_layer_pos + threadIdx.x;
            bool in_range = (this_box_id < state.next_layer_begin);

            bool subdivides = false;
            BoxStack box{};

            if (in_range)
            {
                uint64_t box_data = as_box[this_box_id];
                box = BoxStack{box_data};

                // Only actual boxes subdivide (slot 8 parent refs have is_invalid() == true)
                if (!box.is_invalid())
                {
                    uint32_t begin_idx = 0;
                    uint32_t end_idx = 0;
                    if (box.depth() == 0)
                    {
                        begin_idx = 0;
                        end_idx = used_boxes_size;
                    }
                    else
                    {
                        begin_idx = find_greater_or_equal_index(box.begin(), used_boxes, 0, used_boxes_size);
                        end_idx = find_greater_or_equal_index(box.end(), used_boxes, 0, used_boxes_size);
                    }

                    if (begin_idx < end_idx && box.depth() < BoxStack::max_depth)
                    {
                        for (uint32_t k = begin_idx; k < end_idx; ++k)
                        {
                            if (BoxStack{used_boxes[k]}.depth() > box.depth())
                            {
                                subdivides = true;
                                break;
                            }
                        }
                    }
                }
            }

            int my_alloc = subdivides ? 9 : 0;
            int inc_scan = block_scan_kogge_stone(my_alloc);
            int exc_scan = inc_scan - my_alloc;

            if (threadIdx.x == blockDim.x - 1)
            {
                block_base_offset = state.next_layer_pos;
                state.next_layer_pos += inc_scan;
                assert(state.next_layer_pos < max_capacity);
            }
            __syncthreads();

            if (in_range && !box.is_invalid())
            {
                if (subdivides)
                {
                    uint32_t child_start = block_base_offset + exc_scan;
                    as[this_box_id] = static_cast<int>(child_start);

                    for (int branch = 0; branch < 8; ++branch)
                    {
                        BoxStack child_stack = box;
                        child_stack.descend(branch);
                        as_box[child_start + branch] = child_stack.m_data;
                        as[child_start + branch] = 0;
                    }

                    BoxStack invalid_stack;
                    invalid_stack.set_invalid(true);
                    as_box[child_start + 8] = invalid_stack.m_data;
                    as[child_start + 8] = static_cast<int>(this_box_id);
                }
                else
                {
                    as[this_box_id] = 0;
                }
            }

            if (threadIdx.x == 0)
            {
                state.current_layer_pos += blockDim.x;
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            state.current_layer_begin = state.next_layer_begin;
            state.next_layer_begin = state.next_layer_pos;
            state.current_layer_pos = state.current_layer_begin;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && out_as_size != nullptr)
    {
        *out_as_size = state.next_layer_begin;
    }
}

#define STACK_SIZE 2048
__global__ void expand_number_id(
    soa::soa_view<PrimitiveChunk16> primitives,
    soa::soa_view<Vertex> vertices,
    soa::soa_view<TriangleNumberId> input_number_id,
    uint32_t input_size,
    const int* as,
    const uint64_t* as_box,
    soa::soa_view<TriangleNumberId> result,
    uint32_t* result_counter,
    uint32_t max_result_capacity,
    auto find_triangle_cube_relation)
{
    struct StackElement
    {
        uint32_t t_id;
        uint32_t node;
        BoxStack box;
    };

    __shared__ cuda::atomic<int> stack_size;
    __shared__ StackElement stack[STACK_SIZE];
    __shared__ int batch_start;

    auto push_to_stack = [&](uint32_t id, uint32_t node, BoxStack box)
    {
        auto stack_start = stack_size.fetch_add(1);
        assert(stack_start < STACK_SIZE);
        stack[stack_start].t_id = id;
        stack[stack_start].node = node;
        stack[stack_start].box = box;
    };

    if (threadIdx.x == 0)
    {
        stack_size.store(0);
    }
    __syncthreads();

    auto block_first = blockIdx.x * blockDim.x;
    uint32_t elem_id = block_first + threadIdx.x;

    if (elem_id < input_size)
    {
        uint32_t t_id = input_number_id[elem_id].id;
        uint64_t box_data = input_number_id[elem_id].number;
        BoxStack target_box{box_data};

        // Navigate down 'as' from root to target_box
        uint32_t curr_node = 0;
        for (int lvl = 1; lvl <= target_box.depth(); ++lvl)
        {
            int child_start = as[curr_node];
            if (child_start <= 0)
            {
                break;
            }
            uint32_t branch = target_box.branch(lvl);
            curr_node = child_start + branch;
        }

        if (as[curr_node] <= 0)
        {
            // Already a leaf
            cuda::atomic_ref<uint32_t> out_cnt(*result_counter);
            auto out_idx = out_cnt.fetch_add(1);
            assert(out_idx < max_result_capacity);
            result[out_idx].id = t_id;
            result[out_idx].number = as_box[curr_node];
        }
        else
        {
            // Box was subdivided! Push to stack
            push_to_stack(t_id, curr_node, BoxStack{as_box[curr_node]});
        }
    }

    while (true)
    {
        __syncthreads();
        if (threadIdx.x == 0)
        {
            batch_start = stack_size.load() - blockDim.x;
            stack_size -= blockDim.x;
            if (stack_size.load() < 0)
                stack_size.store(0);
        }
        __syncthreads();

        if (batch_start <= -static_cast<int>(blockDim.x))
        {
            return;
        }

        int element = batch_start + static_cast<int>(threadIdx.x);
        uint32_t t_id = 0;
        uint32_t node = 0;
        BoxStack box{};

        if (element >= 0)
        {
            t_id = stack[element].t_id;
            node = stack[element].node;
            box = stack[element].box;
        }
        __syncthreads();

        if (element >= 0)
        {
            int child_start = as[node];
            if (child_start > 0)
            {
                glm::vec3 v0, v1, v2, n0, n1, n2;
                get_primitive_triangle_vertices(primitives, vertices, t_id, v0, v1, v2, n0, n1, n2);

                for (int branch = 0; branch < 8; ++branch)
                {
                    uint32_t child_node = child_start + branch;
                    BoxStack child_box = box;
                    child_box.descend(branch);

                    auto rel = find_triangle_cube_relation(
                        child_box.get_center(), child_box.half_box_size(), v0, v1, v2, n0, n1, n2);

                    if (rel != TriangleCubeRelation::Outside)
                    {
                        if (as[child_node] > 0)
                        {
                            push_to_stack(t_id, child_node, child_box);
                        }
                        else
                        {
                            cuda::atomic_ref<uint32_t> out_cnt(*result_counter);
                            auto out_idx = out_cnt.fetch_add(1);
                            assert(out_idx < max_result_capacity);
                            result[out_idx].id = t_id;
                            result[out_idx].number = child_box.m_data;
                        }
                    }
                }
            }
            else
            {
                cuda::atomic_ref<uint32_t> out_cnt(*result_counter);
                auto out_idx = out_cnt.fetch_add(1);
                assert(out_idx < max_result_capacity);
                result[out_idx].id = t_id;
                result[out_idx].number = box.m_data;
            }
        }
    }
}

__global__ void count_leaf_primitives_kernel(
    const int* as, const uint64_t* as_box, uint32_t as_size,
    const uint64_t* sorted_numbers, uint32_t expanded_size,
    uint32_t* prim_counts)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= as_size) return;

    if (as[i] <= 0 && !BoxStack{as_box[i]}.is_invalid())
    {
        uint64_t code = as_box[i];
        uint32_t begin = find_greater_or_equal_index(code, sorted_numbers, 0, expanded_size);
        uint32_t end = find_greater_or_equal_index(code + 1, sorted_numbers, begin, expanded_size);
        prim_counts[i] = end - begin;
    }
    else
    {
        prim_counts[i] = 0;
    }
}

__global__ void populate_leaf_primitives_kernel(
    int* as, const uint64_t* as_box, uint32_t as_size,
    const uint64_t* sorted_numbers, const uint32_t* sorted_triangle_ids, uint32_t expanded_size,
    const uint32_t* prim_counts, const uint32_t* prim_offsets,
    soa::soa_view<BoxContent> box_content)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= as_size) return;

    uint32_t p_count = prim_counts[i];
    if (p_count > 0)
    {
        uint32_t bc_start = prim_offsets[i] + 1; // 1-based index (0 is reserved empty)
        as[i] = -static_cast<int>(bc_start);

        uint64_t code = as_box[i];
        uint32_t begin = find_greater_or_equal_index(code, sorted_numbers, 0, expanded_size);

        for (uint32_t p = 0; p < p_count; ++p)
        {
            box_content[bc_start + p] = BoxContent(sorted_triangle_ids[begin + p], (p + 1 == p_count));
        }
    }
    else if (as[i] <= 0 && !BoxStack{as_box[i]}.is_invalid())
    {
        as[i] = 0; // Empty leaf box
    }
}

__global__ void adjust_global_as_offsets_kernel(
    int* local_as, uint32_t as_size, uint32_t as_start, uint32_t box_content_start)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < as_size)
    {
        int child = local_as[i];
        if (child > 0)
            local_as[i] = child + static_cast<int>(as_start);
        else if (child < 0)
            local_as[i] = child - static_cast<int>(box_content_start);
    }
}

void OctreeAS::insert_triangles_gpu(uint32_t space_id, uint32_t primitives_start, uint32_t number_of_triangles, uint32_t occupied_slots,
                                    int depth_min, int depth_max, float triangle_box_stop_ratio, auto find_triangle_cube_relation)
{
    if (number_of_triangles == 0)
        return;

    static constexpr uint32_t block_size=128;
    uint32_t number_of_blocks= (number_of_triangles-1) / block_size + 1;

    soa::cuda_device_vector<TriangleNumberId> d_number_id(number_of_triangles*100);

    uint32_t* d_result_size = nullptr;
    CUDA_CHECK(cudaMalloc(&d_result_size, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_result_size, 0, sizeof(uint32_t)));

    evaluate_triangle_box<<<number_of_blocks,block_size>>>(d_primitives.view(),d_vertices.view(),primitives_start,
       number_of_triangles,occupied_slots,d_number_id.view(),d_result_size,depth_min,depth_max,triangle_box_stop_ratio,find_triangle_cube_relation);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t result_size = 0;
    CUDA_CHECK(cudaMemcpy(&result_size, d_result_size, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result_size));

    printf("\n\nElemetnts: %d\n", result_size);

    thrust::sort_by_key(thrust::device, &d_number_id[0].number, &d_number_id[0].number + result_size, &d_number_id[0].id);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n\nElements: %d\n", result_size);
    printf("Number of triangles: %d\n", number_of_triangles);

    d_number_id.resize(result_size);

    // Step 1: create_as (Single-Block Kernel, Layer-by-Layer)
    uint32_t max_as_capacity = std::max(2048u, result_size * 18 + 1024);
    soa::cuda_device_vector<int> d_local_as(max_as_capacity);
    thrust::fill_n(thrust::device, &d_local_as[0], max_as_capacity, 0);
    thrust::device_vector<uint64_t> d_as_box(max_as_capacity, 0);

    uint32_t* d_as_size = nullptr;
    CUDA_CHECK(cudaMalloc(&d_as_size, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_as_size, 0, sizeof(uint32_t)));

    create_as<<<1, 256>>>(
        &d_number_id[0].number, result_size,
        &d_local_as[0], d_as_box.data().get(),
        max_as_capacity, d_as_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t as_size = 0;
    CUDA_CHECK(cudaMemcpy(&as_size, d_as_size, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_as_size));

    d_local_as.resize(as_size);
    d_as_box.resize(as_size);
    printf("AS built on GPU: %d boxes\n", as_size);

    // Step 2: expand_number_id (Push triangles from divided boxes down to leaf nodes)
    uint32_t max_expanded_capacity = std::max(result_size * 8 + 1024, 2048u);
    soa::cuda_device_vector<TriangleNumberId> d_expanded_number_id(max_expanded_capacity);

    uint32_t* d_expanded_size = nullptr;
    CUDA_CHECK(cudaMalloc(&d_expanded_size, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_expanded_size, 0, sizeof(uint32_t)));

    uint32_t expand_blocks = (result_size - 1) / 256 + 1;
    expand_number_id<<<expand_blocks, 256>>>(
        d_primitives.view(), d_vertices.view(),
        d_number_id.view(), result_size,
        &d_local_as[0], d_as_box.data().get(),
        d_expanded_number_id.view(), d_expanded_size, max_expanded_capacity,
        find_triangle_cube_relation);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t expanded_size = 0;
    CUDA_CHECK(cudaMemcpy(&expanded_size, d_expanded_size, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_expanded_size));

    d_expanded_number_id.resize(expanded_size);
    d_number_id.clear();
    d_number_id.shrink_to_fit();
    printf("Expanded elements: %d\n", expanded_size);

    // Step 3: Sort expanded d_number_id
    thrust::sort_by_key(thrust::device,
        &d_expanded_number_id[0].number,
        &d_expanded_number_id[0].number + expanded_size,
        &d_expanded_number_id[0].id);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 4: insert_primitives (Host-orchestrated count -> scan -> populate directly into box_content)
    thrust::device_vector<uint32_t> d_prim_counts(as_size);
    thrust::device_vector<uint32_t> d_prim_offsets(as_size);

    uint32_t leaf_blocks = (as_size - 1) / 256 + 1;
    count_leaf_primitives_kernel<<<leaf_blocks, 256>>>(
        &d_local_as[0], d_as_box.data().get(), as_size,
        &d_expanded_number_id[0].number, expanded_size,
        d_prim_counts.data().get());
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::exclusive_scan(d_prim_counts.begin(), d_prim_counts.end(), d_prim_offsets.begin());

    uint32_t last_prim_offset = 0, last_prim_count = 0;
    CUDA_CHECK(cudaMemcpy(&last_prim_offset, d_prim_offsets.data().get() + as_size - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_prim_count, d_prim_counts.data().get() + as_size - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    uint32_t total_box_contents = last_prim_offset + last_prim_count;

    soa::cuda_device_vector<BoxContent> d_local_box_content(total_box_contents + 1);
    {
        auto bc_begin = thrust::make_zip_iterator(d_local_box_content.get_tuple());
        auto bc_end = thrust::make_zip_iterator(d_local_box_content.get_tuple(total_box_contents + 1));
        using BcZipValue = typename decltype(bc_begin)::value_type;
        thrust::fill(thrust::device, bc_begin, bc_end, BcZipValue{});
    }

    populate_leaf_primitives_kernel<<<leaf_blocks, 256>>>(
        &d_local_as[0], d_as_box.data().get(), as_size,
        &d_expanded_number_id[0].number, &d_expanded_number_id[0].id, expanded_size,
        d_prim_counts.data().get(), d_prim_offsets.data().get(),
        d_local_box_content.view());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 5: Global pool integration & download
    auto as_start = as.alloc(as_size);
    auto box_content_start = box_content.alloc(total_box_contents + 1);

    spaces[space_id].root_box = static_cast<int>(as_start);
    spaces[space_id].as_start = as_start;
    spaces[space_id].as_size = as_size;
    spaces[space_id].box_content_start = box_content_start;
    spaces[space_id].box_content_size = total_box_contents + 1;

    adjust_global_as_offsets_kernel<<<leaf_blocks, 256>>>(
        &d_local_as[0], as_size, as_start, box_content_start);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (d_as.size() < as.size()) d_as.resize(as.size());
    if (d_box_content.size() < box_content.size()) d_box_content.resize(box_content.size());

    d_as.range(as_start) = d_local_as;
    d_box_content.range(box_content_start) = d_local_box_content;

    // Download to host pools for UI & debug
    as.range(as_start) = d_as.range(as_start, as_size);
    box_content.range(box_content_start) = d_box_content.range(box_content_start, total_box_contents + 1);
}

void OctreeAS::insert_triangles_cpu(uint32_t space_id, uint32_t primitives_start, uint32_t number_of_triangles, uint32_t occupied_slots,
                                    int depth_min, int depth_max, float triangle_box_stop_ratio, auto find_triangle_cube_relation)
{
    if (number_of_triangles == 0)
        return;

    static constexpr uint32_t block_size = 256;
    uint32_t number_of_blocks = (number_of_triangles - 1) / block_size + 1;

    soa::cuda_device_vector<TriangleNumberId> d_number_id(number_of_triangles * 100);

    uint32_t* d_result_size = nullptr;
    CUDA_CHECK(cudaMalloc(&d_result_size, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_result_size, 0, sizeof(uint32_t)));

    evaluate_triangle_box<<<number_of_blocks, block_size>>>(
        d_primitives.view(), d_vertices.view(), primitives_start,
        number_of_triangles, occupied_slots, d_number_id.view(), d_result_size, depth_min, depth_max, triangle_box_stop_ratio,
        find_triangle_cube_relation);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t result_size = 0;
    CUDA_CHECK(cudaMemcpy(&result_size, d_result_size, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result_size));

    thrust::sort_by_key(thrust::device, &d_number_id[0].number, &d_number_id[0].number + result_size, &d_number_id[0].id);
    CUDA_CHECK(cudaDeviceSynchronize());

    d_number_id.resize(result_size);
    soa::soa_vector<TriangleNumberId> number_id = d_number_id;
    d_number_id.clear();
    d_number_id.shrink_to_fit();

    InsertionContext insertion_context;
    insertion_context.num_id = number_id;
    insertion_context.depth_min = depth_min;
    insertion_context.depth_max = depth_max;
    insertion_context.triangle_box_stop_ratio = triangle_box_stop_ratio;
    insertion_context.__as.allocate(1);
    insertion_context.__box_content.allocate(1);
    insertion_context.__as[0] = 0;

    insert_triangles_recursive(0, BoxStack{}, insertion_context, 0, result_size, {}, find_triangle_cube_relation);

    auto as_start = as.alloc(insertion_context.__as.size());
    auto box_content_start = box_content.alloc(insertion_context.__box_content.size());

    spaces[space_id].root_box = as_start;
    spaces[space_id].as_start = as_start;
    spaces[space_id].as_size = insertion_context.__as.size();
    spaces[space_id].box_content_start = box_content_start;
    spaces[space_id].box_content_size = insertion_context.__box_content.size();

    for (size_t i = 0; i < insertion_context.__as.size(); ++i)
    {
        auto&& child = insertion_context.__as[i];
        if (child > 0)
            child += as_start;
        else if (child < 0)
            child -= box_content_start;
        as[as_start + i] = insertion_context.__as[i];
    }

    for (size_t i = 0; i < insertion_context.__box_content.size(); ++i)
    {
        box_content[box_content_start + i] = insertion_context.__box_content[i];
    }
}

void OctreeAS::insert_triangles(uint32_t space_id, uint32_t primitives_start, uint32_t number_of_triangles, uint32_t occupied_slots,
                                int depth_min, int depth_max, float triangle_box_stop_ratio, bool use_gpu)
{
    auto dispatch = [&](auto relation_func)
    {
        if (use_gpu)
        {
            insert_triangles_gpu(space_id, primitives_start, number_of_triangles, occupied_slots, depth_min, depth_max, triangle_box_stop_ratio, relation_func);
        }
        else
        {
            insert_triangles_cpu(space_id, primitives_start, number_of_triangles, occupied_slots, depth_min, depth_max, triangle_box_stop_ratio, relation_func);
        }
    };

    if (config.triangle_type == PrimitiveType::NagataPatch)
    {
        dispatch(FindCubeIntersection<PrimitiveApproximation::Nagata>{});
    }
    else
    {
        dispatch(FindCubeIntersection<PrimitiveApproximation::Precise>{});
    }
}


void OctreeAS::insert_triangles_recursive(int box_id, BoxStack stack, InsertionContext& ic,
                                                       uint32_t start, uint32_t size, const std::vector<std::pair<uint32_t,uint32_t>>& parent_triangles,
                                                       auto find_triangle_cube_relation)
{
    //count triangles from the current depth
    uint32_t exact_count=0;
    for (uint32_t i=0;i<size && BoxStack{.m_data = ic.num_id[start+i].number}.depth() == stack.depth(); ++i)
    {
        ++exact_count;
    }

    if (exact_count==size)
    {
        std::vector<uint32_t> leaf_prims;
        leaf_prims.reserve(size);
        for (uint32_t i = 0; i < size; ++i)
        {
            leaf_prims.push_back(ic.num_id[start + i].id);
        }

        for (auto [p_start, p_size] : parent_triangles)
        {
            for (auto i = p_start; i < p_start + p_size; ++i)
            {
                auto t_id = ic.num_id[i].id;
                glm::vec3 v0, v1, v2, n0, n1, n2;
                get_primitive_triangle_vertices(primitives.view(), vertices.view(), t_id, v0, v1, v2, n0, n1, n2);
                auto rel = find_triangle_cube_relation(stack.get_center(), stack.half_box_size(), v0, v1, v2, n0, n1, n2);
                if (rel != TriangleCubeRelation::Outside)
                {
                    leaf_prims.push_back(t_id);
                }
            }
        }

        if (!leaf_prims.empty())
        {
            add_box_contents(ic, box_id, leaf_prims);
        }
    }
    else
    {
        auto next_parent_triangles=parent_triangles;
        if (exact_count>0)
            next_parent_triangles.emplace_back(start,exact_count);

        start+=exact_count;
        size-=exact_count;
        if (ic.__as[box_id]==0)
        {
            divide_box(ic, {stack.get_center(), stack.depth(),static_cast<uint32_t>(box_id),stack.box_size()});
        }
        assert(ic.__as[box_id]!=0);

        for (int branch=0; branch<8; ++branch)
        {
            stack.descend(branch);
            uint64_t end=stack.end();
            uint32_t begin_pos=start;
            uint32_t end_pos=start+size;
            while (begin_pos < end_pos)
            {
                auto middle_pos = (begin_pos + end_pos) / 2;
                auto number = ic.num_id[middle_pos].number;
                if (number <= end)
                {
                    begin_pos = middle_pos + 1;
                }
                else
                {
                    end_pos = middle_pos;
                }
            }

            if (end_pos>start || !next_parent_triangles.empty())
                insert_triangles_recursive(ic.__as[box_id]+branch, stack,ic,start,end_pos-start, next_parent_triangles, find_triangle_cube_relation);
            size-= (end_pos-start);
            start=end_pos;

            stack.ascend();
        }
    }
}

void OctreeAS::free_box(int box_id)
{
    free_box_content(box_id);
    as.free(box_id);
}


void OctreeAS::free_box_content(int box_id)
{
    auto content = as[box_id];
    if (content > 0)
    {
        for (int branch = 0; branch < 8; ++branch)
        {
            free_box_content(content + branch);
        }
        as.free(content);
    }
    else if (content < 0)
    {
        box_content.free(-content);
    }
}

void OctreeAS::sort_space(uint32_t space_id, soa::cuda_device_vector<Triangle>& d_temp_triangles)
{
    auto triangles_size = static_cast<uint32_t>(d_temp_triangles.size());
    if (triangles_size == 0 || spaces[space_id].vertices_size == 0)
        return;

    auto triangles_range = d_temp_triangles.view();

    thrust::device_vector<uint64_t> morton_codes(triangles_size, thrust::no_init);
    calculate_morton_codes<<<(triangles_size - 1) / 128 + 1, 128>>>(d_vertices.view(), triangles_range, morton_codes.data().get(), triangles_size);
    auto t = d_temp_triangles.get_tuple();
    thrust::sort_by_key(thrust::device, morton_codes.begin(), morton_codes.end(), thrust::zip_iterator{t});

    // free memory
    morton_codes.resize(0);
    morton_codes.shrink_to_fit();

    auto vertices_start = spaces[space_id].vertices_start;
    auto vertices_size = spaces[space_id].vertices_size;

    auto map = sort_by_first_appearance_in_triangles(triangles_range,
        triangles_size, vertices_start, vertices_size,
        thrust::make_zip_iterator(d_vertices.get_tuple(vertices_start)));

    remap_triangle_vertices<<<(triangles_size - 1) / 256 + 1, 256>>>(triangles_range, map.data().get(), vertices_start, triangles_size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void pack_triangles_kernel(
    soa::soa_view<Triangle> src_triangles,
    uint32_t triangles_count,
    soa::soa_view<PrimitiveChunk16> dst_primitives,
    uint32_t primitives_start)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= triangles_count) return;

    Triangle tri = src_triangles[i];
    tri.type = PrimitiveType::Triangle;
    store_primitive(dst_primitives, primitives_start + i, tri);
}

__global__ void pack_full_triangles_kernel(
    soa::soa_view<Triangle> src_triangles,
    soa::soa_view<Vertex> vertices,
    uint32_t triangles_count,
    soa::soa_view<PrimitiveChunk16> dst_primitives,
    uint32_t primitives_start)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= triangles_count) return;

    Triangle tri = src_triangles[i];
    FullTriangle ftri{};
    ftri.type = PrimitiveType::FullTriangle;
    ftri.material_id = tri.material_id;

    glm::vec3 uv0 = vertices[tri.v0].uv;
    glm::vec3 uv1 = vertices[tri.v1].uv;
    glm::vec3 uv2 = vertices[tri.v2].uv;
    ftri.uv_0 = glm::packHalf2x16(glm::vec2(uv0.x, uv0.y));
    ftri.uv_1 = glm::packHalf2x16(glm::vec2(uv1.x, uv1.y));
    ftri.uv_2 = glm::packHalf2x16(glm::vec2(uv2.x, uv2.y));

    ftri.v0 = vertices[tri.v0].position;
    ftri.n0_compressed.pack(vertices[tri.v0].normal);

    ftri.v1 = vertices[tri.v1].position;
    ftri.n1_compressed.pack(vertices[tri.v1].normal);

    ftri.v2 = vertices[tri.v2].position;
    ftri.n2_compressed.pack(vertices[tri.v2].normal);

    uint32_t dst_slot = primitives_start + i * FullTriangle::occupied_slots;
    store_primitive(dst_primitives, dst_slot, ftri);
}

__global__ void pack_nagata_patches_kernel(
    soa::soa_view<Triangle> src_triangles,
    soa::soa_view<Vertex> vertices,
    uint32_t triangles_count,
    soa::soa_view<PrimitiveChunk16> dst_primitives,
    uint32_t primitives_start)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= triangles_count) return;

    Triangle tri = src_triangles[i];
    glm::vec3 v0 = vertices[tri.v0].position;
    glm::vec3 v1 = vertices[tri.v1].position;
    glm::vec3 v2 = vertices[tri.v2].position;
    glm::vec3 n0 = vertices[tri.v0].normal;
    glm::vec3 n1 = vertices[tri.v1].normal;
    glm::vec3 n2 = vertices[tri.v2].normal;

    NagataPatch patch = NagataPatch::fromTriangle(v0, v1, v2, n0, n1, n2);

    NagataPatchPrimitive prim{};
    prim.type = PrimitiveType::NagataPatch;
    prim.material_id = tri.material_id;
    prim.X1 = patch.X1;
    prim.X2 = patch.X2;
    prim.X3 = patch.X3;
    prim.C12 = patch.C12;
    prim.C23 = patch.C23;
    prim.C31 = patch.C31;

    glm::vec3 uv0 = vertices[tri.v0].uv;
    glm::vec3 uv1 = vertices[tri.v1].uv;
    glm::vec3 uv2 = vertices[tri.v2].uv;
    prim.uv_0 = glm::packHalf2x16(glm::vec2(uv0.x, uv0.y));
    prim.uv_1 = glm::packHalf2x16(glm::vec2(uv1.x, uv1.y));
    prim.uv_2 = glm::packHalf2x16(glm::vec2(uv2.x, uv2.y));

    uint32_t dst_slot = primitives_start + i * NagataPatchPrimitive::occupied_slots;
    store_primitive(dst_primitives, dst_slot, prim);
}

void OctreeAS::pack_primitives(uint32_t space_id, const soa::cuda_device_vector<Triangle>& d_temp_triangles, uint32_t triangles_count)
{
    if (triangles_count == 0)
        return;

    if (config.triangle_type == PrimitiveType::FullTriangle)
    {
        uint32_t primitives_size = triangles_count * FullTriangle::occupied_slots;
        auto primitives_start = primitives.alloc(primitives_size);
        spaces[space_id].primitives_start = primitives_start;
        spaces[space_id].primitives_size = primitives_size;

        if (d_primitives.size() < primitives.size())
            d_primitives.resize(primitives.size());

        uint32_t blocks = (triangles_count - 1) / 256 + 1;
        pack_full_triangles_kernel<<<blocks, 256>>>(
            const_cast<soa::cuda_device_vector<Triangle>&>(d_temp_triangles).view(),
            d_vertices.view(),
            triangles_count,
            d_primitives.view(),
            primitives_start);
        CUDA_CHECK(cudaDeviceSynchronize());

        primitives.range(primitives_start) = d_primitives.range(primitives_start, primitives_size);

        // Free vertices for this space since FullTriangle embeds all vertex attributes inline
        if (spaces[space_id].vertices_size > 0)
        {
            vertices.free(spaces[space_id].vertices_start);
            spaces[space_id].vertices_start = 0;
            spaces[space_id].vertices_size = 0;
            d_vertices = vertices;
        }
    }
    else if (config.triangle_type == PrimitiveType::NagataPatch)
    {
        uint32_t primitives_size = triangles_count * NagataPatchPrimitive::occupied_slots;
        auto primitives_start = primitives.alloc(primitives_size);
        spaces[space_id].primitives_start = primitives_start;
        spaces[space_id].primitives_size = primitives_size;

        if (d_primitives.size() < primitives.size())
            d_primitives.resize(primitives.size());

        uint32_t blocks = (triangles_count - 1) / 256 + 1;
        pack_nagata_patches_kernel<<<blocks, 256>>>(
            const_cast<soa::cuda_device_vector<Triangle>&>(d_temp_triangles).view(),
            d_vertices.view(),
            triangles_count,
            d_primitives.view(),
            primitives_start);
        CUDA_CHECK(cudaDeviceSynchronize());

        primitives.range(primitives_start) = d_primitives.range(primitives_start, primitives_size);

        // Free vertices for this space since NagataPatch embeds all vertex attributes inline
        if (spaces[space_id].vertices_size > 0)
        {
            vertices.free(spaces[space_id].vertices_start);
            spaces[space_id].vertices_start = 0;
            spaces[space_id].vertices_size = 0;
            d_vertices = vertices;
        }
    }
    else
    {
        uint32_t primitives_size = triangles_count * Triangle::occupied_slots;
        auto primitives_start = primitives.alloc(primitives_size);
        spaces[space_id].primitives_start = primitives_start;
        spaces[space_id].primitives_size = primitives_size;

        if (d_primitives.size() < primitives.size())
            d_primitives.resize(primitives.size());

        uint32_t blocks = (triangles_count - 1) / 256 + 1;
        pack_triangles_kernel<<<blocks, 256>>>(
            const_cast<soa::cuda_device_vector<Triangle>&>(d_temp_triangles).view(),
            triangles_count,
            d_primitives.view(),
            primitives_start);
        CUDA_CHECK(cudaDeviceSynchronize());

        primitives.range(primitives_start) = d_primitives.range(primitives_start, primitives_size);
        static_cast<soa::soa_vector<Vertex>&>(vertices) = d_vertices;
    }
}
