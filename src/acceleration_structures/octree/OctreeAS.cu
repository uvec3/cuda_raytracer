#include "OctreeAS.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "../../device_algorithms.cuh"

struct InsertionContext
{
    int depth_min;
    int depth_max;
    float triangle_box_stop_ratio;
    soa::soa_view<TriangleNumberId> num_id;

    soa::soa_vector<Box> __as;
    soa::soa_vector<BoxContent> __box_content;
    soa::soa_vector<Primitive> __primitives;
};

OctreeAS::OctreeAS()
{
    //reserve element 0 for empty value and don't use it
    as.alloc(1);
    box_content.alloc(1);
}

uint32_t OctreeAS::create_space()
{
    auto new_space_id=spaces.alloc_element();
    auto new_space=spaces[new_space_id];
    auto root_box_id=as.alloc_element();
    auto root_box=as[root_box_id];

    root_box.child=0;
    root_box.space_id=new_space_id;
    root_box.parent=0;

    new_space.root_box =new_space_id;
    new_space.first_triangle =0;
    new_space.number_of_triangles =0;
    new_space.vertices_start =0;
    new_space.vertices_size =0;
    new_space.normalization_scale =glm::vec3(1);
    new_space.normalization_translate =glm::vec3(0);

    return new_space_id;
}

void OctreeAS::free_space(uint32_t space_id)
{
    auto root_box = spaces[space_id].root_box;
    // free_box(root_box);

    auto triangles_start = spaces[space_id].first_triangle;
    auto triangles_size = spaces[space_id].number_of_triangles;
    if (triangles_size > 0)
        triangles.free(triangles_start);

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
    auto box = insertion_context.__as[box_info.id];
    auto content = box.child;
    const int32_t child_start = insertion_context.__as.allocate(9);
    for (int i = 0; i < 9; ++i)
    {
        auto new_box = insertion_context.__as[child_start + i];
        new_box.child = 0;
        new_box.parent = box_info.id;
        new_box.space_id = insertion_context.__as[box_info.id].space_id;
    }
    insertion_context.__as[child_start + 8].child = static_cast<int>(box_info.id);//place parent id as 9-th element

    if (content > 0)
        throw std::runtime_error("ain't suppose to divide if already split");

    if (content < 0)//pass data to child if has
    {
        exit(1);
    }

    insertion_context.__as[box_info.id].child = child_start;
    return child_start;
}


void OctreeAS::add_box_contents(InsertionContext& ic, uint32_t box_id, PrimitiveType type, std::span<uint32_t> primitives)
{
    if (primitives.empty())
        return;

    auto content_id = ic.__as[box_id].child;
    if (content_id > 0)
        throw std::runtime_error("can't add content");
    if (content_id == 0)//if empty
    {
        content_id = -ic.__box_content.allocate(1);
        ic.__as[box_id].child = content_id;
        ic.__box_content[-content_id].number_of_primitives = 0;
    }

    auto&& content = ic.__box_content[-content_id];
    auto&& start = content.start;
    auto&& number_of_primitives = content.number_of_primitives;

    if (number_of_primitives == 0)
    {
        start = ic.__primitives.allocate(primitives.size());
    }
    else
    {
        exit(1);
    }

    for (int i = 0; i < primitives.size(); ++i)
    {
        auto new_primitive = ic.__primitives[start + number_of_primitives + i];
        new_primitive.type = type;
        new_primitive.id = primitives[i];
    }

    number_of_primitives += primitives.size();
}



void OctreeAS::normalize_space(uint32_t space_id,bool stretch)
{
    auto vertices_start = &vertices[spaces[space_id].vertices_start].position;
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


#define STACK_SIZE 2048
__global__
void evaluate_triangle_box(soa::soa_view<Triangle> triangles, soa::soa_view<Vertex> vertices, uint32_t first_triangle,
    uint32_t number_of_triangles,  soa::soa_view<TriangleNumberId> result, uint32_t* result_counter, int DEPTH_MIN, int DEPTH_MAX, float TRIANGLE_BOX_STOP_RATIO)
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
    uint32_t t_id=first_triangle+block_first+threadIdx.x;
    BoxStack box{.m_data = 0};

    if (t_id<first_triangle+number_of_triangles)
    {
        auto v1=vertices[triangles[t_id].v1].position;
        auto v2=vertices[triangles[t_id].v2].position;
        auto v3=vertices[triangles[t_id].v3].position;
        auto rel=find_triangle_cube_relation_precise(box.get_center(),box.half_box_size(),v1,v2,v3);

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
            bool descend=box.depth()<DEPTH_MAX;
            if (descend && box.depth()>=DEPTH_MIN)
            {
                auto v0=vertices[triangles[t_id].v1].position;
                auto v1=vertices[triangles[t_id].v2].position;
                auto v2=vertices[triangles[t_id].v3].position;

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
                auto v0=vertices[triangles[t_id].v1].position;
                auto v1=vertices[triangles[t_id].v2].position;
                auto v2=vertices[triangles[t_id].v3].position;

                box.descend(0);
                for (int branch=0;branch<8;++branch)
                {
                    box.move(branch);
                    auto rel=find_triangle_cube_relation_precise(box.get_center(),box.half_box_size(),v0,v1,v2);
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


void OctreeAS::insert_triangles(uint32_t space_id, uint32_t triangle_first, uint32_t number_of_triangles,int depth_min, int depth_max, float triangle_box_stop_ratio)
{
    static constexpr uint32_t block_size=128;
    uint32_t number_of_blocks= (number_of_triangles-1) / block_size + 1;

    soa::cuda_unified_vector<TriangleNumberId> number_id(number_of_triangles*100);

    uint32_t* result_size;cudaMallocManaged(&result_size,sizeof(uint32_t));
    *result_size=0;

    evaluate_triangle_box<<<number_of_blocks,block_size>>>(triangles.view(),vertices.view(),triangle_first,
       number_of_triangles,number_id.view(),result_size,depth_min,depth_max,triangle_box_stop_ratio);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n\nElemetnts: %d\n",*result_size);

    thrust::sort_by_key(&number_id[0].number,&number_id[0].number + *result_size,&number_id[0].id);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n\nElemetnts: %d\n",*result_size);
    printf("Nubmer of triangles: %d\n",number_of_triangles);
    
    InsertionContext insertion_context;
    insertion_context.num_id=number_id;
    insertion_context.depth_min=depth_min;
    insertion_context.depth_max=depth_max;
    insertion_context.triangle_box_stop_ratio=triangle_box_stop_ratio;
    insertion_context.__as.allocate(1);
    insertion_context.__box_content.allocate(1);
    insertion_context.__as[0].child=0;
    insertion_context.__as[0].space_id=space_id;
    insertion_context.__as[0].parent=0;

    insert_triangles_recursive(0, BoxStack{}, insertion_context,0,*result_size, {});


    auto as_start=as.alloc(insertion_context.__as.size());
    auto box_content_start=box_content.alloc(insertion_context.__box_content.size());
    auto primitives_start=primitives.alloc(insertion_context.__primitives.size());

    spaces[space_id].root_box=as_start;
    spaces[space_id].as_start=as_start;
    spaces[space_id].as_size=insertion_context.__as.size();
    spaces[space_id].box_content_start=box_content_start;
    spaces[space_id].box_content_size=insertion_context.__box_content.size();
    spaces[space_id].primitives_start=primitives_start;
    spaces[space_id].primitives_size=insertion_context.__primitives.size();

    for (int i=0;i<insertion_context.__as.size();++i)
    {
        auto&& child= insertion_context.__as[i].child;
        if (child>0)//interior
            child+=as_start;
        else if (child<0)//leaf
            child-=box_content_start;
        as[as_start+i]=insertion_context.__as[i];
    }

    for (int i=0;i<insertion_context.__box_content.size();++i)
    {
        insertion_context.__box_content[i].start+=primitives_start;
        box_content[box_content_start+i]=insertion_context.__box_content[i];
    }

    for (int i=0;i<insertion_context.__primitives.size();++i)
    {
        primitives[primitives_start+i]=insertion_context.__primitives[i];
    }

    cudaFree(result_size);
}


void OctreeAS::insert_triangles_recursive(int box_id, BoxStack stack, InsertionContext& ic,
                                                       uint32_t start, uint32_t size,const std::vector<std::pair<uint32_t,uint32_t>>& parent_triangles={})
{
    auto depth=stack.depth();

    //count triangles from the current depth
    uint32_t exact_count=0;
    for (int i=0;i<size && BoxStack{.m_data = ic.num_id[start+i].number}.depth() == stack.depth(); ++i)
    {
        ++exact_count;
    }

    if (exact_count==size)
    {
        add_box_contents(ic,box_id,PrimitiveType::Triangle, std::span<uint32_t>(&ic.num_id[0].id+start,&ic.num_id[0].id+start+size));


        std::vector<uint32_t> additional_unsorted_triangles;
        for (auto [start,size]:parent_triangles)
        {
            for (auto i=start;i<start+size;++i)
            {
                auto t_id=ic.num_id[i].id;
                auto v0=vertices[triangles[t_id].v1].position;
                auto v1=vertices[triangles[t_id].v2].position;
                auto v2=vertices[triangles[t_id].v3].position;
                auto rel=find_triangle_cube_relation_precise(stack.get_center(),stack.half_box_size(),v0,v1,v2);
                if (true)
                {
                    additional_unsorted_triangles.push_back(t_id);
                }
            }
        }
        if (!additional_unsorted_triangles.empty())
        {
            auto content_id=- ic.__as[box_id].child;
            if (content_id==0)
            {
                add_box_contents(ic,box_id,PrimitiveType::Triangle, additional_unsorted_triangles);
            }
            else
            {
                auto&& box_content=ic.__box_content[content_id];
                box_content.number_of_primitives+=additional_unsorted_triangles.size();
                auto start_prim=ic.__primitives.allocate(additional_unsorted_triangles.size());
                for (int i=0;i< additional_unsorted_triangles.size();++i)
                {
                    ic.__primitives[start_prim+i].id=additional_unsorted_triangles[i];
                    ic.__primitives[start_prim+i].type=PrimitiveType::Triangle;
                }
            }
        }
    }
    else
    {
        auto next_parent_triangles=parent_triangles;
        if (exact_count>0)
            next_parent_triangles.emplace_back(start,exact_count);

        start+=exact_count;
        size-=exact_count;
        if (ic.__as[box_id].child==0)
        {
            divide_box(ic, {stack.get_center(), stack.depth(),static_cast<uint32_t>(box_id),stack.box_size()});
        }
        assert(ic.__as[box_id].child!=0);

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
                insert_triangles_recursive(ic.__as[box_id].child+branch, stack,ic,start,end_pos-start, next_parent_triangles);
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
    auto content = as[box_id].child;
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
        auto primitives_start = box_content[-content].start;
        auto primitives_size = box_content[-content].number_of_primitives;
        if (primitives_size > 0)
            primitives.free(primitives_start);
        box_content.free(-content);
    }
}


