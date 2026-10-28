#pragma once


struct TriangleNumberId
{
    uint64_t& number;
    uint32_t& id;
};

#define BOX_SIZE 1

__host__ __device__
inline float get_quarter_block_size(int depth)
{
    return static_cast<float>(BOX_SIZE) / (1<<(depth + 2));
}

__host__ __device__
inline float get_half_block_size(int depth)
{
    return static_cast<float>(BOX_SIZE) / (1<<(depth + 1));
}

__host__ __device__
inline float get_block_size(int depth)
{
    return static_cast<float>(BOX_SIZE) / (1<<(depth));
}



__host__ __device__
inline float ray_flat_intersection_offset(glm::vec3 flat_point,glm::vec3 flat_normal,glm::vec3 ray_point, glm::vec3 ray_direction)
{
    auto distance_to_flat=glm::dot(flat_normal,flat_point-ray_point);
    auto cos=glm::dot(flat_normal, ray_direction);
    auto offset=distance_to_flat/cos;

    return offset;
}



__device__ __host__ inline
bool correct_ray_pos(glm::vec3& pos, glm::vec3 dir,float box_size)
{
    if (pos.x<0&&dir.x>0)
    {
        auto dx=-pos.x;
        pos+=dir*(dx/dir.x);
        pos.x=0;
    }

    if (pos.y<0&&dir.y>0)
    {
        auto dy=-pos.y;
        pos+=dir*(dy/dir.y);
        pos.y=0;
    }

    if (pos.z<0&&dir.z>0)
    {
        auto dz=-pos.z;
        pos+=dir*(dz/dir.z);
        pos.z=0;
    }

    if (pos.x>box_size&&dir.x<0)
    {
        auto dx=box_size-pos.x;
        pos+=dir*(dx/dir.x);
        pos.x=box_size;
    }

    if (pos.y>box_size&&dir.y<0)
    {
        auto dy=box_size-pos.y;
        pos+=dir*(dy/dir.y);
        pos.y=box_size;
    }

    if (pos.z>box_size&&dir.z<0)
    {
        auto dz=box_size-pos.z;
        pos+=dir*(dz/dir.z);
        pos.z=box_size;
    }

    if (pos.x<0||pos.x>box_size||
        pos.y<0||pos.y>box_size||
        pos.z<0||pos.z>box_size)
        return false;
    return  true;
}


struct CurrentBox
{
    int id;
    BoxStack stack;
    float step=0.f;
};

struct TraversalResult
{
    uint32_t content;
    BoxStack stack;
};


__device__ __host__
inline TraversalResult traverse_stackless(const int* as, CurrentBox& current_box, glm::vec3& pos, glm::vec3 dir,int& iterations)
{
    glm::ivec3 center_delta;
    glm::vec3 center;
    int box_content=as[current_box.id];
    if (current_box.stack.depth()==0)//first entrance
    {
        if (box_content<=0)
        {
            return TraversalResult{.content =(uint32_t) -box_content,.stack=current_box.stack};
        }

        if (!correct_ray_pos(pos,dir,BOX_SIZE))
        {
            //no way of hitting anything inside the region
            return  {0,0};
        }

        center=glm::vec3(0.5,0.5,0.5);
    }
    else//not a root box (continue to traverse)
    {
        center = current_box.stack.get_center();
        center_delta = int_to_id_vec(current_box.stack.branch())*2-1;
        pos=pos+dir*current_box.step;
    }


    BoxStack content_stack=current_box.stack;
    while (true)//traversal loop: one iteration is one step through next box on ray's way
    {
        ++iterations;
        //==========================================================================================
        //----------------------------------------DESCEND-------------------------------------------
        //
        //==========================================================================================
        while (box_content>0)
        {
            //determine into which child box to descend
            glm::ivec3 child_box = glm::greaterThanEqual(pos, center);

            //move center to child box center
            center_delta = child_box * 2 - 1;
            center = center + glm::vec3(center_delta) * current_box.stack.quarter_box_size();
            //update indices
            auto child_box_offset = id_vec_to_int(child_box);

            current_box.id=box_content+child_box_offset;
            box_content=as[current_box.id];

            current_box.stack.descend(child_box_offset);
            content_stack=current_box.stack;
        }


        //==========================================================================================
        //----------------------------------------ASCEND--------------------------------------------
        //
        //==========================================================================================

        glm::ivec3 move_dir = side(pos, dir, center, current_box.stack.box_size());
        while (glm::any(glm::equal(center_delta, move_dir)))
        {//get out
            current_box.id=as[current_box.id+8-current_box.stack.branch()];//element number 8 is parent
            //pop from stack
            current_box.stack.ascend();

            if (current_box.stack.depth()==0)
            {
                if (box_content<0)
                    return TraversalResult{.content =(uint32_t) -box_content,.stack = content_stack};
                return {0,0};
            }

            //move center
            center = center - glm::vec3(center_delta) * current_box.stack.quarter_box_size();
            center_delta= int_to_id_vec(current_box.stack.branch())*2-1;
        }

        //==========================================================================================
        //----------------------------------------SWITCH--------------------------------------------
        //at this point transition to the sibling node in move_dir MUST be possible
        //==========================================================================================

        //evaluate new child id
        auto offset=move_dir.x+move_dir.y*2+move_dir.z*4;
        current_box.id+=offset;


        current_box.stack.move(current_box.stack.branch()+offset);

        // //update ray start point(pos)
        auto flat_point=center+glm::vec3(move_dir)*current_box.stack.half_box_size();
        current_box.step=ray_flat_intersection_offset(flat_point, move_dir, pos, dir);

        if (box_content<0)
            return TraversalResult{.content =(uint32_t) -box_content,.stack = content_stack};

        pos=pos+dir*current_box.step;
        box_content=as[current_box.id];
        content_stack=current_box.stack;

        //move center
        center=center+glm::vec3(move_dir)*current_box.stack.box_size();
        center_delta=center_delta+move_dir*2;
    }
}