#pragma once

#include "../../raytracer/rt_shaders.cuh"

struct OctreeASDeviceView
{
    soa::soa_view<int> as;
    soa::soa_view<BoxContent> box_content;

    template <bool HasNagata = true, typename RenderParams>
    __host__ __device__ inline glm::vec4 first_hit(const RenderParams& p, glm::vec3 source, glm::vec3 direction) const;
};

struct TriangleNumberId
{
    uint64_t& number;
    uint32_t& id;
};

struct Box
{
    uint32_t is_leaf:1;
    uint32_t index: 31;
};


#define BOX_SIZE 1


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
    float t_min=0.0f;
    float t_max=0.0f;
};

struct TraversalResult
{
    uint32_t content;
    BoxStack stack;
};


__device__ __host__
inline TraversalResult traverse_stackless(const int* as, CurrentBox& current_box, glm::vec3 pos, glm::vec3& current_pos, glm::vec3 dir, int& iterations)
{
    constexpr float eps = 0x1p-21f;
    glm::vec3 inv_dir = 1.0f / dir;
    glm::vec3 step_dir = glm::mix(glm::vec3(-1.0f), glm::vec3(1.0f), glm::greaterThanEqual(dir, glm::vec3(0.0f)));

    glm::ivec3 center_delta;
    glm::vec3 center;
    int box_content=as[current_box.id];
    if (current_box.stack.depth()==0)//first entrance
    {
        if (box_content<=0)
        {
            return TraversalResult{.content =(uint32_t) -box_content,.stack=current_box.stack};
        }

        current_pos=pos;
        if (!correct_ray_pos(current_pos,dir,BOX_SIZE))
        {
            //no way of hitting anything inside the region
            return  {0,0};
        }
        current_pos = glm::clamp(current_pos, glm::vec3(eps), glm::vec3(BOX_SIZE - eps));

        center=glm::vec3(0.5,0.5,0.5);
    }
    else//not a root box (continue to traverse)
    {
        center = current_box.stack.get_center();
        center_delta = int_to_id_vec(current_box.stack.branch())*2-1;
    }


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
            glm::ivec3 child_box = glm::greaterThanEqual(current_pos, center);

            //move center to child box center
            center_delta = child_box * 2 - 1;
            center = center + glm::vec3(center_delta) * current_box.stack.quarter_box_size();
            //update indices
            auto child_box_offset = id_vec_to_int(child_box);

            current_box.id=box_content+child_box_offset;
            box_content=as[current_box.id];

            current_box.stack.descend(child_box_offset);
        }



        //==========================================================================================
        //--------------------------------------FIND EXIT DIRECTION---------------------------------
        //
        //==========================================================================================
        float leaf_half_size = current_box.stack.half_box_size();
        glm::vec3 leaf_center = center;
        glm::vec3 exit_plane = leaf_center + step_dir * leaf_half_size;
        glm::vec3 t_max = (exit_plane - pos) * inv_dir;
        // Strictly pick a SINGLE exit axis (smallest t_max)
        glm::ivec3 move_dir(0);
        float exit_t;
        if (t_max.x <= t_max.y && t_max.x <= t_max.z)
        {
            move_dir.x = static_cast<int>(step_dir.x);
            exit_t = t_max.x;
        }
        else if (t_max.y <= t_max.z)
        {
            move_dir.y = static_cast<int>(step_dir.y);
            exit_t = t_max.y;
        }
        else
        {
            move_dir.z = static_cast<int>(step_dir.z);
            exit_t = t_max.z;
        }
        // Never allow a negative step (if pos slightly overshot exit_plane by 1 ULP, step = 0)
        current_box.t_min=current_box.t_max;
        current_box.t_max = glm::max(current_box.t_max, exit_t);

        //==========================================================================================
        //----------------------------------------ASCEND--------------------------------------------
        //
        //==========================================================================================

        while (glm::any(glm::equal(center_delta, move_dir)))
        {//get out
            current_box.id=as[current_box.id+8-current_box.stack.branch()];//element number 8 is parent
            //pop from stack
            current_box.stack.ascend();

            if (current_box.stack.depth()==0)
            {
                if (box_content<0)
                    return TraversalResult{.content =(uint32_t) -box_content,.stack = current_box.stack};
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

        // // //update ray start point(pos)
        // auto flat_point=center+glm::vec3(move_dir)*current_box.stack.half_box_size();
        // current_box.step=ray_flat_intersection_offset(flat_point, move_dir, pos, dir);

        float shrink = leaf_half_size - eps;
        glm::vec3 clamped = glm::clamp(pos + dir * current_box.t_max, leaf_center - shrink, leaf_center + shrink);
        glm::vec3 snapped = exit_plane + glm::vec3(move_dir) * eps;
        current_pos = glm::mix(clamped, snapped, glm::notEqual(move_dir, glm::ivec3(0)));

        if (box_content<0)
            return TraversalResult{.content =(uint32_t) -box_content,.stack = current_box.stack};

        box_content=as[current_box.id];

        //move center
        center=center+glm::vec3(move_dir)*current_box.stack.box_size();
        center_delta=center_delta+move_dir*2;
    }
}

__device__ __host__
inline TraversalResult traverse_stackless(const int* as, CurrentBox& current_box, glm::vec3 pos, glm::vec3 dir, int& iterations)
{
    glm::vec3 current_pos = pos;
    if (current_box.stack.depth() > 0)
    {
        constexpr float eps = 0x1p-21f;
        const glm::vec3 c = current_box.stack.get_center();
        const float shrink = current_box.stack.half_box_size() - eps;
        current_pos = glm::clamp(pos + dir * current_box.t_max, c - shrink, c + shrink);
    }
    return traverse_stackless(as, current_box, pos, current_pos, dir, iterations);
}

template <bool HasNagata, typename RenderParams>
__host__ __device__ inline
glm::vec4 OctreeASDeviceView::first_hit(const RenderParams& p, glm::vec3 source, glm::vec3 direction) const
{
    static constexpr int max_iterations=2000;
    //counts how many boxes were hit on the way
    int box_hits=0;
    //records triangles
    glm::vec4 hit_primitive_color{0};

    //current container and box
    int32_t container_id=INT32_MAX;
    CurrentBox current_box{.id = p.root_box};

    //count the number of iterations to prevent infinite or too long propagations
    //one iteration is a one horizontal step through octree
    int it=0;
    bool continue_traversal=true;
    while (container_id!=0 && continue_traversal && it<max_iterations)
    {
        //AS TRAVERSAL STEP (returns after hitting non-empty leaf node)
        TraversalResult result=traverse_stackless(&as[0], current_box, source, direction, it);
        container_id=result.content;

        if (container_id>0)
        {
            uint32_t closest_triangle=UINT32_MAX;
            glm::vec3 closest_tuv{INFINITY,0,0};
            ++box_hits;

            if (p.show_triangles)
            {
                intersect_box_primitives<HasNagata>(
                    box_content,
                    static_cast<uint32_t>(container_id),
                    p,
                    source,
                    direction,
                    current_box.t_min,
                    current_box.t_max,
                    closest_triangle,
                    closest_tuv);
            }

            if (closest_triangle!=UINT32_MAX)
            {
                auto res=fragment_hit_shader<HasNagata>(closest_triangle,source,direction,closest_tuv,p,continue_traversal);
                hit_primitive_color=blend_colors(hit_primitive_color,res);
                if (continue_traversal)
                {
                    current_box.t_min=1e-4f;
                    current_box.t_max=1e-4f;
                }
            }
        }

        if (current_box.stack.depth()==0)//even though the content was return the stack depth 0 means that that was the last box
        {
            break;
        }
    }

    if (it==max_iterations)//return white to indicate that the loop was interrupted
        return glm::vec4{1.f};

    if (!p.show_iterations && !p.show_boxes)
    {
        if (current_box.stack.depth()==0)//call miss shader
            hit_primitive_color=blend_colors(hit_primitive_color,miss_shader(source,direction,p));
        return hit_primitive_color;
    }

    return compose_debug_overlay(p, hit_primitive_color, it, box_hits);
}
