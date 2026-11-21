#pragma once

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include "../../external/soa_vector/src/soa_vector.hpp"
#include <glm/glm.hpp>
#include "../types.hpp"
#include "../types/NagataPatch.cuh"
#include "../geometry_common.cuh"

//returns whether to continue tracing the ray
template <bool HasNagata = true, bool TransformNormal = false, typename RenderParams>
__host__ __device__ inline
glm::vec4 fragment_hit_shader_impl(
    uint32_t primitive_id,
    glm::vec3& ray_src,
    glm::vec3& ray_dir,
    glm::vec3 tuv,
    const RenderParams& p,
    glm::vec4 s2l_row0,
    glm::vec4 s2l_row1,
    glm::vec4 s2l_row2,
    glm::vec3 blas_scale,
    bool& continue_traversal)
{
    using namespace glm;

    float w0 = 1.0f - tuv[1] - tuv[2];
    float w1 = tuv[1];
    float w2 = tuv[2];

    PrimitiveChunk16 chunk0 = p.primitives[primitive_id];
    uint32_t mat_id = chunk0.material_id();
    glm::vec3 normal(0.0f, 0.0f, 1.0f);
    glm::vec2 uv_interp(0.0f);

    if (chunk0.type() == PrimitiveType::Triangle)
    {
        Triangle tri = chunk0.as<Triangle>();

        if (p.use_fake_normals)
        {
            glm::vec3 n0 = glm::vec3(p.vertices[tri.v0].normal);
            glm::vec3 n1 = glm::vec3(p.vertices[tri.v1].normal);
            glm::vec3 n2 = glm::vec3(p.vertices[tri.v2].normal);
            normal = normalize(n0 * w0 + n1 * w1 + n2 * w2);
        }
        else
        {
            glm::vec3 v0 = glm::vec3(p.vertices[tri.v0].position);
            glm::vec3 v1 = glm::vec3(p.vertices[tri.v1].position);
            glm::vec3 v2 = glm::vec3(p.vertices[tri.v2].position);
            glm::vec3 n = glm::cross(v1 - v0, v2 - v0);
            if constexpr (TransformNormal)
                n *= blas_scale;
            normal = glm::normalize(n);
        }

        auto uv0 = glm::vec2(p.vertices[tri.v0].uv);
        auto uv1 = glm::vec2(p.vertices[tri.v1].uv);
        auto uv2 = glm::vec2(p.vertices[tri.v2].uv);
        uv_interp = uv0 * w0 + uv1 * w1 + uv2 * w2;
    }
    else if (chunk0.type() == PrimitiveType::FullTriangle)
    {
        InlinedTriangle ftri = load_primitive<InlinedTriangle>(p.primitives, primitive_id, chunk0);

        if (p.use_fake_normals)
        {
            auto n0 = ftri.n0_compressed.unpack();
            auto n1 = ftri.n1_compressed.unpack();
            auto n2 = ftri.n2_compressed.unpack();
            normal = normalize(n0 * w0 + n1 * w1 + n2 * w2);
        }
        else
        {
            glm::vec3 n = glm::cross(ftri.v1 - ftri.v0, ftri.v2 - ftri.v0);
            if constexpr (TransformNormal)
                n *= blas_scale;
            normal = glm::normalize(n);
        }

        glm::vec2 uv0 = glm::unpackHalf2x16(ftri.uv_0);
        glm::vec2 uv1 = glm::unpackHalf2x16(ftri.uv_1);
        glm::vec2 uv2 = glm::unpackHalf2x16(ftri.uv_2);
        uv_interp = uv0 * w0 + uv1 * w1 + uv2 * w2;
    }
    else if constexpr (HasNagata)
    {
        if (chunk0.type() == PrimitiveType::NagataPatch)
        {
            NagataPatchPrimitive patch = load_primitive<NagataPatchPrimitive>(p.primitives, primitive_id, chunk0);

            if (p.use_fake_normals)
            {
                glm::vec3 n0 = glm::vec3(p.vertices[patch.v0].normal);
                glm::vec3 n1 = glm::vec3(p.vertices[patch.v1].normal);
                glm::vec3 n2 = glm::vec3(p.vertices[patch.v2].normal);
                normal = normalize(n0 * w0 + n1 * w1 + n2 * w2);
            }
            else
            {
                glm::vec3 v0 = glm::vec3(p.vertices[patch.v0].position);
                glm::vec3 v1 = glm::vec3(p.vertices[patch.v1].position);
                glm::vec3 v2 = glm::vec3(p.vertices[patch.v2].position);
                normal = evaluateNagataNormal2(v0,v1,v2, patch.C12,patch.C23,patch.C31,w0,w1);
                if constexpr (TransformNormal)
                    normal = glm::normalize(normal * blas_scale);
            }

            glm::vec2 uv0 = p.vertices[patch.v0].uv;
            glm::vec2 uv1 = p.vertices[patch.v1].uv;
            glm::vec2 uv2 = p.vertices[patch.v2].uv;
            uv_interp = uv0 * w0 + uv1 * w1 + uv2 * w2;
        }
    }

    if constexpr (TransformNormal)
    {
        const glm::vec3 n_scene =
            glm::vec3(s2l_row0) * normal.x +
            glm::vec3(s2l_row1) * normal.y +
            glm::vec3(s2l_row2) * normal.z;
        const float len2 = glm::dot(n_scene, n_scene);
        if (len2 > 1e-20f)
            normal = n_scene / sqrtf(len2);
    }

    vec3 intersection=ray_src+ray_dir*tuv[0];

    float diffuse;
    if (p.fixed_light_source)
    {
        auto l=p.light_source-intersection;
        auto dist=length(l);
        diffuse = clamp( dot(l/dist,normal),0.f,1.f);
    }
    else
    {
        diffuse = abs(dot(normalize(ray_dir),normal));
    }

    glm::vec4 base_color(1.0f);
    glm::vec3 emissive(0.0f);

    if (mat_id < p.materials.size())
    {
        auto&& mat = p.materials[mat_id];
        base_color = mat.baseColor;
        emissive = mat.emissiveFactor;

#if defined(__CUDA_ARCH__)//only use texture on gpu
        cudaTextureObject_t tex = mat.baseColorTexture;
        if (tex != 0)
        {
            float4 sampled = tex2D<float4>(tex, uv_interp.x, uv_interp.y);
            base_color *= glm::vec4(sampled.x, sampled.y, sampled.z, sampled.w);
        }
#endif
    }

    float lighting = diffuse * p.diffuse_light + p.ambient_light;
    glm::vec4 out_color = glm::vec4(glm::vec3(base_color) * lighting + emissive, base_color.a);

    //decide whether to refract and continue
    if (p.enableRefraction && p.materials[mat_id].transmissionFactor>0.f)
    {
        float eta = p.materials[mat_id].ior;

        //in
        if (glm::dot(ray_dir,normal) < 0.0f) {
            eta = 1/eta;

        }
        else//out
        {
            normal=-normal;
        }

        auto refracted=glm::refract(ray_dir,normal,eta);
        if (refracted!=glm::vec3(0))//refraction
        {
            ray_dir=refracted;
            ray_src=intersection;
            continue_traversal = true;
            out_color.a=1.f-p.materials[mat_id].transmissionFactor;
        }
        else
        {
            continue_traversal = false;
        }
    }
    else if (out_color.a<0.95f)//transparent
    {
        ray_src=intersection;
        continue_traversal=true;
    }
    else//opaque
    {
        continue_traversal=false;
    }

    return out_color;
}

template <bool HasNagata = true, typename RenderParams>
__host__ __device__ inline
glm::vec4 fragment_hit_shader(
    uint32_t primitive_id,
    glm::vec3& ray_src,
    glm::vec3& ray_dir,
    glm::vec3 tuv,
    const RenderParams& p,
    bool& continue_traversal)
{
    return fragment_hit_shader_impl<HasNagata, false>(
        primitive_id,
        ray_src,
        ray_dir,
        tuv,
        p,
        glm::vec4(1.0f, 0.0f, 0.0f, 0.0f),
        glm::vec4(0.0f, 1.0f, 0.0f, 0.0f),
        glm::vec4(0.0f, 0.0f, 1.0f, 0.0f),
        glm::vec3(1.0f),
        continue_traversal);
}

template <bool HasNagata = true, typename RenderParams>
__host__ __device__ inline
glm::vec4 fragment_hit_shader_instanced(
    uint32_t primitive_id,
    glm::vec3& ray_src,
    glm::vec3& ray_dir,
    glm::vec3 tuv,
    const RenderParams& p,
    glm::vec4 s2l_row0,
    glm::vec4 s2l_row1,
    glm::vec4 s2l_row2,
    glm::vec3 blas_scale,
    bool& continue_traversal)
{
    return fragment_hit_shader_impl<HasNagata, true>(
        primitive_id,
        ray_src,
        ray_dir,
        tuv,
        p,
        s2l_row0,
        s2l_row1,
        s2l_row2,
        blas_scale,
        continue_traversal);
}

template <typename RenderParams>
__host__ __device__ inline
glm::vec4 miss_shader(glm::vec3 ray_src, glm::vec3 ray_dir, const RenderParams& p)
{
    if (p.skybox==0)
    {
        return p.background_color;
    }
    else
    {
        return glm::vec4{1.f};
    }
}

__host__ __device__ inline
glm::vec4 blend_colors(glm::vec4 current_color, glm::vec4 new_color)
{
    auto base=glm::vec3(current_color)*current_color.a + (1.0f -current_color.a)*glm::vec3(new_color);
    auto alpha = 1- (1-current_color.a)*(1-new_color.a);
    return glm::vec4(base,alpha);
}

template <bool HasNagata = true, bool ClipToBounds = true, typename RenderParams>
__host__ __device__ inline
void intersect_box_primitives(
    soa::soa_view<BoxContent> box_content,
    uint32_t container_id,
    const RenderParams& p,
    glm::vec3 source,
    glm::vec3 direction,
    float t_min,
    float t_max,
    uint32_t& closest_triangle,
    glm::vec3& closest_tuv)
{
    for (;; ++container_id)
    {
        auto primitive = box_content[container_id];
        uint32_t prim_id = primitive.id();
        PrimitiveChunk16 chunk0 = p.primitives[prim_id];
        if (chunk0.type() == PrimitiveType::Triangle)
        {
            Triangle tri = chunk0.as<Triangle>();

            glm::vec3 tuv;
            bool hit = intersectRayTriangle<ClipToBounds>(
                source, direction, t_min, t_max,
                glm::vec3(p.vertices[tri.v0].position), glm::vec3(p.vertices[tri.v1].position), glm::vec3(p.vertices[tri.v2].position),
                tuv.x, tuv.y, tuv.z);

            if (hit && tuv[0] < closest_tuv[0])
            {
                closest_tuv = tuv;
                closest_triangle = prim_id;
            }
        }
        else if (chunk0.type() == PrimitiveType::FullTriangle)
        {
            InlinedTriangle ftri = load_primitive<InlinedTriangle>(p.primitives, prim_id, chunk0);

            glm::vec3 tuv;
            bool hit = intersectRayTriangle<ClipToBounds>(
                source, direction, t_min, t_max,
                ftri.v0, ftri.v1, ftri.v2,
                tuv.x, tuv.y, tuv.z);

            if (hit && tuv[0] < closest_tuv[0])
            {
                closest_tuv = tuv;
                closest_triangle = prim_id;
            }
        }
        else if constexpr (HasNagata)
        {
            if (chunk0.type() == PrimitiveType::NagataPatch)
            {
                NagataPatchPrimitive patch = load_primitive<NagataPatchPrimitive>(p.primitives, prim_id, chunk0);

                glm::vec3 tuv;
                bool hit = intersectNagataPrecomputed(
                    source, direction, t_min, t_max,
                    glm::vec3(p.vertices[patch.v0].position), glm::vec3(p.vertices[patch.v1].position), glm::vec3(p.vertices[patch.v2].position),
                    patch.C12, patch.C23, patch.C31,
                    tuv.x, tuv.y, tuv.z);

                if (hit && tuv[0] < closest_tuv[0])
                {
                    closest_tuv = tuv;
                    closest_triangle = prim_id;
                }
            }
        }

        if (primitive.is_last())//if it is the last in chain
            break;
    }
}

template <typename RenderParams>
__host__ __device__ inline
glm::vec4 compose_debug_overlay(const RenderParams& p, glm::vec4 hit_primitive_color, int it, int box_hits)
{
    //composes the final color result: number of boxes - red, number of leafs - green
    float iter_div = p.iterations_max > 0.0f ? p.iterations_max : 1.0f;
    float box_div = p.boxes_max > 0.0f ? p.boxes_max : 1.0f;
    float iter_overlay = (float(it) / iter_div) * p.show_iterations;
    float box_overlay = (box_hits / box_div) * p.show_boxes;

    auto a = (hit_primitive_color.x + hit_primitive_color.y + hit_primitive_color.z) / 3;
    glm::vec3 base = glm::mix(glm::vec3(p.background_color), glm::vec3(0, 0, 1), a);
    base = glm::mix(base, glm::vec3(1, 0, 0), iter_overlay);
    base = glm::mix(base, glm::vec3(0, 1, 0), box_overlay);

    return glm::vec4(base, 1.0f);
}
