#include "KDTreeAS.cuh"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "imgui.h"
#include "../../cuda_common.hpp"

namespace kdtree_detail
{
    static constexpr int SAH_BINS = 16;

    __host__ __device__ inline PrimitiveType get_primitive_triangle_vertices(
        soa::soa_view<PrimitiveChunk16> primitives,
        soa::soa_view<Vertex> vertices,
        uint32_t prim_slot_id,
        glm::vec3& v0, glm::vec3& v1, glm::vec3& v2,
        glm::vec3& n0, glm::vec3& n1, glm::vec3& n2)
    {
        PrimitiveChunk16 chunk0 = primitives[prim_slot_id];
        PrimitiveType ptype = chunk0.type();
        if (ptype == PrimitiveType::Triangle)
        {
            Triangle tri = chunk0.as<Triangle>();
            v0 = glm::vec3(vertices[tri.v0].position);
            v1 = glm::vec3(vertices[tri.v1].position);
            v2 = glm::vec3(vertices[tri.v2].position);
            n0 = glm::vec3(vertices[tri.v0].normal);
            n1 = glm::vec3(vertices[tri.v1].normal);
            n2 = glm::vec3(vertices[tri.v2].normal);
        }
        else if (ptype == PrimitiveType::FullTriangle)
        {
            InlinedTriangle ftri = load_primitive<InlinedTriangle>(primitives, prim_slot_id, chunk0);
            v0 = ftri.v0;
            v1 = ftri.v1;
            v2 = ftri.v2;
            n0 = ftri.n0_compressed.unpack();
            n1 = ftri.n1_compressed.unpack();
            n2 = ftri.n2_compressed.unpack();
        }
        else if (ptype == PrimitiveType::NagataPatch)
        {
            NagataPatchPrimitive patch = load_primitive<NagataPatchPrimitive>(primitives, prim_slot_id, chunk0);
            v0 = glm::vec3(vertices[patch.v0].position);
            v1 = glm::vec3(vertices[patch.v1].position);
            v2 = glm::vec3(vertices[patch.v2].position);
            n0 = glm::vec3(vertices[patch.v0].normal);
            n1 = glm::vec3(vertices[patch.v1].normal);
            n2 = glm::vec3(vertices[patch.v2].normal);
        }
        else if (ptype == PrimitiveType::Subspace)
        {
            SubspacePrimitive sub = load_primitive<SubspacePrimitive>(primitives, prim_slot_id, chunk0);
            v0 = sub.bounds_min;
            v1 = sub.bounds_max;
            v2 = 0.5f * (sub.bounds_min + sub.bounds_max);
        }
        return ptype;
    }

    __host__ __device__ inline void get_primitive_aabb(
        PrimitiveType type,
        const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
        const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2,
        glm::vec3& out_min,
        glm::vec3& out_max)
    {
        if (type == PrimitiveType::NagataPatch)
        {
            NagataPatch::fromTriangle(v0, v1, v2, n0, n1, n2).computeBoundingBox(out_min, out_max);
        }
        else
        {
            out_min = glm::min(glm::min(v0, v1), v2);
            out_max = glm::max(glm::max(v0, v1), v2);
        }
    }

    __host__ __device__ inline void clip_polygon_axis(
        const glm::vec3* in_verts,
        int in_count,
        glm::vec3* out_verts,
        int& out_count,
        int axis,
        float plane_val,
        bool keep_greater)
    {
        out_count = 0;
        if (in_count <= 0)
            return;
        glm::vec3 prev = in_verts[in_count - 1];
        float prev_d = keep_greater ? (prev[axis] - plane_val) : (plane_val - prev[axis]);
        bool prev_in = (prev_d >= 0.0f);

        for (int i = 0; i < in_count; ++i)
        {
            const glm::vec3 curr = in_verts[i];
            const float curr_d = keep_greater ? (curr[axis] - plane_val) : (plane_val - curr[axis]);
            const bool curr_in = (curr_d >= 0.0f);

            if (curr_in != prev_in)
            {
                const float denom = prev_d - curr_d;
                const float raw_t = (fabsf(denom) > 1e-20f) ? (prev_d / denom) : 0.0f;
                const float t = raw_t < 0.0f ? 0.0f : (raw_t > 1.0f ? 1.0f : raw_t);
                glm::vec3 hit = prev + t * (curr - prev);
                hit[axis] = plane_val;
                if (out_count < 9)
                    out_verts[out_count++] = hit;
            }
            if (curr_in)
            {
                if (out_count < 9)
                    out_verts[out_count++] = curr;
            }
            prev = curr;
            prev_d = curr_d;
            prev_in = curr_in;
        }
    }

    __host__ __device__ inline bool clip_triangle_to_aabb(
        const glm::vec3& v0,
        const glm::vec3& v1,
        const glm::vec3& v2,
        const glm::vec3& clip_min,
        const glm::vec3& clip_max,
        glm::vec3& out_min,
        glm::vec3& out_max)
    {
        glm::vec3 poly_a[9] = {v0, v1, v2};
        glm::vec3 poly_b[9];
        int count = 3;

        for (int a = 0; a < 3; ++a)
        {
            int next_count = 0;
            clip_polygon_axis(poly_a, count, poly_b, next_count, a, clip_min[a], true);
            count = next_count;
            if (count == 0)
                return false;

            next_count = 0;
            clip_polygon_axis(poly_b, count, poly_a, next_count, a, clip_max[a], false);
            count = next_count;
            if (count == 0)
                return false;
        }

        glm::vec3 mn(FLT_MAX);
        glm::vec3 mx(-FLT_MAX);
        for (int i = 0; i < count; ++i)
        {
            mn = glm::min(mn, poly_a[i]);
            mx = glm::max(mx, poly_a[i]);
        }
        out_min = mn;
        out_max = mx;
        return true;
    }

    __host__ __device__ inline void get_clipped_primitive_aabb(
        PrimitiveType ptype,
        const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
        const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2,
        const glm::vec3& prim_min, const glm::vec3& prim_max,
        const glm::vec3& box_min, const glm::vec3& box_max,
        glm::vec3& out_cmin, glm::vec3& out_cmax)
    {
        const glm::vec3 pad = glm::max((box_max - box_min) * 1e-4f, glm::vec3(1e-6f));
        const glm::vec3 clamped_prim_min = glm::min(glm::max(prim_min, box_min), box_max);
        const glm::vec3 clamped_prim_max = glm::max(clamped_prim_min, glm::min(glm::max(prim_max, box_min), box_max));

        if (ptype == PrimitiveType::Subspace)
        {
            out_cmin = clamped_prim_min;
            out_cmax = clamped_prim_max;
            return;
        }

        if (ptype == PrimitiveType::NagataPatch)
        {
            const NagataPatch patch = NagataPatch::fromTriangle(v0, v1, v2, n0, n1, n2);
            const glm::vec3 curv_pad = 0.25f * (glm::abs(patch.C12) + glm::abs(patch.C23) + glm::abs(patch.C31));
            glm::vec3 base_min, base_max;
            if (clip_triangle_to_aabb(v0, v1, v2, box_min - curv_pad - pad, box_max + curv_pad + pad, base_min, base_max))
            {
                out_cmin = glm::min(glm::max(base_min - curv_pad, clamped_prim_min), box_max);
                out_cmax = glm::max(out_cmin, glm::min(base_max + curv_pad, clamped_prim_max));
            }
            else
            {
                out_cmin = clamped_prim_min;
                out_cmax = clamped_prim_max;
            }
            return;
        }

        glm::vec3 clipped_min, clipped_max;
        if (clip_triangle_to_aabb(v0, v1, v2, box_min - pad, box_max + pad, clipped_min, clipped_max))
        {
            out_cmin = glm::min(glm::max(clipped_min, box_min), box_max);
            out_cmax = glm::max(out_cmin, glm::min(glm::max(clipped_max, box_min), box_max));
        }
        else
        {
            out_cmin = clamped_prim_min;
            out_cmax = clamped_prim_max;
        }
    }

    __device__ __forceinline__ void atomic_min_float(float* addr, float val)
    {
        int* addr_as_i = reinterpret_cast<int*>(addr);
        int old = *addr_as_i;
        while (val < __int_as_float(old))
        {
            const int assumed = old;
            old = atomicCAS(addr_as_i, assumed, __float_as_int(val));
            if (old == assumed)
                break;
        }
    }

    __device__ __forceinline__ void atomic_max_float(float* addr, float val)
    {
        int* addr_as_i = reinterpret_cast<int*>(addr);
        int old = *addr_as_i;
        while (val > __int_as_float(old))
        {
            const int assumed = old;
            old = atomicCAS(addr_as_i, assumed, __float_as_int(val));
            if (old == assumed)
                break;
        }
    }

    __host__ __device__ __forceinline__ bool sat_edge_separated(
        const glm::vec3& a,
        const glm::vec3& b,
        const glm::vec3& e,
        const glm::vec3& H)
    {
        const glm::vec3 pa = glm::cross(a, e);
        const glm::vec3 pb = glm::cross(b, e);
        const glm::vec3 abs_e = glm::abs(e);
        const glm::vec3 R(
            H.y * abs_e.z + H.z * abs_e.y,
            H.z * abs_e.x + H.x * abs_e.z,
            H.x * abs_e.y + H.y * abs_e.x);
        return glm::any(glm::greaterThan(glm::min(pa, pb), R)) ||
               glm::any(glm::lessThan(glm::max(pa, pb), -R));
    }

    __host__ __device__ inline bool triangle_overlaps_aabb(
        const glm::vec3& v0,
        const glm::vec3& v1,
        const glm::vec3& v2,
        const glm::vec3& box_min,
        const glm::vec3& box_max)
    {
        const glm::vec3 center = 0.5f * (box_min + box_max);
        const glm::vec3 half_extents = 0.5f * (box_max - box_min);
        const glm::vec3 H = half_extents + glm::max(half_extents * 1e-4f, glm::vec3(1e-6f));

        const glm::vec3 tv0 = v0 - center;
        const glm::vec3 tv1 = v1 - center;
        const glm::vec3 tv2 = v2 - center;

        const glm::vec3 tri_min = glm::min(glm::min(tv0, tv1), tv2);
        const glm::vec3 tri_max = glm::max(glm::max(tv0, tv1), tv2);
        if (!glm::all(glm::lessThanEqual(tri_min, H)) ||
            !glm::all(glm::greaterThanEqual(tri_max, -H)))
        {
            return false;
        }

        const glm::vec3 e0 = tv1 - tv0;
        const glm::vec3 e1 = tv2 - tv1;
        const glm::vec3 e2 = tv0 - tv2;

        const glm::vec3 N = glm::cross(e0, e1);
        const float r = glm::dot(H, glm::abs(N));
        const float d = glm::dot(N, tv0);
        if (glm::abs(d) > r)
            return false;

        if (sat_edge_separated(tv0, tv2, e0, H)) return false;
        if (sat_edge_separated(tv1, tv0, e1, H)) return false;
        if (sat_edge_separated(tv2, tv1, e2, H)) return false;

        return true;
    }

    __host__ __device__ inline bool primitive_overlaps_aabb(
        PrimitiveType ptype,
        const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
        const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2,
        const glm::vec3& tri_min, const glm::vec3& tri_max,
        const glm::vec3& box_min, const glm::vec3& box_max)
    {
        const glm::vec3 eps = glm::max((box_max - box_min) * 1e-4f, glm::vec3(1e-6f));
        const glm::vec3 pad_min = box_min - eps;
        const glm::vec3 pad_max = box_max + eps;

        if (tri_max.x < pad_min.x || tri_min.x > pad_max.x ||
            tri_max.y < pad_min.y || tri_min.y > pad_max.y ||
            tri_max.z < pad_min.z || tri_min.z > pad_max.z)
        {
            return false;
        }

        if (ptype == PrimitiveType::Subspace)
        {
            return true;
        }

        if (ptype == PrimitiveType::NagataPatch)
        {
            NagataPatch patch = NagataPatch::fromTriangle(v0, v1, v2, n0, n1, n2);
            return classifyNagataBoxIntersection(patch, AABB{pad_min, pad_max}, 4) != BoxIntersectionState::OUTSIDE;
        }

        return triangle_overlaps_aabb(v0, v1, v2, box_min, box_max);
    }

    __host__ __device__ inline void accumulate_sah_bins(
        const glm::vec3& bmin,
        const glm::vec3& bmax,
        const glm::vec3& pmin,
        const glm::vec3& pmax,
        int axis,
        int& out_b0,
        int& out_b1,
        bool& out_valid)
    {
        const float extent = bmax[axis] - bmin[axis];
        if (extent < 1e-5f)
        {
            out_valid = false;
            return;
        }

        const float cmin = fminf(fmaxf(pmin[axis], bmin[axis]), bmax[axis]);
        const float cmax = fmaxf(cmin, fminf(fmaxf(pmax[axis], bmin[axis]), bmax[axis]));
        const float scale = static_cast<float>(SAH_BINS) / extent;

        int b0 = static_cast<int>(floorf((cmin - bmin[axis]) * scale));
        int b1 = static_cast<int>(floorf((cmax - bmin[axis]) * scale));
        out_b0 = b0 < 0 ? 0 : (b0 >= SAH_BINS ? SAH_BINS - 1 : b0);
        out_b1 = b1 < 0 ? 0 : (b1 >= SAH_BINS ? SAH_BINS - 1 : b1);
        out_valid = true;
    }

    __host__ __device__ inline uint32_t longest_axis(const glm::vec3& ext)
    {
        uint32_t axis = 0;
        if (ext.y > ext.x && ext.y >= ext.z)
            axis = 1;
        else if (ext.z > ext.x && ext.z > ext.y)
            axis = 2;
        return axis;
    }

    __host__ __device__ inline bool evaluate_split(
        const glm::vec3& bmin,
        const glm::vec3& bmax,
        const glm::vec3& node_pmin,
        const glm::vec3& node_pmax,
        float root_max_ext,
        uint32_t N,
        int depth,
        KDSplitMode split_mode,
        int max_depth,
        int max_leaf_primitives,
        float traversal_cost,
        float intersection_cost,
        const uint32_t* entry_bins, // [3 * SAH_BINS]
        const uint32_t* exit_bins,  // [3 * SAH_BINS]
        bool is_subspace,
        uint32_t& out_axis,
        float& out_split_pos)
    {
        if (depth >= max_depth || N == 0)
            return false;

        const glm::vec3 ext = bmax - bmin;
        const float max_ext = fmaxf(ext.x, fmaxf(ext.y, ext.z));
        const float min_ext = fminf(ext.x, fminf(ext.y, ext.z));
        if (max_ext < 1e-5f)
            return false;

        int best_empty_axis = -1;
        float best_empty_pos = 0.0f;
        float best_empty_margin = 0.0f;

        if (node_pmin.x <= node_pmax.x)
        {
            const float min_margin = fmaxf(0.004f * root_max_ext, 1e-4f);
            const float empty_frac = is_subspace ? 0.02f : 0.15f;
            for (int a = 0; a < 3; ++a)
            {
                if (ext[a] < 1e-5f)
                    continue;

                const float cut_eps = ext[a] * 5e-4f + 2e-5f;
                const float margin_lo = node_pmin[a] - bmin[a];
                const float pos_lo = node_pmin[a] - cut_eps;
                if (margin_lo > empty_frac * ext[a] && margin_lo > min_margin &&
                    pos_lo > bmin[a] + 1e-6f && pos_lo < bmax[a] - 1e-6f &&
                    margin_lo > best_empty_margin)
                {
                    best_empty_margin = margin_lo;
                    best_empty_axis = a;
                    best_empty_pos = pos_lo;
                }

                const float margin_hi = bmax[a] - node_pmax[a];
                const float pos_hi = node_pmax[a] + cut_eps;
                if (margin_hi > empty_frac * ext[a] && margin_hi > min_margin &&
                    pos_hi > bmin[a] + 1e-6f && pos_hi < bmax[a] - 1e-6f &&
                    margin_hi > best_empty_margin)
                {
                    best_empty_margin = margin_hi;
                    best_empty_axis = a;
                    best_empty_pos = pos_hi;
                }
            }

            if (best_empty_axis >= 0 &&
                (best_empty_margin >= 0.25f * ext[best_empty_axis] || N <= static_cast<uint32_t>(max_leaf_primitives)))
            {
                out_axis = static_cast<uint32_t>(best_empty_axis);
                out_split_pos = best_empty_pos;
                return true;
            }
        }

        // Only split thick 3D leaves for 2D triangles (where splitting exposes empty half-spaces);
        // solid 3D SubspacePrimitive AABBs already fill their trimmed leaf box.
        const bool is_thick_3d_box = !is_subspace && (min_ext > 0.035f * root_max_ext);
        if (N <= static_cast<uint32_t>(max_leaf_primitives) && !is_thick_3d_box)
            return false;

        if (split_mode == KDSplitMode::SpatialMedian)
        {
            const uint32_t axis = longest_axis(ext);
            out_axis = axis;
            out_split_pos = 0.5f * (bmin[axis] + bmax[axis]);
            return true;
        }

        const float parent_half_area = ext.x * ext.y + ext.y * ext.z + ext.z * ext.x;
        if (parent_half_area <= 1e-12f)
            return false;

        const float inv_parent_area = 1.0f / parent_half_area;
        int best_axis = -1;
        float best_split_pos = 0.0f;
        float best_cost = FLT_MAX;

        if (entry_bins != nullptr && exit_bins != nullptr)
        {
            for (int a = 0; a < 3; ++a)
            {
                const float axis_ext = ext[a];
                if (axis_ext < 1e-5f)
                    continue;

                const int b_ax = (a + 1) % 3;
                const int c_ax = (a + 2) % 3;
                const float db = ext[b_ax];
                const float dc = ext[c_ax];
                const float side_area = db * dc;
                const float perimeter_other = db + dc;

                const uint32_t* ax_entry = entry_bins + a * SAH_BINS;
                const uint32_t* ax_exit = exit_bins + a * SAH_BINS;

                uint32_t right_suffix[SAH_BINS];
                uint32_t running_right = 0;
                for (int k = SAH_BINS - 1; k >= 0; --k)
                {
                    running_right += ax_exit[k];
                    right_suffix[k] = running_right;
                }

                uint32_t n_left = 0;
                for (int k = 1; k < SAH_BINS; ++k)
                {
                    n_left += ax_entry[k - 1];
                    const uint32_t n_right = right_suffix[k];

                    if ((n_left == N && n_right == N) || (n_left == 0 && n_right == 0))
                        continue;

                    const float frac = static_cast<float>(k) / static_cast<float>(SAH_BINS);
                    const float w_left = frac * axis_ext;
                    const float w_right = (1.0f - frac) * axis_ext;
                    const float area_left = w_left * perimeter_other + side_area;
                    const float area_right = w_right * perimeter_other + side_area;

                    float cost = traversal_cost + intersection_cost * inv_parent_area *
                        (area_left * static_cast<float>(n_left) + area_right * static_cast<float>(n_right));

                    if (n_left == 0 || n_right == 0)
                        cost *= 0.85f;

                    if (cost < best_cost)
                    {
                        best_cost = cost;
                        best_axis = a;
                        best_split_pos = bmin[a] + frac * axis_ext;
                    }
                }
            }
        }

        if (best_axis >= 0)
        {
            const float leaf_cost = intersection_cost * static_cast<float>(N);
            if (best_cost < leaf_cost || N > static_cast<uint32_t>(max_leaf_primitives) || is_thick_3d_box)
            {
                out_axis = static_cast<uint32_t>(best_axis);
                out_split_pos = best_split_pos;
                return true;
            }
        }

        if (best_empty_axis >= 0)
        {
            out_axis = static_cast<uint32_t>(best_empty_axis);
            out_split_pos = best_empty_pos;
            return true;
        }

        if (is_thick_3d_box || N > static_cast<uint32_t>(max_leaf_primitives))
        {
            const uint32_t axis = longest_axis(ext);
            out_axis = axis;
            out_split_pos = 0.5f * (bmin[axis] + bmax[axis]);
            return true;
        }

        return false;
    }

    struct MinVec3Functor
    {
        __host__ __device__ glm::vec3 operator()(const glm::vec3& a, const glm::vec3& b) const
        {
            return glm::min(a, b);
        }
    };

    struct MaxVec3Functor
    {
        __host__ __device__ glm::vec3 operator()(const glm::vec3& a, const glm::vec3& b) const
        {
            return glm::max(a, b);
        }
    };

    struct CpuBuildContext
    {
        soa::soa_view<PrimitiveChunk16> prim_view;
        soa::soa_view<Vertex> vert_view;
        const glm::vec3* prim_min = nullptr;
        const glm::vec3* prim_max = nullptr;
        const uint32_t* prim_slot_ids = nullptr;
        float root_max_ext = 1.0f;
        KDSplitMode split_mode = KDSplitMode::SAH;
        int max_depth = 20;
        int max_leaf_primitives = 8;
        float traversal_cost = 1.0f;
        float intersection_cost = 1.5f;
        bool is_subspace = false;
        std::vector<KDNode>* nodes = nullptr;
        std::vector<std::pair<uint32_t, uint32_t>>* leaf_pairs = nullptr;
        uint32_t total_leaves = 0;
    };

    inline void build_cpu_recursive(
        CpuBuildContext& ctx,
        uint32_t node_idx,
        const glm::vec3& bmin,
        const glm::vec3& bmax,
        int depth,
        const std::vector<uint32_t>& prim_indices)
    {
        uint32_t entry_bins[3 * SAH_BINS]{};
        uint32_t exit_bins[3 * SAH_BINS]{};
        glm::vec3 node_pmin(FLT_MAX);
        glm::vec3 node_pmax(-FLT_MAX);

        const uint32_t n = static_cast<uint32_t>(prim_indices.size());
        if (depth < ctx.max_depth && n > 0)
        {
            for (uint32_t p : prim_indices)
            {
                glm::vec3 v0{}, v1{}, v2{}, n0{}, n1{}, n2{};
                PrimitiveType ptype = PrimitiveType::Subspace;
                if (!ctx.is_subspace)
                {
                    const uint32_t slot_id = ctx.prim_slot_ids[p];
                    ptype = get_primitive_triangle_vertices(
                        ctx.prim_view, ctx.vert_view, slot_id, v0, v1, v2, n0, n1, n2);
                }

                glm::vec3 cmin, cmax;
                get_clipped_primitive_aabb(
                    ptype, v0, v1, v2, n0, n1, n2,
                    ctx.prim_min[p], ctx.prim_max[p],
                    bmin, bmax, cmin, cmax);

                node_pmin = glm::min(node_pmin, cmin);
                node_pmax = glm::max(node_pmax, cmax);

                if (ctx.split_mode == KDSplitMode::SAH)
                {
                    for (int a = 0; a < 3; ++a)
                    {
                        int b0 = 0, b1 = 0;
                        bool valid = false;
                        accumulate_sah_bins(bmin, bmax, cmin, cmax, a, b0, b1, valid);
                        if (valid)
                        {
                            ++entry_bins[a * SAH_BINS + b0];
                            ++exit_bins[a * SAH_BINS + b1];
                        }
                    }
                }
            }
        }

        uint32_t split_axis = 0;
        float split_pos = 0.0f;
        const bool should_split = evaluate_split(
            bmin, bmax, node_pmin, node_pmax, ctx.root_max_ext,
            n, depth,
            ctx.split_mode, ctx.max_depth, ctx.max_leaf_primitives,
            ctx.traversal_cost, ctx.intersection_cost,
            entry_bins, exit_bins,
            ctx.is_subspace,
            split_axis, split_pos);

        if (!should_split)
        {
            ++ctx.total_leaves;
            (*ctx.nodes)[node_idx] = KDNode::make_leaf(0u);
            for (uint32_t p : prim_indices)
            {
                ctx.leaf_pairs->emplace_back(node_idx, ctx.prim_slot_ids[p]);
            }
            return;
        }

        const uint32_t left_idx = static_cast<uint32_t>(ctx.nodes->size());
        ctx.nodes->resize(left_idx + 2);
        (*ctx.nodes)[node_idx] = KDNode::make_interior(split_axis, split_pos, left_idx);

        glm::vec3 left_min = bmin;
        glm::vec3 left_max = bmax;
        left_max[split_axis] = split_pos;

        glm::vec3 right_min = bmin;
        glm::vec3 right_max = bmax;
        right_min[split_axis] = split_pos;

        std::vector<uint32_t> left_prims;
        std::vector<uint32_t> right_prims;
        left_prims.reserve(n / 2 + 1);
        right_prims.reserve(n / 2 + 1);

        for (uint32_t p : prim_indices)
        {
            glm::vec3 v0{}, v1{}, v2{}, n0{}, n1{}, n2{};
            PrimitiveType ptype = PrimitiveType::Subspace;
            if (!ctx.is_subspace)
            {
                const uint32_t slot_id = ctx.prim_slot_ids[p];
                ptype = get_primitive_triangle_vertices(
                    ctx.prim_view, ctx.vert_view, slot_id, v0, v1, v2, n0, n1, n2);
            }

            if (primitive_overlaps_aabb(ptype, v0, v1, v2, n0, n1, n2, ctx.prim_min[p], ctx.prim_max[p], left_min, left_max))
            {
                left_prims.push_back(p);
            }
            if (primitive_overlaps_aabb(ptype, v0, v1, v2, n0, n1, n2, ctx.prim_min[p], ctx.prim_max[p], right_min, right_max))
            {
                right_prims.push_back(p);
            }
        }

        build_cpu_recursive(ctx, left_idx, left_min, left_max, depth + 1, left_prims);
        std::vector<uint32_t>().swap(left_prims);

        build_cpu_recursive(ctx, left_idx + 1, right_min, right_max, depth + 1, right_prims);
    }
} // namespace kdtree_detail

__global__ void compute_kdtree_prim_aabbs_kernel(
    soa::soa_view<PrimitiveChunk16> primitives,
    soa::soa_view<Vertex> vertices,
    uint32_t primitives_start,
    uint32_t number_of_triangles,
    uint32_t occupied_slots,
    glm::vec3* prim_min,
    glm::vec3* prim_max,
    uint32_t* prim_slot_ids)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number_of_triangles)
        return;

    const uint32_t slot_id = primitives_start + idx * occupied_slots;
    glm::vec3 v0, v1, v2, n0, n1, n2;
    const PrimitiveType ptype = kdtree_detail::get_primitive_triangle_vertices(
        primitives, vertices, slot_id, v0, v1, v2, n0, n1, n2);

    glm::vec3 tmin, tmax;
    kdtree_detail::get_primitive_aabb(ptype, v0, v1, v2, n0, n1, n2, tmin, tmax);

    prim_min[idx] = tmin;
    prim_max[idx] = tmax;
    prim_slot_ids[idx] = slot_id;
}

__global__ void bin_kdtree_pairs_kernel(
    uint32_t num_pairs,
    soa::soa_view<PrimitiveChunk16> primitives,
    soa::soa_view<Vertex> vertices,
    const uint32_t* prim_slot_ids,
    const uint32_t* pair_node,
    const uint32_t* pair_prim,
    const glm::vec3* prim_min,
    const glm::vec3* prim_max,
    const glm::vec3* active_bmin,
    const glm::vec3* active_bmax,
    glm::vec3* node_pmin,
    glm::vec3* node_pmax,
    uint32_t* entry_bins,
    uint32_t* exit_bins)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_pairs)
        return;

    const uint32_t u = pair_node[i];
    const uint32_t p = pair_prim[i];
    const glm::vec3 bmin = active_bmin[u];
    const glm::vec3 bmax = active_bmax[u];
    const glm::vec3 pmin = prim_min[p];
    const glm::vec3 pmax = prim_max[p];

    glm::vec3 v0, v1, v2, n0, n1, n2;
    const PrimitiveType ptype = kdtree_detail::get_primitive_triangle_vertices(
        primitives, vertices, prim_slot_ids[p], v0, v1, v2, n0, n1, n2);

    glm::vec3 cmin, cmax;
    kdtree_detail::get_clipped_primitive_aabb(
        ptype, v0, v1, v2, n0, n1, n2,
        pmin, pmax, bmin, bmax, cmin, cmax);

    kdtree_detail::atomic_min_float(&node_pmin[u].x, cmin.x);
    kdtree_detail::atomic_min_float(&node_pmin[u].y, cmin.y);
    kdtree_detail::atomic_min_float(&node_pmin[u].z, cmin.z);
    kdtree_detail::atomic_max_float(&node_pmax[u].x, cmax.x);
    kdtree_detail::atomic_max_float(&node_pmax[u].y, cmax.y);
    kdtree_detail::atomic_max_float(&node_pmax[u].z, cmax.z);

    if (entry_bins != nullptr && exit_bins != nullptr)
    {
        const uint32_t base = u * (3u * kdtree_detail::SAH_BINS);
        for (int a = 0; a < 3; ++a)
        {
            int b0 = 0, b1 = 0;
            bool valid = false;
            kdtree_detail::accumulate_sah_bins(bmin, bmax, cmin, cmax, a, b0, b1, valid);
            if (valid)
            {
                atomicAdd(&entry_bins[base + a * kdtree_detail::SAH_BINS + b0], 1u);
                atomicAdd(&exit_bins[base + a * kdtree_detail::SAH_BINS + b1], 1u);
            }
        }
    }
}

__global__ void decide_kdtree_splits_kernel(
    uint32_t num_active,
    int depth,
    float root_max_ext,
    KDSplitMode split_mode,
    int max_depth,
    int max_leaf_primitives,
    float traversal_cost,
    float intersection_cost,
    const glm::vec3* active_bmin,
    const glm::vec3* active_bmax,
    const glm::vec3* node_pmin,
    const glm::vec3* node_pmax,
    const uint32_t* node_counts,
    const uint32_t* entry_bins,
    const uint32_t* exit_bins,
    bool is_subspace,
    uint8_t* is_split,
    uint32_t* split_axis,
    float* split_pos)
{
    const uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_active)
        return;

    const uint32_t* node_entry = entry_bins ? (entry_bins + u * (3u * kdtree_detail::SAH_BINS)) : nullptr;
    const uint32_t* node_exit = exit_bins ? (exit_bins + u * (3u * kdtree_detail::SAH_BINS)) : nullptr;

    uint32_t axis = 0;
    float pos = 0.0f;
    const bool split = kdtree_detail::evaluate_split(
        active_bmin[u],
        active_bmax[u],
        node_pmin[u],
        node_pmax[u],
        root_max_ext,
        node_counts[u],
        depth,
        split_mode,
        max_depth,
        max_leaf_primitives,
        traversal_cost,
        intersection_cost,
        node_entry,
        node_exit,
        is_subspace,
        axis,
        pos);

    is_split[u] = split ? 1 : 0;
    split_axis[u] = axis;
    split_pos[u] = pos;
}

__global__ void mark_kdtree_leaf_pairs_kernel(
    uint32_t num_pairs,
    const uint32_t* pair_node,
    const uint8_t* is_split,
    uint32_t* leaf_flags)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_pairs)
        return;

    leaf_flags[i] = (is_split[pair_node[i]] == 0) ? 1u : 0u;
}

__global__ void scatter_kdtree_leaf_pairs_kernel(
    uint32_t num_pairs,
    const uint32_t* pair_node,
    const uint32_t* pair_prim,
    const uint32_t* active_node_idx,
    const uint32_t* prim_slot_ids,
    const uint32_t* leaf_flags,
    const uint32_t* leaf_offsets,
    uint32_t* out_leaf_nodes,
    uint32_t* out_leaf_prims)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_pairs || leaf_flags[i] == 0u)
        return;

    const uint32_t dst = leaf_offsets[i];
    out_leaf_nodes[dst] = active_node_idx[pair_node[i]];
    out_leaf_prims[dst] = prim_slot_ids[pair_prim[i]];
}

__global__ void build_kdtree_child_nodes_kernel(
    uint32_t num_active,
    uint32_t next_node_offset,
    const uint8_t* is_split,
    const uint32_t* split_rank,
    const uint32_t* split_axis,
    const float* split_pos,
    const glm::vec3* active_bmin,
    const glm::vec3* active_bmax,
    uint32_t* next_node_idx,
    glm::vec3* next_bmin,
    glm::vec3* next_bmax)
{
    const uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_active || is_split[u] == 0)
        return;

    const uint32_t left_u = 2u * split_rank[u];
    const uint32_t right_u = left_u + 1u;

    next_node_idx[left_u] = next_node_offset + left_u;
    next_node_idx[right_u] = next_node_offset + right_u;

    const glm::vec3 bmin = active_bmin[u];
    const glm::vec3 bmax = active_bmax[u];
    const uint32_t axis = split_axis[u];
    const float pos = split_pos[u];

    glm::vec3 lmax = bmax;
    lmax[axis] = pos;
    next_bmin[left_u] = bmin;
    next_bmax[left_u] = lmax;

    glm::vec3 rmin = bmin;
    rmin[axis] = pos;
    next_bmin[right_u] = rmin;
    next_bmax[right_u] = bmax;
}

__global__ void count_kdtree_child_pairs_kernel(
    uint32_t num_pairs,
    soa::soa_view<PrimitiveChunk16> primitives,
    soa::soa_view<Vertex> vertices,
    const uint32_t* prim_slot_ids,
    const glm::vec3* prim_min,
    const glm::vec3* prim_max,
    const uint32_t* pair_node,
    const uint32_t* pair_prim,
    const uint8_t* is_split,
    const uint32_t* split_axis,
    const float* split_pos,
    const glm::vec3* active_bmin,
    const glm::vec3* active_bmax,
    uint8_t* out_mask,
    uint32_t* out_counts)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_pairs)
        return;

    const uint32_t u = pair_node[i];
    if (is_split[u] == 0)
    {
        out_mask[i] = 0;
        out_counts[i] = 0;
        return;
    }

    const uint32_t p = pair_prim[i];
    const uint32_t slot_id = prim_slot_ids[p];
    const glm::vec3 pmin = prim_min[p];
    const glm::vec3 pmax = prim_max[p];

    glm::vec3 v0, v1, v2, n0, n1, n2;
    const PrimitiveType ptype = kdtree_detail::get_primitive_triangle_vertices(
        primitives, vertices, slot_id, v0, v1, v2, n0, n1, n2);

    const glm::vec3 bmin = active_bmin[u];
    const glm::vec3 bmax = active_bmax[u];
    const uint32_t axis = split_axis[u];
    const float pos = split_pos[u];

    glm::vec3 lmax = bmax;
    lmax[axis] = pos;
    glm::vec3 rmin = bmin;
    rmin[axis] = pos;

    const bool hit_left = kdtree_detail::primitive_overlaps_aabb(
        ptype, v0, v1, v2, n0, n1, n2, pmin, pmax, bmin, lmax);
    const bool hit_right = kdtree_detail::primitive_overlaps_aabb(
        ptype, v0, v1, v2, n0, n1, n2, pmin, pmax, rmin, bmax);

    const uint8_t mask = (hit_left ? 1u : 0u) | (hit_right ? 2u : 0u);
    out_mask[i] = mask;
    out_counts[i] = (hit_left ? 1u : 0u) + (hit_right ? 1u : 0u);
}

__global__ void emit_kdtree_child_pairs_kernel(
    uint32_t num_pairs,
    const uint32_t* pair_node,
    const uint32_t* pair_prim,
    const uint32_t* split_rank,
    const uint8_t* mask,
    const uint32_t* offsets,
    uint32_t* next_pair_node,
    uint32_t* next_pair_prim)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_pairs)
        return;

    const uint8_t m = mask[i];
    if (m == 0)
        return;

    const uint32_t left_u = 2u * split_rank[pair_node[i]];
    const uint32_t p = pair_prim[i];
    uint32_t dst = offsets[i];

    if (m & 1u)
    {
        next_pair_node[dst] = left_u;
        next_pair_prim[dst] = p;
        ++dst;
    }
    if (m & 2u)
    {
        next_pair_node[dst] = left_u + 1u;
        next_pair_prim[dst] = p;
    }
}

__global__ void find_sorted_kdtree_runs_kernel(
    uint32_t num_pairs,
    const uint32_t* sorted_pair_node,
    uint32_t* run_starts,
    uint32_t* run_ends)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_pairs)
        return;

    const uint32_t c = sorted_pair_node[i];
    if (i == 0 || sorted_pair_node[i - 1] != c)
        run_starts[c] = i;
    if (i + 1 == num_pairs || sorted_pair_node[i + 1] != c)
        run_ends[c] = i + 1;
}

__global__ void compute_kdtree_node_counts_kernel(
    uint32_t num_nodes,
    const uint32_t* run_starts,
    const uint32_t* run_ends,
    uint32_t* node_counts)
{
    const uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes)
        return;

    node_counts[u] = run_ends[u] - run_starts[u];
}

KDTreeAS::KDTreeAS(GeometryContext* geometry_context)
    : m_geometry_context(geometry_context)
{
    // Reserve element 0 in pools as sentinel (0 = empty)
    as.alloc(1);
    as[0] = KDNode::make_leaf(0u);
    box_content.alloc(1);
    sync_to_device();
}

void KDTreeAS::config_ui()
{
    const char* split_mode_names[] = {"SAH (16 bins)", "Spatial Median"};
    int split_idx = static_cast<int>(config.split_mode);
    if (ImGui::Combo("Split mode", &split_idx, split_mode_names, 2))
    {
        config.split_mode = static_cast<KDSplitMode>(split_idx);
    }

    ImGui::DragInt("Max depth", &config.max_depth, 0.25f, 1, 30);
    if (config.max_depth < 1) config.max_depth = 1;
    if (config.max_depth > 30) config.max_depth = 30;

    ImGui::DragInt("Max leaf primitives", &config.max_leaf_primitives, 0.5f, 1, 128);
    if (config.max_leaf_primitives < 1) config.max_leaf_primitives = 1;

    if (config.split_mode == KDSplitMode::SAH)
    {
        ImGui::DragFloat("Traversal cost", &config.traversal_cost, 0.05f, 0.01f, 100.0f);
        ImGui::DragFloat("Intersection cost", &config.intersection_cost, 0.05f, 0.01f, 100.0f);
    }

    ImGui::Checkbox("Build on GPU", &config.use_gpu);
}

void KDTreeAS::info_ui(uint32_t space_id, Space space)
{
    auto it = m_build_times_ms.find(space_id);
    if (it != m_build_times_ms.end())
    {
        ImGui::Text("  Build Time: %.2f ms", it->second);
    }

    uint32_t num_leaves = 0;
    if (space.root_box >= 0)
    {
        auto tree = trees[static_cast<size_t>(space.root_box)];
        num_leaves = tree.num_leaves;
        ImGui::Text("  Bounds Min: (%.3f, %.3f, %.3f)", tree.bounds_min.x, tree.bounds_min.y, tree.bounds_min.z);
        ImGui::Text("  Bounds Max: (%.3f, %.3f, %.3f)", tree.bounds_max.x, tree.bounds_max.y, tree.bounds_max.z);
    }

    const size_t nodes_bytes = static_cast<size_t>(space.as_size) * sizeof(KDNode);
    const size_t content_bytes = static_cast<size_t>(space.box_content_size) * sizeof(BoxContent);
    const size_t tree_meta_bytes = (space.root_box >= 0) ? (sizeof(glm::vec3) * 2 + sizeof(uint32_t) * 2) : 0;
    const size_t as_bytes = nodes_bytes + content_bytes + tree_meta_bytes;
    const size_t geom_bytes = space_geometry_bytes(space);
    const size_t total_bytes = as_bytes + geom_bytes;

    ImGui::Text("  Nodes:      %u (%u leaves)", (uint32_t)space.as_size, num_leaves);
    ImGui::Text("  Contents:   %u", (uint32_t)space.box_content_size);
    ImGui::Text("  Primitives: %u slots", (uint32_t)space.primitives_size);
    ImGui::Text("  Vertices:   %u", (uint32_t)space.vertices_size);
    ImGui::Text("  AS Memory:  %s", format_bytes(as_bytes).c_str());
    ImGui::Text("    Nodes:    %s", format_bytes(nodes_bytes).c_str());
    ImGui::Text("    Contents: %s", format_bytes(content_bytes).c_str());
    ImGui::Text("  Geo Memory: %s", format_bytes(geom_bytes).c_str());
    ImGui::Text("  Total Size: %s", format_bytes(total_bytes).c_str());
}

void KDTreeAS::free_space(uint32_t space_id, Space space)
{
    m_build_times_ms.erase(space_id);
    if (space.as_type != ASType::KDTree)
        return;

    if (space.root_box >= 0)
    {
        trees.free(static_cast<size_t>(space.root_box));
    }
    if (space.as_size > 0)
    {
        as.free(space.as_start);
    }
    if (space.box_content_size > 0)
    {
        box_content.free(space.box_content_start);
    }
}

void KDTreeAS::finalize_tree_pools(
    Space space,
    glm::vec3 bounds_min,
    glm::vec3 bounds_max,
    std::vector<KDNode>& host_nodes,
    std::vector<std::pair<uint32_t, uint32_t>>& leaf_pairs,
    uint32_t total_leaves)
{
    std::sort(leaf_pairs.begin(), leaf_pairs.end());

    const uint32_t tree_id = static_cast<uint32_t>(trees.alloc_element());
    const uint32_t total_nodes = static_cast<uint32_t>(host_nodes.size());
    const uint32_t as_start = static_cast<uint32_t>(as.alloc(total_nodes));
    const uint32_t total_pairs = static_cast<uint32_t>(leaf_pairs.size());
    const uint32_t bc_start = static_cast<uint32_t>(box_content.alloc(total_pairs + 1));

    for (size_t i = 0; i < host_nodes.size(); ++i)
    {
        if (!host_nodes[i].is_leaf())
        {
            host_nodes[i] = KDNode::make_interior(
                host_nodes[i].axis(),
                host_nodes[i].split_pos,
                as_start + host_nodes[i].left_child());
        }
    }

    box_content[bc_start] = BoxContent{};
    for (uint32_t i = 0; i < total_pairs; ++i)
    {
        const uint32_t ln = leaf_pairs[i].first;
        const uint32_t pid = leaf_pairs[i].second;
        const bool is_first = (i == 0) || (leaf_pairs[i - 1].first != ln);
        const bool is_last = (i + 1 == total_pairs) || (leaf_pairs[i + 1].first != ln);
        const uint32_t bc_idx = bc_start + 1u + i;
        box_content[bc_idx] = BoxContent(pid, is_last);
        if (is_first)
        {
            host_nodes[ln] = KDNode::make_leaf(bc_idx);
        }
    }

    for (uint32_t i = 0; i < total_nodes; ++i)
    {
        as[as_start + i] = host_nodes[i];
    }

    trees[tree_id].bounds_min = bounds_min;
    trees[tree_id].bounds_max = bounds_max;
    trees[tree_id].as_start = as_start;
    trees[tree_id].num_leaves = total_leaves;

    space.root_box = static_cast<int>(tree_id);
    space.as_start = as_start;
    space.as_size = total_nodes;
    space.box_content_start = bc_start;
    space.box_content_size = total_pairs + 1u;

    soa::upload_soa_range_async(d_trees, trees, tree_id, 1);
    soa::upload_soa_range_async(d_as, as, as_start, total_nodes);
    soa::upload_soa_range_async(d_box_content, box_content, bc_start, total_pairs + 1u);
}

void KDTreeAS::build_gpu(uint32_t /*space_id*/, Space space, uint32_t number_of_triangles, uint32_t occupied_slots, bool is_subspace)
{
    if (number_of_triangles == 0)
        return;

    constexpr uint32_t block_size = 256;
    const uint32_t tri_blocks = (number_of_triangles - 1u) / block_size + 1u;

    thrust::device_vector<glm::vec3> d_prim_min(number_of_triangles);
    thrust::device_vector<glm::vec3> d_prim_max(number_of_triangles);
    thrust::device_vector<uint32_t> d_prim_slot_ids(number_of_triangles);

    compute_kdtree_prim_aabbs_kernel<<<tri_blocks, block_size>>>(
        m_geometry_context->primitives.view(),
        m_geometry_context->vertices.view(),
        space.primitives_start,
        number_of_triangles,
        occupied_slots,
        d_prim_min.data().get(),
        d_prim_max.data().get(),
        d_prim_slot_ids.data().get());
    CUDA_CHECK(cudaDeviceSynchronize());

    glm::vec3 root_min = thrust::reduce(
        thrust::device,
        d_prim_min.begin(),
        d_prim_min.end(),
        glm::vec3(FLT_MAX),
        kdtree_detail::MinVec3Functor{});
    glm::vec3 root_max = thrust::reduce(
        thrust::device,
        d_prim_max.begin(),
        d_prim_max.end(),
        glm::vec3(-FLT_MAX),
        kdtree_detail::MaxVec3Functor{});

    const glm::vec3 pad = glm::max((root_max - root_min) * 1e-4f, glm::vec3(1e-5f));
    root_min -= pad;
    root_max += pad;

    const glm::vec3 root_ext = root_max - root_min;
    const float root_max_ext = fmaxf(root_ext.x, fmaxf(root_ext.y, root_ext.z));

    std::vector<KDNode> host_nodes(1);
    std::vector<std::pair<uint32_t, uint32_t>> leaf_pairs;
    uint32_t total_leaves = 0;

    thrust::device_vector<uint32_t> d_active_node_idx(1, 0u);
    thrust::device_vector<glm::vec3> d_active_bmin(1, root_min);
    thrust::device_vector<glm::vec3> d_active_bmax(1, root_max);
    thrust::device_vector<uint32_t> d_node_counts(1, number_of_triangles);

    thrust::device_vector<uint32_t> d_pair_node(number_of_triangles, 0u);
    thrust::device_vector<uint32_t> d_pair_prim(number_of_triangles);
    thrust::sequence(thrust::device, d_pair_prim.begin(), d_pair_prim.end(), 0u);

    const int max_depth = config.max_depth < 1 ? 1 : (config.max_depth > 30 ? 30 : config.max_depth);
    const int max_leaf_prims = std::max(config.max_leaf_primitives, 1);

    for (int depth = 0; depth <= max_depth; ++depth)
    {
        const uint32_t num_active = static_cast<uint32_t>(d_active_node_idx.size());
        if (num_active == 0)
            break;

        const uint32_t num_pairs = static_cast<uint32_t>(d_pair_node.size());
        const uint32_t node_blocks = (num_active - 1u) / block_size + 1u;

        thrust::device_vector<uint8_t> d_is_split(num_active, 0);
        thrust::device_vector<uint32_t> d_split_axis(num_active, 0);
        thrust::device_vector<float> d_split_pos(num_active, 0.0f);

        thrust::device_vector<glm::vec3> d_node_pmin(num_active, glm::vec3(FLT_MAX));
        thrust::device_vector<glm::vec3> d_node_pmax(num_active, glm::vec3(-FLT_MAX));
        thrust::device_vector<uint32_t> d_entry_bins;
        thrust::device_vector<uint32_t> d_exit_bins;

        if (depth < max_depth && num_pairs > 0)
        {
            if (config.split_mode == KDSplitMode::SAH)
            {
                d_entry_bins.assign(static_cast<size_t>(num_active) * 3u * kdtree_detail::SAH_BINS, 0u);
                d_exit_bins.assign(static_cast<size_t>(num_active) * 3u * kdtree_detail::SAH_BINS, 0u);
            }

            const uint32_t pair_blocks = (num_pairs - 1u) / block_size + 1u;
            bin_kdtree_pairs_kernel<<<pair_blocks, block_size>>>(
                num_pairs,
                m_geometry_context->primitives.view(),
                m_geometry_context->vertices.view(),
                d_prim_slot_ids.data().get(),
                d_pair_node.data().get(),
                d_pair_prim.data().get(),
                d_prim_min.data().get(),
                d_prim_max.data().get(),
                d_active_bmin.data().get(),
                d_active_bmax.data().get(),
                d_node_pmin.data().get(),
                d_node_pmax.data().get(),
                d_entry_bins.empty() ? nullptr : d_entry_bins.data().get(),
                d_exit_bins.empty() ? nullptr : d_exit_bins.data().get());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        decide_kdtree_splits_kernel<<<node_blocks, block_size>>>(
            num_active,
            depth,
            root_max_ext,
            config.split_mode,
            max_depth,
            max_leaf_prims,
            config.traversal_cost,
            config.intersection_cost,
            d_active_bmin.data().get(),
            d_active_bmax.data().get(),
            d_node_pmin.data().get(),
            d_node_pmax.data().get(),
            d_node_counts.data().get(),
            d_entry_bins.empty() ? nullptr : d_entry_bins.data().get(),
            d_exit_bins.empty() ? nullptr : d_exit_bins.data().get(),
            is_subspace,
            d_is_split.data().get(),
            d_split_axis.data().get(),
            d_split_pos.data().get());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<uint32_t> h_active_node_idx(num_active);
        std::vector<uint8_t> h_is_split(num_active);
        std::vector<uint32_t> h_split_axis(num_active);
        std::vector<float> h_split_pos(num_active);
        thrust::copy(d_active_node_idx.begin(), d_active_node_idx.end(), h_active_node_idx.begin());
        thrust::copy(d_is_split.begin(), d_is_split.end(), h_is_split.begin());
        thrust::copy(d_split_axis.begin(), d_split_axis.end(), h_split_axis.begin());
        thrust::copy(d_split_pos.begin(), d_split_pos.end(), h_split_pos.begin());

        const uint32_t next_node_offset = static_cast<uint32_t>(host_nodes.size());
        std::vector<uint32_t> h_split_rank(num_active, 0u);
        uint32_t num_splits = 0;

        for (uint32_t u = 0; u < num_active; ++u)
        {
            if (h_is_split[u] != 0)
            {
                h_split_rank[u] = num_splits++;
                host_nodes[h_active_node_idx[u]] = KDNode::make_interior(
                    h_split_axis[u],
                    h_split_pos[u],
                    next_node_offset + 2u * h_split_rank[u]);
            }
            else
            {
                ++total_leaves;
                host_nodes[h_active_node_idx[u]] = KDNode::make_leaf(0u);
            }
        }

        if (num_splits > 0)
        {
            host_nodes.resize(next_node_offset + 2u * num_splits);
        }

        if (num_splits < num_active && num_pairs > 0)
        {
            const uint32_t pair_blocks = (num_pairs - 1u) / block_size + 1u;
            thrust::device_vector<uint32_t> d_leaf_flags(num_pairs, 0u);
            thrust::device_vector<uint32_t> d_leaf_offsets(num_pairs, 0u);

            mark_kdtree_leaf_pairs_kernel<<<pair_blocks, block_size>>>(
                num_pairs,
                d_pair_node.data().get(),
                d_is_split.data().get(),
                d_leaf_flags.data().get());
            CUDA_CHECK(cudaDeviceSynchronize());

            thrust::exclusive_scan(d_leaf_flags.begin(), d_leaf_flags.end(), d_leaf_offsets.begin());
            const uint32_t last_flag = d_leaf_flags[num_pairs - 1u];
            const uint32_t last_off = d_leaf_offsets[num_pairs - 1u];
            const uint32_t level_leaf_pairs = last_off + last_flag;

            if (level_leaf_pairs > 0)
            {
                thrust::device_vector<uint32_t> d_level_leaf_nodes(level_leaf_pairs);
                thrust::device_vector<uint32_t> d_level_leaf_prims(level_leaf_pairs);

                scatter_kdtree_leaf_pairs_kernel<<<pair_blocks, block_size>>>(
                    num_pairs,
                    d_pair_node.data().get(),
                    d_pair_prim.data().get(),
                    d_active_node_idx.data().get(),
                    d_prim_slot_ids.data().get(),
                    d_leaf_flags.data().get(),
                    d_leaf_offsets.data().get(),
                    d_level_leaf_nodes.data().get(),
                    d_level_leaf_prims.data().get());
                CUDA_CHECK(cudaDeviceSynchronize());

                std::vector<uint32_t> h_leaf_nodes(level_leaf_pairs);
                std::vector<uint32_t> h_leaf_prims(level_leaf_pairs);
                thrust::copy(d_level_leaf_nodes.begin(), d_level_leaf_nodes.end(), h_leaf_nodes.begin());
                thrust::copy(d_level_leaf_prims.begin(), d_level_leaf_prims.end(), h_leaf_prims.begin());

                leaf_pairs.reserve(leaf_pairs.size() + level_leaf_pairs);
                for (uint32_t k = 0; k < level_leaf_pairs; ++k)
                {
                    leaf_pairs.emplace_back(h_leaf_nodes[k], h_leaf_prims[k]);
                }
            }
        }

        if (num_splits == 0)
            break;

        const thrust::device_vector<uint32_t> d_split_rank(h_split_rank.begin(), h_split_rank.end());
        const uint32_t num_children = 2u * num_splits;

        thrust::device_vector<uint32_t> d_next_node_idx(num_children);
        thrust::device_vector<glm::vec3> d_next_bmin(num_children);
        thrust::device_vector<glm::vec3> d_next_bmax(num_children);
        thrust::device_vector<uint32_t> d_next_node_counts(num_children, 0u);

        build_kdtree_child_nodes_kernel<<<node_blocks, block_size>>>(
            num_active,
            next_node_offset,
            d_is_split.data().get(),
            d_split_rank.data().get(),
            d_split_axis.data().get(),
            d_split_pos.data().get(),
            d_active_bmin.data().get(),
            d_active_bmax.data().get(),
            d_next_node_idx.data().get(),
            d_next_bmin.data().get(),
            d_next_bmax.data().get());
        CUDA_CHECK(cudaDeviceSynchronize());

        thrust::device_vector<uint32_t> d_next_pair_node;
        thrust::device_vector<uint32_t> d_next_pair_prim;

        if (num_pairs > 0)
        {
            const uint32_t pair_blocks = (num_pairs - 1u) / block_size + 1u;
            thrust::device_vector<uint8_t> d_mask(num_pairs, 0);
            thrust::device_vector<uint32_t> d_counts(num_pairs, 0u);
            thrust::device_vector<uint32_t> d_offsets(num_pairs, 0u);

            count_kdtree_child_pairs_kernel<<<pair_blocks, block_size>>>(
                num_pairs,
                m_geometry_context->primitives.view(),
                m_geometry_context->vertices.view(),
                d_prim_slot_ids.data().get(),
                d_prim_min.data().get(),
                d_prim_max.data().get(),
                d_pair_node.data().get(),
                d_pair_prim.data().get(),
                d_is_split.data().get(),
                d_split_axis.data().get(),
                d_split_pos.data().get(),
                d_active_bmin.data().get(),
                d_active_bmax.data().get(),
                d_mask.data().get(),
                d_counts.data().get());
            CUDA_CHECK(cudaDeviceSynchronize());

            thrust::exclusive_scan(d_counts.begin(), d_counts.end(), d_offsets.begin());
            const uint32_t total_next_pairs = d_offsets[num_pairs - 1u] + d_counts[num_pairs - 1u];

            if (total_next_pairs > 0)
            {
                d_next_pair_node.resize(total_next_pairs);
                d_next_pair_prim.resize(total_next_pairs);

                emit_kdtree_child_pairs_kernel<<<pair_blocks, block_size>>>(
                    num_pairs,
                    d_pair_node.data().get(),
                    d_pair_prim.data().get(),
                    d_split_rank.data().get(),
                    d_mask.data().get(),
                    d_offsets.data().get(),
                    d_next_pair_node.data().get(),
                    d_next_pair_prim.data().get());
                CUDA_CHECK(cudaDeviceSynchronize());

                thrust::sort_by_key(
                    thrust::device,
                    d_next_pair_node.begin(),
                    d_next_pair_node.end(),
                    d_next_pair_prim.begin());

                thrust::device_vector<uint32_t> d_run_starts(num_children, 0u);
                thrust::device_vector<uint32_t> d_run_ends(num_children, 0u);
                const uint32_t next_pair_blocks = (total_next_pairs - 1u) / block_size + 1u;
                find_sorted_kdtree_runs_kernel<<<next_pair_blocks, block_size>>>(
                    total_next_pairs,
                    d_next_pair_node.data().get(),
                    d_run_starts.data().get(),
                    d_run_ends.data().get());

                const uint32_t child_blocks = (num_children - 1u) / block_size + 1u;
                compute_kdtree_node_counts_kernel<<<child_blocks, block_size>>>(
                    num_children,
                    d_run_starts.data().get(),
                    d_run_ends.data().get(),
                    d_next_node_counts.data().get());
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }

        d_active_node_idx = std::move(d_next_node_idx);
        d_active_bmin = std::move(d_next_bmin);
        d_active_bmax = std::move(d_next_bmax);
        d_node_counts = std::move(d_next_node_counts);
        d_pair_node = std::move(d_next_pair_node);
        d_pair_prim = std::move(d_next_pair_prim);
    }

    finalize_tree_pools(space, root_min, root_max, host_nodes, leaf_pairs, total_leaves);
}

void KDTreeAS::build_cpu(uint32_t /*space_id*/, Space space, uint32_t number_of_triangles, uint32_t occupied_slots, bool is_subspace)
{
    if (number_of_triangles == 0)
        return;

    soa::soa_vector<PrimitiveChunk16> host_primitives = m_geometry_context->primitives;
    soa::soa_vector<Vertex> host_vertices = m_geometry_context->vertices;
    const auto prim_view = host_primitives.view();
    const auto vert_view = host_vertices.view();

    std::vector<glm::vec3> prim_min(number_of_triangles);
    std::vector<glm::vec3> prim_max(number_of_triangles);
    std::vector<uint32_t> prim_slot_ids(number_of_triangles);

    glm::vec3 root_min(FLT_MAX);
    glm::vec3 root_max(-FLT_MAX);
    for (uint32_t i = 0; i < number_of_triangles; ++i)
    {
        const uint32_t slot_id = space.primitives_start + i * occupied_slots;
        prim_slot_ids[i] = slot_id;

        glm::vec3 v0, v1, v2, n0, n1, n2;
        const PrimitiveType ptype = kdtree_detail::get_primitive_triangle_vertices(
            prim_view, vert_view, slot_id, v0, v1, v2, n0, n1, n2);
        kdtree_detail::get_primitive_aabb(ptype, v0, v1, v2, n0, n1, n2, prim_min[i], prim_max[i]);

        root_min = glm::min(root_min, prim_min[i]);
        root_max = glm::max(root_max, prim_max[i]);
    }

    const glm::vec3 pad = glm::max((root_max - root_min) * 1e-4f, glm::vec3(1e-5f));
    root_min -= pad;
    root_max += pad;

    const glm::vec3 root_ext = root_max - root_min;
    const float root_max_ext = fmaxf(root_ext.x, fmaxf(root_ext.y, root_ext.z));

    std::vector<uint32_t> root_prims(number_of_triangles);
    std::iota(root_prims.begin(), root_prims.end(), 0u);

    std::vector<KDNode> host_nodes(1);
    std::vector<std::pair<uint32_t, uint32_t>> leaf_pairs;

    kdtree_detail::CpuBuildContext ctx{
        .prim_view = prim_view,
        .vert_view = vert_view,
        .prim_min = prim_min.data(),
        .prim_max = prim_max.data(),
        .prim_slot_ids = prim_slot_ids.data(),
        .root_max_ext = root_max_ext,
        .split_mode = config.split_mode,
        .max_depth = config.max_depth < 1 ? 1 : (config.max_depth > 30 ? 30 : config.max_depth),
        .max_leaf_primitives = std::max(config.max_leaf_primitives, 1),
        .traversal_cost = config.traversal_cost,
        .intersection_cost = config.intersection_cost,
        .is_subspace = is_subspace,
        .nodes = &host_nodes,
        .leaf_pairs = &leaf_pairs,
        .total_leaves = 0
    };

    kdtree_detail::build_cpu_recursive(ctx, 0u, root_min, root_max, 0, root_prims);
    finalize_tree_pools(space, root_min, root_max, host_nodes, leaf_pairs, ctx.total_leaves);
}

void KDTreeAS::build_subspaces_cpu(uint32_t space_id, Space space, std::span<const SubspacePrimitive> subspaces)
{
    if (subspaces.empty())
        return;

    const auto t0 = std::chrono::high_resolution_clock::now();
    const uint32_t count = static_cast<uint32_t>(subspaces.size());

    std::vector<glm::vec3> prim_min(count);
    std::vector<glm::vec3> prim_max(count);
    std::vector<uint32_t> prim_slot_ids(count);

    glm::vec3 root_min(FLT_MAX);
    glm::vec3 root_max(-FLT_MAX);
    for (uint32_t i = 0; i < count; ++i)
    {
        prim_slot_ids[i] = space.primitives_start + i * SubspacePrimitive::occupied_slots;
        prim_min[i] = subspaces[i].bounds_min;
        prim_max[i] = subspaces[i].bounds_max;
        root_min = glm::min(root_min, prim_min[i]);
        root_max = glm::max(root_max, prim_max[i]);
    }

    const glm::vec3 pad = glm::max((root_max - root_min) * 1e-4f, glm::vec3(1e-5f));
    root_min -= pad;
    root_max += pad;

    const glm::vec3 root_ext = root_max - root_min;
    const float root_max_ext = fmaxf(root_ext.x, fmaxf(root_ext.y, root_ext.z));

    std::vector<uint32_t> root_prims(count);
    std::iota(root_prims.begin(), root_prims.end(), 0u);

    std::vector<KDNode> host_nodes(1);
    host_nodes.reserve(count * 4u);
    std::vector<std::pair<uint32_t, uint32_t>> leaf_pairs;
    leaf_pairs.reserve(count * 2u);

    kdtree_detail::CpuBuildContext ctx{
        .prim_view = {},
        .vert_view = {},
        .prim_min = prim_min.data(),
        .prim_max = prim_max.data(),
        .prim_slot_ids = prim_slot_ids.data(),
        .root_max_ext = root_max_ext,
        .split_mode = config.split_mode,
        .max_depth = config.max_depth < 1 ? 1 : (config.max_depth > 30 ? 30 : config.max_depth),
        .max_leaf_primitives = std::max(config.max_leaf_primitives, 1),
        .traversal_cost = config.traversal_cost,
        .intersection_cost = config.intersection_cost,
        .is_subspace = true,
        .nodes = &host_nodes,
        .leaf_pairs = &leaf_pairs,
        .total_leaves = 0
    };

    kdtree_detail::build_cpu_recursive(ctx, 0u, root_min, root_max, 0, root_prims);
    finalize_tree_pools(space, root_min, root_max, host_nodes, leaf_pairs, ctx.total_leaves);

    const auto t1 = std::chrono::high_resolution_clock::now();
    m_build_times_ms[space_id] = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void KDTreeAS::build(uint32_t space_id, Space space, PrimitiveType primitive_type)
{
    assert(m_geometry_context != nullptr);
    auto t0 = std::chrono::high_resolution_clock::now();

    uint32_t occupied_slots = Triangle::occupied_slots;
    if (primitive_type == PrimitiveType::FullTriangle)
        occupied_slots = InlinedTriangle::occupied_slots;
    else if (primitive_type == PrimitiveType::NagataPatch)
        occupied_slots = NagataPatchPrimitive::occupied_slots;
    else if (primitive_type == PrimitiveType::Subspace)
        occupied_slots = SubspacePrimitive::occupied_slots;

    const uint32_t number_of_triangles = space.primitives_size / occupied_slots;
    if (number_of_triangles == 0)
        return;

    const bool is_subspace = (primitive_type == PrimitiveType::Subspace);
    if (config.use_gpu)
    {
        build_gpu(space_id, space, number_of_triangles, occupied_slots, is_subspace);
    }
    else
    {
        build_cpu(space_id, space, number_of_triangles, occupied_slots, is_subspace);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    m_build_times_ms[space_id] = std::chrono::duration<double, std::milli>(t1 - t0).count();
}
