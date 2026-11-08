#pragma once

#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#define NOMINMAX
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/config.h>

#include "../cuda_common.hpp"
#include "../../external/soa_vector/src/cuda_common.hpp"

struct Ray_ {
    glm::vec3 origin;
    glm::vec3 dir; // Must be unit length (normalized)
    float tMin = 1e-4f;
    float tMax = 1e30f;
};

struct HitRecord {
    float t;
    float u, v;
    glm::vec3 position;
    glm::vec3 normal;
};

struct AABB {
    glm::vec3 min;
    glm::vec3 max;

    CUDA_HOST_DEVICE bool containsPoint(const glm::vec3& p) const {
        return (p.x >= min.x && p.x <= max.x) &&
               (p.y >= min.y && p.y <= max.y) &&
               (p.z >= min.z && p.z <= max.z);
    }

    CUDA_HOST_DEVICE bool containsBox(const AABB& other) const {
        return (other.min.x >= min.x && other.max.x <= max.x) &&
               (other.min.y >= min.y && other.max.y <= max.y) &&
               (other.min.z >= min.z && other.max.z <= max.z);
    }

    CUDA_HOST_DEVICE bool overlapsBox(const AABB& other) const {
        return (min.x <= other.max.x && max.x >= other.min.x) &&
               (min.y <= other.max.y && max.y >= other.min.y) &&
               (min.z <= other.max.z && max.z >= other.min.z);
    }
};


struct NagataPatch {
    glm::vec3 X1, X2, X3;
    glm::vec3 C12, C23, C31;

    CUDA_HOST_DEVICE glm::vec3 evaluate(float u, float v) const {
        float w = 1.0f - u - v;
        return (u * X1 + v * X2 + w * X3) - ((u * v) * C12 + (v * w) * C23 + (w * u) * C31);
    }

    // Conservative 6-point Quadratic Bezier Control-Hull AABB
    CUDA_HOST_DEVICE AABB controlHullAABB() const {
        glm::vec3 b12 = 0.5f * (X1 + X2 - C12);
        glm::vec3 b23 = 0.5f * (X2 + X3 - C23);
        glm::vec3 b31 = 0.5f * (X3 + X1 - C31);

        glm::vec3 mn = glm::min(glm::min(X1, X2), glm::min(X3, glm::min(b12, glm::min(b23, b31))));
        glm::vec3 mx = glm::max(glm::max(X1, X2), glm::max(X3, glm::max(b12, glm::max(b23, b31))));
        return { mn, mx };
    }

    // Exact Analytic AABB of the Nagata Patch (Closed-form quadratic extrema)
    // Also returns true if ANY evaluated extremum point on the patch lies inside `targetBox`
    CUDA_HOST_DEVICE AABB exactAABB(const AABB& targetBox, bool& outAnyExtremumInBox) const {
        glm::vec3 mn = glm::min(glm::min(X1, X2), X3);
        glm::vec3 mx = glm::max(glm::max(X1, X2), X3);

        auto recordSample = [&](float u, float v) {
            glm::vec3 p = evaluate(u, v);
            mn = glm::min(mn, p);
            mx = glm::max(mx, p);
            if (targetBox.containsPoint(p)) {
                outAnyExtremumInBox = true;
            }
        };

        // Check per-axis 1D edge critical points and 2D interior critical points
        for (int k = 0; k < 3; ++k) {
            // 1. Edge 1-2: u = 1-t, v = t, w = 0
            if (glm::abs(C12[k]) > 1e-15f) {
                float t = 0.5f + (X1[k] - X2[k]) / (2.0f * C12[k]);
                if (t > 0.0f && t < 1.0f) recordSample(1.0f - t, t);
            }
            // 2. Edge 2-3: u = 0, v = 1-t, w = t
            if (glm::abs(C23[k]) > 1e-15f) {
                float t = 0.5f + (X2[k] - X3[k]) / (2.0f * C23[k]);
                if (t > 0.0f && t < 1.0f) recordSample(0.0f, 1.0f - t);
            }
            // 3. Edge 3-1: u = t, v = 0, w = 1-t
            if (glm::abs(C31[k]) > 1e-15f) {
                float t = 0.5f + (X3[k] - X1[k]) / (2.0f * C31[k]);
                if (t > 0.0f && t < 1.0f) recordSample(t, 0.0f);
            }

            // 4. 2D Interior Stationary Point: df/du = 0, df/dv = 0
            float Auu = 2.0f * C31[k];
            float Avv = 2.0f * C23[k];
            float Auv = C23[k] + C31[k] - C12[k];
            float det = Auu * Avv - Auv * Auv;

            if (glm::abs(det) > 1e-30f) {
                float bu = -(X1[k] - X3[k] - C31[k]);
                float bv = -(X2[k] - X3[k] - C23[k]);
                float u  = (bu * Avv - bv * Auv) / det;
                float v  = (Auu * bv - Auv * bu) / det;
                if (u > 0.0f && v > 0.0f && (u + v) < 1.0f) {
                    recordSample(u, v);
                }
            }
        }
        return { mn, mx };
    }

    // Helper: Compute the Nagata curvature vector for a directed edge (Xi -> Xj)
    CUDA_HOST_DEVICE static glm::vec3 computeEdgeCurvature(
        const glm::vec3& Xi, const glm::vec3& Xj,
        const glm::vec3& Ni, const glm::vec3& Nj)
    {
        glm::vec3 D = Xj - Xi;
        glm::vec3 N_avg  = 0.5f * (Ni + Nj);
        glm::vec3 N_diff = 0.5f * (Ni - Nj);

        float c = glm::dot(Ni, Nj); // Cosine of angle between normals
        float max_d = 0.5f * glm::length(D);
        float d1 = glm::clamp(glm::dot(D, Ni), -max_d, max_d);
        float d2 = glm::clamp(glm::dot(D, Nj), -max_d, max_d);
        float d = 0.5f * (d1 + d2);
        float delta_d = 0.5f * (d1 - d2);

        // Guard against sharp creases / opposing normals (angle >= 90 degrees)
        if (c <= 0.0f) {
            return glm::vec3(0.0f);
        }

        // Regularized circular-arc Nagata coefficients (singularity-free at 90 deg)
        // Note: In Nagata's original piecewise formulation, for small angles:
        // alpha = delta_d / (1 - c) when c != 1, and beta = d / c.
        // Using 2.0f / (1.0f + c) avoids division-by-zero at both c = 1 and c = 0
        // while preserving exact second-order circular curvature.
        float inv_denom = 2.0f / (1.0f + c);

        // If normals are nearly identical (flat or pure S-curve), delta_d -> 0
        float alpha = delta_d * inv_denom;
        float beta  = d * inv_denom;

        return alpha * N_avg + beta * N_diff;
    }

    // Construct the patch from 3 vertices and 3 unit normals
    CUDA_HOST_DEVICE static NagataPatch fromTriangle(
        const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3,
        const glm::vec3& n1, const glm::vec3& n2, const glm::vec3& n3)
    {
        NagataPatch patch;
        patch.X1 = v1;
        patch.X2 = v2;
        patch.X3 = v3;

        // Ensure normals are safely normalized
        glm::vec3 Ng = glm::cross(v2 - v1, v3 - v1);
        float lenNg2 = glm::dot(Ng, Ng);
        if (lenNg2 > 1e-20f) {
            Ng *= glm::inversesqrt(lenNg2);
        }
        auto safeNorm = [&](const glm::vec3& n) {
            float l2 = glm::dot(n, n);
            return (l2 > 1e-12f) ? (n * glm::inversesqrt(l2)) : Ng;
        };
        glm::vec3 N1 = safeNorm(n1);
        glm::vec3 N2 = safeNorm(n2);
        glm::vec3 N3 = safeNorm(n3);

        // Compute the 3 edge curvature vectors
        // Note: C_ij is symmetric with respect to (i, j) because switching i and j
        // negates both D and N_diff, keeping all dot products and signs identical!
        patch.C12 = computeEdgeCurvature(v1, v2, N1, N2);
        patch.C23 = computeEdgeCurvature(v2, v3, N2, N3);
        patch.C31 = computeEdgeCurvature(v3, v1, N3, N1);

        return patch;
    }

    // Evaluate 3D surface position at barycentric coordinates (u, v), where w = 1 - u - v
    CUDA_HOST_DEVICE glm::vec3 evaluatePosition(float u, float v) const {
        float w = 1.0f - u - v;

        glm::vec3 linearPos = u * X1 + v * X2 + w * X3;
        glm::vec3 bulge     = (u * v) * C12 + (v * w) * C23 + (w * u) * C31;

        return linearPos - bulge;
    }

    // Evaluate exact analytic surface tangents and normal at (u, v)
    CUDA_HOST_DEVICE glm::vec3 evaluateNormal(float u, float v) const {
        // Partial derivative dX/du (with w = 1 - u - v)
        glm::vec3 dXdu = (X1 - X3)
                       - v * C12
                       + v * C23
                       - (1.0f - 2.0f * u - v) * C31;

        // Partial derivative dX/dv (with w = 1 - u - v)
        glm::vec3 dXdv = (X2 - X3)
                       - u * C12
                       - (1.0f - u - 2.0f * v) * C23
                       + u * C31;

        return glm::normalize(glm::cross(dXdu, dXdv));
    }

    // Convert to a 6-point Quadratic Bezier Triangle to compute a guaranteed tight AABB
    CUDA_HOST_DEVICE void computeBoundingBox(glm::vec3& outMin, glm::vec3& outMax) const {
        // Quadratic Bezier corner control points are X1, X2, X3.
        // Edge midpoint Bezier control points are:
        glm::vec3 B12 = 0.5f * (X1 + X2 - C12);
        glm::vec3 B23 = 0.5f * (X2 + X3 - C23);
        glm::vec3 B31 = 0.5f * (X3 + X1 - C31);

        outMin = glm::min(glm::min(X1, X2), X3);
        outMin = glm::min(outMin, glm::min(glm::min(B12, B23), B31));

        outMax = glm::max(glm::max(X1, X2), X3);
        outMax = glm::max(outMax, glm::max(glm::max(B12, B23), B31));
    }
};


// Branchless orthonormal basis (Pixar / Duff et al. 2017)
// Given unit ray direction D, produces two unit vectors E1, E2 perpendicular to D
inline void buildOrthonormalBasis(const glm::vec3& D, glm::vec3& E1, glm::vec3& E2) {
    float sign = std::copysign(1.0f, D.z);
    const float a = -1.0f / (sign + D.z);
    const float b = D.x * D.y * a;
    E1 = glm::vec3(1.0f + sign * D.x * D.x * a, sign * b, -sign * D.x);
    E2 = glm::vec3(b, sign + D.y * D.y * a, -D.y);
}

// Fast 2D Ray-Plane Projected Newton-Raphson Intersection for a Nagata Patch
inline bool intersectNagataPatch(const Ray_& ray, const NagataPatch& patch, HitRecord& hit) {
    // -------------------------------------------------------------------------
    // STEP 1: Build 2D Ray-Centric Coordinate System & Project Control Vectors
    // -------------------------------------------------------------------------
    glm::vec3 E1, E2;
    buildOrthonormalBasis(ray.dir, E1, E2);

    auto projectDir = [&](const glm::vec3& vec) -> glm::vec2 {
        return glm::vec2(glm::dot(vec, E1), glm::dot(vec, E2));
    };

    glm::vec3 relX1 = patch.X1 - ray.origin;
    glm::vec3 relX2 = patch.X2 - ray.origin;
    glm::vec3 relX3 = patch.X3 - ray.origin;

    // 2D projected vertices (relative to ray origin) and 2D curvature vectors
    glm::vec2 x1 = projectDir(relX1);
    glm::vec2 x2 = projectDir(relX2);
    glm::vec2 x3 = projectDir(relX3);

    glm::vec2 c12 = projectDir(patch.C12);
    glm::vec2 c23 = projectDir(patch.C23);
    glm::vec2 c31 = projectDir(patch.C31);

    // -------------------------------------------------------------------------
    // STEP 2: Early 2D Bezier Control-Hull Reject (Tight Ray-Aligned Bound)
    // -------------------------------------------------------------------------
    glm::vec2 b12 = 0.5f * (x1 + x2 - c12);
    glm::vec2 b23 = 0.5f * (x2 + x3 - c23);
    glm::vec2 b31 = 0.5f * (x3 + x1 - c31);

    glm::vec2 hullMin = glm::min(glm::min(x1, x2), glm::min(x3, glm::min(b12, glm::min(b23, b31))));
    glm::vec2 hullMax = glm::max(glm::max(x1, x2), glm::max(x3, glm::max(b12, glm::max(b23, b31))));

    if (hullMin.x > 0.0f || hullMax.x < 0.0f || hullMin.y > 0.0f || hullMax.y < 0.0f) {
        return false; // Early exit: ray misses the 2D convex hull of the patch
    }

    // -------------------------------------------------------------------------
    // STEP 3: Initial Guess (u, v) from the Projected 2D Flat Triangle
    // -------------------------------------------------------------------------
    glm::vec2 edge1 = x1 - x3;
    glm::vec2 edge2 = x2 - x3;
    float scale2 = glm::dot(edge1, edge1) + glm::dot(edge2, edge2);
    float detFlat = edge1.x * edge2.y - edge1.y * edge2.x;

    float u = 1.0f / 3.0f;
    float v = 1.0f / 3.0f;

    if (std::abs(detFlat) > 1e-6f * scale2 && std::abs(detFlat) > 1e-20f) {
        float invDet = 1.0f / detFlat;
        u = (-x3.x * edge2.y + x3.y * edge2.x) * invDet;
        v = ( edge1.x * (-x3.y) - edge1.y * (-x3.x)) * invDet;

        u = glm::clamp(u, 0.0f, 1.0f);
        v = glm::clamp(v, 0.0f, 1.0f);
        if (u + v > 1.0f) {
            float sum = u + v;
            u /= sum;
            v /= sum;
        }
    }

    // -------------------------------------------------------------------------
    // STEP 4: 2D Newton-Raphson Iteration
    // -------------------------------------------------------------------------
    const int MAX_ITER = 5;
    const float CONVERGENCE_EPS2 = 1e-12f * scale2;
    bool converged = false;

    for (int i = 0; i < MAX_ITER; ++i) {
        float w = 1.0f - u - v;

        glm::vec2 F = (u * x1 + v * x2 + w * x3)
                    - ((u * v) * c12 + (v * w) * c23 + (w * u) * c31);

        if (glm::dot(F, F) < CONVERGENCE_EPS2) {
            converged = true;
            break;
        }

        glm::vec2 J_u = edge1 - v * c12 + v * c23 - (1.0f - 2.0f * u - v) * c31;
        glm::vec2 J_v = edge2 - u * c12 - (1.0f - u - 2.0f * v) * c23 + u * c31;

        float detJ = J_u.x * J_v.y - J_u.y * J_v.x;
        if (std::abs(detJ) <= 1e-7f * scale2 || std::abs(detJ) <= 1e-20f) {
            break;
        }

        float invDetJ = 1.0f / detJ;
        float du = ( J_v.y * F.x - J_v.x * F.y) * invDetJ;
        float dv = (-J_u.y * F.x + J_u.x * F.y) * invDetJ;

        u = glm::clamp(u - du, -0.5f, 1.5f);
        v = glm::clamp(v - dv, -0.5f, 1.5f);
    }

    if (!converged) {
        float w = 1.0f - u - v;
        glm::vec2 F = (u * x1 + v * x2 + w * x3)
                    - ((u * v) * c12 + (v * w) * c23 + (w * u) * c31);
        if (glm::dot(F, F) > 1e-9f * scale2) {
            return false;
        }
    }

    // -------------------------------------------------------------------------
    // STEP 5: Barycentric Domain & Ray Interval [tMin, tMax] Validation
    // -------------------------------------------------------------------------
    const float BARY_EPS = -1e-4f;
    float w = 1.0f - u - v;
    if (u < BARY_EPS || v < BARY_EPS || w < BARY_EPS) {
        return false;
    }

    glm::vec3 hitPos = patch.evaluatePosition(u, v);
    float t = glm::dot(hitPos - ray.origin, ray.dir);

    if (t < ray.tMin || t > ray.tMax) {
        return false;
    }

    hit.t = t;
    hit.u = u;
    hit.v = v;
    hit.position = hitPos;
    hit.normal = patch.evaluateNormal(u, v);
    return true;
}

//---------------------------------------------------------------------

// Packed 16-byte aligned structure for GPU StructuredBuffer / SSBO
// Using vec4 packs cleanly into 4x 128-bit memory loads (v_load_dwordx4)
struct alignas(16) GPUNagataPatch {
    glm::vec4 X1_pad;  // xyz = X1
    glm::vec4 X2_pad;  // xyz = X2
    glm::vec4 X3_pad;  // xyz = X3
    glm::vec4 C12_pad; // xyz = C12
    glm::vec4 C23_pad; // xyz = C23
    glm::vec4 C31_pad; // xyz = C31
};

// GPU Hardware Intrinsics Emulation (Single-cycle on NVIDIA/AMD GPUs)
inline CUDA_HOST_DEVICE glm::vec2 min3(const glm::vec2& a, const glm::vec2& b, const glm::vec2& c) {
    return glm::min(glm::min(a, b), c);
}
inline CUDA_HOST_DEVICE glm::vec2 max3(const glm::vec2& a, const glm::vec2& b, const glm::vec2& c) {
    return glm::max(glm::max(a, b), c);
}

// Branchless Orthonormal Basis
inline CUDA_HOST_DEVICE
void buildBasisGPU(const glm::vec3& D, glm::vec3& E1, glm::vec3& E2) {
    float s = (D.z >= 0.0f) ? 1.0f : -1.0f;
    float a = -1.0f / (s + D.z);
    float b = D.x * D.y * a;
    E1 = glm::vec3(glm::fma(s * D.x * D.x, a, 1.0f), s * b, -s * D.x);
    E2 = glm::vec3(b, glm::fma(D.y * D.y, a, s), -D.y);
}

// =============================================================================
// GPU-Optimized Intersection Kernel (Minimized VGPR Register Pressure)
// Returns true and outputs (t, u, v) ONLY. Normal is deferred to Closest-Hit!
// =============================================================================
inline bool intersectNagataGPU(
    const glm::vec3& rayOrigin,
    const glm::vec3& rayDir,
    float tMin,
    float tMax,
    const GPUNagataPatch& patch, // Passed by const ref (SSBO/Global Load on GPU)
    float& outT,
    glm::vec2& outUV)
{
    // 1. Build 2D Ray Basis (6 VGPRs, immediately consumed during projection)
    glm::vec3 E1, E2;
    buildBasisGPU(rayDir, E1, E2);

    auto proj2D = [&](const glm::vec3& v) -> glm::vec2 {
        return glm::vec2(glm::dot(v, E1), glm::dot(v, E2));
    };

    // 2. Stream & Project Directly to 2D (Saves 6 VGPRs by not keeping 3D coords live)
    glm::vec2 x3 = proj2D(glm::vec3(patch.X3_pad) - rayOrigin);
    glm::vec2 e1 = proj2D(glm::vec3(patch.X1_pad) - glm::vec3(patch.X3_pad)); // x1 - x3
    glm::vec2 e2 = proj2D(glm::vec3(patch.X2_pad) - glm::vec3(patch.X3_pad)); // x2 - x3

    glm::vec2 c12 = proj2D(glm::vec3(patch.C12_pad));
    glm::vec2 c23 = proj2D(glm::vec3(patch.C23_pad));
    glm::vec2 c31 = proj2D(glm::vec3(patch.C31_pad));

    // 3. Branchless 2D Control-Hull Compute + Single Warp-Coherent Early Exit
    glm::vec2 x1 = x3 + e1;
    glm::vec2 x2 = x3 + e2;
    glm::vec2 b12 = 0.5f * (x1 + x2 - c12);
    glm::vec2 b23 = 0.5f * (x2 + x3 - c23);
    glm::vec2 b31 = 0.5f * (x3 + x1 - c31);

    glm::vec2 hMin = glm::min(min3(x1, x2, x3), min3(b12, b23, b31));
    glm::vec2 hMax = glm::max(max3(x1, x2, x3), max3(b12, b23, b31));

    if ((hMin.x > 0.0f) | (hMax.x < 0.0f) | (hMin.y > 0.0f) | (hMax.y < 0.0f)) {
        return false;
    }

    // 4. Branchless Flat-Triangle Initial Guess + Triangle Domain Clamp
    float scale2 = glm::dot(e1, e1) + glm::dot(e2, e2);
    float detFlat = glm::fma(e1.x, e2.y, -e1.y * e2.x);
    float u = 1.0f / 3.0f;
    float v = 1.0f / 3.0f;
    if (std::abs(detFlat) > 1e-6f * scale2 && std::abs(detFlat) > 1e-20f) {
        float invDetFlat = 1.0f / detFlat;
        u = glm::clamp(glm::fma(-x3.x, e2.y,  x3.y * e2.x) * invDetFlat, 0.0f, 1.0f);
        v = glm::clamp(glm::fma( e1.x, -x3.y, -e1.y * -x3.x) * invDetFlat, 0.0f, 1.0f);
        float sumUV = u + v;
        if (sumUV > 1.0f) {
            float invSum = 1.0f / sumUV;
            u *= invSum;
            v *= invSum;
        }
    }

    // 5. Unrolled 5-Step Branchless Newton-Raphson Solver
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        float w = 1.0f - u - v;

        glm::vec2 F = x3 + u * e1 + v * e2
                    - ((u * v) * c12 + (v * w) * c23 + (w * u) * c31);

        glm::vec2 J_u = e1 - v * c12 + v * c23 - (w - u) * c31;
        glm::vec2 J_v = e2 - u * c12 - (w - v) * c23 + u * c31;

        float detJ = glm::fma(J_u.x, J_v.y, -J_u.y * J_v.x);
        float invDetJ = (std::abs(detJ) > 1e-7f * scale2 && std::abs(detJ) > 1e-20f) ? (1.0f / detJ) : 0.0f;

        u = glm::clamp(u - glm::fma( J_v.y, F.x, -J_v.x * F.y) * invDetJ, -0.5f, 1.5f);
        v = glm::clamp(v - glm::fma(-J_u.y, F.x,  J_u.x * F.y) * invDetJ, -0.5f, 1.5f);
    }

    // 6. Fast Barycentric & Scale-Invariant Residual Verification
    float w = 1.0f - u - v;
    const float BARY_EPS = -1e-4f;
    if ((u < BARY_EPS) | (v < BARY_EPS) | (w < BARY_EPS)) {
        return false;
    }

    glm::vec2 finalF = x3 + u * e1 + v * e2
                     - ((u * v) * c12 + (v * w) * c23 + (w * u) * c31);
    if (glm::dot(finalF, finalF) > 1e-9f * scale2) {
        return false; // Did not converge to the ray
    }

    // 7. Reconstruct 3D Hit Depth `t` ONLY for Validated Hits!
    glm::vec3 hitPos = u * glm::vec3(patch.X1_pad)
                     + v * glm::vec3(patch.X2_pad)
                     + w * glm::vec3(patch.X3_pad)
                     - ((u * v) * glm::vec3(patch.C12_pad)
                      + (v * w) * glm::vec3(patch.C23_pad)
                      + (w * u) * glm::vec3(patch.C31_pad));

    float t = glm::dot(hitPos - rayOrigin, rayDir);
    if (t < tMin || t > tMax) {
        return false;
    }

    outT  = t;
    outUV = glm::vec2(u, v);
    return true;
}

// =============================================================================
// Direct Precomputed Nagata Intersection (X1..X3, C12..C31)
// Used when NagataPatch is stored as a first-class primitive in PrimitiveChunk16
// =============================================================================
inline CUDA_HOST_DEVICE bool intersectNagataPrecomputed(
    const glm::vec3& rayOrigin,
    const glm::vec3& rayDir,
    float tMin,
    float tMax,
    const glm::vec3& X1,
    const glm::vec3& X2,
    const glm::vec3& X3,
    const glm::vec3& C12,
    const glm::vec3& C23,
    const glm::vec3& C31,
    float& outT,
    float& outU,
    float& outV)
{
    // 1. Build 2D Ray Basis (Pixar Duff et al.)
    glm::vec3 E1, E2;
    buildBasisGPU(rayDir, E1, E2);

    auto proj2D = [&](const glm::vec3& v) -> glm::vec2 {
        return glm::vec2(glm::dot(v, E1), glm::dot(v, E2));
    };

    // 2. Stream & Project Directly to 2D
    glm::vec2 x3 = proj2D(X3 - rayOrigin);
    glm::vec2 e1 = proj2D(X1 - X3);
    glm::vec2 e2 = proj2D(X2 - X3);

    glm::vec2 c12 = proj2D(C12);
    glm::vec2 c23 = proj2D(C23);
    glm::vec2 c31 = proj2D(C31);

    // 3. Branchless 2D Control-Hull Compute + Early Exit
    glm::vec2 x1 = x3 + e1;
    glm::vec2 x2 = x3 + e2;
    glm::vec2 b12 = 0.5f * (x1 + x2 - c12);
    glm::vec2 b23 = 0.5f * (x2 + x3 - c23);
    glm::vec2 b31 = 0.5f * (x3 + x1 - c31);

    glm::vec2 hMin = glm::min(min3(x1, x2, x3), min3(b12, b23, b31));
    glm::vec2 hMax = glm::max(max3(x1, x2, x3), max3(b12, b23, b31));

    if ((hMin.x > 0.0f) | (hMax.x < 0.0f) | (hMin.y > 0.0f) | (hMax.y < 0.0f)) {
        return false;
    }

    // 4. Branchless Flat-Triangle Initial Guess
    float scale2 = glm::dot(e1, e1) + glm::dot(e2, e2);
    float detFlat = glm::fma(e1.x, e2.y, -e1.y * e2.x);
    float u = 1.0f / 3.0f;
    float v = 1.0f / 3.0f;
    if (std::abs(detFlat) > 1e-6f * scale2 && std::abs(detFlat) > 1e-20f) {
        float invDetFlat = 1.0f / detFlat;
        u = glm::clamp(glm::fma(-x3.x, e2.y,  x3.y * e2.x) * invDetFlat, 0.0f, 1.0f);
        v = glm::clamp(glm::fma( e1.x, -x3.y, -e1.y * -x3.x) * invDetFlat, 0.0f, 1.0f);
        float sumUV = u + v;
        if (sumUV > 1.0f) {
            float invSum = 1.0f / sumUV;
            u *= invSum;
            v *= invSum;
        }
    }

    // 5. Unrolled 5-Step Branchless Newton-Raphson Solver
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        float w = 1.0f - u - v;

        glm::vec2 F = x3 + u * e1 + v * e2
                    - ((u * v) * c12 + (v * w) * c23 + (w * u) * c31);

        glm::vec2 J_u = e1 - v * c12 + v * c23 - (w - u) * c31;
        glm::vec2 J_v = e2 - u * c12 - (w - v) * c23 + u * c31;

        float detJ = glm::fma(J_u.x, J_v.y, -J_u.y * J_v.x);
        float invDetJ = (std::abs(detJ) > 1e-7f * scale2 && std::abs(detJ) > 1e-20f) ? (1.0f / detJ) : 0.0f;

        u = glm::clamp(u - glm::fma( J_v.y, F.x, -J_v.x * F.y) * invDetJ, -0.5f, 1.5f);
        v = glm::clamp(v - glm::fma(-J_u.y, F.x,  J_u.x * F.y) * invDetJ, -0.5f, 1.5f);
    }

    // 6. Fast Barycentric & Scale-Invariant Residual Verification
    float w = 1.0f - u - v;
    const float BARY_EPS = -1e-4f;
    if ((u < BARY_EPS) | (v < BARY_EPS) | (w < BARY_EPS)) {
        return false;
    }

    glm::vec2 finalF = x3 + u * e1 + v * e2
                     - ((u * v) * c12 + (v * w) * c23 + (w * u) * c31);
    if (glm::dot(finalF, finalF) > 1e-9f * scale2) {
        return false;
    }

    // 7. Reconstruct 3D Hit Depth `t`
    glm::vec3 hitPos = u * X1 + v * X2 + w * X3
                     - ((u * v) * C12 + (v * w) * C23 + (w * u) * C31);

    float t = glm::dot(hitPos - rayOrigin, rayDir);
    if (t < tMin || t > tMax) {
        return false;
    }

    outT = t;
    outU = v;             // Barycentric weight of X2 (vertex 1), matching tuv[1] / w1 convention
    outV = 1.0f - u - v;  // Barycentric weight of X3 (vertex 2), matching tuv[2] / w2 convention
    return true;
}


//------------

// 16-Byte Indexed Vertex (1 single 128-bit load per vertex!)
// Pos takes 12B (FP32x3), Normal is packed into 4B (Octahedral 16-bit snorm x2)
struct alignas(16) GPUVertex {
    glm::vec3 pos;
    uint32_t  octNormal;

    // Pack unit normal into 32-bit octahedral representation (Cigolle et al. 2014)
    inline static uint32_t packOctNormal(glm::vec3 n) {
        n /= (std::abs(n.x) + std::abs(n.y) + std::abs(n.z));
        if (n.z < 0.0f) {
            float ox = (1.0f - std::abs(n.y)) * (n.x >= 0.0f ? 1.0f : -1.0f);
            float oy = (1.0f - std::abs(n.x)) * (n.y >= 0.0f ? 1.0f : -1.0f);
            n.x = ox; n.y = oy;
        }
        return glm::packSnorm2x16(glm::vec2(n.x, n.y));
    }

    // Unpack in ~6 GPU instructions
    inline static glm::vec3 unpackOctNormal(uint32_t packed) {
        glm::vec2 f = glm::unpackSnorm2x16(packed);
        glm::vec3 n(f.x, f.y, 1.0f - std::abs(f.x) - std::abs(f.y));
        float t = glm::clamp(-n.z, 0.0f, 1.0f);
        n.x += (n.x >= 0.0f) ? -t : t;
        n.y += (n.y >= 0.0f) ? -t : t;
        return glm::normalize(n);
    }
};


// Compute the two scalar weights (w_i, w_j) such that C_ij = w_i * N_i + w_j * N_j
// Takes only 12 FLOPs per edge and avoids allocating 3D N_avg / N_diff vectors!
inline CUDA_HOST_DEVICE
glm::vec2 computeNagataEdgeWeights(
    const glm::vec3& D_ij, const glm::vec3& Ni, const glm::vec3& Nj)
{
    float c = glm::dot(Ni, Nj);
    float max_d = 0.5f * glm::length(D_ij);
    float d1 = glm::clamp(glm::dot(D_ij, Ni), -max_d, max_d);
    float d2 = glm::clamp(glm::dot(D_ij, Nj), -max_d, max_d);

    // Since N_avg = 0.5*(Ni+Nj) and N_diff = 0.5*(Ni-Nj):
    // d = 0.5 * (d1 + d2), and delta_d = 0.5 * (d1 - d2)
    float inv_1_plus_c = (c > 0.0f) ? (1.0f / (1.0f + c)) : 0.0f;
    float beta = (d1 + d2) * inv_1_plus_c;

    float alpha = (1.0f - c > 1e-5f)
                ? ((d1 - d2) * inv_1_plus_c)
                : 0.0f;

    // C_ij = alpha * N_avg + beta * N_diff = 0.5*(alpha + beta)*Ni + 0.5*(alpha - beta)*Nj
    return glm::vec2(0.5f * (alpha + beta), 0.5f * (alpha - beta));
}

// =============================================================================
// On-the-Fly Indexed Nagata Intersection (Zero Precomputed Patch Buffers!)
// =============================================================================
inline CUDA_HOST_DEVICE
bool intersectNagataOnTheFly(
    const glm::vec3& rayOrigin,
    const glm::vec3& rayDir,
    float tMin,
    float tMax,
    const glm::vec3 v1, const glm::vec3 v2, const glm::vec3 v3,
    glm::vec3 N1, glm::vec3 N2, glm::vec3 N3,
    float& outT,
    float& outU,
    float& outV)
{
    glm::vec3 localOrigin = rayOrigin + rayDir * tMin;
    float localTMax = tMax - tMin;

    // 1. Compute scalar Nagata weights (w_i, w_j) for the 3 edges (~36 FLOPs total)
    glm::vec2 w12 = computeNagataEdgeWeights(v2 - v1, N1, N2);
    glm::vec2 w23 = computeNagataEdgeWeights(v3 - v2, N2, N3);
    glm::vec2 w31 = computeNagataEdgeWeights(v1 - v3, N3, N1);

    // 2. Project vertices AND normals into 2D Ray Space
    glm::vec3 E1, E2;
    buildBasisGPU(rayDir, E1, E2);
    auto proj2D = [&](const glm::vec3& v) -> glm::vec2 {
        return glm::vec2(glm::dot(v, E1), glm::dot(v, E2));
    };

    glm::vec2 x3 = proj2D(v3 - localOrigin);
    glm::vec2 e1 = proj2D(v1 - v3);
    glm::vec2 e2 = proj2D(v2 - v3);

    glm::vec2 n1_2d = proj2D(N1);
    glm::vec2 n2_2d = proj2D(N2);
    glm::vec2 n3_2d = proj2D(N3);

    // Construct 2D curvature vectors directly from 2D normals (Saves 3D C_ij!)
    glm::vec2 c12 = w12.x * n1_2d + w12.y * n2_2d;
    glm::vec2 c23 = w23.x * n2_2d + w23.y * n3_2d;
    glm::vec2 c31 = w31.x * n3_2d + w31.y * n1_2d;

    // 3. 2D Bezier Control-Hull Early Reject
    glm::vec2 x1 = x3 + e1;
    glm::vec2 x2 = x3 + e2;
    glm::vec2 b12 = 0.5f * (x1 + x2 - c12);
    glm::vec2 b23 = 0.5f * (x2 + x3 - c23);
    glm::vec2 b31 = 0.5f * (x3 + x1 - c31);

    glm::vec2 hMin = glm::min(glm::min(glm::min(x1, x2), x3), glm::min(glm::min(b12, b23), b31));
    glm::vec2 hMax = glm::max(glm::max(glm::max(x1, x2), x3), glm::max(glm::max(b12, b23), b31));
    if ((hMin.x > 0.0f) | (hMax.x < 0.0f) | (hMin.y > 0.0f) | (hMax.y < 0.0f)) {
        return false;
    }

    // 4. Dual-Root (Front & Back Slope) Unrolled Newton-Raphson with Fold Reflection
    float scale2 = glm::dot(e1, e1) + glm::dot(e2, e2);
    float tol2 = 1e-9f * scale2;
    float detFlat = glm::fma(e1.x, e2.y, -e1.y * e2.x);
    float u0 = 1.0f / 3.0f, v0 = 1.0f / 3.0f;
    if (std::abs(detFlat) > 1e-2f * scale2 && std::abs(detFlat) > 1e-20f) {
        float invDetFlat = 1.0f / detFlat;
        float uFlat = glm::fma(-x3.x, e2.y,  x3.y * e2.x) * invDetFlat;
        float vFlat = glm::fma( e1.x, -x3.y, -e1.y * -x3.x) * invDetFlat;
        float wFlat = 1.0f - uFlat - vFlat;
        if (uFlat >= -0.25f && vFlat >= -0.25f && wFlat >= -0.25f) {
            u0 = glm::clamp(uFlat, 0.0f, 1.0f);
            v0 = glm::clamp(vFlat, 0.0f, 1.0f);
            if (u0 + v0 > 1.0f) { float invS = 1.0f / (u0 + v0); u0 *= invS; v0 *= invS; }
        }
    }

    auto runNewton = [&](float& u, float& v) {
        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            float w = 1.0f - u - v;
            glm::vec2 F   = x3 + u * e1 + v * e2 - ((u * v) * c12 + (v * w) * c23 + (w * u) * c31);
            glm::vec2 J_u = e1 - v * c12 + v * c23 - (w - u) * c31;
            glm::vec2 J_v = e2 - u * c12 - (w - v) * c23 + u * c31;

            float detJ = glm::fma(J_u.x, J_v.y, -J_u.y * J_v.x);
            float invDetJ = (std::abs(detJ) > 1e-7f * scale2 && std::abs(detJ) > 1e-20f) ? (1.0f / detJ) : 0.0f;
            float stepU = glm::fma( J_v.y, F.x, -J_v.x * F.y) * invDetJ;
            float stepV = glm::fma(-J_u.y, F.x,  J_u.x * F.y) * invDetJ;
            float maxStep = glm::max(std::abs(stepU), std::abs(stepV));
            if (maxStep > 0.5f) {
                float scaleStep = 0.5f / maxStep;
                stepU *= scaleStep;
                stepV *= scaleStep;
            }
            u = glm::clamp(u - stepU, -1.0f, 2.0f);
            v = glm::clamp(v - stepV, -1.0f, 2.0f);
        }
    };

    float ua = u0, va = v0;
    runNewton(ua, va);

    // Solve 1D quadratic across the silhouette fold to find both the near and opposite-slope roots
    float wa = 1.0f - ua - va;
    glm::vec2 Fa   = x3 + ua * e1 + va * e2 - ((ua * va) * c12 + (va * wa) * c23 + (wa * ua) * c31);
    glm::vec2 J_ua = e1 - va * c12 + va * c23 - (wa - ua) * c31;
    glm::vec2 J_va = e2 - ua * c12 - (wa - va) * c23 + ua * c31;

    glm::vec2 q0 = n1_2d + n2_2d + n3_2d;
    if (glm::dot(q0, q0) < 1e-12f) q0 = c12 + c23 + c31;
    glm::vec2 q0Perp(-q0.y, q0.x);
    float du = -glm::dot(q0Perp, J_va);
    float dv =  glm::dot(q0Perp, J_ua);
    float dLen2 = du * du + dv * dv;
    if (dLen2 > 1e-30f) {
        float invDLen = glm::inversesqrt(dLen2);
        du *= invDLen;
        dv *= invDLen;
    }
    float dw = -du - dv;
    glm::vec2 Q0 = -((du * dv) * c12 + (dv * dw) * c23 + (dw * du) * c31);
    if (glm::dot(Q0, Q0) > 1e-30f) {
        glm::vec2 qPerp(-Q0.y, Q0.x);
        du = -glm::dot(qPerp, J_va);
        dv =  glm::dot(qPerp, J_ua);
        dLen2 = du * du + dv * dv;
        if (dLen2 > 1e-30f) {
            float invDLen = glm::inversesqrt(dLen2);
            du *= invDLen;
            dv *= invDLen;
        }
        dw = -du - dv;
    }
    glm::vec2 L_vec = J_ua * du + J_va * dv;
    glm::vec2 Q_vec = -((du * dv) * c12 + (dv * dw) * c23 + (dw * du) * c31);
    float q2 = glm::dot(Q_vec, Q_vec);
    float q1 = glm::dot(L_vec, Q_vec);
    float q0_val = glm::dot(Fa, Q_vec);

    float sNear = 0.0f;
    float sFar  = 0.0f;
    if (q2 > 1e-30f) {
        float sFold = -0.5f * q1 / q2;
        float disc  = glm::max(0.0f, q1 * q1 - 4.0f * q2 * q0_val);
        float ds    = 0.5f * std::sqrt(disc) / q2;
        float signFold = (sFold >= 0.0f) ? 1.0f : -1.0f;
        sNear = sFold - signFold * ds;
        sFar  = sFold + signFold * ds;
    }

    const float BARY_EPS = -1e-4f;
    float ub = ua + sFar * du;
    float vb = va + sFar * dv;

    // If (ua, va) stalled near the fold or converged to an out-of-bounds ghost root, jump it to sNear (or centroid) and polish
    bool uaOut = (ua < BARY_EPS) | (va < BARY_EPS) | (wa < BARY_EPS);
    if (glm::dot(Fa, Fa) > tol2 || uaOut) {
        if (std::abs(sNear) > 1e-4f) {
            ua = glm::clamp(ua + sNear * du, 0.0f, 1.0f);
            va = glm::clamp(va + sNear * dv, 0.0f, 1.0f);
            if (ua + va > 1.0f) { float invS = 1.0f / (ua + va); ua *= invS; va *= invS; }
        } else {
            ua = 1.0f / 3.0f;
            va = 1.0f / 3.0f;
        }
        runNewton(ua, va);
    }

    auto setVertexSeed = [&](float& uSeed, float& vSeed) {
        float z1 = glm::dot(v1 - localOrigin, rayDir);
        float z2 = glm::dot(v2 - localOrigin, rayDir);
        float z3 = glm::dot(v3 - localOrigin, rayDir);
        if (z1 <= z2 && z1 <= z3)      { uSeed = 0.8f; vSeed = 0.1f; }
        else if (z2 <= z1 && z2 <= z3) { uSeed = 0.1f; vSeed = 0.8f; }
        else                           { uSeed = 0.1f; vSeed = 0.1f; }
    };

    if (std::abs(ub - ua) + std::abs(vb - va) < 1e-2f) {
        // Fallback if both candidates are identical: seed near the front-most vertex
        setVertexSeed(ub, vb);
    } else {
        ub = glm::clamp(ub, 0.0f, 1.0f);
        vb = glm::clamp(vb, 0.0f, 1.0f);
        if (ub + vb > 1.0f) { float invS = 1.0f / (ub + vb); ub *= invS; vb *= invS; }
    }
    runNewton(ub, vb);

    // If (ub, vb) converged to an out-of-bounds ghost root or collapsed back onto (ua, va), retry from front-most vertex
    float wb = 1.0f - ub - vb;
    bool ubOut = (ub < BARY_EPS) | (vb < BARY_EPS) | (wb < BARY_EPS);
    if (ubOut || (std::abs(ub - ua) + std::abs(vb - va) < 1e-2f)) {
        setVertexSeed(ub, vb);
        runNewton(ub, vb);
    }

    // 5. Validate Both Candidates (Scale-Invariant Residual) & Pick Closest Depth `t`
    float n1_z = glm::dot(N1, rayDir);
    float n2_z = glm::dot(N2, rayDir);
    float n3_z = glm::dot(N3, rayDir);

    float c12_z = w12.x * n1_z + w12.y * n2_z;
    float c23_z = w23.x * n2_z + w23.y * n3_z;
    float c31_z = w31.x * n3_z + w31.y * n1_z;

    float t_eps = glm::max(1e-6f, localTMax * 1e-2f);
    bool found = false;
    float bestLocalT = localTMax + t_eps;
    float bestOutU = 0.0f, bestOutV = 0.0f;

    float candU[2] = { ua, ub };
    float candV[2] = { va, vb };
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        float uk = candU[k];
        float vk = candV[k];
        float wk = 1.0f - uk - vk;
        if ((uk < BARY_EPS) | (vk < BARY_EPS) | (wk < BARY_EPS)) continue;

        glm::vec2 finalF = x3 + uk * e1 + vk * e2 - ((uk * vk) * c12 + (vk * wk) * c23 + (wk * uk) * c31);
        if (glm::dot(finalF, finalF) > tol2) continue;

        uk = glm::max(uk, 0.0f);
        vk = glm::max(vk, 0.0f);
        float sumUV = uk + vk;
        if (sumUV > 1.0f) { float invS = 1.0f / sumUV; uk *= invS; vk *= invS; }
        wk = 1.0f - uk - vk;

        glm::vec3 linearPos = uk * v1 + vk * v2 + wk * v3;
        float tk = glm::dot(linearPos - localOrigin, rayDir)
                 - ((uk * vk) * c12_z + (vk * wk) * c23_z + (wk * uk) * c31_z);

        if (tk >= -t_eps && tk <= bestLocalT && (tMin + tk) > 1e-7f) {
            bestLocalT = tk;
            bestOutU = vk;
            bestOutV = wk;
            found = true;
        }
    }

    if (!found) return false;

    outT = tMin + bestLocalT;
    outU = bestOutU;
    outV = bestOutV;
    return true;
}

// =============================================================================
// Analytical 2D Conic-Resultant Solver + 2D Damped Newton Polish
// Finds all valid intersections (u, v) of the ray with the quadratic Nagata patch
// =============================================================================
inline CUDA_HOST_DEVICE
int solveNagata2DAnalytical(
    const glm::vec2& x3,
    const glm::vec2& e1,
    const glm::vec2& e2,
    const glm::vec2& c12,
    const glm::vec2& c23,
    const glm::vec2& c31,
    glm::vec2 outUVs[4])
{
    // F(u, v) = a*u^2 + b*u*v + c*v^2 + d*u + e*v + g = 0
    glm::vec2 a = c31;
    glm::vec2 b = c23 + c31 - c12;
    glm::vec2 c = c23;
    glm::vec2 d = e1 - c31;
    glm::vec2 e = e2 - c23;
    glm::vec2 g = x3;

    // Pivot (u <-> v) so ||c|| >= ||a|| for maximum numerical stability
    bool swapped = false;
    if (glm::dot(a, a) > glm::dot(c, c)) {
        glm::vec2 tmpA = a; a = c; c = tmpA;
        glm::vec2 tmpD = d; d = e; e = tmpD;
        swapped = true;
    }

    auto cross2 = [](const glm::vec2& U, const glm::vec2& V) -> float {
        return glm::fma(U.x, V.y, -U.y * V.x);
    };

    // D_BC(u) = B(u) x C(u) = r3*u^3 + r2*u^2 + r1*u + r0
    float r3 = cross2(b, a);
    float r2 = cross2(b, d) + cross2(e, a);
    float r1 = cross2(b, g) + cross2(e, d);
    float r0 = cross2(e, g);

    float cLen2  = glm::dot(c, c);
    float scale2 = glm::dot(e1, e1) + glm::dot(e2, e2);

    float k4, k3, k2, k1, k0;
    float cLen = 0.0f;
    glm::vec2 cHat(0.0f);
    float p1 = 0.0f, p0 = 0.0f, q2 = 0.0f, q1 = 0.0f, q0 = 0.0f;

    if (cLen2 > 1e-16f * scale2 && cLen2 > 1e-30f) {
        cLen = std::sqrt(cLen2);
        cHat = c * (1.0f / cLen);
        p1 = cross2(cHat, b);
        p0 = cross2(cHat, e);
        q2 = cross2(cHat, a);
        q1 = cross2(cHat, d);
        q0 = cross2(cHat, g);

        // Normalized Bezout resultant: P(u) = D_AB(u)*D_BC(u) - ||c|| * D_AC(u)^2
        k4 = p1 * r3 - cLen * q2 * q2;
        k3 = p1 * r2 + p0 * r3 - 2.0f * cLen * q2 * q1;
        k2 = p1 * r1 + p0 * r2 - cLen * (q1 * q1 + 2.0f * q2 * q0);
        k1 = p1 * r0 + p0 * r1 - 2.0f * cLen * q1 * q0;
        k0 = p0 * r0 - cLen * q0 * q0;
    } else {
        // Both c23 and c31 are zero -> system is already linear in v!
        k4 = 0.0f;
        k3 = r3;
        k2 = r2;
        k1 = r1;
        k0 = r0;
    }

    float maxK = glm::max(glm::max(std::abs(k4), std::abs(k3)),
                          glm::max(std::abs(k2), glm::max(std::abs(k1), std::abs(k0))));
    if (maxK > 1e-30f) {
        float invK = 1.0f / maxK;
        k4 *= invK; k3 *= invK; k2 *= invK; k1 *= invK; k0 *= invK;
    }

    auto evalP = [&](float u) -> float {
        return (((k4 * u + k3) * u + k2) * u + k1) * u + k0;
    };
    auto evaldP = [&](float u) -> float {
        return ((4.0f * k4 * u + 3.0f * k3) * u + 2.0f * k2) * u + k1;
    };
    auto evalddP = [&](float u) -> float {
        return (12.0f * k4 * u + 6.0f * k3) * u + 2.0f * k2;
    };

    const float uMin = -0.02f;
    const float uMax =  1.02f;

    // 1. Inflection points: P''(u) = 0  <=>  6*k4*u^2 + 3*k3*u + k2 = 0
    float infl[4];
    int nInfl = 0;
    infl[nInfl++] = uMin;
    {
        float A2 = 6.0f * k4;
        float B2 = 3.0f * k3;
        float C2 = k2;
        float rA = -10.0f, rB = -10.0f;
        if (std::abs(A2) <= 1e-7f) {
            if (std::abs(B2) > 1e-7f) rA = -C2 / B2;
        } else {
            float disc = B2 * B2 - 4.0f * A2 * C2;
            if (disc >= 0.0f) {
                float s = (B2 >= 0.0f) ? 1.0f : -1.0f;
                float q = -0.5f * (B2 + s * std::sqrt(disc));
                rA = q / A2;
                if (std::abs(q) > 1e-12f) rB = C2 / q;
            }
        }
        if (rA > rB) { float t = rA; rA = rB; rB = t; }
        if (rA > uMin && rA < uMax) infl[nInfl++] = rA;
        if (rB > uMin && rB < uMax && std::abs(rB - rA) > 1e-6f) infl[nInfl++] = rB;
    }
    infl[nInfl++] = uMax;

    // 2. Extrema / Turning points: P'(u) = 0 on each monotone interval of P'(u)
    float crit[5];
    int nCrit = 0;
    crit[nCrit++] = uMin;
    for (int i = 0; i < nInfl - 1; ++i) {
        float lo = infl[i], hi = infl[i + 1];
        float flo = evaldP(lo), fhi = evaldP(hi);
        if (flo * fhi <= 0.0f && (std::abs(flo) > 1e-12f || std::abs(fhi) > 1e-12f)) {
            float u = 0.5f * (lo + hi);
            #pragma unroll
            for (int iter = 0; iter < 5; ++iter) {
                float f  = evaldP(u);
                float df = evalddP(u);
                if (f * flo <= 0.0f) { hi = u; fhi = f; }
                else                 { lo = u; flo = f; }
                float step = (std::abs(df) > 1e-10f) ? (f / df) : 0.0f;
                float uNext = u - step;
                u = (uNext > lo && uNext < hi) ? uNext : 0.5f * (lo + hi);
            }
            if (u > crit[nCrit - 1] + 1e-6f && u < uMax - 1e-6f) {
                crit[nCrit++] = u;
            }
        }
    }
    crit[nCrit++] = uMax;

    // 3. Isolate roots of P(u) = 0 on each monotone bracket + grazing extrema
    float candU[6];
    int nCandU = 0;
    for (int i = 0; i < nCrit - 1; ++i) {
        float lo = crit[i], hi = crit[i + 1];
        float flo = evalP(lo), fhi = evalP(hi);
        if (flo * fhi <= 0.0f) {
            float u = 0.5f * (lo + hi);
            #pragma unroll
            for (int iter = 0; iter < 6; ++iter) {
                float f  = evalP(u);
                float df = evaldP(u);
                if (f * flo <= 0.0f) { hi = u; fhi = f; }
                else                 { lo = u; flo = f; }
                float step = (std::abs(df) > 1e-10f) ? (f / df) : 0.0f;
                float uNext = u - step;
                u = (uNext > lo && uNext < hi) ? uNext : 0.5f * (lo + hi);
            }
            candU[nCandU++] = u;
        }
    }
    // Include near-zero turning points (grazing / silhouette double-roots)
    for (int i = 1; i < nCrit - 1; ++i) {
        if (std::abs(evalP(crit[i])) < 1e-2f && nCandU < 6) {
            candU[nCandU++] = crit[i];
        }
    }
    if (nCandU == 0) {
        candU[nCandU++] = 1.0f / 3.0f;
    }

    // 4. Recover v for each candidate u and polish (u, v) in 2D
    int numRoots = 0;
    float tol2 = 1e-9f * scale2;

    for (int idx = 0; idx < nCandU; ++idx) {
        float u0 = candU[idx];
        glm::vec2 B_vec = b * u0 + e;
        glm::vec2 C_vec = (a * u0 + d) * u0 + g;

        float vCands[2];
        int nVCands = 0;

        if (cLen > 0.0f) {
            float denom = p1 * u0 + p0;
            float numer = (q2 * u0 + q1) * u0 + q0;
            float bNorm = std::sqrt(glm::dot(B_vec, B_vec));
            if (std::abs(denom) > 1e-4f * bNorm && std::abs(denom) > 1e-12f) {
                vCands[nVCands++] = -numer / denom;
            } else {
                // Degenerate fiber (two roots sharing the same u): solve 1D quadratic along cHat
                float qa = cLen;
                float qb = glm::dot(cHat, B_vec);
                float qc = glm::dot(cHat, C_vec);
                float disc = qb * qb - 4.0f * qa * qc;
                if (disc >= 0.0f) {
                    float s = (qb >= 0.0f) ? 1.0f : -1.0f;
                    float q = -0.5f * (qb + s * std::sqrt(disc));
                    vCands[nVCands++] = q / qa;
                    if (std::abs(q) > 1e-12f) vCands[nVCands++] = qc / q;
                } else {
                    vCands[nVCands++] = -0.5f * qb / qa;
                }
            }
        } else {
            float bLen2 = glm::dot(B_vec, B_vec);
            vCands[nVCands++] = (bLen2 > 1e-20f) ? (-glm::dot(B_vec, C_vec) / bLen2) : (1.0f / 3.0f);
        }

        for (int vi = 0; vi < nVCands; ++vi) {
            float u = swapped ? vCands[vi] : u0;
            float v = swapped ? u0 : vCands[vi];

            // Clamp initial candidate to neighborhood of triangle before polish
            u = glm::clamp(u, -0.25f, 1.25f);
            v = glm::clamp(v, -0.25f, 1.25f);

            // 2-step Levenberg-Marquardt / Damped Newton polish (handles grazing folds detJ -> 0)
            #pragma unroll
            for (int iter = 0; iter < 2; ++iter) {
                float w = 1.0f - u - v;
                glm::vec2 F   = x3 + u * e1 + v * e2 - ((u * v) * c12 + (v * w) * c23 + (w * u) * c31);
                glm::vec2 J_u = e1 - v * c12 + v * c23 - (w - u) * c31;
                glm::vec2 J_v = e2 - u * c12 - (w - v) * c23 + u * c31;

                float detJ   = glm::fma(J_u.x, J_v.y, -J_u.y * J_v.x);
                float normJ2 = glm::dot(J_u, J_u) + glm::dot(J_v, J_v);

                if (std::abs(detJ) > 1e-5f * normJ2 && std::abs(detJ) > 1e-20f) {
                    float invDetJ = 1.0f / detJ;
                    u -= glm::fma( J_v.y, F.x, -J_v.x * F.y) * invDetJ;
                    v -= glm::fma(-J_u.y, F.x,  J_u.x * F.y) * invDetJ;
                } else if (normJ2 > 1e-20f) {
                    float lam = 1e-4f * normJ2;
                    float a11 = glm::dot(J_u, J_u) + lam;
                    float a12 = glm::dot(J_u, J_v);
                    float a22 = glm::dot(J_v, J_v) + lam;
                    float b1  = glm::dot(J_u, F);
                    float b2  = glm::dot(J_v, F);
                    float invDet = 1.0f / (a11 * a22 - a12 * a12);
                    u -= ( a22 * b1 - a12 * b2) * invDet;
                    v -= (-a12 * b1 + a11 * b2) * invDet;
                }
                u = glm::clamp(u, -0.5f, 1.5f);
                v = glm::clamp(v, -0.5f, 1.5f);
            }

            float w = 1.0f - u - v;
            const float BARY_EPS = -1e-4f;
            if ((u < BARY_EPS) | (v < BARY_EPS) | (w < BARY_EPS)) continue;

            glm::vec2 finalF = x3 + u * e1 + v * e2 - ((u * v) * c12 + (v * w) * c23 + (w * u) * c31);
            if (glm::dot(finalF, finalF) > tol2) continue;

            // Clamp tiny boundary epsilon excursions onto [0, 1] simplex
            u = glm::max(u, 0.0f);
            v = glm::max(v, 0.0f);
            float sumUV = u + v;
            if (sumUV > 1.0f) { float invS = 1.0f / sumUV; u *= invS; v *= invS; }

            // Deduplicate against already found roots
            bool dup = false;
            for (int r = 0; r < numRoots; ++r) {
                glm::vec2 diff = outUVs[r] - glm::vec2(u, v);
                if (glm::dot(diff, diff) < 1e-6f) { dup = true; break; }
            }
            if (!dup && numRoots < 4) {
                outUVs[numRoots++] = glm::vec2(u, v);
            }
        }
    }

    return numRoots;
}

// =============================================================================
// Analytical Intersection for CPU/Host NagataPatch
// =============================================================================
inline CUDA_HOST_DEVICE
bool intersectNagataPatchAnalytical(const Ray_& ray, const NagataPatch& patch, HitRecord& hit)
{
    glm::vec3 E1, E2;
    buildBasisGPU(ray.dir, E1, E2);
    auto proj2D = [&](const glm::vec3& v) -> glm::vec2 {
        return glm::vec2(glm::dot(v, E1), glm::dot(v, E2));
    };

    glm::vec2 x3  = proj2D(patch.X3 - ray.origin);
    glm::vec2 e1  = proj2D(patch.X1 - patch.X3);
    glm::vec2 e2  = proj2D(patch.X2 - patch.X3);
    glm::vec2 c12 = proj2D(patch.C12);
    glm::vec2 c23 = proj2D(patch.C23);
    glm::vec2 c31 = proj2D(patch.C31);

    glm::vec2 x1 = x3 + e1;
    glm::vec2 x2 = x3 + e2;
    glm::vec2 b12 = 0.5f * (x1 + x2 - c12);
    glm::vec2 b23 = 0.5f * (x2 + x3 - c23);
    glm::vec2 b31 = 0.5f * (x3 + x1 - c31);

    glm::vec2 hMin = glm::min(glm::min(glm::min(x1, x2), x3), glm::min(glm::min(b12, b23), b31));
    glm::vec2 hMax = glm::max(glm::max(glm::max(x1, x2), x3), glm::max(glm::max(b12, b23), b31));
    if ((hMin.x > 0.0f) | (hMax.x < 0.0f) | (hMin.y > 0.0f) | (hMax.y < 0.0f)) {
        return false;
    }

    glm::vec2 roots[4];
    int numRoots = solveNagata2DAnalytical(x3, e1, e2, c12, c23, c31, roots);

    bool found = false;
    float bestT = ray.tMax;
    glm::vec2 bestUV(0.0f);
    glm::vec3 bestPos(0.0f);

    for (int i = 0; i < numRoots; ++i) {
        float u = roots[i].x;
        float v = roots[i].y;
        glm::vec3 pos = patch.evaluatePosition(u, v);
        float t = glm::dot(pos - ray.origin, ray.dir);
        if (t >= ray.tMin && t <= bestT) {
            bestT = t;
            bestUV = roots[i];
            bestPos = pos;
            found = true;
        }
    }

    if (!found) return false;

    hit.t = bestT;
    hit.u = bestUV.x;
    hit.v = bestUV.y;
    hit.position = bestPos;
    hit.normal = patch.evaluateNormal(bestUV.x, bestUV.y);
    return true;
}

// =============================================================================
// Analytical Intersection for Precomputed GPUNagataPatch
// =============================================================================
inline CUDA_HOST_DEVICE
bool intersectNagataGPUAnalytical(
    const glm::vec3& rayOrigin,
    const glm::vec3& rayDir,
    float tMin,
    float tMax,
    const GPUNagataPatch& patch,
    float& outT,
    glm::vec2& outUV)
{
    glm::vec3 E1, E2;
    buildBasisGPU(rayDir, E1, E2);
    auto proj2D = [&](const glm::vec3& v) -> glm::vec2 {
        return glm::vec2(glm::dot(v, E1), glm::dot(v, E2));
    };

    glm::vec2 x3  = proj2D(glm::vec3(patch.X3_pad) - rayOrigin);
    glm::vec2 e1  = proj2D(glm::vec3(patch.X1_pad) - glm::vec3(patch.X3_pad));
    glm::vec2 e2  = proj2D(glm::vec3(patch.X2_pad) - glm::vec3(patch.X3_pad));
    glm::vec2 c12 = proj2D(glm::vec3(patch.C12_pad));
    glm::vec2 c23 = proj2D(glm::vec3(patch.C23_pad));
    glm::vec2 c31 = proj2D(glm::vec3(patch.C31_pad));

    glm::vec2 x1 = x3 + e1;
    glm::vec2 x2 = x3 + e2;
    glm::vec2 b12 = 0.5f * (x1 + x2 - c12);
    glm::vec2 b23 = 0.5f * (x2 + x3 - c23);
    glm::vec2 b31 = 0.5f * (x3 + x1 - c31);

    glm::vec2 hMin = glm::min(glm::min(glm::min(x1, x2), x3), glm::min(glm::min(b12, b23), b31));
    glm::vec2 hMax = glm::max(glm::max(glm::max(x1, x2), x3), glm::max(glm::max(b12, b23), b31));
    if ((hMin.x > 0.0f) | (hMax.x < 0.0f) | (hMin.y > 0.0f) | (hMax.y < 0.0f)) {
        return false;
    }

    glm::vec2 roots[4];
    int numRoots = solveNagata2DAnalytical(x3, e1, e2, c12, c23, c31, roots);
    if (numRoots == 0) return false;

    bool found = false;
    float bestT = tMax;
    glm::vec2 bestUV(0.0f);

    for (int i = 0; i < numRoots; ++i) {
        float u = roots[i].x;
        float v = roots[i].y;
        float w = 1.0f - u - v;
        glm::vec3 hitPos = u * glm::vec3(patch.X1_pad)
                         + v * glm::vec3(patch.X2_pad)
                         + w * glm::vec3(patch.X3_pad)
                         - ((u * v) * glm::vec3(patch.C12_pad)
                          + (v * w) * glm::vec3(patch.C23_pad)
                          + (w * u) * glm::vec3(patch.C31_pad));
        float t = glm::dot(hitPos - rayOrigin, rayDir);
        if (t >= tMin && t <= bestT) {
            bestT = t;
            bestUV = roots[i];
            found = true;
        }
    }

    if (!found) return false;

    outT  = bestT;
    outUV = bestUV;
    return true;
}

// =============================================================================
// Analytical On-the-Fly Nagata Intersection
// =============================================================================
inline CUDA_HOST_DEVICE
bool intersectNagataOnTheFlyAnalytical(
    const glm::vec3& rayOrigin,
    const glm::vec3& rayDir,
    float tMin,
    float tMax,
    const glm::vec3 v1, const glm::vec3 v2, const glm::vec3 v3,
    const glm::vec3 N1, const glm::vec3 N2, const glm::vec3 N3,
    float& outT,
    float& outU,
    float& outV)
{
    // 1. Compute scalar Nagata weights (w_i, w_j) for the 3 edges
    glm::vec2 w12 = computeNagataEdgeWeights(v2 - v1, N1, N2);
    glm::vec2 w23 = computeNagataEdgeWeights(v3 - v2, N2, N3);
    glm::vec2 w31 = computeNagataEdgeWeights(v1 - v3, N3, N1);

    // 2. Project vertices AND normals into 2D Ray Space
    glm::vec3 E1, E2;
    buildBasisGPU(rayDir, E1, E2);
    auto proj2D = [&](const glm::vec3& v) -> glm::vec2 {
        return glm::vec2(glm::dot(v, E1), glm::dot(v, E2));
    };

    glm::vec2 x3 = proj2D(v3 - rayOrigin);
    glm::vec2 e1 = proj2D(v1 - v3);
    glm::vec2 e2 = proj2D(v2 - v3);

    glm::vec2 n1_2d = proj2D(N1);
    glm::vec2 n2_2d = proj2D(N2);
    glm::vec2 n3_2d = proj2D(N3);

    glm::vec2 c12 = w12.x * n1_2d + w12.y * n2_2d;
    glm::vec2 c23 = w23.x * n2_2d + w23.y * n3_2d;
    glm::vec2 c31 = w31.x * n3_2d + w31.y * n1_2d;

    // 3. 2D Bezier Control-Hull Early Reject
    glm::vec2 x1 = x3 + e1;
    glm::vec2 x2 = x3 + e2;
    glm::vec2 b12 = 0.5f * (x1 + x2 - c12);
    glm::vec2 b23 = 0.5f * (x2 + x3 - c23);
    glm::vec2 b31 = 0.5f * (x3 + x1 - c31);

    glm::vec2 hMin = glm::min(glm::min(glm::min(x1, x2), x3), glm::min(glm::min(b12, b23), b31));
    glm::vec2 hMax = glm::max(glm::max(glm::max(x1, x2), x3), glm::max(glm::max(b12, b23), b31));
    if ((hMin.x > 0.0f) | (hMax.x < 0.0f) | (hMin.y > 0.0f) | (hMax.y < 0.0f)) {
        return false;
    }

    // 4. Solve 2D Conic Intersection Analytically + Polish
    glm::vec2 roots[4];
    int numRoots = solveNagata2DAnalytical(x3, e1, e2, c12, c23, c31, roots);
    if (numRoots == 0) return false;

    // 5. Evaluate 1D Ray Depth `t` for all valid roots and pick the closest in [tMin, tMax]
    float n1_z = glm::dot(N1, rayDir);
    float n2_z = glm::dot(N2, rayDir);
    float n3_z = glm::dot(N3, rayDir);

    float c12_z = w12.x * n1_z + w12.y * n2_z;
    float c23_z = w23.x * n2_z + w23.y * n3_z;
    float c31_z = w31.x * n3_z + w31.y * n1_z;

    bool found = false;
    float bestT = tMax;
    float bestBaryV2 = 0.0f;
    float bestBaryV3 = 0.0f;

    for (int i = 0; i < numRoots; ++i) {
        float u = roots[i].x;
        float v = roots[i].y;
        float w = 1.0f - u - v;

        glm::vec3 linearPos = u * v1 + v * v2 + w * v3;
        float t = glm::dot(linearPos - rayOrigin, rayDir)
                - ((u * v) * c12_z + (v * w) * c23_z + (w * u) * c31_z);

        if (t >= tMin && t <= bestT) {
            bestT = t;
            bestBaryV2 = v;
            bestBaryV3 = w;
            found = true;
        }
    }

    if (!found) return false;

    outT = bestT;
    outU = bestBaryV2;
    outV = bestBaryV3;
    return true;
}

// =============================================================================
// On-the-Fly Analytical Surface Normal Evaluator (dX/du x dX/dv)
// Takes the (baryV2, baryV3) convention returned in (outU, outV)
// =============================================================================
inline CUDA_HOST_DEVICE
glm::vec3 evaluateNagataNormalOnTheFly(
    const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3,
    const glm::vec3& N1, const glm::vec3& N2, const glm::vec3& N3,
    float baryV2, float baryV3)
{
    float v = baryV2;
    float w = baryV3;
    float u = 1.0f - v - w;

    glm::vec3 E1 = v1 - v3;
    glm::vec3 E2 = v2 - v3;
    glm::vec2 w12 = computeNagataEdgeWeights(v2 - v1, N1, N2);
    glm::vec2 w23 = computeNagataEdgeWeights(v3 - v2, N2, N3);
    glm::vec2 w31 = computeNagataEdgeWeights(E1,      N3, N1);

    float wu = w - u;
    float wv = w - v;

    float a1 = v * w12.x + wu * w31.y;
    float a2 = v * (w12.y - w23.x);
    float a3 = wu * w31.x - v * w23.y;
    glm::vec3 dXdu = E1 - (a1 * N1 + a2 * N2 + a3 * N3);

    float b1 = u * (w12.x - w31.y);
    float b2 = u * w12.y + wv * w23.x;
    float b3 = wv * w23.y - u * w31.x;
    glm::vec3 dXdv = E2 - (b1 * N1 + b2 * N2 + b3 * N3);

    return glm::normalize(glm::cross(dXdu, dXdv));
}

// Uses the same GPUVertex and computeNagataEdgeWeights from the Intersection Shader
inline void evaluateNagataClosestHit(
    uint32_t primID,
    glm::vec2 uv,
    const glm::uvec3* indexBuffer,
    const GPUVertex* vertexBuffer,
    glm::vec3& outHitPos,
    glm::vec3& outAnalyticNormal,
    glm::vec3& outShadingNormal)
{
    float u = uv.x;
    float v = uv.y;
    float w = 1.0f - u - v;

    // 1. Fetch the 3 vertices (Already hot in L1/L2 cache from the winning hit!)
    glm::uvec3 tri = indexBuffer[primID];
    GPUVertex v1 = vertexBuffer[tri.x];
    GPUVertex v2 = vertexBuffer[tri.y];
    GPUVertex v3 = vertexBuffer[tri.z];

    glm::vec3 N1 = GPUVertex::unpackOctNormal(v1.octNormal);
    glm::vec3 N2 = GPUVertex::unpackOctNormal(v2.octNormal);
    glm::vec3 N3 = GPUVertex::unpackOctNormal(v3.octNormal);

    // 2. Compute scalar edge weights (w_i, w_j)
    glm::vec3 E1 = v1.pos - v3.pos;
    glm::vec3 E2 = v2.pos - v3.pos;
    glm::vec2 w12 = computeNagataEdgeWeights(v2.pos - v1.pos, N1, N2);
    glm::vec2 w23 = computeNagataEdgeWeights(v3.pos - v2.pos, N2, N3);
    glm::vec2 w31 = computeNagataEdgeWeights(E1,              N3, N1);

    // 3. Evaluate 3D Hit Position using combined normal weights (No 3D C_ij needed!)
    float p1 = (u * v) * w12.x + (w * u) * w31.y;
    float p2 = (u * v) * w12.y + (v * w) * w23.x;
    float p3 = (v * w) * w23.y + (w * u) * w31.x;

    outHitPos = (u * v1.pos + v * v2.pos + w * v3.pos) - (p1 * N1 + p2 * N2 + p3 * N3);

    // 4. Evaluate Exact Analytic Tangents dX/du and dX/dv via scalar normal weights
    float wu = w - u; // 1 - 2u - v
    float wv = w - v; // 1 - u - 2v

    // Scalar weights for dX/du
    float a1 = v * w12.x + wu * w31.y;
    float a2 = v * (w12.y - w23.x);
    float a3 = wu * w31.x - v * w23.y;
    glm::vec3 dXdu = E1 - (a1 * N1 + a2 * N2 + a3 * N3);

    // Scalar weights for dX/dv
    float b1 = u * (w12.x - w31.y);
    float b2 = u * w12.y + wv * w23.x;
    float b3 = wv * w23.y - u * w31.x;
    glm::vec3 dXdv = E2 - (b1 * N1 + b2 * N2 + b3 * N3);

    // Exact geometric normal of the quadratic Nagata surface
    outAnalyticNormal = glm::normalize(glm::cross(dXdu, dXdv));

    // Smooth G1-continuous Phong normal for crease-free specular highlights
    outShadingNormal  = glm::normalize(u * N1 + v * N2 + w * N3);
}


enum class BoxIntersectionState {
    OUTSIDE,    // Entire curved patch is strictly outside the AABB
    INSIDE,     // Entire curved patch is inside (or on the boundary of) the AABB
    INTERSECTS  // Curved patch partially overlaps / pierces the AABB boundary
};


// Standard 13-Axis Separating Axis Theorem (SAT) for Flat Triangle vs. AABB
inline CUDA_HOST_DEVICE bool flatTriangleIntersectsAABB(
    const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const AABB& box)
{
    glm::vec3 c = 0.5f * (box.min + box.max);
    glm::vec3 e = 0.5f * (box.max - box.min);

    glm::vec3 a = v0 - c, b = v1 - c, d = v2 - c;
    glm::vec3 f0 = b - a, f1 = d - b, f2 = a - d;

    // 1. Test the 9 cross-product axes (3 triangle edges x 3 box basis axes)
    auto testEdgeAxis = [&](float p0, float p1, float p2, float r) -> bool {
        float max_p = glm::max(glm::max(p0, p1), p2);
        float min_p = glm::min(glm::min(p0, p1), p2);
        return glm::max(-max_p, min_p) <= r;
    };

    const glm::vec3 edges[3] = { f0, f1, f2 };
    for (int i = 0; i < 3; ++i) {
        const glm::vec3& f = edges[i];
        // Cross X-axis (0, -f.z, f.y)
        if (!testEdgeAxis(-a.y*f.z + a.z*f.y, -b.y*f.z + b.z*f.y, -d.y*f.z + d.z*f.y,
                          e.y * glm::abs(f.z) + e.z * glm::abs(f.y))) return false;
        // Cross Y-axis (f.z, 0, -f.x)
        if (!testEdgeAxis( a.x*f.z - a.z*f.x,  b.x*f.z - b.z*f.x,  d.x*f.z - d.z*f.x,
                          e.x * glm::abs(f.z) + e.z * glm::abs(f.x))) return false;
        // Cross Z-axis (-f.y, f.x, 0)
        if (!testEdgeAxis(-a.x*f.y + a.y*f.x, -b.x*f.y + b.y*f.x, -d.x*f.y + d.y*f.x,
                          e.x * glm::abs(f.y) + e.y * glm::abs(f.x))) return false;
    }

    // 2. Test the 3 AABB face normals
    if (glm::max(glm::max(a.x, b.x), d.x) < -e.x || glm::min(glm::min(a.x, b.x), d.x) > e.x) return false;
    if (glm::max(glm::max(a.y, b.y), d.y) < -e.y || glm::min(glm::min(a.y, b.y), d.y) > e.y) return false;
    if (glm::max(glm::max(a.z, b.z), d.z) < -e.z || glm::min(glm::min(a.z, b.z), d.z) > e.z) return false;

    // 3. Test the triangle plane normal
    glm::vec3 n = glm::cross(f0, f1);
    float r = e.x * glm::abs(n.x) + e.y * glm::abs(n.y) + e.z * glm::abs(n.z);
    float s = glm::dot(n, a);
    return glm::abs(s) <= r;
}

// =============================================================================
// Main 3-State Classifier: Nagata Patch vs. Axis-Aligned Bounding Box
// =============================================================================
inline CUDA_HOST_DEVICE BoxIntersectionState classifyNagataBoxIntersection(
    const NagataPatch& patch, const AABB& box, int maxSubdivDepth = 4)
{
    // -------------------------------------------------------------------------
    // STEP 1: Fast Control-Hull AABB Check
    // -------------------------------------------------------------------------
    AABB hullBox = patch.controlHullAABB();
    if (!box.overlapsBox(hullBox)) {
        return BoxIntersectionState::OUTSIDE; // Guaranteed disjoint
    }
    if (box.containsBox(hullBox)) {
        return BoxIntersectionState::INSIDE;  // Entire control hull is inside box
    }

    // -------------------------------------------------------------------------
    // STEP 2: Exact Closed-Form `INSIDE` Test & Surface Point Sampling
    // -------------------------------------------------------------------------
    bool anyPointInside = box.containsPoint(patch.X1) ||
                          box.containsPoint(patch.X2) ||
                          box.containsPoint(patch.X3);

    bool anyExtremumInside = false;
    AABB exactPatchBox = patch.exactAABB(box, anyExtremumInside);
    anyPointInside = anyPointInside || anyExtremumInside;

    // Necessary & sufficient condition for INSIDE: exact patch AABB is inside box
    if (box.containsBox(exactPatchBox)) {
        return BoxIntersectionState::INSIDE;
    }
    // If even the exact tight AABB of the patch misses the box, it's OUTSIDE
    if (!box.overlapsBox(exactPatchBox)) {
        return BoxIntersectionState::OUTSIDE;
    }
    // If ANY surface point is inside the box, and the patch is not completely
    // INSIDE (checked above), then the continuous surface MUST cross the boundary!
    if (anyPointInside) {
        return BoxIntersectionState::INTERSECTS;
    }

    // -------------------------------------------------------------------------
    // STEP 3: Iterative Midpoint Subdivision to Resolve OUTSIDE vs. INTERSECTS
    // (Reached only when all corners/extrema are outside, but AABB overlaps)
    // -------------------------------------------------------------------------
    struct SubPatchNode {
        glm::vec3 v1, v2, v3;
        int depth;
    };

    // Max stack size for depth 4 is 3 * 4 + 1 = 13 nodes
    maxSubdivDepth = glm::min(maxSubdivDepth, 10);
    SubPatchNode stack[32];
    int stackPtr = 0;
    stack[stackPtr++] = { patch.X1, patch.X2, patch.X3, 0 };

    while (stackPtr > 0) {
        SubPatchNode node = stack[--stackPtr];

        // Scale curvature by 4^(-depth) = 1 / (1 << (2 * depth))
        float cScale = 1.0f / static_cast<float>(1 << (2 * node.depth));
        glm::vec3 c12 = patch.C12 * cScale;
        glm::vec3 c23 = patch.C23 * cScale;
        glm::vec3 c31 = patch.C31 * cScale;

        // Compute the 6 Bezier control points of this sub-patch
        glm::vec3 b12 = 0.5f * (node.v1 + node.v2 - c12);
        glm::vec3 b23 = 0.5f * (node.v2 + node.v3 - c23);
        glm::vec3 b31 = 0.5f * (node.v3 + node.v1 - c31);

        AABB subHull = {
            glm::min(glm::min(node.v1, node.v2), glm::min(node.v3, glm::min(b12, glm::min(b23, b31)))),
            glm::max(glm::max(node.v1, node.v2), glm::max(node.v3, glm::max(b12, glm::max(b23, b31))))
        };

        // Prune sub-patch if its tightened control hull misses the box
        if (!box.overlapsBox(subHull)) {
            continue;
        }

        // Evaluate the 3 true surface edge midpoints M_ij
        glm::vec3 m12 = 0.5f * (0.5f * (node.v1 + node.v2) + b12);
        glm::vec3 m23 = 0.5f * (0.5f * (node.v2 + node.v3) + b23);
        glm::vec3 m31 = 0.5f * (0.5f * (node.v3 + node.v1) + b31);

        // If any midpoint or the entire sub-hull lands inside the box -> INTERSECTS!
        if (box.containsPoint(m12) || box.containsPoint(m23) ||
            box.containsPoint(m31) || box.containsBox(subHull))
        {
            return BoxIntersectionState::INTERSECTS;
        }

        // At max depth, the sub-patch curvature is < 1/256th of original -> run exact SAT
        if (node.depth >= maxSubdivDepth) {
            glm::vec3 sagitta = 0.35f * glm::max(glm::max(glm::abs(c12), glm::abs(c23)), glm::abs(c31));
            AABB inflatedBox = { box.min - sagitta, box.max + sagitta };
            if (flatTriangleIntersectsAABB(node.v1, node.v2, node.v3, inflatedBox)) {
                return BoxIntersectionState::INTERSECTS;
            }
            continue;
        }

        // Push the 4 child sub-patches (notice how (m23, m31, m12) preserves C12, C23, C31 order!)
        int nextDepth = node.depth + 1;
        stack[stackPtr++] = { node.v1, m12, m31, nextDepth };
        stack[stackPtr++] = { m12, node.v2, m23, nextDepth };
        stack[stackPtr++] = { m31, m23, node.v3, nextDepth };
        stack[stackPtr++] = { m23, m31, m12, nextDepth };
    }

    return BoxIntersectionState::OUTSIDE;
}

//
// soa::cuda_device_vector<NagataPatch> build_nagata_patches(soa::soa_view<Triangle> triangles_d,soa::soa_view<Vertex> vertices_d)
// {
//     soa::cuda_device_vector<NagataPatch> result(triangles_d.size());
//
//
// }