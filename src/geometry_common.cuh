#pragma once
#include <cuda.h>
#define GLM_FORCE_CUDA
#define GLM_FORCE_PURE
#include <glm/glm.hpp>
#include <cstdint>

#undef min
#undef max



#ifdef NDEBUG
    #define TEST_ASSERT(x)
#else
    #define TEST_ASSERT(x) assert((x))
#endif


__host__ __device__ __forceinline__ bool intersectRayTriangle(
    const glm::vec3 orig, const glm::vec3 dir,
    const glm::vec3 v0, const glm::vec3 v1, const glm::vec3 v2,
    float& t, float& u, float& v)
{
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;

    glm::vec3 pvec = glm::cross(dir, edge2);
    float det = glm::dot(edge1, pvec);

    // Compute the inverse determinant.
    // If det is 0, this results in Infinity, which is perfectly fine.
    // The subsequent math will yield Inf or NaN, which will fail the boolean checks below.
    float invDet = 1.0f / det;

    glm::vec3 tvec = orig - v0;
    glm::vec3 qvec = glm::cross(tvec, edge1);

    // Speculatively compute barycentric coordinates and distance
    float u_tmp = glm::dot(tvec, pvec) * invDet;
    float v_tmp = glm::dot(dir, qvec) * invDet;
    float t_tmp = glm::dot(edge2, qvec) * invDet;

    // Use bitwise AND (&) instead of logical AND (&&).
    // This prevents the compiler from generating short-circuit branching.
    // The conditions map directly to sequential predicate assignments in PTX.
    bool is_valid = (glm::abs(det) > 1e-8f) &
                    (u_tmp >= 0.0f) &
                    (u_tmp <= 1.0f) &
                    (v_tmp >= 0.0f) &
                    (u_tmp + v_tmp <= 1.0f) &
                    (t_tmp > 1e-8f);

    // Only write to the output variables if the intersection is valid.
    if (is_valid) {
        t = t_tmp;
        u = u_tmp;
        v = v_tmp;
    }

    return is_valid;
}

__device__ __host__
inline glm::vec3 side(glm::vec3 p, glm::vec3 dir,glm::vec3 boxCenter, float boxSize)
{
    using namespace glm;

    vec3 checkDirection= glm::sign(dir);


    vec3 remainedBox=checkDirection*(boxSize*0.5f)+boxCenter-p;
    vec3 qubeizedDir= glm::abs(dir/remainedBox);//can be NAN
    float maxCoord=
        glm::max(glm::max(qubeizedDir[0],qubeizedDir[1]),qubeizedDir[2]);
    vec3 mask=glm::sign(qubeizedDir-maxCoord)+1.0f;
    return mask*checkDirection;
}

__device__ __host__
inline glm::vec3 ray_triangle_intersection(glm::vec3 pos, glm::vec3 dir, glm::vec3 triangle[3])
{
    return glm::vec3(0);
}

enum class TriangleCubeRelation
{
    Inside,
    Outside,
    Intersecting
};


__host__ __device__ __forceinline__
TriangleCubeRelation find_triangle_cube_relation_approximate(glm::vec3 cube_center, float cube_half_size, const glm::vec3 triangle_in[3])
{
    glm::vec3 triangle[3];
    //get relative coordinates
    for (int i=0;i<3;++i)
    {
        triangle[i]=triangle_in[i]-cube_center;
    }

    bool inside=true;
    for (int i=0;i<3;++i)
    {
        inside= inside && glm::all(glm::lessThanEqual(glm::abs(triangle[i]),glm::vec3(cube_half_size)));
    }
    if (inside)
        return TriangleCubeRelation::Inside;

    glm::vec3 aabb_max=glm::max(glm::max(triangle[0],triangle[1]),triangle[2]);
    glm::vec3 aabb_min=glm::min(glm::min(triangle[0],triangle[1]),triangle[2]);
    if (    !(glm::all(glm::lessThan(aabb_min,glm::vec3(cube_half_size))) &&
            glm::all(glm::greaterThan(aabb_max,glm::vec3(-cube_half_size))))         )
    {
        return TriangleCubeRelation::Outside;
    }


    return TriangleCubeRelation::Intersecting;
}

__host__ __device__ __forceinline__
TriangleCubeRelation find_triangle_cube_relation_precise(glm::vec3 cube_center, float cube_half_size,
    glm::vec3 v0, glm::vec3 v1, glm::vec3 v2)
{
    using namespace glm;
    v0-=cube_center;
    v1-=cube_center;
    v2-=cube_center;


    bool inside=true;
    inside= inside && glm::all(glm::lessThanEqual(glm::abs(v0),glm::vec3(cube_half_size)));
    inside= inside && glm::all(glm::lessThanEqual(glm::abs(v1),glm::vec3(cube_half_size)));
    inside= inside && glm::all(glm::lessThanEqual(glm::abs(v2),glm::vec3(cube_half_size)));
    if (inside)
        return TriangleCubeRelation::Inside;

    glm::vec3 aabb_max=glm::max(glm::max(v0,v1),v2);
    glm::vec3 aabb_min=glm::min(glm::min(v0,v1),v2);
    if (    !(glm::all(glm::lessThanEqual(aabb_min,glm::vec3(cube_half_size))) &&
            glm::all(glm::greaterThanEqual(aabb_max,glm::vec3(-cube_half_size))))         )
    {
        return TriangleCubeRelation::Outside;
    }


    vec3 H = vec3(cube_half_size);

    // Precompute triangle edges
    vec3 e0 = v1 - v0;
    vec3 e1 = v2 - v1;
    vec3 e2 = v0 - v2;


    // ==========================================
    // TEST 2: Triangle Face Normal (1 axis)
    // ==========================================
    vec3 N = cross(e0, e1);

    // Project the box half-extents onto the triangle normal
    float r = dot(H, abs(N));
    // Distance of the triangle plane from the origin
    float d = dot(N, v0);

    bool sepNorm = abs(d) > r;

    // ==========================================
    // TEST 3: Edge Cross Products (9 axes)
    // ==========================================
    // THE SWIZZLE TRICK:
    // cross(v, e) elegantly produces the exact projection values for all 3
    // cross-product axes (XxE, YxE, ZxE) in a single hardware instruction.
    // We then swizzle the edge components to calculate the 3 radius bounds simultaneously.

    // Edge 0 (tests v0 and v2)
    vec3 P0_e0 = cross(v0, e0); // Projections of v0 (and v1, they are equal)
    vec3 P2_e0 = cross(v2, e0); // Projections of v2
    vec3 R_e0  = H * abs(vec3(e0.z,e0.x,e0.y)) + H * abs(vec3(e0.y,e0.z,e0.x));
    bool sepE0 = any(greaterThan(min(P0_e0, P2_e0), R_e0)) ||
                 any(lessThan(max(P0_e0, P2_e0), -R_e0));

    // Edge 1 (tests v1 and v0)
    vec3 P1_e1 = cross(v1, e1);
    vec3 P0_e1 = cross(v0, e1);
    vec3 R_e1  = H * abs(vec3(e1.z,e1.x,e1.y)) + H * abs(vec3(e1.y,e1.z,e1.x));
    bool sepE1 = any(greaterThan(min(P1_e1, P0_e1), R_e1)) ||
                 any(lessThan(max(P1_e1, P0_e1), -R_e1));

    // Edge 2 (tests v2 and v1)
    vec3 P2_e2 = cross(v2, e2);
    vec3 P1_e2 = cross(v1, e2);
    vec3 R_e2  = H * abs(vec3(e2.z,e2.x,e2.y)) + H * abs(vec3(e2.y,e2.z,e2.x));
    bool sepE2 = any(greaterThan(min(P2_e2, P1_e2), R_e2)) ||
                 any(lessThan(max(P2_e2, P1_e2), -R_e2));



    if (sepNorm || sepE0 || sepE1 || sepE2)
        return TriangleCubeRelation::Outside;

    return TriangleCubeRelation::Intersecting;
}


__host__ __device__ __forceinline__
int id_vec_to_int(glm::ivec3 id_vec)
{
    return id_vec.x+id_vec.y*2+id_vec.z*4;
}

__host__ __device__ __forceinline__
glm::ivec3 int_to_id_vec(int id)
{
    return {(id&1)!=0?1:0, (id&2)!=0?1:0, (id&4)!=0?1:0};
}

struct BoxInfo
{
    glm::vec3 center;
    int depth;
    uint32_t id;
    float size;

    BoxInfo descend(int child_offset, uint32_t child_id)
    {
        auto offset=glm::vec3(int_to_id_vec(child_offset))*2.f-1.f;

        return BoxInfo{
            .center = center+offset*(size/4),
            .depth = depth+1,
            .id=child_id,
            .size=size/2
        };
    }
};


__device__ __host__ __forceinline__
uint32_t deinterleave_left(uint64_t x)
{
    union double_uint32
    {
        uint64_t m_data;
        struct {
            uint32_t m_lo; // Bits  0..31
            uint32_t m_hi; // Bits 32..63
        };
    } x2{.m_data = x};


    x2.m_data &= 0x9249249249249249ULL;
    x2.m_data = (x2.m_data | (x2.m_data <<  2)) & 0xC30C30C30C30C30CULL;
    x2.m_data=  (x2.m_data | (x2.m_data <<  4)) & 0xF00F00F00F00F00FULL;
    x2.m_data = (x2.m_data | (x2.m_data <<  8)) & 0xFF0000FF0000FF00ULL;
    x2.m_data = (x2.m_data | (x2.m_data << 16)) & 0xFFFF00000000FFFFULL;
    x2.m_lo =   (x2.m_hi | x2.m_lo) & 0xFFFFFC00UL;

    return x2.m_lo;
}

__device__ __host__ __forceinline__ uint64_t spread_left_aligned(uint32_t val)
{

    uint64_t x = static_cast<uint64_t>(val) << 32;

    // 2. Spread the bits rightwards
    x = (x | (x >> 32)) & 0xFFFF00000000FFFFULL;
    x = (x | (x >> 16)) & 0xFF0000FF0000FF00ULL;
    x = (x | (x >>  8)) & 0xF00F00F00F00F00FULL;
    x = (x | (x >>  4)) & 0xC30C30C30C30C30CULL;
    x = (x | (x >>  2)) & 0x9249249249249249ULL;

    return x;
}


__device__ __host__ __forceinline__
float bits_division_to_float(uint32_t x)
{
    return static_cast<float>(x) * 0x1p-32f;
}

__device__ __host__ __forceinline__
uint32_t bits_division_from_float(float x)
{
    return static_cast<uint32_t>(x * 0x1p+32f);
}

struct BoxStack
{
    union {
        uint64_t m_data = 0;
        struct {
            uint32_t m_lo; // Bits  0..31 (contains depth bits)
            uint32_t m_hi; // Bits 32..63
        };
    };

    static constexpr uint32_t total_bits=sizeof(m_data)*8;
    static constexpr uint32_t depth_number_of_bits=5;
    static constexpr uint32_t depth_bits=(1<<depth_number_of_bits)-1; //00011111
    static constexpr uint32_t outside_bit=0b10000000;//64th bit or 32th in the m_lo
    static constexpr uint32_t invalid_bit=0b00100000;//6th bit

    static constexpr uint32_t mask_3bits=0b111;
    static constexpr uint32_t max_depth= (total_bits-depth_number_of_bits-2)/3;

    __host__ __device__ [[nodiscard]] __forceinline__
    int depth() const
    {
        return static_cast<int>(m_lo & depth_bits);
    }

    __host__ __device__ [[nodiscard]] __forceinline__
    uint32_t branch(int level) const
    {
        TEST_ASSERT(level>0 && level<=depth());
        return static_cast<uint32_t>( m_data >> (63-level*3) ) & mask_3bits;
    }

    __host__ __device__ [[nodiscard]] __forceinline__
    uint32_t branch() const
    {
        return branch(depth());
    }

    __host__ __device__ __forceinline__
    void descend(uint32_t child)
    {
        TEST_ASSERT(depth()<max_depth);
        ++m_lo;//increment depth
        move(child);
    }

    __host__ __device__ __forceinline__
    void ascend()
    {
        TEST_ASSERT(depth()>0);
        --m_lo;
    }

    __host__ __device__ __forceinline__
    void move(uint32_t child)
    {
        TEST_ASSERT(child<8);
        uint32_t shift = 63 - depth() * 3;
        m_data = (m_data & ~(static_cast<uint64_t>(mask_3bits) << shift))
               | (static_cast<uint64_t>(child) << shift);
    }

    __host__ __device__  __forceinline__ [[nodiscard]]
    glm::vec3 get_center(int target_depth) const
    {
        TEST_ASSERT(target_depth>=0 && target_depth<=this->depth());
        uint32_t last_bit=1<<(31-target_depth);
        uint32_t clean_mask = -(last_bit << 1);

        glm::vec3  center;
        center.z = bits_division_to_float(deinterleave_left(m_data << 1) & clean_mask | last_bit);
        center.y = bits_division_to_float(deinterleave_left(m_data << 2) & clean_mask | last_bit);
        center.x = bits_division_to_float(deinterleave_left(m_data << 3) & clean_mask | last_bit);

        return center;
    }

    __host__ __device__  __forceinline__ [[nodiscard]]
    glm::vec3 get_center() const
    {
        return  get_center(depth());
    }

    __host__ __device__  __forceinline__ [[nodiscard]]
    static BoxStack getContainingBoxStack(glm::vec3 point, int depth)
    {
        BoxStack result;
        // {
        //     auto x_bits=spread_left_aligned(bits_division_from_float(point.x)>>3);
        //     result.m_data|=x_bits;
        // }
        // {
        //     auto y_bits=spread_left_aligned(bits_division_from_float(point.y)>>2);
        //     result.m_data|=y_bits;
        // }
        // {
        //     auto z_bits=spread_left_aligned(bits_division_from_float(point.z) >> 1 );
        //     result.m_data|=z_bits;
        // }

        return result;
    }

    __host__ __device__ __forceinline__ [[nodiscard]]
    float box_size() const
    {
        return static_cast<float>(1) / (1<<depth());
    }

    __host__ __device__ __forceinline__ [[nodiscard]]
    float half_box_size() const
    {
        return static_cast<float>(1) / (1<<(depth()+1));
    }

    __host__ __device__ __forceinline__ [[nodiscard]]
    float quarter_box_size() const
    {
        return static_cast<float>(1) / (1<<(depth()+2));
    }

    __host__ __device__ __forceinline__
    void set_outside(bool outside)
    {
        if (outside)
            m_hi= outside_bit;
        else
            m_hi&= (~outside_bit);
    }

    __host__ __device__ __forceinline__ [[nodiscard]]
    bool is_outside() const
    {
        return m_hi&outside_bit;
    }

    __host__ __device__ __forceinline__
    void set_invalid(bool invalid)
    {
        if (invalid)
            m_lo|= invalid_bit;
        else
            m_lo&= (~invalid_bit);
    }

    __host__ __device__ __forceinline__ [[nodiscard]]
    bool is_invalid() const
    {
        return m_lo&invalid_bit;
    }


    __host__ __device__ __forceinline__ [[nodiscard]]
    uint64_t begin() const
    {
        uint64_t mask= (static_cast<uint64_t>(1)<<(63 - depth()*3))-1;
        return m_data& ~mask;
    }

    __host__ __device__ __forceinline__ [[nodiscard]]
    uint64_t end() const
    {
        uint64_t bit = (static_cast<uint64_t>(1)<<(63 - depth()*3));
        return (m_data & ~(bit-1)) + bit;
    }

    __host__ __device__
    void print() const
    {
        printf("Depth=%d  ",depth());
        auto center=get_center(depth());
        printf("Center=(%f, %f, %f)   ",center.x, center.y, center.z);
        printf("Is invalid: %s    ",is_invalid()?"True":"False");
        printf("Is outside: %s    ",is_outside()?"True":"False");
        // printf("bits: %s    ",std::format("{:b}", m_data).c_str() );
        // printf("begin: %s    ",std::format("{:b}", begin()).c_str() );
        // printf("end: %s    ",std::format("{:b}", end()).c_str() );

        printf("Stack: ");

        for (int i=1;i<=depth();++i)
        {
            printf("%d  ",branch(i));
        }
        printf("\n\n");
    }
};