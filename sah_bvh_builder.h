// sah_bvh_builder.h
#pragma once

#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <cmath>
#include <float.h>
#include "vector_math.h"

struct BVHPackedNodeHalf
{
    float3 lower;
    float3 upper;
};

struct BVHPackedNode
{
    BVHPackedNodeHalf left;
    BVHPackedNodeHalf right;

    union
    {
        unsigned int child;
        int prim_index;
    };

    union
    {
        unsigned int right_child;
        int prim_count;
    };

    unsigned int flags; // 0 = internal, 1 = leaf
};

struct AABB
{
    float3 lower = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 upper = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    void grow(const float3 &p)
    {
        lower.x = fminf(lower.x, p.x);
        lower.y = fminf(lower.y, p.y);
        lower.z = fminf(lower.z, p.z);
        upper.x = fmaxf(upper.x, p.x);
        upper.y = fmaxf(upper.y, p.y);
        upper.z = fmaxf(upper.z, p.z);
    }

    void grow(const AABB &box)
    {
        grow(box.lower);
        grow(box.upper);
    }

    float surface_area() const
    {
        float3 extent = upper - lower;
        return 2.0f * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x);
    }

    int max_extent() const
    {
        float3 extent = upper - lower;
        if (extent.x > extent.y && extent.x > extent.z)
            return 0;
        if (extent.y > extent.z)
            return 1;
        return 2;
    }
};

struct PrimRef
{
    int tri_index;
    AABB bounds;
    float3 centroid;
};

struct PrimInfo
{
    AABB bounds;
    AABB centroid_bounds;
    int start;
    int end;

    PrimInfo(const std::vector<PrimRef> &prims, int s, int e)
        : start(s), end(e)
    {
        for (int i = s; i < e; ++i)
        {
            bounds.grow(prims[i].bounds);
            centroid_bounds.grow(prims[i].centroid);
        }
    }

    int size() const { return end - start; }
};

void build_bvh_sah(
    const float3 *points,
    const int *indices,
    int num_tris,
    std::vector<BVHPackedNode> &out_nodes);
