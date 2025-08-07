#include "bvh.h"
#include <cstdio>
namespace
{

    constexpr int BIN_COUNT = 16;

    struct Bin
    {
        AABB bounds;
        int count = 0;
    };

    static float4 f4(const float3 &a) { return make_float4(a.x, a.y, a.z, 0.0f); }

    int sah_split(std::vector<PrimRef> &prims, int start, int end, int axis, float split_pos)
    {
        auto mid = std::partition(prims.begin() + start, prims.begin() + end,
                                  [axis, split_pos](const PrimRef &p)
                                  { return ((&p.centroid.x)[axis] < split_pos); });
        return int(mid - prims.begin());
    }

    float find_sah_split(std::vector<PrimRef> &prims, int start, int end, int &best_axis, int &best_split)
    {
        float best_cost = FLT_MAX;
        best_axis = -1;
        best_split = -1;
        PrimInfo pinfo(prims, start, end);
        if (pinfo.size() <= 1)
            return best_cost;
        for (int axis = 0; axis < 3; ++axis)
        {
            Bin bins[BIN_COUNT] = {};
            AABB cb = pinfo.centroid_bounds;
            float min_c = (&cb.lower.x)[axis], max_c = (&cb.upper.x)[axis];
            if (max_c - min_c < 1e-5f)
                continue;
            float scale = BIN_COUNT / (max_c - min_c);
            for (int i = start; i < end; ++i)
            {
                int b = int(((&prims[i].centroid.x)[axis] - min_c) * scale);
                b = std::min(std::max(b, 0), BIN_COUNT - 1);
                bins[b].count++;
                bins[b].bounds.grow(prims[i].bounds);
            }
            AABB left_bounds[BIN_COUNT], right_bounds[BIN_COUNT];
            int left_count[BIN_COUNT] = {}, right_count[BIN_COUNT] = {};
            AABB lb;
            int lc = 0;
            for (int i = 0; i < BIN_COUNT; ++i)
            {
                lb.grow(bins[i].bounds);
                lc += bins[i].count;
                left_bounds[i] = lb;
                left_count[i] = lc;
            }
            AABB rb;
            int rc = 0;
            for (int i = BIN_COUNT - 1; i >= 0; --i)
            {
                rb.grow(bins[i].bounds);
                rc += bins[i].count;
                right_bounds[i] = rb;
                right_count[i] = rc;
            }
            for (int i = 0; i < BIN_COUNT - 1; ++i)
            {
                float cost = left_count[i] * left_bounds[i].surface_area() + right_count[i + 1] * right_bounds[i + 1].surface_area();
                if (cost < best_cost)
                {
                    best_cost = cost;
                    best_axis = axis;
                    float split_val = min_c + (i + 1) / scale;
                    best_split = sah_split(prims, start, end, axis, split_val);
                }
            }
        }
        return best_cost;
    }

    static int build_node(
        std::vector<BVHPackedNode> &out,
        std::vector<PrimRef> &prims,
        int start,
        int end)
    {
        // Remember index, push placeholder
        const int node_index = (int)out.size();
        out.emplace_back(); // placeholder

        // Range info
        PrimInfo info(prims, start, end);

        // LEAF
        if (info.size() <= 1)
        {
            BVHPackedNode &node = out[node_index]; // safe here (no recursion before)
            node.flags = 1;                        // leaf

            node.left.lower = make_float4(info.bounds.lower.x,
                                          info.bounds.lower.y,
                                          info.bounds.lower.z, 0.0f);
            node.left.upper = make_float4(info.bounds.upper.x,
                                          info.bounds.upper.y,
                                          info.bounds.upper.z, 0.0f);

            node.prim_index = (info.size() == 1) ? prims[start].tri_index : -1;
            node.prim_count = (info.size() == 1) ? 1 : 0;

            node.right.lower = make_float4(0.f, 0.f, 0.f, 0.f);
            node.right.upper = make_float4(0.f, 0.f, 0.f, 0.f);
            return node_index;
        }

        // INTERNAL: choose split
        int axis = -1, split_index = -1;
        (void)find_sah_split(prims, start, end, axis, split_index);
        if (split_index <= start || split_index >= end)
            split_index = start + (info.size() / 2);

        // Recurse (may reallocate 'out')
        const int left_idx = build_node(out, prims, start, split_index);
        const int right_idx = build_node(out, prims, split_index, end);

        // Re-acquire reference AFTER recursion
        BVHPackedNode &node = out[node_index];
        node.flags = 0;
        node.child = (unsigned)left_idx;
        node.right_child = (unsigned)right_idx;

        // Tight bounds for each child range
        PrimInfo left_info(prims, start, split_index);
        PrimInfo right_info(prims, split_index, end);

        node.left.lower = make_float4(left_info.bounds.lower.x,
                                      left_info.bounds.lower.y,
                                      left_info.bounds.lower.z, 0.0f);
        node.left.upper = make_float4(left_info.bounds.upper.x,
                                      left_info.bounds.upper.y,
                                      left_info.bounds.upper.z, 0.0f);

        node.right.lower = make_float4(right_info.bounds.lower.x,
                                       right_info.bounds.lower.y,
                                       right_info.bounds.lower.z, 0.0f);
        node.right.upper = make_float4(right_info.bounds.upper.x,
                                       right_info.bounds.upper.y,
                                       right_info.bounds.upper.z, 0.0f);

        return node_index;
    }

} // anon

void build_sah_bvh(const float3 *points, const uint3 *indices, int num_tris, std::vector<BVHPackedNode> &out_nodes)
{
    std::vector<PrimRef> prims(num_tris);
    for (int i = 0; i < num_tris; ++i)
    {
        uint3 tri = indices[i];
        float3 v0 = points[tri.x];
        float3 v1 = points[tri.y];
        float3 v2 = points[tri.z];
        PrimRef &p = prims[i];
        p.tri_index = i;
        p.bounds.grow(v0);
        p.bounds.grow(v1);
        p.bounds.grow(v2);
        p.centroid = (v0 + v1 + v2) * (1.0f / 3.0f);
    }

    // DEBUG: dump overall prim bounds
    AABB dbg_all;
    for (int i = 0; i < num_tris; ++i)
    {
        dbg_all.grow(prims[i].bounds);
    }
    printf("DBG prim bounds: lower=(%f,%f,%f) upper=(%f,%f,%f)  tris=%d\n",
           dbg_all.lower.x, dbg_all.lower.y, dbg_all.lower.z,
           dbg_all.upper.x, dbg_all.upper.y, dbg_all.upper.z, num_tris);

    out_nodes.clear();
    out_nodes.reserve(std::max(1, 2 * num_tris)); // avoid realloc while building
    build_node(out_nodes, prims, 0, num_tris);
}
