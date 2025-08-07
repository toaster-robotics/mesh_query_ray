#include "bvh.h"

namespace
{

    constexpr int BIN_COUNT = 16;

    struct Bin
    {
        AABB bounds;
        int count = 0;
    };

    int sah_split(
        std::vector<PrimRef> &prims,
        int start, int end,
        int axis,
        float split_pos)
    {
        auto mid = std::partition(prims.begin() + start, prims.begin() + end,
                                  [axis, split_pos](const PrimRef &p)
                                  {
                                      return ((&p.centroid.x)[axis] < split_pos);
                                  });

        return int(mid - prims.begin());
    }

    float find_sah_split(
        std::vector<PrimRef> &prims,
        int start, int end,
        int &best_axis,
        int &best_split)
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
            AABB centroid_bounds = pinfo.centroid_bounds;
            float min_c = (&centroid_bounds.lower.x)[axis];
            float max_c = (&centroid_bounds.upper.x)[axis];

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
                float cost =
                    left_count[i] * left_bounds[i].surface_area() +
                    right_count[i + 1] * right_bounds[i + 1].surface_area();

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

    void build_node(
        std::vector<BVHPackedNode> &out,
        std::vector<PrimRef> &prims,
        int start,
        int end)
    {
        int node_index = int(out.size());
        out.emplace_back();

        PrimInfo info(prims, start, end);

        if (info.size() <= 1)
        {
            BVHPackedNode &leaf = out[node_index];
            leaf.left.lower = info.bounds.lower;
            leaf.left.upper = info.bounds.upper;
            leaf.flags = 1;
            leaf.prim_index = prims[start].tri_index;
            leaf.prim_count = 1;
            return;
        }

        int axis, split_index;
        float split_cost = find_sah_split(prims, start, end, axis, split_index);

        if (split_index <= start || split_index >= end) // fallback
            split_index = start + (info.size() / 2);

        build_node(out, prims, start, split_index);
        build_node(out, prims, split_index, end);

        BVHPackedNode &node = out[node_index];
        node.flags = 0;
        node.left.lower = out[out[node_index + 1].flags ? node_index + 1 : out[node_index + 1].child].left.lower;
        node.left.upper = out[out[node_index + 1].flags ? node_index + 1 : out[node_index + 1].child].left.upper;
        node.right.lower = out.back().left.lower;
        node.right.upper = out.back().left.upper;
        node.child = node_index + 1;
        node.right_child = (unsigned int)(out.size() - 1);
    }

} // anonymous namespace

void build_bvh_sah(
    const float3 *points,
    const int *indices,
    int num_tris,
    std::vector<BVHPackedNode> &out_nodes)
{
    std::vector<PrimRef> prims(num_tris);

    for (int i = 0; i < num_tris; ++i)
    {
        int i0 = indices[i * 3 + 0];
        int i1 = indices[i * 3 + 1];
        int i2 = indices[i * 3 + 2];

        float3 v0 = points[i0];
        float3 v1 = points[i1];
        float3 v2 = points[i2];

        PrimRef &prim = prims[i];
        prim.tri_index = i;
        prim.bounds.grow(v0);
        prim.bounds.grow(v1);
        prim.bounds.grow(v2);
        prim.centroid = (v0 + v1 + v2) * (1.0f / 3.0f);
    }

    out_nodes.clear();
    build_node(out_nodes, prims, 0, num_tris);
}
